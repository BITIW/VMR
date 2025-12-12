use rayon::prelude::*;
use std::fmt;
use std::sync::atomic::{AtomicU8, Ordering};

mod rans;
use rans::{rans_compress_blob_auto_mt, rans_decompress_blob_auto_mt, RansError};

const VMR_MAGIC: &[u8; 4] = b"VMRQ";
const HEADER_SIZE: usize = 16;

// ====== knobs ======
const BLOCK_FRAMES: usize = 2048;
const MAX_MODE: u8 = 8;

// QR split tuning
const MAX_K: u8 = 12; // remainder bits per residual (0..12)

// Header flags (byte 6)
const FLAG_STEREO_SIDE: u8 = 1 << 0;
const FLAG_SHUFFLE16: u8 = 1 << 1; // residual u16 stored as [low...][high...]

// payload flags (byte inside channel payload)
const PAYLOAD_FLAG_SHUFFLE16: u8 = 1 << 0;

static VERBOSITY: AtomicU8 = AtomicU8::new(0);

#[inline]
fn v() -> u8 {
    VERBOSITY.load(Ordering::Relaxed)
}
#[inline]
fn vlog(level: u8, msg: impl FnOnce() -> String) {
    if v() >= level {
        eprintln!("[vmr] {}", msg());
    }
}

#[derive(Debug)]
pub enum VmrError {
    InvalidInput(String),
    CodecError(String),
    FormatError(String),
}
impl fmt::Display for VmrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VmrError::InvalidInput(s) => write!(f, "Invalid input: {s}"),
            VmrError::CodecError(s) => write!(f, "Codec error: {s}"),
            VmrError::FormatError(s) => write!(f, "Format error: {s}"),
        }
    }
}
impl std::error::Error for VmrError {}

pub type Result<T> = std::result::Result<T, VmrError>;

#[derive(Debug, Clone, Copy)]
pub struct EncodeStats {
    pub frames: usize,
    pub channels: u8,
    pub sample_rate: u32,
    pub raw_size: usize,
    pub payload_size: usize,    // uncompressed channel payloads total (meta+residual bytes)
    pub compressed_size: usize, // meta_blob + k_blob + qL + q0[k] + q1[k] + q2[k] + r_bits
}

pub fn set_verbosity(level: u8) {
    VERBOSITY.store(level, Ordering::Relaxed);
    rans::set_verbosity(level);
}

pub fn vmr_encode(pcm: &[u8], channels: u8, sample_rate: u32) -> Result<Vec<u8>> {
    Ok(vmr_encode_with_stats(pcm, channels, sample_rate)?.0)
}

pub fn vmr_encode_with_stats(
    pcm: &[u8],
    channels: u8,
    sample_rate: u32,
) -> Result<(Vec<u8>, EncodeStats)> {
    if channels == 0 || channels > 2 {
        return Err(VmrError::InvalidInput("Only mono/stereo supported".into()));
    }
    if pcm.len() % (2 * channels as usize) != 0 {
        return Err(VmrError::InvalidInput(
            "PCM length must be aligned to 16-bit * channels".into(),
        ));
    }

    let frames = pcm.len() / (2 * channels as usize);
    let chn = channels as usize;

    // interleaved -> planar i16
    let mut ch_data: Vec<Vec<i16>> = (0..channels)
        .map(|_| Vec::with_capacity(frames))
        .collect();

    for f in 0..frames {
        for ch in 0..channels {
            let i = (f * chn + ch as usize) * 2;
            ch_data[ch as usize].push(i16::from_le_bytes([pcm[i], pcm[i + 1]]));
        }
    }

    // Flags
    let mut flags: u8 = 0;

    // Stereo decorrelation: Side if it looks cheaper than Right
    if channels == 2 && frames > 0 {
        let l = &ch_data[0];
        let r = &ch_data[1];

        let mut e_r: i64 = 0;
        let mut e_s: i64 = 0;
        for i in 0..frames {
            e_r += (r[i] as i32).abs() as i64;
            e_s += (r[i].wrapping_sub(l[i]) as i32).abs() as i64;
        }

        if e_s < e_r {
            flags |= FLAG_STEREO_SIDE;
            let (l_slice, r_slice) = ch_data.split_at_mut(1);
            let left = &mut l_slice[0];
            let right = &mut r_slice[0];
            for i in 0..frames {
                right[i] = right[i].wrapping_sub(left[i]);
            }
        }
    }

    // shuffle16 always enabled for residual bytes
    flags |= FLAG_SHUFFLE16;

    // Encode per channel payloads (uncompressed, old internal format)
    let payloads: Vec<Vec<u8>> = ch_data
        .par_iter()
        .map(|samples| encode_channel_payload(samples, flags))
        .collect();

    let payload_total: usize = payloads.iter().map(|p| p.len()).sum();

    // Extract meta slices and residual_u16 streams; build QR streams
    let mut meta_sizes: Vec<u32> = Vec::with_capacity(chn);
    let mut meta_stream: Vec<u8> = Vec::new();

    let mut k_stream: Vec<u8> = Vec::new();

    // multi-stream varint decomposition:
    let mut ql_stream: Vec<u8> = Vec::new(); // 0/1/2 per residual (global)

    // ====== NEW: split q0/q1/q2 by k ======
    let kmax = MAX_K as usize;
    let mut q0_by_k: Vec<Vec<u8>> = (0..=kmax).map(|_| Vec::new()).collect();
    let mut q1_by_k: Vec<Vec<u8>> = (0..=kmax).map(|_| Vec::new()).collect();
    let mut q2_by_k: Vec<Vec<u8>> = (0..=kmax).map(|_| Vec::new()).collect();

    let mut r_bits = BitWriter::new();

    let mut total_blocks: usize = 0;
    let mut total_residuals: usize = 0;
    let mut k_hist = [0usize; (MAX_K as usize) + 1];

    // debug stats for -vv
    let mut q1_total: usize = 0;
    let mut q2_total: usize = 0;
    let mut q0_counts_by_k = vec![0usize; kmax + 1];
    let mut q1_counts_by_k = vec![0usize; kmax + 1];
    let mut q2_counts_by_k = vec![0usize; kmax + 1];

    for p in &payloads {
        let pm = parse_payload_meta(p)?;
        if !pm.shuffle {
            return Err(VmrError::FormatError("Expected shuffle16 residual layout".into()));
        }

        meta_sizes.push(pm.meta_end as u32);
        meta_stream.extend_from_slice(&p[..pm.meta_end]);

        let residual_len = pm.residual_len;
        total_residuals += residual_len;
        total_blocks += pm.blocks;

        // residual bytes (shuffle): [low..][high..]
        let rb = &p[pm.meta_end..pm.meta_end + residual_len * 2];
        let low = &rb[..residual_len];
        let high = &rb[residual_len..];

        // residual_u16 (zigzag-coded residuals)
        let mut uvals: Vec<u16> = Vec::with_capacity(residual_len);
        for i in 0..residual_len {
            uvals.push((low[i] as u16) | ((high[i] as u16) << 8));
        }

        // per-block: choose k, then emit q-streams and r(bits)
        let mut rp = 0usize;
        for b in 0..pm.blocks {
            let rc = pm.res_counts[b];
            let slice = &uvals[rp..rp + rc];

            let k = choose_k_for_block(slice);
            k_stream.push(k);
            k_hist[k as usize] += 1;

            let kk = k as usize;
            let k_u32 = k as u32;
            let mask: u16 = if k == 0 { 0 } else { (1u16 << k) - 1 };

            for &u in slice {
                let q = (u >> k) as u16;

                // varint-ish split (base-128 groups)
                let q0 = (q & 0x7F) as u8;
                let q1 = ((q >> 7) & 0x7F) as u8;
                let q2 = (q >> 14) as u8; // 0..3 for u16

                let cls: u8 = if q < 0x80 { 0 } else if q < 0x4000 { 1 } else { 2 };

                ql_stream.push(cls);

                q0_by_k[kk].push(q0);
                q0_counts_by_k[kk] += 1;

                if cls >= 1 {
                    q1_by_k[kk].push(q1);
                    q1_counts_by_k[kk] += 1;
                    q1_total += 1;
                }
                if cls == 2 {
                    q2_by_k[kk].push(q2);
                    q2_counts_by_k[kk] += 1;
                    q2_total += 1;
                }

                if k != 0 {
                    let r = (u & mask) as u32;
                    r_bits.write_bits(r, k_u32);
                }
            }

            rp += rc;
        }
        if rp != residual_len {
            return Err(VmrError::FormatError("Residual walk mismatch".into()));
        }
    }

    let r_bits_blob = r_bits.finish();

    vlog(2, || {
        let mut s = String::new();
        s.push_str(&format!(
            "[qr] blocks={} residuals={} qL={} q0(sum)={} q1(sum)={} q2(sum)={} r_bytes={} k_hist=",
            total_blocks,
            total_residuals,
            ql_stream.len(),
            q0_by_k.iter().map(|v| v.len()).sum::<usize>(),
            q1_total,
            q2_total,
            r_bits_blob.len(),
        ));
        for k in 0..=MAX_K {
            let n = k_hist[k as usize];
            if n != 0 {
                s.push_str(&format!("{k}:{n} "));
            }
        }
        if v() >= 3 {
            s.push_str(" q0_by_k=");
            for k in 0..=MAX_K {
                let n = q0_counts_by_k[k as usize];
                if n != 0 {
                    s.push_str(&format!("{k}:{n} "));
                }
            }
            s.push_str(" q1_by_k=");
            for k in 0..=MAX_K {
                let n = q1_counts_by_k[k as usize];
                if n != 0 {
                    s.push_str(&format!("{k}:{n} "));
                }
            }
        }
        s
    });

    // Compress meta/k/q* via rANS (planner inside)
    let meta_blob = rans_compress_blob_auto_mt(&meta_stream, "meta");
    let k_blob = rans_compress_blob_auto_mt(&k_stream, "k");
    let ql_blob = rans_compress_blob_auto_mt(&ql_stream, "qL");

    let mut q0_blobs: Vec<Vec<u8>> = Vec::with_capacity(kmax + 1);
    let mut q1_blobs: Vec<Vec<u8>> = Vec::with_capacity(kmax + 1);
    let mut q2_blobs: Vec<Vec<u8>> = Vec::with_capacity(kmax + 1);

    for kk in 0..=kmax {
        let tag0 = format!("q0[{}]", kk);
        let tag1 = format!("q1[{}]", kk);
        let tag2 = format!("q2[{}]", kk);

        q0_blobs.push(rans_compress_blob_auto_mt(&q0_by_k[kk], &tag0));
        q1_blobs.push(rans_compress_blob_auto_mt(&q1_by_k[kk], &tag1));
        q2_blobs.push(rans_compress_blob_auto_mt(&q2_by_k[kk], &tag2));
    }

    let compressed_size =
        meta_blob.len()
        + k_blob.len()
        + ql_blob.len()
        + q0_blobs.iter().map(|b| b.len()).sum::<usize>()
        + q1_blobs.iter().map(|b| b.len()).sum::<usize>()
        + q2_blobs.iter().map(|b| b.len()).sum::<usize>()
        + r_bits_blob.len();

    // Container layout:
    // [magic4][channels][bps][flags][max_mode]
    // [sample_rate u32][frames u32]
    // [u32 meta_len_ch0..]
    //
    // [u32 meta_blob_len][u32 k_blob_len][u32 ql_blob_len]
    // for k=0..MAX_K:
    //   [u32 q0_blob_len[k]][u32 q1_blob_len[k]][u32 q2_blob_len[k]]
    // [u32 r_bits_len]
    //
    // [meta_blob][k_blob][ql_blob]
    // for k=0..MAX_K:
    //   [q0_blob[k]][q1_blob[k]][q2_blob[k]]
    // [r_bits]
    let mut out = Vec::with_capacity(
        HEADER_SIZE + 4 * chn + (3 + 3 * (kmax + 1) + 1) * 4 + compressed_size
    );

    out.extend_from_slice(VMR_MAGIC);
    out.push(channels);
    out.push(16);
    out.push(flags);
    out.push(MAX_MODE);

    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(frames as u32).to_le_bytes());

    for ch in 0..chn {
        out.extend_from_slice(&meta_sizes[ch].to_le_bytes());
    }

    out.extend_from_slice(&(meta_blob.len() as u32).to_le_bytes());
    out.extend_from_slice(&(k_blob.len() as u32).to_le_bytes());
    out.extend_from_slice(&(ql_blob.len() as u32).to_le_bytes());

    for kk in 0..=kmax {
        out.extend_from_slice(&(q0_blobs[kk].len() as u32).to_le_bytes());
        out.extend_from_slice(&(q1_blobs[kk].len() as u32).to_le_bytes());
        out.extend_from_slice(&(q2_blobs[kk].len() as u32).to_le_bytes());
    }

    out.extend_from_slice(&(r_bits_blob.len() as u32).to_le_bytes());

    out.extend_from_slice(&meta_blob);
    out.extend_from_slice(&k_blob);
    out.extend_from_slice(&ql_blob);

    for kk in 0..=kmax {
        out.extend_from_slice(&q0_blobs[kk]);
        out.extend_from_slice(&q1_blobs[kk]);
        out.extend_from_slice(&q2_blobs[kk]);
    }

    out.extend_from_slice(&r_bits_blob);

    Ok((
        out,
        EncodeStats {
            frames,
            channels,
            sample_rate,
            raw_size: pcm.len(),
            payload_size: payload_total,
            compressed_size,
        },
    ))
}

pub fn vmr_decode(vmr: &[u8]) -> Result<(Vec<u8>, u8, u32)> {
    if vmr.len() < HEADER_SIZE {
        return Err(VmrError::FormatError("VMR too short".into()));
    }
    if &vmr[0..4] != VMR_MAGIC {
        return Err(VmrError::FormatError("Bad VMR magic".into()));
    }

    let channels = vmr[4];
    let bps = vmr[5];
    let flags = vmr[6];
    let max_mode_in_file = vmr[7];

    if bps != 16 {
        return Err(VmrError::FormatError("Only 16-bit supported".into()));
    }
    if channels == 0 || channels > 2 {
        return Err(VmrError::FormatError("Invalid channels".into()));
    }
    if max_mode_in_file > MAX_MODE {
        return Err(VmrError::FormatError(
            "File requires higher MAX_MODE than decoder supports".into(),
        ));
    }
    if (flags & FLAG_SHUFFLE16) == 0 {
        return Err(VmrError::FormatError("Expected shuffle16 flag".into()));
    }

    let sample_rate = u32::from_le_bytes([vmr[8], vmr[9], vmr[10], vmr[11]]);
    let frames = u32::from_le_bytes([vmr[12], vmr[13], vmr[14], vmr[15]]) as usize;

    let chn = channels as usize;
    let mut off = HEADER_SIZE;

    let kmax = MAX_K as usize;

    let need_table_bytes = 4 * chn + (3 + 3 * (kmax + 1) + 1) * 4;
    if vmr.len() < off + need_table_bytes {
        return Err(VmrError::FormatError("VMR too short for tables".into()));
    }

    let mut meta_sizes = vec![0usize; chn];
    for ch in 0..chn {
        meta_sizes[ch] = read_u32(vmr, &mut off)?;
    }

    let meta_blob_len = read_u32(vmr, &mut off)?;
    let k_blob_len = read_u32(vmr, &mut off)?;
    let ql_blob_len = read_u32(vmr, &mut off)?;

    let mut q0_lens = vec![0usize; kmax + 1];
    let mut q1_lens = vec![0usize; kmax + 1];
    let mut q2_lens = vec![0usize; kmax + 1];

    for kk in 0..=kmax {
        q0_lens[kk] = read_u32(vmr, &mut off)?;
        q1_lens[kk] = read_u32(vmr, &mut off)?;
        q2_lens[kk] = read_u32(vmr, &mut off)?;
    }

    let r_bits_len = read_u32(vmr, &mut off)?;

    let mut need_total = meta_blob_len + k_blob_len + ql_blob_len + r_bits_len;
    for kk in 0..=kmax {
        need_total += q0_lens[kk] + q1_lens[kk] + q2_lens[kk];
    }

    if vmr.len() < off + need_total {
        return Err(VmrError::FormatError("VMR truncated in blobs".into()));
    }

    let meta_blob = &vmr[off..off + meta_blob_len];
    off += meta_blob_len;
    let k_blob = &vmr[off..off + k_blob_len];
    off += k_blob_len;
    let ql_blob = &vmr[off..off + ql_blob_len];
    off += ql_blob_len;

    let mut q0_blobs: Vec<&[u8]> = Vec::with_capacity(kmax + 1);
    let mut q1_blobs: Vec<&[u8]> = Vec::with_capacity(kmax + 1);
    let mut q2_blobs: Vec<&[u8]> = Vec::with_capacity(kmax + 1);

    for kk in 0..=kmax {
        let b0 = &vmr[off..off + q0_lens[kk]];
        off += q0_lens[kk];
        let b1 = &vmr[off..off + q1_lens[kk]];
        off += q1_lens[kk];
        let b2 = &vmr[off..off + q2_lens[kk]];
        off += q2_lens[kk];

        q0_blobs.push(b0);
        q1_blobs.push(b1);
        q2_blobs.push(b2);
    }

    let r_bits_data = &vmr[off..off + r_bits_len];

    let meta_stream = rans_decompress_blob_auto_mt(meta_blob).map_err(map_rans_err)?;
    let k_stream = rans_decompress_blob_auto_mt(k_blob).map_err(map_rans_err)?;
    let ql_stream = rans_decompress_blob_auto_mt(ql_blob).map_err(map_rans_err)?;

    let mut q0_by_k: Vec<Vec<u8>> = Vec::with_capacity(kmax + 1);
    let mut q1_by_k: Vec<Vec<u8>> = Vec::with_capacity(kmax + 1);
    let mut q2_by_k: Vec<Vec<u8>> = Vec::with_capacity(kmax + 1);

    for kk in 0..=kmax {
        q0_by_k.push(rans_decompress_blob_auto_mt(q0_blobs[kk]).map_err(map_rans_err)?);
        q1_by_k.push(rans_decompress_blob_auto_mt(q1_blobs[kk]).map_err(map_rans_err)?);
        q2_by_k.push(rans_decompress_blob_auto_mt(q2_blobs[kk]).map_err(map_rans_err)?);
    }

    let meta_total: usize = meta_sizes.iter().sum();
    if meta_stream.len() != meta_total {
        return Err(VmrError::FormatError("meta_stream size mismatch".into()));
    }

    // Compute expected blocks/residuals from meta headers
    let mut total_blocks = 0usize;
    let mut total_residuals = 0usize;

    let mut meta_off = 0usize;
    for ch in 0..chn {
        let mlen = meta_sizes[ch];
        if mlen < 15 || meta_stream.len() < meta_off + mlen {
            return Err(VmrError::FormatError("meta slice truncated".into()));
        }
        let ms = &meta_stream[meta_off..meta_off + mlen];
        let blocks = u32::from_le_bytes([ms[6], ms[7], ms[8], ms[9]]) as usize;
        let residual_len = u32::from_le_bytes([ms[10], ms[11], ms[12], ms[13]]) as usize;
        total_blocks += blocks;
        total_residuals += residual_len;
        meta_off += mlen;
    }

    if k_stream.len() != total_blocks {
        return Err(VmrError::FormatError("k_stream length mismatch".into()));
    }
    if ql_stream.len() != total_residuals {
        return Err(VmrError::FormatError("qL length mismatch".into()));
    }

    // Rebuild channel payloads and decode them
    let mut payloads: Vec<Vec<u8>> = Vec::with_capacity(chn);

    let mut kpos = 0usize;

    let mut ql_pos = 0usize;
    let mut q0_pos = vec![0usize; kmax + 1];
    let mut q1_pos = vec![0usize; kmax + 1];
    let mut q2_pos = vec![0usize; kmax + 1];

    let mut br = BitReader::new(r_bits_data);

    meta_off = 0usize;

    for ch in 0..chn {
        let mlen = meta_sizes[ch];
        let meta = &meta_stream[meta_off..meta_off + mlen];

        let pm = parse_payload_meta_from_meta_only(meta, frames)?;

        // rebuild residual_u16
        let mut residual_u16: Vec<u16> = Vec::with_capacity(pm.residual_len);

        for b in 0..pm.blocks {
            if kpos >= k_stream.len() {
                return Err(VmrError::FormatError("k_stream underflow".into()));
            }
            let k = k_stream[kpos];
            kpos += 1;
            if k > MAX_K {
                return Err(VmrError::FormatError("Bad k in stream".into()));
            }
            let kk = k as usize;

            let rc = pm.res_counts[b];

            for _ in 0..rc {
                if ql_pos >= ql_stream.len() {
                    return Err(VmrError::FormatError("qL underflow".into()));
                }
                let cls = ql_stream[ql_pos];
                ql_pos += 1;
                if cls > 2 {
                    return Err(VmrError::FormatError("Bad qL class".into()));
                }

                // q0 from stream of this k
                let q0 = *q0_by_k[kk]
                    .get(q0_pos[kk])
                    .ok_or_else(|| VmrError::FormatError("q0 underflow".into()))?;
                q0_pos[kk] += 1;

                let mut q: u16 = q0 as u16;

                if cls >= 1 {
                    let q1 = *q1_by_k[kk]
                        .get(q1_pos[kk])
                        .ok_or_else(|| VmrError::FormatError("q1 underflow".into()))?;
                    q1_pos[kk] += 1;
                    q |= (q1 as u16) << 7;
                }
                if cls == 2 {
                    let q2 = *q2_by_k[kk]
                        .get(q2_pos[kk])
                        .ok_or_else(|| VmrError::FormatError("q2 underflow".into()))?;
                    q2_pos[kk] += 1;
                    q |= (q2 as u16) << 14;
                }

                let r = if k == 0 { 0u16 } else { br.read_bits(k as u32)? as u16 };
                let u = (((q as u32) << (k as u32)) | (r as u32)) as u16;
                residual_u16.push(u);
            }
        }

        if residual_u16.len() != pm.residual_len {
            return Err(VmrError::FormatError("Residual rebuild length mismatch".into()));
        }

        let residual_bytes = shuffle16_u16_to_bytes(&residual_u16);

        // payload = meta + residual_bytes (shuffle16)
        let mut p = Vec::with_capacity(mlen + residual_bytes.len());
        p.extend_from_slice(meta);
        p.extend_from_slice(&residual_bytes);

        payloads.push(p);

        meta_off += mlen;
    }

    if kpos != k_stream.len() {
        return Err(VmrError::FormatError("k_stream leftover / mismatch".into()));
    }
    if ql_pos != ql_stream.len() {
        return Err(VmrError::FormatError("qL leftover / mismatch".into()));
    }
    for kk in 0..=kmax {
        if q0_pos[kk] != q0_by_k[kk].len() {
            return Err(VmrError::FormatError(format!("q0[{}] leftover / mismatch", kk)));
        }
        if q1_pos[kk] != q1_by_k[kk].len() {
            return Err(VmrError::FormatError(format!("q1[{}] leftover / mismatch", kk)));
        }
        if q2_pos[kk] != q2_by_k[kk].len() {
            return Err(VmrError::FormatError(format!("q2[{}] leftover / mismatch", kk)));
        }
    }

    // decode channels in parallel
    let mut ch_data: Vec<Vec<i16>> = payloads
        .par_iter()
        .map(|p| decode_channel_payload(p, frames))
        .collect::<Result<Vec<_>>>()?;

    // undo Side
    if channels == 2 && (flags & FLAG_STEREO_SIDE) != 0 {
        let (l_slice, s_slice) = ch_data.split_at_mut(1);
        let left = &mut l_slice[0];
        let side = &mut s_slice[0];
        for i in 0..frames {
            side[i] = left[i].wrapping_add(side[i]);
        }
    }

    // planar -> interleaved PCM LE
    let mut pcm = Vec::with_capacity(frames * chn * 2);
    for f in 0..frames {
        for ch in 0..chn {
            pcm.extend_from_slice(&ch_data[ch][f].to_le_bytes());
        }
    }

    Ok((pcm, channels, sample_rate))
}

/* ===================== Stable k selection ===================== */

#[inline]
fn choose_k_for_block(u: &[u16]) -> u8 {
    let n = u.len();
    if n == 0 {
        return 0;
    }

    // hist[bitlen], bitlen=0 for u=0, else 1..16
    let mut hist = [0u32; 17];
    for &x in u {
        let bl = if x == 0 { 0 } else { (16 - x.leading_zeros()) as usize };
        hist[bl] += 1;
    }

    // prefix sums for q==0: u < 2^k  <=> bitlen <= k
    let mut pref = [0u32; 17];
    let mut acc: u32 = 0;
    for i in 0..=16 {
        acc += hist[i];
        pref[i] = acc;
    }

    // suffix sums for count(bitlen >= t)
    let mut suf = [0u32; 18]; // suf[17]=0
    acc = 0;
    for i in (0..=16).rev() {
        acc += hist[i];
        suf[i] = acc;
    }

    let n_f = n as f64;

    // heuristic cost(k) ≈ remainder bits + approx q cost
    let mut best_k: u8 = 0;
    let mut best_cost: f64 = f64::INFINITY;

    for k in 0..=MAX_K {
        let k_us = k as usize;

        // p0 = P(q==0)
        let q0 = pref[k_us.min(16)] as f64;
        let p0 = q0 / n_f;

        // expected count that need q1/q2 (based on bitlen thresholds)
        let t_q1 = 8 + k_us;   // q>=128
        let t_q2 = 15 + k_us;  // q>=16384

        let count_q1 = if t_q1 <= 16 { suf[t_q1] as f64 } else { 0.0 };
        let count_q2 = if t_q2 <= 16 { suf[t_q2] as f64 } else { 0.0 };

        // "bytes" in our split varint model: q0 always, q1 sometimes, q2 rarely
        let q_bytes = n_f + count_q1 + count_q2;

        // slightly aggressive (we do rANS anyway)
        let factor = 0.25 + 0.75 * (1.0 - p0);

        let cost_bits = (n_f * (k as f64)) + (8.0 * q_bytes * factor);

        if cost_bits < best_cost - 1e-9
            || ((cost_bits - best_cost).abs() <= 1e-9 && k < best_k)
        {
            best_cost = cost_bits;
            best_k = k;
        }
    }

    best_k
}

/* ===================== Bit IO ===================== */

struct BitWriter {
    out: Vec<u8>,
    acc: u64,
    bits: u32,
}
impl BitWriter {
    fn new() -> Self {
        Self { out: Vec::new(), acc: 0, bits: 0 }
    }
    #[inline]
    fn write_bits(&mut self, v: u32, k: u32) {
        if k == 0 { return; }
        self.acc |= (v as u64) << self.bits;
        self.bits += k;
        while self.bits >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.bits -= 8;
        }
    }
    fn finish(mut self) -> Vec<u8> {
        if self.bits > 0 {
            self.out.push(self.acc as u8);
        }
        self.out
    }
}

struct BitReader<'a> {
    buf: &'a [u8],
    pos: usize,
    acc: u64,
    bits: u32,
}
impl<'a> BitReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0, acc: 0, bits: 0 }
    }
    #[inline]
    fn read_bits(&mut self, k: u32) -> Result<u32> {
        if k == 0 { return Ok(0); }
        while self.bits < k {
            if self.pos >= self.buf.len() {
                return Err(VmrError::FormatError("r_bits underflow".into()));
            }
            self.acc |= (self.buf[self.pos] as u64) << self.bits;
            self.bits += 8;
            self.pos += 1;
        }
        let mask = if k == 32 { u64::MAX } else { (1u64 << k) - 1 };
        let v = (self.acc & mask) as u32;
        self.acc >>= k;
        self.bits -= k;
        Ok(v)
    }
}

/* ===================== helpers: container parsing ===================== */

fn read_u32(buf: &[u8], off: &mut usize) -> Result<usize> {
    if buf.len() < *off + 4 {
        return Err(VmrError::FormatError("Truncated u32".into()));
    }
    let v = u32::from_le_bytes([buf[*off], buf[*off + 1], buf[*off + 2], buf[*off + 3]]) as usize;
    *off += 4;
    Ok(v)
}

fn map_rans_err(e: RansError) -> VmrError {
    VmrError::CodecError(e.to_string())
}

/* ===================== Payload meta parsing ===================== */

#[derive(Clone)]
struct ParsedMeta {
    block_frames: usize,
    frames: usize,
    blocks: usize,
    residual_len: usize,
    meta_end: usize,
    shuffle: bool,
    res_counts: Vec<usize>,
}

fn parse_payload_meta(payload: &[u8]) -> Result<ParsedMeta> {
    if payload.len() < 15 {
        return Err(VmrError::FormatError("Channel payload too short".into()));
    }

    let block_frames = u16::from_le_bytes([payload[0], payload[1]]) as usize;
    let frames = u32::from_le_bytes([payload[2], payload[3], payload[4], payload[5]]) as usize;
    let blocks = u32::from_le_bytes([payload[6], payload[7], payload[8], payload[9]]) as usize;
    let residual_len = u32::from_le_bytes([payload[10], payload[11], payload[12], payload[13]]) as usize;
    let payload_flags = payload[14];

    if block_frames == 0 {
        return Err(VmrError::FormatError("Bad block_frames".into()));
    }
    let expected_blocks = (frames + block_frames - 1) / block_frames;
    if blocks != expected_blocks {
        return Err(VmrError::FormatError("Blocks mismatch".into()));
    }

    let shuffle = (payload_flags & PAYLOAD_FLAG_SHUFFLE16) != 0;

    let mut off = 15usize;
    let mut res_counts = Vec::with_capacity(blocks);
    let mut res_sum = 0usize;

    for b in 0..blocks {
        if off >= payload.len() {
            return Err(VmrError::FormatError("Payload truncated in metadata".into()));
        }
        let mode = payload[off];
        off += 1;
        if mode > MAX_MODE {
            return Err(VmrError::FormatError("Bad mode".into()));
        }

        let start = b * block_frames;
        let end = (start + block_frames).min(frames);
        let len = end - start;

        let m = (mode as usize).min(len);
        let need = 2 * m;
        if payload.len() < off + need {
            return Err(VmrError::FormatError("Payload truncated in warmup".into()));
        }
        off += need;

        let rc = len.saturating_sub(m);
        res_sum += rc;
        res_counts.push(rc);
    }

    if res_sum != residual_len {
        return Err(VmrError::FormatError("Residual length mismatch".into()));
    }
    if payload.len() < off + residual_len * 2 {
        return Err(VmrError::FormatError("Payload truncated in residual bytes".into()));
    }

    Ok(ParsedMeta {
        block_frames,
        frames,
        blocks,
        residual_len,
        meta_end: off,
        shuffle,
        res_counts,
    })
}

fn parse_payload_meta_from_meta_only(meta: &[u8], frames_expected: usize) -> Result<ParsedMeta> {
    if meta.len() < 15 {
        return Err(VmrError::FormatError("Meta slice too short".into()));
    }

    let block_frames = u16::from_le_bytes([meta[0], meta[1]]) as usize;
    let frames = u32::from_le_bytes([meta[2], meta[3], meta[4], meta[5]]) as usize;
    let blocks = u32::from_le_bytes([meta[6], meta[7], meta[8], meta[9]]) as usize;
    let residual_len = u32::from_le_bytes([meta[10], meta[11], meta[12], meta[13]]) as usize;
    let payload_flags = meta[14];

    if frames != frames_expected {
        return Err(VmrError::FormatError("Frames mismatch".into()));
    }
    if block_frames == 0 {
        return Err(VmrError::FormatError("Bad block_frames".into()));
    }
    let expected_blocks = (frames + block_frames - 1) / block_frames;
    if blocks != expected_blocks {
        return Err(VmrError::FormatError("Blocks mismatch".into()));
    }

    let shuffle = (payload_flags & PAYLOAD_FLAG_SHUFFLE16) != 0;
    if !shuffle {
        return Err(VmrError::FormatError("Expected shuffle16 payload".into()));
    }

    let mut off = 15usize;
    let mut res_counts = Vec::with_capacity(blocks);
    let mut res_sum = 0usize;

    for b in 0..blocks {
        if off >= meta.len() {
            return Err(VmrError::FormatError("Meta truncated in modes".into()));
        }
        let mode = meta[off];
        off += 1;
        if mode > MAX_MODE {
            return Err(VmrError::FormatError("Bad mode".into()));
        }

        let start = b * block_frames;
        let end = (start + block_frames).min(frames);
        let len = end - start;

        let m = (mode as usize).min(len);
        let need = 2 * m;
        if meta.len() < off + need {
            return Err(VmrError::FormatError("Meta truncated in warmup".into()));
        }
        off += need;

        let rc = len.saturating_sub(m);
        res_sum += rc;
        res_counts.push(rc);
    }

    if res_sum != residual_len {
        return Err(VmrError::FormatError("Residual length mismatch".into()));
    }

    Ok(ParsedMeta {
        block_frames,
        frames,
        blocks,
        residual_len,
        meta_end: meta.len(),
        shuffle,
        res_counts,
    })
}

/* ===================== Channel payload (warmup + residual_u16 bytes) ===================== */

#[derive(Clone)]
struct BlockMeta {
    mode: u8,
    warmup: Vec<i16>,
    res_count: usize,
}

fn encode_channel_payload(samples: &[i16], container_flags: u8) -> Vec<u8> {
    let frames = samples.len();
    let blocks = (frames + BLOCK_FRAMES - 1) / BLOCK_FRAMES;

    let shuffle = (container_flags & FLAG_SHUFFLE16) != 0;
    let mut payload_flags: u8 = 0;
    if shuffle {
        payload_flags |= PAYLOAD_FLAG_SHUFFLE16;
    }

    let blocks_enc: Vec<(BlockMeta, Vec<u16>)> = (0..blocks)
        .into_par_iter()
        .map(|b| {
            let start = b * BLOCK_FRAMES;
            let end = (start + BLOCK_FRAMES).min(frames);
            let block = &samples[start..end];
            let len = block.len();

            let mode = choose_mode_warmup(block);
            let m = (mode as usize).min(len);

            let warmup = block[..m].to_vec();

            let coeffs = coeffs_for_mode(mode);
            let mut residual: Vec<u16> = Vec::with_capacity(len.saturating_sub(m));

            for i in m..len {
                let pred = predict_from_prev(coeffs, block, i);
                let r = block[i].wrapping_sub(pred);
                residual.push(zigzag_i16_to_u16(r));
            }

            (
                BlockMeta { mode, warmup, res_count: residual.len() },
                residual,
            )
        })
        .collect();

    let mut meta = Vec::with_capacity(blocks * (1 + 2 * MAX_MODE as usize));
    let mut residual_u16: Vec<u16> = Vec::with_capacity(frames);

    for (bm, r) in &blocks_enc {
        meta.push(bm.mode);
        for &s in &bm.warmup {
            meta.extend_from_slice(&s.to_le_bytes());
        }
        residual_u16.extend_from_slice(r);
    }

    let residual_len = residual_u16.len();
    let residual_bytes = if shuffle {
        shuffle16_u16_to_bytes(&residual_u16)
    } else {
        u16s_to_le_bytes(&residual_u16)
    };

    let mut out = Vec::with_capacity(2 + 4 + 4 + 4 + 1 + meta.len() + residual_bytes.len());
    out.extend_from_slice(&(BLOCK_FRAMES as u16).to_le_bytes());
    out.extend_from_slice(&(frames as u32).to_le_bytes());
    out.extend_from_slice(&(blocks as u32).to_le_bytes());
    out.extend_from_slice(&(residual_len as u32).to_le_bytes());
    out.push(payload_flags);
    out.extend_from_slice(&meta);
    out.extend_from_slice(&residual_bytes);

    out
}

fn decode_channel_payload(payload: &[u8], frames_expected: usize) -> Result<Vec<i16>> {
    if payload.len() < 2 + 4 + 4 + 4 + 1 {
        return Err(VmrError::FormatError("Channel payload too short".into()));
    }

    let block_frames = u16::from_le_bytes([payload[0], payload[1]]) as usize;
    let frames = u32::from_le_bytes([payload[2], payload[3], payload[4], payload[5]]) as usize;
    let blocks = u32::from_le_bytes([payload[6], payload[7], payload[8], payload[9]]) as usize;
    let residual_len = u32::from_le_bytes([payload[10], payload[11], payload[12], payload[13]]) as usize;
    let payload_flags = payload[14];

    if block_frames == 0 {
        return Err(VmrError::FormatError("Bad block_frames".into()));
    }
    if frames != frames_expected {
        return Err(VmrError::FormatError("Frames mismatch".into()));
    }
    let expected_blocks = (frames + block_frames - 1) / block_frames;
    if blocks != expected_blocks {
        return Err(VmrError::FormatError("Blocks mismatch".into()));
    }

    let shuffle = (payload_flags & PAYLOAD_FLAG_SHUFFLE16) != 0;

    let mut off = 15usize;

    let mut metas: Vec<BlockMeta> = Vec::with_capacity(blocks);
    let mut res_offsets = vec![0usize; blocks + 1];

    for b in 0..blocks {
        if off >= payload.len() {
            return Err(VmrError::FormatError("Payload truncated in metadata".into()));
        }
        let mode = payload[off];
        off += 1;
        if mode > MAX_MODE {
            return Err(VmrError::FormatError("Bad mode".into()));
        }

        let start = b * block_frames;
        let end = (start + block_frames).min(frames);
        let len = end - start;

        let m = (mode as usize).min(len);
        let need = 2 * m;

        if payload.len() < off + need {
            return Err(VmrError::FormatError("Payload truncated in warmup".into()));
        }

        let mut warmup = Vec::with_capacity(m);
        for i in 0..m {
            let b0 = payload[off + i * 2];
            let b1 = payload[off + i * 2 + 1];
            warmup.push(i16::from_le_bytes([b0, b1]));
        }
        off += need;

        let res_count = len.saturating_sub(m);
        res_offsets[b + 1] = res_offsets[b] + res_count;

        metas.push(BlockMeta { mode, warmup, res_count });
    }

    if res_offsets[blocks] != residual_len {
        return Err(VmrError::FormatError("Residual length mismatch".into()));
    }

    let need_bytes = residual_len * 2;
    if payload.len() < off + need_bytes {
        return Err(VmrError::FormatError("Payload truncated in residual bytes".into()));
    }
    let rb = &payload[off..off + need_bytes];

    let residual_u16 = if shuffle {
        unshuffle16_bytes_to_u16(rb, residual_len)?
    } else {
        le_bytes_to_u16s(rb, residual_len)?
    };

    let mut out = vec![0i16; frames];

    out.par_chunks_mut(block_frames)
        .enumerate()
        .for_each(|(b, out_block)| {
            let start = b * block_frames;
            let end = (start + block_frames).min(frames);
            let len = end - start;

            let bm = &metas[b];
            let mode = bm.mode;
            let m = (mode as usize).min(len);
            let coeffs = coeffs_for_mode(mode);

            for i in 0..m {
                out_block[i] = bm.warmup[i];
            }

            let r0 = res_offsets[b];
            let r = &residual_u16[r0..r0 + bm.res_count];

            for i in m..len {
                let pred = predict_from_prev(coeffs, &out_block[..len], i);
                let rr = unzigzag_u16_to_i16(r[i - m]);
                out_block[i] = pred.wrapping_add(rr);
            }
        });

    Ok(out)
}

/* ===================== Mode choice / predictors ===================== */

fn choose_mode_warmup(block: &[i16]) -> u8 {
    let len = block.len();
    let max_try = (MAX_MODE as usize).min(len);

    let mut best_mode = 0u8;
    let mut best_max_u: u16 = u16::MAX;
    let mut best_score: u64 = u64::MAX;

    for m in 0..=max_try {
        let mode = m as u8;
        let coeffs = coeffs_for_mode(mode);

        let mut max_u: u16 = 0;
        let mut sum_u: u64 = 0;

        for i in m..len {
            let pred = predict_from_prev(coeffs, block, i);
            let r = block[i].wrapping_sub(pred);
            let u = zigzag_i16_to_u16(r);
            max_u = max_u.max(u);
            sum_u += u as u64;
        }

        let penalty = (m as u64) * 65_536;
        let score = sum_u.saturating_add(penalty);

        if max_u < best_max_u || (max_u == best_max_u && score < best_score) {
            best_max_u = max_u;
            best_score = score;
            best_mode = mode;
        }
    }

    best_mode
}

fn coeffs_for_mode(mode: u8) -> &'static [i32] {
    match mode {
        0 => &[],
        1 => &[1],
        2 => &[2, -1],
        3 => &[3, -3, 1],
        4 => &[4, -6, 4, -1],
        5 => &[5, -10, 10, -5, 1],
        6 => &[6, -15, 20, -15, 6, -1],
        7 => &[7, -21, 35, -35, 21, -7, 1],
        8 => &[8, -28, 56, -70, 56, -28, 8, -1],
        _ => &[],
    }
}

#[inline]
fn predict_from_prev(coeffs: &[i32], block: &[i16], i: usize) -> i16 {
    if coeffs.is_empty() {
        return 0;
    }
    let mut acc: i64 = 0;
    for (k, &c) in coeffs.iter().enumerate() {
        let idx = i.wrapping_sub(1 + k);
        let s = if idx < block.len() { block[idx] } else { 0 };
        acc += (c as i64) * (s as i64);
    }
    acc as i16
}

/* ===================== u16 <-> bytes (LE and shuffle) ===================== */

fn u16s_to_le_bytes(v: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn le_bytes_to_u16s(bytes: &[u8], n: usize) -> Result<Vec<u16>> {
    if bytes.len() != n * 2 {
        return Err(VmrError::FormatError("LE residual bytes length mismatch".into()));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]));
    }
    Ok(out)
}

// shuffle: [low bytes...][high bytes...]
fn shuffle16_u16_to_bytes(v: &[u16]) -> Vec<u8> {
    let n = v.len();
    let mut out = vec![0u8; n * 2];
    for i in 0..n {
        let x = v[i];
        out[i] = (x & 0xFF) as u8;
        out[i + n] = (x >> 8) as u8;
    }
    out
}

fn unshuffle16_bytes_to_u16(bytes: &[u8], n: usize) -> Result<Vec<u16>> {
    if bytes.len() != n * 2 {
        return Err(VmrError::FormatError("Shuffle residual bytes length mismatch".into()));
    }
    let lows = &bytes[..n];
    let highs = &bytes[n..];
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push((lows[i] as u16) | ((highs[i] as u16) << 8));
    }
    Ok(out)
}

/* ===================== ZigZag ===================== */

#[inline]
fn zigzag_i16_to_u16(x: i16) -> u16 {
    ((x << 1) ^ (x >> 15)) as u16
}
#[inline]
fn unzigzag_u16_to_i16(v: u16) -> i16 {
    ((v >> 1) as i16) ^ (-((v & 1) as i16))
}
