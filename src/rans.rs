//! rans.rs — byte-oriented rANS + PACK7 fallback (битпак 7-bit потоков)
//!
//! Экспорты под твой lib.rs:
//!   - rans_compress_blob_auto_mt(data, name) -> Vec<u8>   // НЕ Result
//!   - rans_decompress_blob_auto_mt(blob) -> Result<Vec<u8>, RansError>
//!   - set_verbosity(level)

use rayon::prelude::*;
use std::fmt;
use std::sync::atomic::{AtomicU8, Ordering};

/* ============================= Public API ============================= */

static VERBOSITY: AtomicU8 = AtomicU8::new(0);

pub fn set_verbosity(level: u8) {
    VERBOSITY.store(level, Ordering::Relaxed);
}

fn verbosity() -> u8 {
    VERBOSITY.load(Ordering::Relaxed)
}

#[derive(Debug)]
pub enum RansError {
    Format(String),
    Codec(String),
}

impl fmt::Display for RansError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RansError::Format(s) => write!(f, "Format error: {s}"),
            RansError::Codec(s) => write!(f, "Codec error: {s}"),
        }
    }
}
impl std::error::Error for RansError {}

pub type Result<T> = std::result::Result<T, RansError>;

/// Infallible encode (как у тебя в lib.rs): всегда вернёт blob.
/// Если что-то пошло не так — упадём на baseline (PACK7 для 7-bit, иначе RAW).
pub fn rans_compress_blob_auto_mt(data: &[u8], name: &str) -> Vec<u8> {
    match encode_bytes_auto(name, data) {
        Ok(v) => v,
        Err(e) => {
            if verbosity() >= 1 {
                eprintln!("[vmr] [plan] {name}: encoder error -> fallback baseline: {e}");
            }
            // baseline: если поток реально 7-bit — pack7, иначе raw
            let is7 = data.iter().all(|&b| b < 128);
            if is7 {
                encode_pack7_blob(data).unwrap_or_else(|_| encode_raw_blob(data, true))
            } else {
                encode_raw_blob(data, false)
            }
        }
    }
}

/// Decode может фейлиться — это нормально, тут Result.
pub fn rans_decompress_blob_auto_mt(blob: &[u8]) -> Result<Vec<u8>> {
    decode_bytes_auto(blob)
}

/* ============================= Planner knobs ============================= */

// твоя эвристика: шумно => вместо rANS уходим в bitpack (PACK7)
const ENTROPY_BITPACK_THRESHOLD: f64 = 6.6;
const UNIQUE_BITPACK_THRESHOLD: usize = 120;

// rANS candidates
const CHUNK_CANDIDATES: &[usize] = &[8192, 16384, 32768, 65536, 131072];
const MIN_RANS_LEN: usize = 2048;

// margin: rANS должен быть хотя бы на 0.5% лучше baseline
const MARGIN_NUM: u64 = 995;
const MARGIN_DEN: u64 = 1000;

/* ============================= Container constants ============================= */

const MAGIC: &[u8; 4] = b"RNS1";
const CODEC_RAW: u8 = 0;
const CODEC_PACK7: u8 = 1;
const CODEC_RANS: u8 = 2;

/* ============================= Stats ============================= */

#[derive(Clone, Copy, Debug)]
struct Stats {
    unique: usize,
    entropy: f64, // bits/symbol
    p0: f64,
    max_sym: u8,
    is_7bit: bool,
}

fn stats_u8(data: &[u8]) -> Stats {
    if data.is_empty() {
        return Stats {
            unique: 0,
            entropy: 0.0,
            p0: 0.0,
            max_sym: 0,
            is_7bit: true,
        };
    }

    let mut freq = [0u32; 256];
    let mut maxv = 0u8;
    for &b in data {
        freq[b as usize] += 1;
        if b > maxv {
            maxv = b;
        }
    }

    let n = data.len() as f64;
    let mut u = 0usize;
    let mut h = 0.0f64;
    for &c in &freq {
        if c != 0 {
            u += 1;
            let p = (c as f64) / n;
            h -= p * p.log2();
        }
    }

    let p0 = (freq[0] as f64) / n;

    Stats {
        unique: u,
        entropy: h,
        p0,
        max_sym: maxv,
        is_7bit: maxv < 128,
    }
}

fn approx_model_overhead_bytes(st: &Stats) -> usize {
    // pairs: u16 count + each pair (u8 sym + u16 freq) => 2 + 3*U
    2 + st.unique * 3 + 8
}

fn estimate_rans_size(len: usize, entropy_bits: f64, overhead_per_chunk: usize, chunk: usize) -> usize {
    let n_chunks = (len + chunk - 1) / chunk;
    let payload = (entropy_bits / 8.0 * (len as f64)) as usize;
    let per_chunk = overhead_per_chunk + 8; // state(u32)+renorm_len(u32)
    payload + n_chunks * per_chunk
}

/* ============================= Decision ============================= */

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Decision {
    Raw,
    Pack7,
    Rans { chunk: usize },
}

/* ============================= Auto encode/decode ============================= */

fn encode_bytes_auto(name: &str, data: &[u8]) -> Result<Vec<u8>> {
    let st = stats_u8(data);
    let overhead = approx_model_overhead_bytes(&st);

    let decision = choose_decision(name, data.len(), &st, overhead);

    match decision {
        Decision::Raw => Ok(encode_raw_blob(data, st.is_7bit)),
        Decision::Pack7 => encode_pack7_blob(data),
        Decision::Rans { chunk } => encode_rans_chunks_blob(data, chunk, st.is_7bit),
    }
}

fn decode_bytes_auto(blob: &[u8]) -> Result<Vec<u8>> {
    if blob.len() < 12 {
        return Err(RansError::Format("blob too small".into()));
    }
    if &blob[0..4] != MAGIC {
        return Err(RansError::Format("bad magic".into()));
    }
    let codec = blob[4];

    match codec {
        CODEC_RAW => decode_raw_blob(blob),
        CODEC_PACK7 => decode_pack7_blob(blob),
        CODEC_RANS => decode_rans_chunks_blob(blob),
        _ => Err(RansError::Format(format!("unknown codec id: {codec}"))),
    }
}

fn choose_decision(name: &str, len: usize, st: &Stats, overhead: usize) -> Decision {
    // 1) Выбираем baseline:
    //    - если поток 7-bit: baseline обычно PACK7 (он никогда не хуже RAW по размеру)
    //    - если поток не 7-bit: baseline RAW
    //
    // 2) ТВОЯ эвристика H/U:
    //    Она НЕ "force PACK7", а "baseline = PACK7 (noise)",
    //    при этом rANS может обогнать baseline и будет выбран.

    let mut baseline_kind = if st.is_7bit { Decision::Pack7 } else { Decision::Raw };
    let mut baseline_cost: u64 = if st.is_7bit {
        pack7_len_bytes(len) as u64
    } else {
        len as u64
    };

    let noise_pack7 = st.is_7bit
        && st.entropy > ENTROPY_BITPACK_THRESHOLD
        && st.unique > UNIQUE_BITPACK_THRESHOLD;

    // Если вдруг ты НЕ хочешь PACK7 всегда для 7-bit, а только по H/U — переключай тут:
    // if st.is_7bit && noise_pack7 { baseline_kind = Decision::Pack7; baseline_cost = pack7_len_bytes(len) as u64; }
    // В текущем варианте baseline и так PACK7 для 7-bit, но флаг noise_pack7 пригодится для логов.

    // маленькие потоки: baseline
    if len < MIN_RANS_LEN {
        if verbosity() >= 2 {
            eprintln!(
                "[vmr] [plan] {name}: len={} U={} H={:.3} p0={:.3} overhead≈{}B/chunk baseline={}({}B) -> baseline (too small)",
                len,
                st.unique,
                st.entropy,
                st.p0,
                overhead,
                match baseline_kind { Decision::Pack7 => "PACK7", _ => "RAW" },
                baseline_cost,
            );
        }
        return baseline_kind;
    }

    // Поиск лучшего rANS chunk по estimate
    let mut best_est = u64::MAX;
    let mut best_chunk: Option<usize> = None;

    for &chunk in CHUNK_CANDIDATES {
        let est = estimate_rans_size(len, st.entropy, overhead, chunk) as u64;
        if est < best_est {
            best_est = est;
            best_chunk = Some(chunk);
        }
    }

    // Принимаем rANS только если он реально лучше baseline (с margin)
    let accept_rans = if let Some(chunk) = best_chunk {
        best_est * MARGIN_DEN < baseline_cost * MARGIN_NUM
            && len >= MIN_RANS_LEN
            && chunk > 0
    } else {
        false
    };

    let decision = if accept_rans {
        Decision::Rans { chunk: best_chunk.unwrap() }
    } else {
        baseline_kind
    };

    if verbosity() >= 2 {
        let base = match baseline_kind { Decision::Pack7 => "PACK7", _ => "RAW" };
        let noise_note = if noise_pack7 { " (noise baseline)" } else { "" };

        match decision {
            Decision::Rans { chunk } => {
                eprintln!(
                    "[vmr] [plan] {name}: len={} U={} H={:.3} p0={:.3} overhead≈{}B/chunk baseline={base}({}B){noise_note} -> rANS chunk={} (est≈{}B)",
                    len, st.unique, st.entropy, st.p0, overhead, baseline_cost, chunk, best_est
                );
            }
            Decision::Pack7 => {
                eprintln!(
                    "[vmr] [plan] {name}: len={} U={} H={:.3} p0={:.3} overhead≈{}B/chunk baseline={base}({}B){noise_note} -> PACK7",
                    len, st.unique, st.entropy, st.p0, overhead, baseline_cost
                );
            }
            Decision::Raw => {
                eprintln!(
                    "[vmr] [plan] {name}: len={} U={} H={:.3} p0={:.3} overhead≈{}B/chunk baseline={base}({}B){noise_note} -> RAW",
                    len, st.unique, st.entropy, st.p0, overhead, baseline_cost
                );
            }
        }
    }

    decision
}

/* ============================= RAW codec ============================= */

fn encode_raw_blob(data: &[u8], stream_is_7bit_info: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(12 + data.len());
    out.extend_from_slice(MAGIC);
    out.push(CODEC_RAW);
    out.push(if stream_is_7bit_info { 1 } else { 0 }); // flags bit0 informational
    out.push(0);
    out.push(0);
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(data);
    out
}

fn decode_raw_blob(blob: &[u8]) -> Result<Vec<u8>> {
    if blob.len() < 12 {
        return Err(RansError::Format("RAW blob too small".into()));
    }
    let n = u32::from_le_bytes([blob[8], blob[9], blob[10], blob[11]]) as usize;
    if blob.len() != 12 + n {
        return Err(RansError::Format("RAW length mismatch".into()));
    }
    Ok(blob[12..].to_vec())
}

/* ============================= PACK7 codec ============================= */

fn pack7_len_bytes(n_syms: usize) -> usize {
    (n_syms * 7 + 7) / 8
}

fn encode_pack7_blob(data: &[u8]) -> Result<Vec<u8>> {
    if data.iter().any(|&b| b > 127) {
        return Err(RansError::Codec("PACK7 used but data contains values > 127".into()));
    }
    let packed = pack7(data);

    let mut out = Vec::with_capacity(12 + packed.len());
    out.extend_from_slice(MAGIC);
    out.push(CODEC_PACK7);
    out.push(1); // flags bit0 = 7bit
    out.push(0);
    out.push(0);
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&packed);
    Ok(out)
}

fn decode_pack7_blob(blob: &[u8]) -> Result<Vec<u8>> {
    if blob.len() < 12 {
        return Err(RansError::Format("PACK7 blob too small".into()));
    }
    let n = u32::from_le_bytes([blob[8], blob[9], blob[10], blob[11]]) as usize;
    let packed = &blob[12..];
    let need = pack7_len_bytes(n);
    if packed.len() != need {
        return Err(RansError::Format("PACK7 length mismatch".into()));
    }
    Ok(unpack7(packed, n))
}

/// Pack 7-bit symbols (0..127) into bytes (LSB-first).
fn pack7(src: &[u8]) -> Vec<u8> {
    let n = src.len();
    let mut out = vec![0u8; pack7_len_bytes(n)];

    let mut bitpos = 0usize;
    for &v in src {
        let x = (v & 0x7F) as u32;
        let byte = bitpos >> 3;
        let shift = (bitpos & 7) as u32;

        let cur = x << shift;
        out[byte] |= (cur & 0xFF) as u8;

        if shift > 1 && byte + 1 < out.len() {
            out[byte + 1] |= ((cur >> 8) & 0xFF) as u8;
        }

        bitpos += 7;
    }

    out
}

fn unpack7(src: &[u8], n_syms: usize) -> Vec<u8> {
    let mut out = vec![0u8; n_syms];
    let mut bitpos = 0usize;

    for i in 0..n_syms {
        let byte = bitpos >> 3;
        let shift = (bitpos & 7) as u32;

        let lo = src[byte] as u32;
        let hi = if byte + 1 < src.len() { src[byte + 1] as u32 } else { 0 };
        let w = lo | (hi << 8);

        out[i] = ((w >> shift) & 0x7F) as u8;
        bitpos += 7;
    }

    out
}

/* ============================= rANS codec ============================= */

// TOTFREQ = 2^12
const SCALE_BITS: u32 = 12;
const TOTFREQ: u32 = 1 << SCALE_BITS;
const MASK: u32 = TOTFREQ - 1;
const RANS_L: u32 = 1 << 23;

#[derive(Clone)]
struct Model {
    freq: [u16; 256],
    cum: [u32; 257],
    dec: Vec<DecEntry>, // len TOTFREQ
}

#[derive(Clone, Copy)]
struct DecEntry {
    sym: u8,
    freq: u16,
    cum: u32,
}

fn encode_rans_chunks_blob(data: &[u8], chunk_size: usize, stream_is_7bit_info: bool) -> Result<Vec<u8>> {
    let n = data.len();
    let chunks = (n + chunk_size - 1) / chunk_size;

    let slices: Vec<&[u8]> = (0..chunks)
        .map(|i| {
            let s = i * chunk_size;
            let e = (s + chunk_size).min(n);
            &data[s..e]
        })
        .collect();

    let enc_chunks: Vec<ChunkEncoded> = slices
        .par_iter()
        .map(|sl| encode_rans_chunk(sl))
        .collect::<Result<Vec<_>>>()?;

    let mut comp_lens: Vec<u32> = Vec::with_capacity(chunks);
    let mut total_body = 0usize;
    for c in &enc_chunks {
        let t = c.model_bytes.len() + c.payload.len();
        comp_lens.push(t as u32);
        total_body += t;
    }

    // header:
    // MAGIC(4) codec(1) flags(1) rsv(2) out_len(u32) chunk(u16) chunks(u32) lens[chunks]*u32 ...
    let mut out = Vec::with_capacity(18 + chunks * 4 + total_body);
    out.extend_from_slice(MAGIC);
    out.push(CODEC_RANS);
    out.push(if stream_is_7bit_info { 1 } else { 0 });
    out.push(0);
    out.push(0);

    out.extend_from_slice(&(n as u32).to_le_bytes());
    out.extend_from_slice(&(chunk_size as u16).to_le_bytes());
    out.extend_from_slice(&(chunks as u32).to_le_bytes());
    for &cl in &comp_lens {
        out.extend_from_slice(&cl.to_le_bytes());
    }
    for c in enc_chunks {
        out.extend_from_slice(&c.model_bytes);
        out.extend_from_slice(&c.payload);
    }

    Ok(out)
}

fn decode_rans_chunks_blob(blob: &[u8]) -> Result<Vec<u8>> {
    if blob.len() < 18 {
        return Err(RansError::Format("RANS blob too small".into()));
    }
    let n = u32::from_le_bytes([blob[8], blob[9], blob[10], blob[11]]) as usize;
    let chunk_size = u16::from_le_bytes([blob[12], blob[13]]) as usize;
    let chunks = u32::from_le_bytes([blob[14], blob[15], blob[16], blob[17]]) as usize;

    let header_lens_off = 18;
    let header_lens_bytes = chunks
        .checked_mul(4)
        .ok_or_else(|| RansError::Format("chunks overflow".into()))?;
    if blob.len() < header_lens_off + header_lens_bytes {
        return Err(RansError::Format("RANS blob truncated (lens)".into()));
    }

    let mut comp_lens = Vec::with_capacity(chunks);
    let mut off = header_lens_off;
    for _ in 0..chunks {
        let cl = u32::from_le_bytes([blob[off], blob[off + 1], blob[off + 2], blob[off + 3]]) as usize;
        comp_lens.push(cl);
        off += 4;
    }

    let mut chunk_blobs: Vec<&[u8]> = Vec::with_capacity(chunks);
    for &cl in &comp_lens {
        if blob.len() < off + cl {
            return Err(RansError::Format("RANS blob truncated (chunk)".into()));
        }
        chunk_blobs.push(&blob[off..off + cl]);
        off += cl;
    }
    if off != blob.len() {
        return Err(RansError::Format("RANS blob has trailing bytes".into()));
    }

    let mut out = vec![0u8; n];

    out.par_chunks_mut(chunk_size)
        .enumerate()
        .try_for_each(|(i, out_chunk)| -> Result<()> {
            if i >= chunk_blobs.len() {
                return Ok(());
            }
            let start = i * chunk_size;
            let chunk_len = (n - start).min(chunk_size);
            let dst = &mut out_chunk[..chunk_len];
            decode_rans_chunk_into(chunk_blobs[i], dst)
        })?;

    Ok(out)
}

/* ============================= Chunk encoding/decoding ============================= */

struct ChunkEncoded {
    model_bytes: Vec<u8>,
    payload: Vec<u8>,
}

fn encode_rans_chunk(src: &[u8]) -> Result<ChunkEncoded> {
    // пустой чанк: модель empty + payload empty
    if src.is_empty() {
        let mut model_bytes = Vec::new();
        model_bytes.extend_from_slice(&(0u16).to_le_bytes()); // 0 pairs
        let mut payload = Vec::new();
        payload.extend_from_slice(&(RANS_L as u32).to_le_bytes());
        payload.extend_from_slice(&(0u32).to_le_bytes()); // renorm_len
        return Ok(ChunkEncoded { model_bytes, payload });
    }

    let model = build_model_from_data(src)?;
    let model_bytes = serialize_model(&model);
    let payload = rans_encode_bytes(src, &model);
    Ok(ChunkEncoded { model_bytes, payload })
}

fn decode_rans_chunk_into(chunk_blob: &[u8], dst: &mut [u8]) -> Result<()> {
    let (model, payload_off) = deserialize_model(chunk_blob)?;
    let payload = &chunk_blob[payload_off..];
    rans_decode_bytes(payload, &model, dst)?;
    Ok(())
}

/* ============================= Model building ============================= */

fn build_model_from_data(data: &[u8]) -> Result<Model> {
    let mut hist = [0u32; 256];
    for &b in data {
        hist[b as usize] += 1;
    }
    let freq_u16 = normalize_histogram(&hist, data.len())?;

    let mut cum = [0u32; 257];
    let mut acc = 0u32;
    for s in 0..256 {
        cum[s] = acc;
        acc += freq_u16[s] as u32;
    }
    cum[256] = acc;
    if acc != TOTFREQ {
        return Err(RansError::Codec(format!("normalized sum != TOTFREQ: {acc}")));
    }

    let mut dec = vec![DecEntry { sym: 0, freq: 0, cum: 0 }; TOTFREQ as usize];
    for s in 0..256usize {
        let f = freq_u16[s];
        if f == 0 {
            continue;
        }
        let start = cum[s] as usize;
        let end = (cum[s] + f as u32) as usize;
        for x in start..end {
            dec[x] = DecEntry { sym: s as u8, freq: f, cum: cum[s] };
        }
    }

    Ok(Model { freq: freq_u16, cum, dec })
}

fn normalize_histogram(hist: &[u32; 256], n: usize) -> Result<[u16; 256]> {
    let mut out = [0u16; 256];
    if n == 0 {
        return Ok(out);
    }

    let mut nonzero: Vec<usize> = Vec::new();
    for s in 0..256 {
        if hist[s] != 0 {
            nonzero.push(s);
        }
    }
    if nonzero.is_empty() {
        return Ok(out);
    }

    // initial scaling
    let n64 = n as u64;
    let mut sum = 0u32;
    for &s in &nonzero {
        let scaled = ((hist[s] as u64) * (TOTFREQ as u64)) / n64;
        let mut f = scaled as u32;
        if f == 0 {
            f = 1;
        }
        if f > 0xFFFF {
            f = 0xFFFF;
        }
        out[s] = f as u16;
        sum += f;
    }

    // deterministic adjust by descending hist
    nonzero.sort_by_key(|&s| std::cmp::Reverse(hist[s]));

    if sum < TOTFREQ {
        let mut need = TOTFREQ - sum;
        let mut idx = 0usize;
        while need > 0 {
            let s = nonzero[idx % nonzero.len()];
            let v = out[s] as u32;
            if v < 0xFFFF {
                out[s] = (v + 1) as u16;
                need -= 1;
            }
            idx += 1;
        }
    } else if sum > TOTFREQ {
        let mut need = sum - TOTFREQ;
        let mut idx = 0usize;
        while need > 0 {
            let s = nonzero[idx % nonzero.len()];
            let v = out[s] as u32;
            if v > 1 {
                out[s] = (v - 1) as u16;
                need -= 1;
            }
            idx += 1;
        }
    }

    let mut final_sum = 0u32;
    for s in 0..256 {
        final_sum += out[s] as u32;
    }
    if final_sum != TOTFREQ {
        return Err(RansError::Codec(format!("normalize failed: sum={final_sum}")));
    }

    Ok(out)
}

fn serialize_model(m: &Model) -> Vec<u8> {
    let mut pairs: Vec<(u8, u16)> = Vec::new();
    for s in 0..256 {
        let f = m.freq[s];
        if f != 0 {
            pairs.push((s as u8, f));
        }
    }

    let mut out = Vec::with_capacity(2 + pairs.len() * 3);
    out.extend_from_slice(&(pairs.len() as u16).to_le_bytes());
    for (sym, f) in pairs {
        out.push(sym);
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

fn deserialize_model(buf: &[u8]) -> Result<(Model, usize)> {
    if buf.len() < 2 {
        return Err(RansError::Format("model truncated".into()));
    }
    let pairs = u16::from_le_bytes([buf[0], buf[1]]) as usize;
    let need = 2 + pairs * 3;
    if buf.len() < need {
        return Err(RansError::Format("model truncated (pairs)".into()));
    }

    let mut freq = [0u16; 256];
    let mut off = 2usize;
    for _ in 0..pairs {
        let sym = buf[off];
        let f = u16::from_le_bytes([buf[off + 1], buf[off + 2]]);
        off += 3;
        freq[sym as usize] = f;
    }

    let mut cum = [0u32; 257];
    let mut acc = 0u32;
    for s in 0..256 {
        cum[s] = acc;
        acc += freq[s] as u32;
    }
    cum[256] = acc;
    if acc != TOTFREQ {
        return Err(RansError::Format(format!("model sum != TOTFREQ: {acc}")));
    }

    let mut dec = vec![DecEntry { sym: 0, freq: 0, cum: 0 }; TOTFREQ as usize];
    for s in 0..256usize {
        let f = freq[s];
        if f == 0 {
            continue;
        }
        let start = cum[s] as usize;
        let end = (cum[s] + f as u32) as usize;
        for x in start..end {
            dec[x] = DecEntry { sym: s as u8, freq: f, cum: cum[s] };
        }
    }

    Ok((Model { freq, cum, dec }, off))
}

/* ============================= rANS core ============================= */

fn rans_encode_bytes(src: &[u8], m: &Model) -> Vec<u8> {
    // rANS кодирует символы с конца, renorm байты кладём в стек (LIFO)
    let mut state: u32 = RANS_L;
    let mut renorm: Vec<u8> = Vec::with_capacity(src.len() / 2);

    for &sym in src.iter().rev() {
        let s = sym as usize;
        let f = m.freq[s] as u32;
        let c = m.cum[s] as u32;

        let x_max = ((RANS_L >> SCALE_BITS) * f) << 8;
        while state >= x_max {
            renorm.push((state & 0xFF) as u8);
            state >>= 8;
        }

        let q = state / f;
        let r = state - q * f;
        state = q * TOTFREQ + r + c;
    }

    let mut payload = Vec::with_capacity(8 + renorm.len());
    payload.extend_from_slice(&state.to_le_bytes());
    payload.extend_from_slice(&(renorm.len() as u32).to_le_bytes());
    payload.extend_from_slice(&renorm);
    payload
}

fn rans_decode_bytes(payload: &[u8], m: &Model, dst: &mut [u8]) -> Result<()> {
    if payload.len() < 8 {
        return Err(RansError::Format("payload too small".into()));
    }
    let mut state = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
    let renorm_len = u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]) as usize;
    if payload.len() != 8 + renorm_len {
        return Err(RansError::Format("payload length mismatch".into()));
    }

    // renorm читаем как стек: с конца к началу
    let renorm = &payload[8..];
    let mut pos = renorm.len();

    for i in 0..dst.len() {
        let x = (state & MASK) as usize;
        let e = m.dec[x];
        dst[i] = e.sym;

        state = (e.freq as u32) * (state >> SCALE_BITS) + (state & MASK) - e.cum;

        while state < RANS_L {
            if pos == 0 {
                return Err(RansError::Format("payload truncated during renorm".into()));
            }
            pos -= 1;
            state = (state << 8) | (renorm[pos] as u32);
        }
    }

    if pos != 0 {
        return Err(RansError::Format("payload has trailing renorm bytes".into()));
    }
    Ok(())
}
