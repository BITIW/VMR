use zstd::stream::{encode_all, decode_all};
use rayon::prelude::*;
use std::fmt;

const VMR_MAGIC: &[u8; 4] = b"VMR1";
const VMR_HEADER_SIZE: usize = 16;
const BLOCK_FRAMES: usize = 4096; // длина блока предсказателя в фреймах

#[derive(Debug)]
pub enum VmrError {
    InvalidInput(String),
    CodecError(String),
    FormatError(String),
}

impl fmt::Display for VmrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VmrError::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
             VmrError::CodecError(msg) => write!(f, "Codec error: {msg}"),
            VmrError::FormatError(msg) => write!(f, "Format error: {msg}"),
        }
    }
}

impl std::error::Error for VmrError {}

pub type Result<T> = std::result::Result<T, VmrError>;

/// Статы для логов
#[derive(Debug, Clone, Copy)]
pub struct EncodeStats {
    pub frames: usize,
    pub channels: u8,
    pub sample_rate: u32,
    pub raw_size: usize,
    /// размер residual-потока (predictors + bitplane per channel)
    pub bitplane_size: usize,
    /// суммарный размер всех lz4-блоков
    pub compressed_size: usize,
}

/// Простое API без статистики
pub fn vmr_encode(pcm: &[u8], channels: u8, sample_rate: u32) -> Result<Vec<u8>> {
    Ok(vmr_encode_with_stats(pcm, channels, sample_rate)?.0)
}

/// Полное API: данные + статистика
pub fn vmr_encode_with_stats(
    pcm: &[u8],
    channels: u8,
    sample_rate: u32,
) -> Result<(Vec<u8>, EncodeStats)> {
    if channels == 0 || channels > 2 {
        return Err(VmrError::InvalidInput(
            "Only 1 or 2 channels are supported".into(),
        ));
    }

    let bytes_per_sample = 2usize; // 16-bit
    if pcm.len() % (bytes_per_sample * channels as usize) != 0 {
        return Err(VmrError::InvalidInput(
            "PCM length is not aligned to sample size * channels".into(),
        ));
    }

    let frames = pcm.len() / (bytes_per_sample * channels as usize);
    let channels_usize = channels as usize;

    // ---------- 1) interleaved -> planar i16 ----------
    let mut channels_data: Vec<Vec<i16>> = (0..channels)
        .map(|_| Vec::with_capacity(frames))
        .collect();

    for frame in 0..frames {
        for ch in 0..channels {
            let byte_index =
                (frame * channels as usize + ch as usize) * bytes_per_sample;
            let sample_bytes = [pcm[byte_index], pcm[byte_index + 1]];
            let s = i16::from_le_bytes(sample_bytes);
            channels_data[ch as usize].push(s);
        }
    }

    // ---------- 2) Left + Side для стерео ----------
    if channels == 2 {
        let (left_slice, right_slice) = channels_data.split_at_mut(1);
        let left = &mut left_slice[0];
        let right = &mut right_slice[0];

        for i in 0..frames {
            let l = left[i];
            let r = right[i];
            let side = r.wrapping_sub(l);
            right[i] = side; // второй канал теперь side
        }
    }

    // ---------- 3) Кодируем каждый канал отдельно ----------
    let encoded_channels: Vec<Vec<u8>> = channels_data
        .par_iter()
        .map(|samples| encode_channel(samples, frames))
        .collect();

    let mut residual_size = 0usize;
    for ch_data in &encoded_channels {
        residual_size += ch_data.len();
    }

    // ---------- 4) LZ4 per-channel ----------
    let mut compressed_channels: Vec<Vec<u8>> = Vec::with_capacity(channels_usize);
    let mut compressed_size = 0usize;
    
    // уровень zstd: 3–5 — нормально; 10+ — уже прям жирно, но медленнее
    let zstd_level: i32 = 7;
    
    for ch_data in &encoded_channels {
        let compressed = encode_all(std::io::Cursor::new(ch_data.as_slice()), zstd_level)
            .map_err(|e| VmrError::CodecError(e.to_string()))?;
        compressed_size += compressed.len();
        compressed_channels.push(compressed);
    }

    // ---------- 5) Хедер + размеры + данные ----------
    // Формат:
    // [16 байт заголовок]
    // [u32 size_ch0][u32 size_ch1]...
    // [ch0 compressed bytes][ch1 compressed bytes]...
    let mut out = Vec::with_capacity(
        VMR_HEADER_SIZE + 4 * channels_usize + compressed_size,
    );

    // заголовок
    out.extend_from_slice(VMR_MAGIC);              // 0..4
    out.push(channels);                            // 4
    out.push(16);                                  // 5 bits_per_sample
    out.extend_from_slice(&0u16.to_le_bytes());    // 6..8 reserved
    out.extend_from_slice(&sample_rate.to_le_bytes()); // 8..12
    out.extend_from_slice(&(frames as u32).to_le_bytes()); // 12..16

    // длины каналов
    for ch in 0..channels_usize {
        let len = compressed_channels[ch].len() as u32;
        out.extend_from_slice(&len.to_le_bytes());
    }

    // данные каналов
    for ch in 0..channels_usize {
        out.extend_from_slice(&compressed_channels[ch]);
    }

    let stats = EncodeStats {
        frames,
        channels,
        sample_rate,
        raw_size: pcm.len(),
        bitplane_size: residual_size,
        compressed_size,
    };

    Ok((out, stats))
}

/// Декодер: VMR -> (PCM 16-bit LE interleaved, channels, sample_rate)
pub fn vmr_decode(vmr: &[u8]) -> Result<(Vec<u8>, u8, u32)> {
    if vmr.len() < VMR_HEADER_SIZE {
        return Err(VmrError::FormatError("VMR data too short".into()));
    }

    if &vmr[0..4] != VMR_MAGIC {
        return Err(VmrError::FormatError("Bad VMR magic".into()));
    }

    let channels = vmr[4];
    let bits_per_sample = vmr[5];
    if bits_per_sample != 16 {
        return Err(VmrError::FormatError(
            "Only 16-bit VMR is supported in this implementation".into(),
        ));
    }

    let _reserved = u16::from_le_bytes([vmr[6], vmr[7]]);
    let sample_rate = u32::from_le_bytes([vmr[8], vmr[9], vmr[10], vmr[11]]);
    let frames = u32::from_le_bytes([vmr[12], vmr[13], vmr[14], vmr[15]]) as usize;

    if channels == 0 || channels > 2 {
        return Err(VmrError::FormatError(
            "Invalid channel count in VMR header".into(),
        ));
    }

    let channels_usize = channels as usize;

    let mut offset = VMR_HEADER_SIZE;

    if vmr.len() < offset + 4 * channels_usize {
        return Err(VmrError::FormatError(
            "VMR too short for channel sizes".into(),
        ));
    }

    // читаем размеры сжатых каналов
    let mut ch_sizes: Vec<usize> = Vec::with_capacity(channels_usize);
    for _ in 0..channels_usize {
        let len = u32::from_le_bytes([
            vmr[offset],
            vmr[offset + 1],
            vmr[offset + 2],
            vmr[offset + 3],
        ]) as usize;
        ch_sizes.push(len);
        offset += 4;
    }

    // ---------- 1) Распаковываем и декодим каждый канал ----------
    let mut channels_data: Vec<Vec<i16>> = Vec::with_capacity(channels_usize);

    for ch in 0..channels_usize {
        let len = ch_sizes[ch];
        if vmr.len() < offset + len {
            return Err(VmrError::FormatError(
                "VMR truncated in channel data".into(),
            ));
        }
        let compressed = &vmr[offset..offset + len];
        offset += len;

        let ch_payload = decode_all(std::io::Cursor::new(compressed))
            .map_err(|e| VmrError::CodecError(e.to_string()))?;

        let ch_samples = decode_channel(&ch_payload, frames)?;
        channels_data.push(ch_samples);
    }

    // ---------- 2) Обратно из Left+Side в Left+Right ----------
    if channels == 2 {
        let (left_slice, side_slice) = channels_data.split_at_mut(1);
        let left = &mut left_slice[0];
        let side = &mut side_slice[0];

        for i in 0..frames {
            let l = left[i];
            let s = side[i];
            let r = l.wrapping_add(s);
            side[i] = r; // side становится правым каналом
        }
    }

    // ---------- 3) Планар -> interleaved PCM LE ----------
    let total_samples = frames * channels_usize;
    let mut pcm = Vec::with_capacity(total_samples * 2);

    for frame in 0..frames {
        for ch in 0..channels_usize {
            let sample = channels_data[ch][frame];
            pcm.extend_from_slice(&sample.to_le_bytes());
        }
    }

    Ok((pcm, channels, sample_rate))
}

/* ========= Кодек одного канала (блоки + предиктор + bitplane) ========= */

fn encode_channel(samples: &[i16], frames: usize) -> Vec<u8> {
    if frames == 0 {
        return Vec::new();
    }

    let blocks = (frames + BLOCK_FRAMES - 1) / BLOCK_FRAMES;

    // 1) считаем предикторы и residuals
    let mut predictors: Vec<u8> = Vec::with_capacity(blocks);
    let mut residuals: Vec<i16> = Vec::with_capacity(frames);

    let mut start = 0;
    while start < frames {
        let end = (start + BLOCK_FRAMES).min(frames);
        let block_slice = &samples[start..end];

        let (mode, block_residuals) = encode_block_best_predictor(block_slice);

        predictors.push(mode);
        residuals.extend_from_slice(&block_residuals);

        start = end;
    }

    debug_assert_eq!(residuals.len(), frames);

    // 2) residuals -> ZigZag(u16) -> bitplane
    let mut residual_u16: Vec<u16> = Vec::with_capacity(frames);
    for r in residuals {
        residual_u16.push(zigzag_i16_to_u16(r));
    }

    let bitplanes = bitplane_pack_u16(&residual_u16);

    // 3) склейка: [predictors][bitplanes]
    let mut out = Vec::with_capacity(predictors.len() + bitplanes.len());
    out.extend_from_slice(&predictors);
    out.extend_from_slice(&bitplanes);

    out
}

fn decode_channel(data: &[u8], frames: usize) -> Result<Vec<i16>> {
    if frames == 0 {
        if !data.is_empty() {
            return Err(VmrError::FormatError(
                "Non-empty channel data for zero frames".into(),
            ));
        }
        return Ok(Vec::new());
    }

    let blocks = (frames + BLOCK_FRAMES - 1) / BLOCK_FRAMES;
    if data.len() < blocks {
        return Err(VmrError::FormatError(
            "Channel data too short for predictors".into(),
        ));
    }

    let predictors = &data[..blocks];
    let bitplane_bytes = &data[blocks..];

    // ожидаемая длина битпланов
    let chunks = (frames + 7) / 8;
    let expected_bitplane_len = 16 * chunks;

    if bitplane_bytes.len() != expected_bitplane_len {
        return Err(VmrError::FormatError(format!(
            "Bitplane data length mismatch: expected {}, got {}",
            expected_bitplane_len,
            bitplane_bytes.len()
        )));
    }

    let residual_u16 = bitplane_unpack_u16(bitplane_bytes, frames)?;
    debug_assert_eq!(residual_u16.len(), frames);

    let mut residuals: Vec<i16> = Vec::with_capacity(frames);
    for v in residual_u16 {
        residuals.push(unzigzag_u16_to_i16(v));
    }

    let mut out = Vec::with_capacity(frames);
    let mut cursor = 0usize;

    for block_idx in 0..blocks {
        let start_frame = block_idx * BLOCK_FRAMES;
        let block_len = (frames - start_frame).min(BLOCK_FRAMES);

        let mode = predictors[block_idx];
        let end_cursor = cursor + block_len;
        if end_cursor > residuals.len() {
            return Err(VmrError::FormatError(
                "Residuals index out of range".into(),
            ));
        }

        let block_residuals = &residuals[cursor..end_cursor];
        cursor = end_cursor;

        let block_samples = decode_block_predictor(mode, block_residuals)?;
        out.extend_from_slice(&block_samples);
    }

    if cursor != frames {
        return Err(VmrError::FormatError(format!(
            "Residuals cursor mismatch: expected {}, got {}",
            frames, cursor
        )));
    }

    Ok(out)
}

/* ========= Предикторы ========= */

#[derive(Clone, Copy)]
enum Predictor {
    P0, // raw
    P1, // s[n] - s[n-1]
    P2, // s[n] - (2*s[n-1] - s[n-2])
}

fn encode_block_best_predictor(samples: &[i16]) -> (u8, Vec<i16>) {
    let (m0, r0, c0) = encode_block_with_predictor(samples, Predictor::P0);
    let (m1, r1, c1) = encode_block_with_predictor(samples, Predictor::P1);
    let (m2, r2, c2) = encode_block_with_predictor(samples, Predictor::P2);

    let mut best_mode = m0;
    let mut best_residuals = r0;
    let mut best_cost = c0;

    if c1 < best_cost {
        best_mode = m1;
        best_residuals = r1;
        best_cost = c1;
    }
    if c2 < best_cost {
        best_mode = m2;
        best_residuals = r2;
        best_cost = c2;
    }

    (best_mode, best_residuals)
}

fn encode_block_with_predictor(
    samples: &[i16],
    predictor: Predictor,
) -> (u8, Vec<i16>, i64) {
    let n = samples.len();
    let mut residuals = Vec::with_capacity(n);
    let mut cost: i64 = 0;

    match predictor {
        Predictor::P0 => {
            for &s in samples {
                let r = s;
                residuals.push(r);
                cost += (r as i32).abs() as i64;
            }
            (0u8, residuals, cost)
        }
        Predictor::P1 => {
            let mut prev: i32 = 0;
            for &s16 in samples {
                let s = s16 as i32;
                let r16 = (s - prev) as i16;
                residuals.push(r16);
                cost += (r16 as i32).abs() as i64;
                prev = s;
            }
            (1u8, residuals, cost)
        }
        Predictor::P2 => {
            let mut prev1: i32 = 0;
            let mut prev2: i32 = 0;

            for (i, &s16) in samples.iter().enumerate() {
                let s = s16 as i32;
                let pred = if i == 0 {
                    0
                } else if i == 1 {
                    prev1
                } else {
                    2 * prev1 - prev2
                };

                let r16 = (s - pred) as i16;
                residuals.push(r16);
                cost += (r16 as i32).abs() as i64;

                prev2 = prev1;
                prev1 = s;
            }

            (2u8, residuals, cost)
        }
    }
}

fn decode_block_predictor(mode: u8, residuals: &[i16]) -> Result<Vec<i16>> {
    match mode {
        0 => Ok(residuals.to_vec()),
        1 => Ok(decode_p1(residuals)),
        2 => Ok(decode_p2(residuals)),
        _ => Err(VmrError::FormatError(format!(
            "Unknown predictor mode: {}",
            mode
        ))),
    }
}

fn decode_p1(residuals: &[i16]) -> Vec<i16> {
    let mut out = Vec::with_capacity(residuals.len());
    let mut prev: i32 = 0;
    for &r16 in residuals {
        let r = r16 as i32;
        let s = (prev + r) as i16;
        out.push(s);
        prev = s as i32;
    }
    out
}

fn decode_p2(residuals: &[i16]) -> Vec<i16> {
    let mut out = Vec::with_capacity(residuals.len());
    let mut prev1: i32 = 0;
    let mut prev2: i32 = 0;

    for (i, &r16) in residuals.iter().enumerate() {
        let r = r16 as i32;
        let pred = if i == 0 {
            0
        } else if i == 1 {
            prev1
        } else {
            2 * prev1 - prev2
        };

        let s = (pred + r) as i16;
        out.push(s);
        prev2 = prev1;
        prev1 = s as i32;
    }

    out
}

/* ========= ZigZag ========= */

#[inline]
fn zigzag_i16_to_u16(x: i16) -> u16 {
    ((x << 1) ^ (x >> 15)) as u16
}

#[inline]
fn unzigzag_u16_to_i16(v: u16) -> i16 {
    ((v >> 1) as i16) ^ (-((v & 1) as i16))
}

/* ========= Bitplane pack/unpack ========= */

fn bitplane_pack_u16(samples: &[u16]) -> Vec<u8> {
    let n = samples.len();
    if n == 0 {
        return Vec::new();
    }

    let chunks = (n + 7) / 8;
    let mut out = Vec::with_capacity(16 * chunks);

    for plane in (0..16).rev() {
        for chunk in 0..chunks {
            let mut byte = 0u8;
            for bit_idx in 0..8 {
                let sample_idx = chunk * 8 + bit_idx;
                let bit = if sample_idx < n {
                    ((samples[sample_idx] >> plane) & 1) as u8
                } else {
                    0
                };
                byte |= bit << (7 - bit_idx);
            }
            out.push(byte);
        }
    }

    out
}

fn bitplane_unpack_u16(data: &[u8], n_samples: usize) -> Result<Vec<u16>> {
    if n_samples == 0 {
        if !data.is_empty() {
            return Err(VmrError::FormatError(
                "Non-empty bitplane data for zero samples".into(),
            ));
        }
        return Ok(Vec::new());
    }

    let chunks = (n_samples + 7) / 8;
    let expected_len = 16 * chunks;
    if data.len() != expected_len {
        return Err(VmrError::FormatError(format!(
            "Bitplane data length mismatch: expected {}, got {}",
            expected_len,
            data.len()
        )));
    }

    let mut samples = vec![0u16; n_samples];
    let mut offset = 0;

    for plane in (0..16).rev() {
        for chunk in 0..chunks {
            let byte = data[offset];
            offset += 1;
            for bit_idx in 0..8 {
                let sample_idx = chunk * 8 + bit_idx;
                if sample_idx >= n_samples {
                    break;
                }
                let bit = (byte >> (7 - bit_idx)) & 1;
                samples[sample_idx] |= (bit as u16) << plane;
            }
        }
    }

    Ok(samples)
}