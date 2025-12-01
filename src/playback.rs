use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SizedSample};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[derive(Debug)]
pub enum PlayError {
    NoDevice,
    Cpal(String),
}

impl std::fmt::Display for PlayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlayError::NoDevice => write!(f, "No default output device"),
            PlayError::Cpal(msg) => write!(f, "CPAL error: {msg}"),
        }
    }
}

impl std::error::Error for PlayError {}

/// samples: i16 interleaved, src_channels: 1/2, src_sample_rate: из VMR
pub fn play_i16(
    samples: &[i16],
    src_channels: u16,
    src_sample_rate: u32,
) -> Result<(), PlayError> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or(PlayError::NoDevice)?;

    // Берём конфиг, который железка точно умеет
    let supported_config = device
        .default_output_config()
        .map_err(|e| PlayError::Cpal(e.to_string()))?;

    let sample_format = supported_config.sample_format();
    let config = supported_config.config(); // StreamConfig

    let dst_channels = config.channels;
    let dst_sample_rate = config.sample_rate.0;

    // Подгоняем наш буфер под формат устройства (каналы + sample_rate)
    let converted: Vec<i16> = convert_audio(
        samples,
        src_channels,
        src_sample_rate,
        dst_channels,
        dst_sample_rate,
    );

    let data = Arc::new(converted);
    let len = data.len();
    let index = Arc::new(AtomicUsize::new(0));

    let err_fn = |e| eprintln!("[vmr] Audio stream error: {e}");

    let stream = match sample_format {
        cpal::SampleFormat::I16 => build_stream::<i16>(
            &device,
            &config,
            data.clone(),
            index.clone(),
            len,
            err_fn,
        ),
        cpal::SampleFormat::U16 => build_stream::<u16>(
            &device,
            &config,
            data.clone(),
            index.clone(),
            len,
            err_fn,
        ),
        cpal::SampleFormat::F32 => build_stream::<f32>(
            &device,
            &config,
            data.clone(),
            index.clone(),
            len,
            err_fn,
        ),
        _ => {
            return Err(PlayError::Cpal(
                "Unsupported sample format".into(),
            ))
        }
    }?;

    stream
        .play()
        .map_err(|e| PlayError::Cpal(e.to_string()))?;

    // Ждём, пока все сэмплы не отдадим
    while index.load(Ordering::SeqCst) < len {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    Ok(())
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    samples: Arc<Vec<i16>>,
    index: Arc<AtomicUsize>,
    len: usize,
    err_fn: impl Fn(cpal::StreamError) + Send + 'static,
) -> Result<cpal::Stream, PlayError>
where
    T: Sample + SizedSample + FromSample<i16>,
{
    let channels = config.channels as usize;

    let stream = device
        .build_output_stream(
            config,
            move |output: &mut [T], _info: &cpal::OutputCallbackInfo| {
                let mut i = index.load(Ordering::SeqCst);

                for frame in output.chunks_mut(channels) {
                    for sample_out in frame {
                        let v = if i < len {
                            let s = samples[i];
                            i += 1;
                            s
                        } else {
                            0
                        };

                        *sample_out = T::from_sample(v);
                    }
                }

                index.store(i, Ordering::SeqCst);
            },
            err_fn,
            None,
        )
        .map_err(|e| PlayError::Cpal(e.to_string()))?;

    Ok(stream)
}

/// Конвертация каналов + простейший ресемплинг под формат устройства.
fn convert_audio(
    samples: &[i16],
    src_channels: u16,
    src_sample_rate: u32,
    dst_channels: u16,
    dst_sample_rate: u32,
) -> Vec<i16> {
    let src_ch = src_channels.max(1) as usize;
    let dst_ch = dst_channels.max(1) as usize;

    let src_frames = samples.len() / src_ch;

    // 1) Сначала приводим количество каналов к dst_ch, частоту пока не трогаем
    let mut tmp: Vec<i16> = Vec::with_capacity(src_frames * dst_ch);

    for f in 0..src_frames {
        let base = f * src_ch;

        match (src_channels, dst_channels) {
            (1, 1) => {
                let s = samples[base];
                tmp.push(s);
            }
            (1, 2) => {
                let s = samples[base];
                tmp.push(s);
                tmp.push(s);
            }
            (2, 1) => {
                let l = samples[base] as i32;
                let r = samples[base + 1] as i32;
                let m = ((l + r) / 2)
                    .clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                tmp.push(m);
            }
            (2, 2) => {
                tmp.push(samples[base]);
                tmp.push(samples[base + 1]);
            }
            // Общий случай: копируем min(ch), остальные забиваем нулями
            (sc, dc) => {
                let min_ch = (sc.min(dc)).max(1) as usize;

                for ch in 0..dst_ch {
                    if ch < min_ch && ch < src_ch {
                        tmp.push(samples[base + ch]);
                    } else {
                        tmp.push(0);
                    }
                }
            }
        }
    }

    // 2) Если sample rate совпадает — ресэмплинг не нужен
    if src_sample_rate == dst_sample_rate {
        return tmp;
    }

    // 3) Простейший ресэмплинг (линейная интерполяция по фреймам)
    let src_frames2 = src_frames;
    if src_frames2 == 0 {
        return Vec::new();
    }

    let dst_frames =
        ((src_frames2 as u64 * dst_sample_rate as u64) / src_sample_rate as u64)
            as usize;

    let mut out: Vec<i16> = Vec::with_capacity(dst_frames * dst_ch);

    for i in 0..dst_frames {
        let t = (i as f64) * (src_sample_rate as f64) / (dst_sample_rate as f64);
        let i0 = t.floor() as usize;
        let frac = t - (i0 as f64);
        let i1 = if i0 + 1 < src_frames2 { i0 + 1 } else { i0 };

        for ch in 0..dst_ch {
            let s0 = tmp[i0 * dst_ch + ch] as f32;
            let s1 = tmp[i1 * dst_ch + ch] as f32;
            let v_f = s0 + (s1 - s0) * (frac as f32);

            let v_i = if v_f > i16::MAX as f32 {
                i16::MAX
            } else if v_f < i16::MIN as f32 {
                i16::MIN
            } else {
                v_f.round() as i16
            };

            out.push(v_i);
        }
    }

    out
}
