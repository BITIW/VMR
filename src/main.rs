use clap::{Parser, Subcommand};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use vmr::{vmr_decode, vmr_encode_with_stats};

mod playback;

#[derive(Parser)]
#[command(
    name = "vmr",
    version,
    about = "VMR (Very Maximal Rust) audio coder"
)]
struct Cli {
    /// Verbose logs (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Encode raw PCM (16-bit LE) into VMR
    Encode {
        /// Input raw PCM file (16-bit LE, mono/stereo)
        #[arg(short, long)]
        input: PathBuf,

        /// Output VMR file
        #[arg(short, long)]
        output: PathBuf,

        /// Channels: 1 = mono, 2 = stereo
        #[arg(short, long)]
        channels: u8,

        /// Sample rate, e.g. 44100
        #[arg(short = 'r', long)]
        sample_rate: u32,
    },

    /// Encode WAV (16-bit PCM) into VMR
    EncodeWav {
        /// Input WAV file
        #[arg(short, long)]
        input: PathBuf,

        /// Output VMR file
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Decode VMR into raw PCM (16-bit LE)
    Decode {
        /// Input VMR file
        #[arg(short, long)]
        input: PathBuf,

        /// Output raw PCM
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Decode VMR into WAV (16-bit PCM)
    DecodeWav {
        /// Input VMR file
        #[arg(short, long)]
        input: PathBuf,

        /// Output WAV file
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Decode and play VMR through default audio output
    Play {
        /// Input VMR file
        #[arg(short, long)]
        input: PathBuf,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let verbose = cli.verbose > 0;

    match cli.command {
        // ---------- RAW PCM -> VMR ----------
        Commands::Encode {
            input,
            output,
            channels,
            sample_rate,
        } => {
            let input_str = input.to_string_lossy();
            let output_str = output.to_string_lossy();

            let pcm = fs::read(&input)?;
            let raw_size = pcm.len();

            if verbose {
                println!(
                    "[vmr] Encoding RAW {} (16-bit PCM, {} ch, {} Hz)",
                    input_str, channels, sample_rate
                );
                println!("[vmr] Raw size: {} bytes", raw_size);
            }

            let t0 = Instant::now();
            let (encoded, stats) =
                vmr_encode_with_stats(&pcm, channels, sample_rate)?;
            let elapsed = t0.elapsed();

            fs::write(&output, &encoded)?;

            if verbose {
                log_encode_stats(
                    &input_str,
                    &output_str,
                    raw_size,
                    &encoded,
                    &stats,
                    elapsed.as_secs_f64(),
                    Some("RAW"),
                );
            }

            Ok(())
        }

        // ---------- WAV -> VMR ----------
        Commands::EncodeWav { input, output } => {
            use hound::WavReader;

            let input_str = input.to_string_lossy();
            let output_str = output.to_string_lossy();

            let wav_file_size = fs::metadata(&input)?.len() as usize;

            let mut reader = WavReader::open(&input)?;
            let spec = reader.spec();

            if spec.bits_per_sample != 16 || spec.sample_format != hound::SampleFormat::Int {
                return Err("Only 16-bit PCM WAV is supported".into());
            }

            if spec.channels == 0 || spec.channels > 2 {
                return Err("Only mono or stereo WAV is supported".into());
            }

            let channels = spec.channels;
            let sample_rate = spec.sample_rate;

            if verbose {
                println!(
                    "[vmr] Encoding WAV {} ({} Hz, {} ch, 16-bit PCM)",
                    input_str, sample_rate, channels
                );
                println!("[vmr] WAV file size: {} bytes", wav_file_size);
            }

            let t_read0 = Instant::now();

            let samples_result: Result<Vec<i16>, _> = reader.samples::<i16>().collect();
            let samples = samples_result?;
            let read_elapsed = t_read0.elapsed();

            let mut pcm = Vec::with_capacity(samples.len() * 2);
            for s in samples {
                pcm.extend_from_slice(&s.to_le_bytes());
            }

            let raw_size = pcm.len();

            if verbose {
                println!(
                    "[vmr] Extracted raw PCM from WAV: {} bytes (read time: {:.3} sec)",
                    raw_size,
                    read_elapsed.as_secs_f64()
                );
            }

            let t0 = Instant::now();
            let (encoded, stats) =
                vmr_encode_with_stats(&pcm, channels as u8, sample_rate)?;
            let elapsed = t0.elapsed();

            fs::write(&output, &encoded)?;

            if verbose {
                println!(
                    "[vmr] Original WAV size: {:.2} MB",
                    wav_file_size as f64 / (1024.0 * 1024.0)
                );
                log_encode_stats(
                    &input_str,
                    &output_str,
                    raw_size,
                    &encoded,
                    &stats,
                    elapsed.as_secs_f64(),
                    Some("WAV"),
                );
            }

            Ok(())
        }

        // ---------- VMR -> RAW PCM ----------
        Commands::Decode { input, output } => {
            let input_str = input.to_string_lossy();
            let output_str = output.to_string_lossy();

            let vmr_data = fs::read(&input)?;
            if verbose {
                println!(
                    "[vmr] Decoding {} ({} bytes) -> RAW PCM",
                    input_str,
                    vmr_data.len()
                );
            }

            let t0 = Instant::now();
            let (pcm, channels, sample_rate) = vmr_decode(&vmr_data)?;
            let elapsed = t0.elapsed();

            fs::write(&output, &pcm)?;

            if verbose {
                let raw_size = pcm.len();
                let raw_mb = raw_size as f64 / (1024.0 * 1024.0);
                println!(
                    "[vmr] Decoded {} -> {}",
                    input_str, output_str
                );
                println!(
                    "[vmr] Output RAW: {} bytes ({:.2} MB), {} ch, {} Hz",
                    raw_size, raw_mb, channels, sample_rate
                );
                println!(
                    "[vmr] Total decode time: {:.3} sec",
                    elapsed.as_secs_f64()
                );
            }

            Ok(())
        }

        // ---------- VMR -> WAV ----------
        Commands::DecodeWav { input, output } => {
            use hound::{WavSpec, WavWriter, SampleFormat};

            let input_str = input.to_string_lossy();
            let output_str = output.to_string_lossy();

            let vmr_data = fs::read(&input)?;
            if verbose {
                println!(
                    "[vmr] Decoding {} ({} bytes) -> WAV",
                    input_str,
                    vmr_data.len()
                );
            }

            let t0 = Instant::now();
            let (pcm, channels, sample_rate) = vmr_decode(&vmr_data)?;
            let decode_elapsed = t0.elapsed();

            if pcm.len() % 2 != 0 {
                return Err("Decoded PCM size is not aligned to 16-bit samples".into());
            }

            let samples: Vec<i16> = pcm
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                .collect();

            let spec = WavSpec {
                channels: channels as u16,
                sample_rate,
                bits_per_sample: 16,
                sample_format: SampleFormat::Int,
            };

            let t_wav0 = Instant::now();
            let mut writer = WavWriter::create(&output, spec)?;
            for s in samples {
                writer.write_sample(s)?;
            }
            writer.finalize()?;
            let wav_elapsed = t_wav0.elapsed();

            if verbose {
                let raw_size = pcm.len();
                let raw_mb = raw_size as f64 / (1024.0 * 1024.0);
                let wav_file_size = fs::metadata(&output)?.len() as f64 / (1024.0 * 1024.0);

                println!(
                    "[vmr] Decoded {} -> {}",
                    input_str, output_str
                );
                println!(
                    "[vmr] Raw PCM: {} bytes ({:.2} MB), {} ch, {} Hz",
                    raw_size, raw_mb, channels, sample_rate
                );
                println!(
                    "[vmr] WAV file size: {:.2} MB",
                    wav_file_size
                );
                println!(
                    "[vmr] Decode time (VMR -> PCM): {:.3} sec",
                    decode_elapsed.as_secs_f64()
                );
                println!(
                    "[vmr] WAV write time: {:.3} sec",
                    wav_elapsed.as_secs_f64()
                );
            }

            Ok(())
        }

        // ---------- VMR -> play ----------
        Commands::Play { input } => {
            let input_str = input.to_string_lossy();
            let vmr_data = fs::read(&input)?;

            if verbose {
                println!(
                    "[vmr] Decoding {} for playback ({} bytes)",
                    input_str,
                    vmr_data.len()
                );
            }

            let (pcm, channels, sample_rate) = vmr_decode(&vmr_data)?;

            if pcm.len() % 2 != 0 {
                return Err("Decoded PCM size is not aligned to 16-bit samples".into());
            }

            let samples: Vec<i16> = pcm
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                .collect();

            let frames = samples.len() / channels as usize;
            let duration_sec = frames as f64 / sample_rate as f64;

            if verbose {
                println!(
                    "[vmr] Playing: {} ch, {} Hz, ~{:.2} sec",
                    channels, sample_rate, duration_sec
                );
            }

            playback::play_i16(&samples, channels as u16, sample_rate)?;

            Ok(())
        }
    }
}

/// Красивый лог в духе примера, с разными источниками (RAW/WAV)
fn log_encode_stats(
    input_str: &str,
    output_str: &str,
    raw_size: usize,
    encoded: &[u8],
    stats: &vmr::EncodeStats,
    secs: f64,
    source_label: Option<&str>,
) {
    let vmr_size = encoded.len();
    let bitplane_mb = stats.bitplane_size as f64 / (1024.0 * 1024.0);
    let raw_mb = raw_size as f64 / (1024.0 * 1024.0);
    let vmr_mb = vmr_size as f64 / (1024.0 * 1024.0);
    let ratio = if vmr_size > 0 {
        raw_size as f64 / vmr_size as f64
    } else {
        0.0
    };

    let label = source_label.unwrap_or("RAW");

    println!(
        "[vmr] Encoded {} -> {}",
        input_str,
        output_str
    );
    println!(
        "[vmr] {} -> RAW planar -> VMR: {:.2}MB -> {:.2}MB -> {:.2}MB",
        label, raw_mb, bitplane_mb, vmr_mb
    );
    println!(
        "[vmr] RAW planar size: {} bytes",
        stats.raw_size
    );
    println!(
        "[vmr] Bitplane size: {} bytes",
        stats.bitplane_size
    );
    println!(
        "[vmr] Total VMR size: {} bytes",
        vmr_size
    );
    println!(
        "[vmr] Compression ratio vs raw: {:.2}x smaller",
        ratio
    );
    println!(
        "[vmr] Total encode time (full pipeline): {:.3} sec",
        secs
    );
    if secs > 0.0 {
        let speed_mb_s = raw_mb / secs;
        println!("[vmr] Speed: {:.2} MB/s", speed_mb_s);
    }
}