use clap::{Parser, Subcommand};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use vmr::{vmr_decode, vmr_encode_with_stats};

mod playback;

#[derive(Parser)]
#[command(name = "vmr", version, about = "VMR audio coder")]
struct Cli {
    #[arg(short, long = "verbose", action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    EncodeWav {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    DecodeWav {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    Play {
        #[arg(short, long)]
        input: PathBuf,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let verbose = cli.verbose > 0;
    vmr::set_verbosity(cli.verbose);

    match cli.command {
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
                return Err("Only mono/stereo WAV supported".into());
            }

            let channels = spec.channels as u8;
            let sample_rate = spec.sample_rate;

            if verbose {
                println!(
                    "[vmr] Encoding WAV {} ({} Hz, {} ch, 16-bit PCM)",
                    input_str, sample_rate, channels
                );
                println!("[vmr] WAV file size: {} bytes", wav_file_size);
            }

            let t_read0 = Instant::now();
            let samples: Vec<i16> = reader.samples::<i16>().collect::<Result<_, _>>()?;
            let read_elapsed = t_read0.elapsed();

            let mut pcm = Vec::with_capacity(samples.len() * 2);
            for s in samples {
                pcm.extend_from_slice(&s.to_le_bytes());
            }

            let raw_size = pcm.len();

            if verbose {
                println!(
                    "[vmr] Extracted raw PCM from WAV: {} bytes (read time: {:.} sec)",
                    raw_size,
                    read_elapsed.as_secs_f64()
                );
                println!(
                    "[vmr] Original WAV size: {:.3} MB",
                    wav_file_size as f64 / (1024.0 * 1024.0)
                );
            }

            let t0 = Instant::now();
            let (encoded, stats) = vmr_encode_with_stats(&pcm, channels, sample_rate)?;
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
                    "WAV",
                );
            }

            Ok(())
        }

        Commands::DecodeWav { input, output } => {
            use hound::{SampleFormat, WavSpec, WavWriter};

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
                return Err("Decoded PCM size not aligned to 16-bit".into());
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
                let wav_out_mb = fs::metadata(&output)?.len() as f64 / (1024.0 * 1024.0);

                println!("[vmr] Decoded {} -> {}", input_str, output_str);
                println!(
                    "[vmr] Raw PCM: {} bytes ({:.3} MB), {} ch, {} Hz",
                    raw_size, raw_mb, channels, sample_rate
                );
                println!("[vmr] WAV file size: {:.3} MB", wav_out_mb);
                println!(
                    "[vmr] Decode time (VMR -> PCM): {:.4} sec",
                    decode_elapsed.as_secs_f64()
                );
                println!("[vmr] WAV write time: {:.4} sec", wav_elapsed.as_secs_f64());
            }

            Ok(())
        }

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

            let samples: Vec<i16> = pcm
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                .collect();

            if verbose {
                let frames = samples.len() / channels as usize;
                let duration_sec = frames as f64 / sample_rate as f64;
                println!(
                    "[vmr] Playing: {} ch, {} Hz, ~{:.3} sec",
                    channels, sample_rate, duration_sec
                );
            }

            playback::play_i16(&samples, channels as u16, sample_rate)?;
            Ok(())
        }
    }
}

fn log_encode_stats(
    input_str: &str,
    output_str: &str,
    raw_size: usize,
    encoded: &[u8],
    stats: &vmr::EncodeStats,
    secs: f64,
    source_label: &str,
) {
    let vmr_size = encoded.len();
    let raw_mb = raw_size as f64 / (1024.0 * 1024.0);
    let payload_mb = stats.payload_size as f64 / (1024.0 * 1024.0);
    let vmr_mb = vmr_size as f64 / (1024.0 * 1024.0);

    let ratio = if vmr_size > 0 {
        raw_size as f64 / vmr_size as f64
    } else {
        0.0
    };

    println!("[vmr] Encoded {} -> {}", input_str, output_str);
    println!(
        "[vmr] {} -> RAW planar -> VMR: {:.3}MB -> {:.3}MB -> {:.3}MB",
        source_label, raw_mb, payload_mb, vmr_mb
    );
    println!("[vmr] RAW planar size: {} bytes", stats.raw_size);
    println!("[vmr] Payload size: {} bytes", stats.payload_size);
    println!("[vmr] Total VMR size: {} bytes", vmr_size);
    println!("[vmr] Compression ratio vs raw: {:.3}x smaller", ratio);
    println!(
        "[vmr] Total encode time (full pipeline): {:.4} sec",
        secs
    );
    if secs > 0.0 {
        println!("[vmr] Speed: {:.3} MB/s", raw_mb / secs);
    }
}