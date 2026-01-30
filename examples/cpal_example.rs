
use std::{collections::HashMap, error::Error};

use cpal::traits::{DeviceTrait, HostTrait};
use futures::executor::block_on;
use hound::{SampleFormat as HoundSampleFormat, WavReader, WavSpec, WavWriter};
use melaec3::cpal_aec::{AecConfig, AecStream, InputDeviceConfig, OutputDeviceConfig};

const FRAME_SIZE_MS: u32 = 10;
const FILTER_LENGTH_MS: u32 = 100;
const AEC_SAMPLE_RATE: u32 = 16_000;
const RESAMPLER_QUALITY: i32 = 5;
const HISTORY_LEN: usize = 100;
const CALIBRATION_PACKETS: u32 = 20;
const AUDIO_BUFFER_SECONDS: u32 = 20;
const MAX_OFFSET_FRAMES: usize = 1_000;
const MAX_CALIBRATION_ATTEMPTS : usize = 3;

fn main() -> Result<(), Box<dyn Error>> {
    block_on(async_main())
}

async fn async_main() -> Result<(), Box<dyn Error>> {
    let frame_size = (AEC_SAMPLE_RATE * FRAME_SIZE_MS / 1000) as usize;
    let filter_length = (AEC_SAMPLE_RATE * FILTER_LENGTH_MS / 1000) as usize;

    let aec_config = AecConfig::new(AEC_SAMPLE_RATE, frame_size, filter_length);
    let mut stream = AecStream::new(aec_config)?;

    if let Err(err) = list_devices() {
        eprintln!("Could not list devices: {err}");
    }

    let host = find_host_by_name("ALSA").unwrap_or_else(cpal::default_host);

    let input_device_config = InputDeviceConfig::from_default(
        host.id(),
        "front:CARD=Beyond,DEV=0".to_string(),
        HISTORY_LEN,
        CALIBRATION_PACKETS,
        AUDIO_BUFFER_SECONDS,
        RESAMPLER_QUALITY,
    )
    .await?;

    let output_device_config = OutputDeviceConfig::from_default(
        host.id(),
        "default".to_string(),
        HISTORY_LEN,
        CALIBRATION_PACKETS,
        AUDIO_BUFFER_SECONDS,
        RESAMPLER_QUALITY,
        3,
    )
    .await?;

    stream.add_input_device(&input_device_config).await?;
    let mut stream_output_creator = stream.add_output_device(&output_device_config).await?;

    let pcm_spec_input = WavSpec {
        channels: input_device_config.channels as u16,
        sample_rate: AEC_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: HoundSampleFormat::Int,
    };

    let pcm_spec_output = WavSpec {
        channels: output_device_config.channels as u16,
        ..pcm_spec_input
    };

    let aec_spec = WavSpec {
        sample_format: HoundSampleFormat::Float,
        bits_per_sample: 32,
        ..pcm_spec_input
    };

    let mut in_wav = WavWriter::create("aligned_input.wav", pcm_spec_input)?;
    let mut out_wav = WavWriter::create("aligned_output.wav", pcm_spec_output)?;
    let mut aec_wav = WavWriter::create("aec_applied.wav", aec_spec)?;

    let mut wav = WavReader::open("examples/example_talking.wav")?;
    let spec = wav.spec();
    let wav_channels = spec.channels as usize;
    let wav_rate = spec.sample_rate;
    let wav_samples: Vec<f32> = match spec.sample_format {
        HoundSampleFormat::Int => wav
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / i16::MAX as f32)
            .collect(),
        HoundSampleFormat::Float => wav.samples::<f32>().map(|s| s.unwrap()).collect(),
    };

    let mut channel_map = HashMap::new();
    for i in 0..wav_channels {
        channel_map.insert(i, vec![0]);
    }

    let mut stream_output = stream_output_creator.begin_audio_stream(
        wav_channels,
        channel_map,
        ((wav_samples.len() / wav_rate as usize + 1) * 2000) as u32,
        wav_rate,
        RESAMPLER_QUALITY,
    )?;

    // Wait until both input and output devices have reported ready at least once.
    // update_debug returns the *newly* ready devices for that frame, so we need to track across calls.
    let (mut input_ready, mut output_ready) = (false, false);
    loop {
        let (ready_in, ready_out, _, _, _, _, _) = stream.update_debug().await?;
        if !ready_in.is_empty() {
            input_ready = true;
        }
        if !ready_out.is_empty() {
            output_ready = true;
        }
        if input_ready && output_ready {
            break;
        }
    }

    println!("Computing calibration");
    let calibrated = stream
        .calibrate(MAX_OFFSET_FRAMES, MAX_CALIBRATION_ATTEMPTS, std::slice::from_mut(&mut stream_output_creator), true)
        .await?;
    println!("calibrated: {calibrated}");

    let silence = vec![0.0f32; wav_samples.len()];
    stream_output.queue_audio(wav_samples.as_slice())?;
    stream_output.queue_audio(&silence)?;
    stream_output.queue_audio(&silence)?;
    stream_output.queue_audio(wav_samples.as_slice())?;
    stream_output.queue_audio(&silence)?;
    stream_output.queue_audio(&silence)?;
    stream_output.queue_audio(wav_samples.as_slice())?;
    stream_output.queue_audio(&silence)?;
    stream_output.queue_audio(wav_samples.as_slice())?;

    for _ in 0..6000 {
        let num_input_channels = stream.num_input_channels();
        let (_ready_in, _ready_out, aligned_input, aligned_output, aec_applied, _start, _end) =
            stream.update_debug().await?;
        let _chunk_size = aligned_input.len() / num_input_channels;

        for &s in aligned_input.iter() {
            in_wav.write_sample(s)?;
        }
        for &s in aligned_output.iter() {
            out_wav.write_sample(s)?;
        }
        for &s in aec_applied.iter() {
            aec_wav.write_sample(s)?;
        }
    }

    stream_output_creator.end_audio_stream(&stream_output)?;
    stream_output_creator.interrupt_all_streams()?;

    stream.remove_input_device(&input_device_config)?;
    stream.remove_output_device(&output_device_config)?;

    in_wav.finalize()?;
    out_wav.finalize()?;
    aec_wav.finalize()?;

    Ok(())
}

fn find_host_by_name(target: &str) -> Option<cpal::Host> {
    for host_id in cpal::available_hosts() {
        if host_id.name().eq_ignore_ascii_case(target) {
            if let Ok(host) = cpal::host_from_id(host_id) {
                return Some(host);
            }
        }
    }
    None
}

fn supported_device_configs_to_string(
    device: &cpal::Device,
    direction: &'static str,
) -> Result<String, Box<dyn Error>> {
    let configs: Vec<_> = match direction {
        "Input" => device
            .supported_input_configs()
            .map(|configs| configs.collect())
            .map_err(|err| format!("Unable to enumerate input configs: {err}"))?,
        "Output" => device
            .supported_output_configs()
            .map(|configs| configs.collect())
            .map_err(|err| format!("Unable to enumerate output configs: {err}"))?,
        other => return Err(format!("Unknown direction '{other}'").into()),
    };

    Ok(configs
        .iter()
        .map(|cfg| {
            let min_rate = cfg.min_sample_rate().0;
            let max_rate = cfg.max_sample_rate().0;
            let rate_desc = if min_rate == max_rate {
                format!("{min_rate} Hz")
            } else {
                format!("{min_rate}-{max_rate} Hz")
            };
            format!(
                "{} channel(s), {:?}, sample rates: {rate_desc}",
                cfg.channels(),
                cfg.sample_format()
            )
        })
        .collect::<Vec<_>>()
        .join("; "))
}

fn list_devices() -> Result<(), Box<dyn Error>> {
    for host_id in cpal::available_hosts() {
        println!("Host: '{}'", host_id.name());
        let host = cpal::host_from_id(host_id)?;

        for dev in host.input_devices()? {
            let name = dev.name()?;
            println!("  input: '{name}'");
            match supported_device_configs_to_string(&dev, "Input") {
                Ok(cfgs) => println!("      {cfgs}"),
                Err(err) => println!("      {err}"),
            }
        }

        for dev in host.output_devices()? {
            let name = dev.name()?;
            println!("  output: '{name}'");
            match supported_device_configs_to_string(&dev, "Output") {
                Ok(cfgs) => println!("      {cfgs}"),
                Err(err) => println!("      {err}"),
            }
        }
    }

    Ok(())
}
