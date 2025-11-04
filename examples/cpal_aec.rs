//! Example demonstrating how to wire `speex-rust-aec` into a CPAL input/output
//! pipeline.
//!
//! Requires building with `--features cpal-example` so the optional `cpal`
//! dependency is enabled:
//!
//! ```text
//! cargo run --example cpal_aec --features cpal-example
//! ```
//!
//! You can optionally pass the desired input and output device names (substring
//! match) as the first and second command-line arguments:
//!
//! ```text
//! cargo run --example cpal_aec --features cpal-example "USB Microphone" "Loopback"
//! ```
//!
//! ```text
//! cargo run --example cpal_aec --features cpal-example "USB Microphone" "Loopback" 48000
//! ```
//!
//! The optional third argument selects the internal echo-canceller sample rate (Hz). When
//! omitted, the demo defaults to 48 kHz and transparently resamples the devices if needed.

use std::{
    collections::VecDeque,
    env,
    error::Error,
    ffi::c_void,
    mem,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Device, SampleFormat, Stream, StreamConfig,
};
use hound::{self, WavSpec};
use speex_rust_aec::{
    speex_echo_cancellation, speex_echo_ctl, EchoCanceller, Resampler, SPEEX_ECHO_SET_SAMPLING_RATE,
};

const DEFAULT_AEC_RATE: u32 = 48_000;
const RESAMPLER_QUALITY: i32 = 5;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    let host = cpal::default_host();

    let target_sample_rate = args
        .get(2)
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(DEFAULT_AEC_RATE);

    let far_audio_path = args
        .get(3)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("examples/example_talking.wav"));

    let input_device = if let Some(name) = args.get(0) {
        select_device(host.input_devices(), name, "Input")?
    } else {
        host.default_input_device()
            .ok_or("No default input device available")?
    };

    let output_device = if let Some(name) = args.get(1) {
        select_device(host.output_devices(), name, "Output")?
    } else {
        host.default_output_device()
            .ok_or("No default output device available")?
    };

    let input_config = input_device.default_input_config()?;
    let output_config = output_device.default_output_config()?;

    let input_stream_config: StreamConfig = input_config.config();
    let output_stream_config: StreamConfig = output_config.config();

    if input_stream_config.channels != output_stream_config.channels {
        eprintln!(
            "Input/output channel mismatch: {} in vs {} out. Matching channels are required.",
            input_stream_config.channels, output_stream_config.channels
        );
        return Ok(());
    }

    let input_rate = input_stream_config.sample_rate.0;
    let output_rate = output_stream_config.sample_rate.0;
    let channels = input_stream_config.channels as usize;

    println!(
        "Input rate: {input_rate} Hz, output rate: {output_rate} Hz, AEC rate: {target_sample_rate} Hz"
    );

    let far_audio = load_far_audio(&far_audio_path, channels, output_rate)
        .map_err(|e| format!("Failed to load far-end audio '{}': {e}", far_audio_path.display()))?;
    println!("Far-end audio source: {}", far_audio_path.display());
    let far_source = Arc::new(Mutex::new(far_audio));

    let mut mic_resampler = if input_rate != target_sample_rate {
        println!("Resampling input stream to match AEC rate.");
        Some(
            Resampler::new(
                channels as u32,
                input_rate,
                target_sample_rate,
                RESAMPLER_QUALITY,
            )
            .map_err(|e| format!("Failed to create input resampler: {e}"))?,
        )
    } else {
        None
    };

    if let Some(resampler) = mic_resampler.as_mut() {
        if let Err(err) = resampler.skip_zeros() {
            eprintln!("Unable to prime input resampler: {err}");
        }
    }

    let mut far_resampler = if output_rate != target_sample_rate {
        println!("Resampling output reference stream to match AEC rate.");
        Some(
            Resampler::new(
                channels as u32,
                output_rate,
                target_sample_rate,
                RESAMPLER_QUALITY,
            )
            .map_err(|e| format!("Failed to create output resampler: {e}"))?,
        )
    } else {
        None
    };

    if let Some(resampler) = far_resampler.as_mut() {
        if let Err(err) = resampler.skip_zeros() {
            eprintln!("Unable to prime output resampler: {err}");
        }
    }

    let frame_size = (target_sample_rate / 100).max(1) as usize; // ~10 ms frames
    let filter_length = frame_size * 20; // 200 ms echo tail

    let canceller =
        EchoCanceller::new_multichannel(frame_size, filter_length, channels, channels)
            .ok_or("Failed to allocate echo canceller state")?;

    unsafe {
        let mut rate = target_sample_rate as i32;
        speex_echo_ctl(
            canceller.as_ptr(),
            SPEEX_ECHO_SET_SAMPLING_RATE,
            &mut rate as *mut _ as *mut c_void,
        );
    }

    let shared = Arc::new(Mutex::new(SharedCanceller::new(
        canceller,
        channels,
        frame_size,
        mic_resampler,
        far_resampler,
    )));

    let output_stream = build_output_stream(
        &output_device,
        &output_stream_config,
        Arc::clone(&shared),
        Arc::clone(&far_source),
        channels,
        output_config.sample_format(),
    )?;

    let input_stream = build_input_stream(
        &input_device,
        &input_stream_config,
        Arc::clone(&shared),
        input_config.sample_format(),
    )?;

    output_stream.play()?;
    input_stream.play()?;

    println!("Running Speex AEC demo for five seconds...");
    std::thread::sleep(Duration::from_secs(5));
    println!("Done.");

    drop(output_stream);
    drop(input_stream);

    let recording_path = PathBuf::from("examples/aec_output.wav");
    let recorded_samples = {
        let mut guard = shared.lock().unwrap();
        guard.take_recording()
    };

    if recorded_samples.is_empty() {
        eprintln!("No echo-cancelled audio captured; skipping write to {}", recording_path.display());
    } else {
        let spec = WavSpec {
            channels: channels as u16,
            sample_rate: target_sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&recording_path, spec)?;
        for sample in recorded_samples {
            writer.write_sample(sample)?;
        }
        writer.finalize()?;
        println!(
            "Wrote echo-cancelled audio to {}",
            recording_path.display()
        );
    }

    Ok(())
}

struct SharedCanceller {
    aec: EchoCanceller,
    frame_samples: usize,
    mic_resampler: Option<Resampler>,
    far_resampler: Option<Resampler>,
    mic_resampler_pending: Vec<i16>,
    far_resampler_pending: Vec<i16>,
    mic_resample_buf: Vec<i16>,
    far_resample_buf: Vec<i16>,
    input_convert: Vec<i16>,
    far_queue: VecDeque<i16>,
    mic_queue: VecDeque<i16>,
    mic_frame: Vec<i16>,
    far_frame: Vec<i16>,
    out_frame: Vec<i16>,
    processed_frames: usize,
    recorded: Vec<i16>,
}

impl SharedCanceller {
    fn new(
        aec: EchoCanceller,
        channels: usize,
        frame_size: usize,
        mic_resampler: Option<Resampler>,
        far_resampler: Option<Resampler>,
    ) -> Self {
        let frame_samples = frame_size * channels;
        Self {
            aec,
            frame_samples,
            mic_resampler,
            far_resampler,
            mic_resampler_pending: Vec::with_capacity(frame_samples),
            far_resampler_pending: Vec::with_capacity(frame_samples),
            mic_resample_buf: Vec::with_capacity(frame_samples),
            far_resample_buf: Vec::with_capacity(frame_samples),
            input_convert: Vec::with_capacity(frame_samples),
            far_queue: VecDeque::with_capacity(frame_samples * 8),
            mic_queue: VecDeque::with_capacity(frame_samples * 8),
            mic_frame: vec![0; frame_samples],
            far_frame: vec![0; frame_samples],
            out_frame: vec![0; frame_samples],
            processed_frames: 0,
            recorded: Vec::with_capacity(frame_samples * 8),
        }
    }

    fn push_far_end(&mut self, samples: &[i16]) {
        if let Some(resampler) = self.far_resampler.as_mut() {
            Self::queue_resampled(
                resampler,
                &mut self.far_resampler_pending,
                samples,
                &mut self.far_resample_buf,
                &mut self.far_queue,
            );
        } else {
            self.far_queue.extend(samples.iter().copied());
        }
        let max_len = self.frame_samples * 16;
        while self.far_queue.len() > max_len {
            self.far_queue.pop_front();
        }
    }

    fn process_capture(&mut self, samples: &[i16]) {
        if let Some(resampler) = self.mic_resampler.as_mut() {
            Self::queue_resampled(
                resampler,
                &mut self.mic_resampler_pending,
                samples,
                &mut self.mic_resample_buf,
                &mut self.mic_queue,
            );
        } else {
            self.mic_queue.extend(samples.iter().copied());
        }
        while self.mic_queue.len() >= self.frame_samples {
            for sample in self.mic_frame.iter_mut() {
                *sample = self.mic_queue.pop_front().unwrap();
            }
            for sample in self.far_frame.iter_mut() {
                *sample = self.far_queue.pop_front().unwrap_or(0);
            }
            unsafe {
                speex_echo_cancellation(
                    self.aec.as_ptr(),
                    self.mic_frame.as_ptr(),
                    self.far_frame.as_ptr(),
                    self.out_frame.as_mut_ptr(),
                );
            }
            self.processed_frames += 1;
            self.recorded
                .extend(self.out_frame.iter().copied());
            if self.processed_frames % 50 == 0 {
                let rms = (self
                    .out_frame
                    .iter()
                    .map(|s| (*s as f64) * (*s as f64))
                    .sum::<f64>()
                    / self.out_frame.len() as f64)
                    .sqrt();
                println!("Echo-cancelled frame RMS: {rms:.2}");
            }
        }
    }

    fn queue_resampled(
        resampler: &mut Resampler,
        pending: &mut Vec<i16>,
        new_samples: &[i16],
        scratch: &mut Vec<i16>,
        queue: &mut VecDeque<i16>,
    ) {
        if !new_samples.is_empty() {
            pending.extend_from_slice(new_samples);
        }
        let (in_rate, out_rate) = resampler.get_rate();
        let channels = resampler.channels();

        loop {
            if pending.len() < channels {
                break;
            }

            let available_frames = pending.len() / channels;
            if available_frames == 0 {
                break;
            }
            let available_samples = available_frames * channels;

            let mut expected_frames = (available_frames as u64 * out_rate as u64
                + in_rate as u64
                - 1)
                / in_rate as u64;
            expected_frames = expected_frames.max(1);
            let expected_samples = expected_frames as usize * channels;
            scratch.resize(expected_samples, 0);

            match resampler.process_interleaved_i16(
                &pending[..available_samples],
                scratch.as_mut_slice(),
            ) {
                Ok((consumed, produced)) => {
                    if produced > 0 {
                        queue.extend(scratch[..produced].iter().copied());
                    }
                    if consumed == 0 {
                        break;
                    }
                    if consumed >= pending.len() {
                        pending.clear();
                    } else {
                        pending.drain(..consumed);
                    }
                }
                Err(err) => {
                    eprintln!("Resampler error: {err}");
                    break;
                }
            }
        }
    }

    fn take_recording(&mut self) -> Vec<i16> {
        mem::take(&mut self.recorded)
    }
}

struct FarAudioSource {
    samples: Vec<i16>,
    channels: usize,
    position: usize,
    resampler: Option<Resampler>,
    pending: Vec<i16>,
    scratch: Vec<i16>,
    queue: VecDeque<i16>,
}

impl FarAudioSource {
    fn new(
        samples: Vec<i16>,
        source_rate: u32,
        output_rate: u32,
        channels: usize,
    ) -> Result<Self, String> {
        if channels == 0 {
            return Err("Output channel count must be greater than zero".into());
        }
        if samples.is_empty() {
            return Err("Audio file contained no samples".into());
        }

        let resampler = if source_rate != output_rate {
            let mut r = Resampler::new(
                channels as u32,
                source_rate,
                output_rate,
                RESAMPLER_QUALITY,
            )
            .map_err(|e| format!("Failed to create playback resampler: {e}"))?;
            if let Err(err) = r.skip_zeros() {
                eprintln!("Unable to prime playback resampler: {err}");
            }
            Some(r)
        } else {
            None
        };

        Ok(Self {
            samples,
            channels,
            position: 0,
            resampler,
            pending: Vec::with_capacity(channels * 1024),
            scratch: Vec::with_capacity(channels * 1024),
            queue: VecDeque::with_capacity(channels * 1024),
        })
    }

    fn next_samples(&mut self, out: &mut [i16]) {
        if out.is_empty() {
            return;
        }
        if let Some(_) = self.resampler {
            self.fill_resampled(out.len());
            for sample in out.iter_mut() {
                if let Some(value) = self.queue.pop_front() {
                    *sample = value;
                } else {
                    *sample = 0;
                }
            }
        } else {
            let len = self.samples.len();
            let mut pos = self.position;
            for sample in out.iter_mut() {
                *sample = self.samples[pos];
                pos += 1;
                if pos >= len {
                    pos = 0;
                }
            }
            self.position = pos;
        }
    }

    fn fill_resampled(&mut self, required_samples: usize) {
        let Some(resampler) = self.resampler.as_mut() else {
            return;
        };
        if self.samples.is_empty() {
            return;
        }
        let len = self.samples.len();
        while self.queue.len() < required_samples {
            let before = self.queue.len();
            let chunk_frames = ((required_samples.saturating_sub(before)) / self.channels)
                .max(1);
            let chunk_samples = chunk_frames * self.channels;
            self.pending.reserve(chunk_samples);
            for _ in 0..chunk_samples {
                self.pending.push(self.samples[self.position]);
                self.position += 1;
                if self.position >= len {
                    self.position = 0;
                }
            }
            SharedCanceller::queue_resampled(
                resampler,
                &mut self.pending,
                &[],
                &mut self.scratch,
                &mut self.queue,
            );
            if self.queue.len() == before {
                break;
            }
        }
    }
}

fn load_far_audio(
    path: &Path,
    output_channels: usize,
    output_rate: u32,
) -> Result<FarAudioSource, Box<dyn Error>> {
    if output_channels == 0 {
        return Err("Output channel count must be greater than zero".into());
    }

    let mut reader = hound::WavReader::open(path)
        .map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    let spec = reader.spec();

    if spec.channels == 0 {
        return Err("WAV file reports zero channels".into());
    }

    let source_channels = spec.channels as usize;
    let mut frame = vec![0i16; source_channels];
    let mut frame_index = 0usize;
    let mut samples: Vec<i16> = Vec::new();

    match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => {
            for sample in reader.samples::<i16>() {
                let sample = sample?;
                frame[frame_index] = sample;
                frame_index += 1;
                if frame_index == source_channels {
                    push_mapped_frame(
                        &frame,
                        source_channels,
                        output_channels,
                        &mut samples,
                    );
                    frame_index = 0;
                }
            }
        }
        (hound::SampleFormat::Float, 32) => {
            for sample in reader.samples::<f32>() {
                let sample = sample?;
                let clamped = sample.clamp(-1.0, 1.0);
                let as_i16 = (clamped * i16::MAX as f32) as i16;
                frame[frame_index] = as_i16;
                frame_index += 1;
                if frame_index == source_channels {
                    push_mapped_frame(
                        &frame,
                        source_channels,
                        output_channels,
                        &mut samples,
                    );
                    frame_index = 0;
                }
            }
        }
        (format, bits) => {
            return Err(format!(
                "Unsupported WAV format: {:?} at {} bits per sample",
                format, bits
            )
            .into());
        }
    }

    if samples.is_empty() {
        return Err("WAV file contained no complete frames".into());
    }

    FarAudioSource::new(samples, spec.sample_rate, output_rate, output_channels)
        .map_err(|e| e.into())
}

fn push_mapped_frame(
    frame: &[i16],
    source_channels: usize,
    output_channels: usize,
    out: &mut Vec<i16>,
) {
    if source_channels == 0 || output_channels == 0 {
        return;
    }
    for dst in 0..output_channels {
        let src_idx = if source_channels == 1 {
            0
        } else if dst < source_channels {
            dst
        } else {
            source_channels - 1
        };
        out.push(frame[src_idx]);
    }
}

fn build_output_stream(
    device: &Device,
    config: &StreamConfig,
    shared: Arc<Mutex<SharedCanceller>>,
    far_source: Arc<Mutex<FarAudioSource>>,
    channels: usize,
    format: SampleFormat,
) -> Result<Stream, cpal::BuildStreamError> {
    match format {
        SampleFormat::I16 => {
            build_output_stream_i16(device, config, shared, far_source, channels)
        }
        SampleFormat::F32 => {
            build_output_stream_f32(device, config, shared, far_source, channels)
        }
        SampleFormat::U16 => {
            build_output_stream_u16(device, config, shared, far_source, channels)
        }
        other => {
            eprintln!("Unsupported output sample format: {other:?}");
            Err(cpal::BuildStreamError::StreamConfigNotSupported)
        }
    }
}

fn build_output_stream_i16(
    device: &Device,
    config: &StreamConfig,
    shared: Arc<Mutex<SharedCanceller>>,
    far_source: Arc<Mutex<FarAudioSource>>,
    channels: usize,
) -> Result<Stream, cpal::BuildStreamError> {
    let mut far_buffer = Vec::<i16>::new();
    device.build_output_stream(
        config,
        move |data: &mut [i16], _| {
            if far_buffer.len() != data.len() {
                far_buffer.resize(data.len(), 0);
            }
            {
                let mut source = far_source.lock().unwrap();
                source.next_samples(&mut far_buffer);
            }
            data.copy_from_slice(&far_buffer);
            let mut state = shared.lock().unwrap();
            state.push_far_end(&far_buffer);
        },
        move |err| eprintln!("Output stream error: {err}"),
        None,
    )
}

fn build_output_stream_f32(
    device: &Device,
    config: &StreamConfig,
    shared: Arc<Mutex<SharedCanceller>>,
    far_source: Arc<Mutex<FarAudioSource>>,
    channels: usize,
) -> Result<Stream, cpal::BuildStreamError> {
    let mut far_buffer = Vec::<i16>::new();
    device.build_output_stream(
        config,
        move |data: &mut [f32], _| {
            if far_buffer.len() != data.len() {
                far_buffer.resize(data.len(), 0);
            }
            {
                let mut source = far_source.lock().unwrap();
                source.next_samples(&mut far_buffer);
            }
            for (dst, src) in data.iter_mut().zip(far_buffer.iter()) {
                *dst = (*src as f32) / i16::MAX as f32;
            }
            let mut state = shared.lock().unwrap();
            state.push_far_end(&far_buffer);
        },
        move |err| eprintln!("Output stream error: {err}"),
        None,
    )
}

fn build_output_stream_u16(
    device: &Device,
    config: &StreamConfig,
    shared: Arc<Mutex<SharedCanceller>>,
    far_source: Arc<Mutex<FarAudioSource>>,
    channels: usize,
) -> Result<Stream, cpal::BuildStreamError> {
    let mut far_buffer = Vec::<i16>::new();
    device.build_output_stream(
        config,
        move |data: &mut [u16], _| {
            if far_buffer.len() != data.len() {
                far_buffer.resize(data.len(), 0);
            }
            {
                let mut source = far_source.lock().unwrap();
                source.next_samples(&mut far_buffer);
            }
            for (dst, src) in data.iter_mut().zip(far_buffer.iter()) {
                let normalized =
                    (*src as f32 / i16::MAX as f32).clamp(-1.0_f32, 1.0_f32);
                *dst = (((normalized + 1.0) * 0.5) * u16::MAX as f32)
                    .clamp(0.0, u16::MAX as f32) as u16;
            }
            let mut state = shared.lock().unwrap();
            state.push_far_end(&far_buffer);
        },
        move |err| eprintln!("Output stream error: {err}"),
        None,
    )
}

fn build_input_stream(
    device: &Device,
    config: &StreamConfig,
    shared: Arc<Mutex<SharedCanceller>>,
    format: SampleFormat,
) -> Result<Stream, cpal::BuildStreamError> {
    match format {
        SampleFormat::I16 => build_input_stream_i16(device, config, shared),
        SampleFormat::F32 => build_input_stream_f32(device, config, shared),
        SampleFormat::U16 => build_input_stream_u16(device, config, shared),
        other => {
            eprintln!("Unsupported input sample format: {other:?}");
            Err(cpal::BuildStreamError::StreamConfigNotSupported)
        }
    }
}

fn build_input_stream_i16(
    device: &Device,
    config: &StreamConfig,
    shared: Arc<Mutex<SharedCanceller>>,
) -> Result<Stream, cpal::BuildStreamError> {
    device.build_input_stream(
        config,
        move |data: &[i16], _| {
            let mut state = shared.lock().unwrap();
            state.process_capture(data);
        },
        move |err| eprintln!("Input stream error: {err}"),
        None,
    )
}

fn build_input_stream_f32(
    device: &Device,
    config: &StreamConfig,
    shared: Arc<Mutex<SharedCanceller>>,
) -> Result<Stream, cpal::BuildStreamError> {
    device.build_input_stream(
        config,
        move |data: &[f32], _| {
            let mut state = shared.lock().unwrap();
            let mut buffer = mem::take(&mut state.input_convert);
            buffer.resize(data.len(), 0);
            for (dst, src) in buffer.iter_mut().zip(data.iter()) {
                let clamped = (*src).clamp(-1.0, 1.0);
                *dst = (clamped * i16::MAX as f32) as i16;
            }
            state.process_capture(&buffer);
            state.input_convert = buffer;
        },
        move |err| eprintln!("Input stream error: {err}"),
        None,
    )
}

fn build_input_stream_u16(
    device: &Device,
    config: &StreamConfig,
    shared: Arc<Mutex<SharedCanceller>>,
) -> Result<Stream, cpal::BuildStreamError> {
    device.build_input_stream(
        config,
        move |data: &[u16], _| {
            let mut state = shared.lock().unwrap();
            let mut buffer = mem::take(&mut state.input_convert);
            buffer.resize(data.len(), 0);
            for (dst, src) in buffer.iter_mut().zip(data.iter()) {
                let normalized = (*src as f32) / u16::MAX as f32;
                let centered = (normalized * 2.0) - 1.0;
                let clamped = centered.clamp(-1.0, 1.0);
                *dst = (clamped * i16::MAX as f32) as i16;
            }
            state.process_capture(&buffer);
            state.input_convert = buffer;
        },
        move |err| eprintln!("Input stream error: {err}"),
        None,
    )
}

fn select_device<I>(
    devices: Result<I, cpal::DevicesError>,
    target: &str,
    kind: &str,
) -> Result<Device, Box<dyn Error>>
where
    I: Iterator<Item = Device>,
{
    let target_lower = target.to_lowercase();
    let mut available = Vec::new();
    let mut selected: Option<(String, Device)> = None;

    match devices {
        Ok(list) => {
            for device in list {
                let name = device
                    .name()
                    .unwrap_or_else(|_| "<unknown device>".to_string());
                if selected.is_none() && name.to_lowercase().contains(&target_lower) {
                    selected = Some((name.clone(), device));
                }
                available.push(name);
            }
        }
        Err(err) => return Err(format!("Failed to enumerate {kind} devices: {err}").into()),
    }

    if let Some((name, device)) = selected {
        println!("{kind} device selected: {name}");
        Ok(device)
    } else if available.is_empty() {
        Err(format!("{kind} device matching '{target}' not found (no devices available)").into())
    } else {
        let quoted = available
            .iter()
            .map(|name| format!("'{name}'"))
            .collect::<Vec<_>>()
            .join(", ");
        Err(format!(
            "{kind} device matching '{target}' not found. Available: {quoted}"
        )
        .into())
    }
}
