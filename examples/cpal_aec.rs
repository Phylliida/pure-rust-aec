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
    collections::{hash_map::Entry, HashMap, VecDeque},
    env,
    error::Error,
    ffi::c_void,
    mem::{self, MaybeUninit},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Device, FromSample, InputCallbackInfo, Sample, SampleFormat, SampleRate, SizedSample, Stream,
    StreamConfig, StreamInstant, SupportedStreamConfigRange,
};
use hound::{self, WavSpec};
use ringbuf::{
    traits::{Consumer, Observer, Producer, RingBuffer, Split},
    HeapCons, HeapProd, HeapRb, LocalRb,
};
use ringbuf::storage::Heap;
use speex_rust_aec::{
    speex_echo_cancellation, speex_echo_ctl, EchoCanceller, Resampler, SPEEX_ECHO_SET_SAMPLING_RATE,
};

use std::time::Instant;



const DEFAULT_AEC_RATE: u32 = 48_000;
const RESAMPLER_QUALITY: i32 = 5;

type AecCallback = Box<dyn FnMut(&[i16]) + Send + 'static>;

struct ResampleContext {
    resampler: Resampler,
    channels: usize,
}

#[inline]
unsafe fn assume_init_slice_mut<T>(slice: &mut [MaybeUninit<T>]) -> &mut [T] {
    &mut *(slice as *mut [MaybeUninit<T>] as *mut [T])
}

impl ResampleContext {
    fn new(resampler: Resampler) -> Self {
        let channels = resampler.channels() as usize;
        Self {
            resampler,
            channels,
        }
    }

    fn process(&mut self, new_samples: &[i16], out: &mut Vec<i16>) {
        if new_samples.is_empty() {
            return;
        }
        if self.channels == 0 {
            eprintln!("ResampleContext invoked with zero channels; skipping.");
            return;
        }
        if new_samples.len() % self.channels != 0 {
            eprintln!(
                "ResampleContext received samples not aligned to {} channel(s); dropping remainder.",
                self.channels
            );
            return;
        }

        let (in_rate, out_rate) = self.resampler.get_rate();
        let input_frames = new_samples.len() / self.channels;
        if input_frames == 0 {
            return;
        }

        let mut offset = 0usize;
        while offset < new_samples.len() {
            let remaining_samples = new_samples.len() - offset;
            let remaining_frames = remaining_samples / self.channels;
            if remaining_frames == 0 {
                break;
            }

            let mut expected_frames =
                (remaining_frames as u64 * out_rate as u64 + in_rate as u64 - 1) / in_rate as u64;
            expected_frames = expected_frames.max(1) + 4; // add headroom
            let mut chunk_samples = (expected_frames as usize).saturating_mul(self.channels);
            if chunk_samples == 0 {
                chunk_samples = self.channels.max(1);
            }

            let mut attempts = 0usize;
            loop {
                let start = out.len();
                out.resize(start + chunk_samples, 0);
                match self
                    .resampler
                    .process_interleaved_i16(&new_samples[offset..], &mut out[start..])
                {
                    Ok((consumed, produced)) => {
                        out.truncate(start + produced);
                        if consumed == 0 {
                            if produced == chunk_samples && attempts < 8 {
                                attempts += 1;
                                chunk_samples += self.channels.max(1) * 32;
                                continue;
                            }
                            if produced == 0 {
                                out.truncate(start);
                            }
                            return;
                        }
                        offset += consumed;
                        break;
                    }
                    Err(err) => {
                        out.truncate(start);
                        eprintln!("Resampler error: {err}");
                        return;
                    }
                }
            }
        }
    }

    fn flush_into(&mut self, out: &mut Vec<i16>) {
        let latency_samples = self.resampler.output_latency().saturating_mul(self.channels);
        if latency_samples == 0 {
            return;
        }
        if self.channels == 0 {
            return;
        }
        let zeros = vec![0i16; latency_samples];
        self.process(zeros.as_slice(), out);
    }
}

fn gather_supported_input_configs(
    device: &Device,
) -> Result<Vec<SupportedStreamConfigRange>, cpal::SupportedStreamConfigsError> {
    device.supported_input_configs().map(|configs| configs.collect())
}

fn gather_supported_output_configs(
    device: &Device,
) -> Result<Vec<SupportedStreamConfigRange>, cpal::SupportedStreamConfigsError> {
    device
        .supported_output_configs()
        .map(|configs| configs.collect())
}

fn log_supported_configs(
    kind: &str,
    device_name: &str,
    configs: &[SupportedStreamConfigRange],
) {
    if configs.is_empty() {
        println!(
            "{kind} device '{device_name}' reported no supported stream configurations."
        );
        return;
    }

    println!("{kind} device '{device_name}' supported configs:");
    for (idx, cfg) in configs.iter().enumerate() {
        let min_rate = cfg.min_sample_rate().0;
        let max_rate = cfg.max_sample_rate().0;
        let rate_desc = if min_rate == max_rate {
            format!("{min_rate} Hz")
        } else {
            format!("{min_rate}-{max_rate} Hz")
        };
        println!(
            "  #{idx}: {} channel(s), format: {:?}, sample rates: {rate_desc}",
            cfg.channels(),
            cfg.sample_format(),
        );
    }
}

fn supports_rate(
    configs: &[SupportedStreamConfigRange],
    channels: u16,
    rate: u32,
    format: SampleFormat,
) -> bool {
    let sample_rate = SampleRate(rate);
    configs
        .iter()
        .filter(|cfg| cfg.channels() == channels && cfg.sample_format() == format)
        .any(|cfg| cfg.clone().try_with_sample_rate(sample_rate).is_some())
}

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

    let input_device_name = input_device
        .name()
        .unwrap_or_else(|_| "<unknown input device>".to_string());
    let output_device_name = output_device
        .name()
        .unwrap_or_else(|_| "<unknown output device>".to_string());

    let input_supported_configs = match gather_supported_input_configs(&input_device) {
        Ok(configs) => configs,
        Err(err) => {
            eprintln!(
                "Unable to enumerate input configs for '{input_device_name}': {err}"
            );
            Vec::new()
        }
    };
    log_supported_configs("Input", &input_device_name, &input_supported_configs);

    let output_supported_configs = match gather_supported_output_configs(&output_device) {
        Ok(configs) => configs,
        Err(err) => {
            eprintln!(
                "Unable to enumerate output configs for '{output_device_name}': {err}"
            );
            Vec::new()
        }
    };
    log_supported_configs("Output", &output_device_name, &output_supported_configs);

    let input_config = input_device.default_input_config()?;
    let output_config = output_device.default_output_config()?;

    let input_stream_config: StreamConfig = input_config.config();
    let output_stream_config: StreamConfig = output_config.config();

    let input_rate = input_stream_config.sample_rate.0;
    let output_rate = output_stream_config.sample_rate.0;
    let input_channels = input_stream_config.channels as usize;
    let output_channels = output_stream_config.channels as usize;

    if input_channels == 0 {
        return Err("Input device reports zero channels; cannot run AEC.".into());
    }
    if output_channels == 0 {
        return Err("Output device reports zero channels; cannot run AEC.".into());
    }

    if !input_supported_configs.is_empty() && !output_supported_configs.is_empty() {
        let input_format = input_config.sample_format();
        let output_format = output_config.sample_format();
        if !supports_rate(
            &input_supported_configs,
            input_stream_config.channels,
            target_sample_rate,
            input_format,
        ) || !supports_rate(
            &output_supported_configs,
            output_stream_config.channels,
            target_sample_rate,
            output_format,
        ) {
            let mut reasons = Vec::new();
            if !supports_rate(
                &input_supported_configs,
                input_stream_config.channels,
                target_sample_rate,
                input_format,
            ) {
                reasons.push(format!(
                    "input device '{}' ({} channel[s], {:?})",
                    input_device_name, input_stream_config.channels, input_format
                ));
            }
            if !supports_rate(
                &output_supported_configs,
                output_stream_config.channels,
                target_sample_rate,
                output_format,
            ) {
                reasons.push(format!(
                    "output device '{}' ({} channel[s], {:?})",
                    output_device_name, output_stream_config.channels, output_format
                ));
            }
            let details = if reasons.is_empty() {
                "unknown devices".to_string()
            } else {
                reasons.join(" and ")
            };
            return Err(format!(
                "AEC sample rate {} Hz is not supported by {}. \
                Please rerun with a supported rate.",
                target_sample_rate, details
            )
            .into());
        }
    }

    println!(
        "Input: {input_channels} ch @ {input_rate} Hz, Output: {output_channels} ch @ {output_rate} Hz, AEC rate: {target_sample_rate} Hz"
    );

    let far_audio = load_far_audio(&far_audio_path, output_channels, output_rate)
        .map_err(|e| format!("Failed to load far-end audio '{}': {e}", far_audio_path.display()))?;
    println!("Far-end audio source: {}", far_audio_path.display());

    let default_frame_size = (target_sample_rate / 100).max(1) as usize;
    let default_filter_length = default_frame_size * 20;

    /*
    let shared_stream = SharedAecStream::new(
        &input_device,
        &output_device,
        &input_stream_config,
        input_config.sample_format(),
        &output_stream_config,
        output_config.sample_format(),
        target_sample_rate,
        default_frame_size,
        default_filter_length,
    )?;
    shared_stream.add_audio(0, &far_audio, output_rate);

    let shared_state = shared_stream.shared_state();

    let mut frame_counter = 0usize;
    shared_stream.register_callback(move |frame| {
        frame_counter += 1;
        if frame_counter % 50 == 0 {
            let rms = (frame
                .iter()
                .map(|s| (*s as f64) * (*s as f64))
                .sum::<f64>()
                / frame.len() as f64)
                .sqrt();
            println!("Echo-cancelled frame RMS: {rms:.2}");
        }
    });

    println!("Running Speex AEC demo for five seconds...");
    std::thread::sleep(Duration::from_secs(5));
    println!("Done.");

    drop(shared_stream);

    let recording_path = PathBuf::from("examples/aec_output.wav");
    let (recorded_samples, capture_frames, last_capture_timestamp) = {
        let mut guard = shared_state.lock().unwrap();
        guard.flush();
        let frames = guard.capture_frames_received;
        let timestamp = guard.last_capture_timestamp;
        (guard.take_recording(), frames, timestamp)
    };

    match last_capture_timestamp {
        Some(ts) => println!(
            "Captured {capture_frames} AEC-rate frame(s); last capture timestamp: {ts:?}"
        ),
        None => println!("Captured {capture_frames} AEC-rate frame(s); no timestamp reported."),
    }

    if recorded_samples.is_empty() {
        eprintln!("No echo-cancelled audio captured; skipping write to {}", recording_path.display());
    } else {
        let spec = WavSpec {
            channels: input_channels as u16,
            sample_rate: target_sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&recording_path, spec)?;
        println!("Writing samples of length {}", recorded_samples.len());
        for sample in recorded_samples {
            writer.write_sample(sample)?;
        }
        writer.finalize()?;
        println!(
            "Wrote echo-cancelled audio to {}",
            recording_path.display()
        );
    }
    */

    Ok(())
}

struct SharedCanceller {
    aec: EchoCanceller,
    input_channels: usize,
    output_channels: usize,
    input_frame_samples: usize,
    output_frame_samples: usize,
    far_queue: VecDeque<i16>,
    input_queue: VecDeque<i16>,
    input_frame: Vec<i16>,
    far_frame: Vec<i16>,
    out_frame: Vec<i16>,
    processed_frames: usize,
    recorded: Vec<i16>,
    callbacks: Vec<AecCallback>,
    capture_frames_received: u64,
    last_capture_timestamp: Option<StreamInstant>,
}

impl SharedCanceller {
    fn new(
        aec: EchoCanceller,
        input_channels: usize,
        output_channels: usize,
        frame_size: usize,
    ) -> Self {
        let input_frame_samples = frame_size * input_channels;
        let output_frame_samples = frame_size * output_channels;
        Self {
            aec,
            input_channels,
            output_channels,
            input_frame_samples,
            output_frame_samples,
            far_queue: VecDeque::with_capacity(output_frame_samples * 8),
            input_queue: VecDeque::with_capacity(input_frame_samples * 8),
            input_frame: vec![0; input_frame_samples],
            far_frame: vec![0; output_frame_samples.max(1)],
            out_frame: vec![0; input_frame_samples],
            processed_frames: 0,
            recorded: Vec::with_capacity(input_frame_samples * 8),
            callbacks: Vec::new(),
            capture_frames_received: 0,
            last_capture_timestamp: None,
        }
    }

    fn push_far_end_resampled(&mut self, samples: &[i16]) {
        if samples.is_empty() {
            return;
        }
        if self.output_channels == 0 {
            return;
        }
        let remainder = samples.len() % self.output_channels;
        if remainder != 0 {
            eprintln!(
                "Far-end reference samples ({}) not divisible by output channel count {}; dropping remainder.",
                samples.len(),
                self.output_channels
            );
        }
        let usable = samples.len() - remainder;
        if usable == 0 {
            return;
        }
        self.far_queue
            .extend(samples[..usable].iter().copied());
        let max_len = self.output_frame_samples * 16;
        while self.far_queue.len() > max_len {
            self.far_queue.pop_front();
        }
    }

    fn process_capture_resampled(
        &mut self,
        samples: &[i16],
        capture_timestamp: Option<StreamInstant>,
    ) {
        if let Some(timestamp) = capture_timestamp {
            self.last_capture_timestamp = Some(timestamp);
        }
        if samples.is_empty() {
            return;
        }
        if self.input_channels == 0 {
            return;
        }
        let remainder = samples.len() % self.input_channels;
        if remainder != 0 {
            eprintln!(
                "Capture samples ({}) not divisible by input channel count {}; dropping remainder.",
                samples.len(),
                self.input_channels
            );
        }
        let usable = samples.len() - remainder;
        if usable == 0 {
            return;
        }
        self.input_queue
            .extend(samples[..usable].iter().copied());
        let frames = usable / self.input_channels;
        self.capture_frames_received = self
            .capture_frames_received
            .saturating_add(frames as u64);
        self.process_ready_frames();
    }

    fn process_ready_frames(&mut self) {
        if self.input_frame_samples == 0 {
            return;
        }
        while self.input_queue.len() >= self.input_frame_samples {
            for sample in self.input_frame.iter_mut() {
                *sample = self.input_queue.pop_front().unwrap();
            }
            for sample in self.far_frame.iter_mut() {
                *sample = self.far_queue.pop_front().unwrap_or(0);
            }
            unsafe {
                speex_echo_cancellation(
                    self.aec.as_ptr(),
                    self.input_frame.as_ptr(),
                    if self.output_channels == 0 {
                        std::ptr::null()
                    } else {
                        self.far_frame.as_ptr()
                    },
                    self.out_frame.as_mut_ptr(),
                );
            }
            self.processed_frames += 1;
            self.recorded
                .extend(self.out_frame.iter().copied());
            for callback in self.callbacks.iter_mut() {
                callback(self.out_frame.as_slice());
            }
        }
    }

    fn flush(&mut self) {
        self.process_ready_frames();

        if !self.input_queue.is_empty() && self.input_frame_samples != 0 {
            let needed = self.input_frame_samples - self.input_queue.len();
            self.input_queue
                .extend(std::iter::repeat(0i16).take(needed));
            self.process_ready_frames();
        }
    }

    fn take_recording(&mut self) -> Vec<i16> {
        mem::take(&mut self.recorded)
    }

    fn add_callback<F>(&mut self, callback: F)
    where
        F: FnMut(&[i16]) + Send + 'static,
    {
        self.callbacks.push(Box::new(callback));
    }
}

struct PlaybackBuffer {
    queue: VecDeque<i16>,
    channels: usize,
}

impl PlaybackBuffer {
    fn new(channels: usize) -> Self {
        Self {
            queue: VecDeque::with_capacity(channels * 1024),
            channels,
        }
    }

    fn push_samples(&mut self, samples: &[i16]) {
        if samples.is_empty() {
            return;
        }
        if self.channels == 0 {
            eprintln!("PlaybackBuffer received samples but channel count is zero; discarding.");
            return;
        }
        if samples.len() % self.channels != 0 {
            eprintln!(
                "PlaybackBuffer received samples not aligned to {} channel(s); truncating remainder.",
                self.channels
            );
        }
        let usable = samples.len() - (samples.len() % self.channels);
        self.queue.extend(samples[..usable].iter().copied());
    }

    fn pop_into(&mut self, out: &mut [i16]) {
        for sample in out.iter_mut() {
            *sample = self.queue.pop_front().unwrap_or(0);
        }
    }
}

fn load_far_audio(
    path: &Path,
    output_channels: usize,
    output_rate: u32,
) -> Result<Vec<f32>, Box<dyn Error>> {
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

    let mut processed = if spec.sample_rate != output_rate {
        let resampler = Resampler::new(
            output_channels as u32,
            spec.sample_rate,
            output_rate,
            RESAMPLER_QUALITY,
        )
        .map_err(|e| format!("Failed to create playback resampler: {e}"))?;
        let mut ctx = ResampleContext::new(resampler);
        let mut converted = Vec::with_capacity(samples.len());
        ctx.process(&samples, &mut converted);
        ctx.flush_into(&mut converted);
        converted
    } else {
        samples
    };

    if processed.is_empty() {
        return Err("Resampler produced no audio samples".into());
    }

    let mut as_f32 = Vec::with_capacity(processed.len());
    for sample in processed.drain(..) {
        as_f32.push(sample as f32 / i16::MAX as f32);
    }

    Ok(as_f32)
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
    playback: Arc<Mutex<PlaybackBuffer>>,
    resampler: Option<Arc<Mutex<ResampleContext>>>,
    format: SampleFormat,
) -> Result<Stream, cpal::BuildStreamError> {
    match format {
        SampleFormat::I16 => build_output_stream_typed::<i16>(
            device,
            config,
            shared,
            Arc::clone(&playback),
            resampler.clone(),
        ),
        SampleFormat::F32 => build_output_stream_typed::<f32>(
            device,
            config,
            shared,
            Arc::clone(&playback),
            resampler.clone(),
        ),
        SampleFormat::U16 => build_output_stream_typed::<u16>(
            device,
            config,
            shared,
            playback,
            resampler,
        ),
        other => {
            eprintln!("Unsupported output sample format: {other:?}");
            Err(cpal::BuildStreamError::StreamConfigNotSupported)
        }
    }
}

fn build_output_stream_typed<T>(
    device: &Device,
    config: &StreamConfig,
    shared: Arc<Mutex<SharedCanceller>>,
    playback: Arc<Mutex<PlaybackBuffer>>,
    resampler: Option<Arc<Mutex<ResampleContext>>>,
) -> Result<Stream, cpal::BuildStreamError>
where
    T: Sample + SizedSample + FromSample<i16>,
{
    let mut far_buffer = Vec::<i16>::new();
    let mut resampled = Vec::<i16>::new();
    device.build_output_stream(
        config,
        move |data: &mut [T], _| {
            if far_buffer.len() != data.len() {
                far_buffer.resize(data.len(), 0);
            }
            {
                let mut buffer = playback.lock().unwrap();
                buffer.pop_into(&mut far_buffer);
            }
            for (dst, src) in data.iter_mut().zip(far_buffer.iter()) {
                *dst = T::from_sample(*src);
            }

            let produced = if let Some(resampler_handle) = &resampler {
                resampled.clear();
                if let Ok(mut ctx) = resampler_handle.lock() {
                    ctx.process(&far_buffer, &mut resampled);
                }
                resampled.as_slice()
            } else {
                far_buffer.as_slice()
            };

            if !produced.is_empty() {
                if let Ok(mut state) = shared.lock() {
                    state.push_far_end_resampled(produced);
                }
            }
        },
        move |err| eprintln!("Output stream error: {err}"),
        None,
    )
}

fn build_input_stream(
    device: &Device,
    config: &StreamConfig,
    shared: Arc<Mutex<SharedCanceller>>,
    resampler: Option<Arc<Mutex<ResampleContext>>>,
    format: SampleFormat,
) -> Result<Stream, cpal::BuildStreamError> {
    match format {
        SampleFormat::I16 => build_input_stream_typed::<i16>(
            device,
            config,
            shared,
            resampler.clone(),
        ),
        SampleFormat::F32 => build_input_stream_typed::<f32>(
            device,
            config,
            shared,
            resampler.clone(),
        ),
        SampleFormat::U16 => build_input_stream_typed::<u16>(
            device,
            config,
            shared,
            resampler,
        ),
        other => {
            eprintln!("Unsupported input sample format: {other:?}");
            Err(cpal::BuildStreamError::StreamConfigNotSupported)
        }
    }
}

fn build_input_stream_typed<T>(
    device: &Device,
    config: &StreamConfig,
    shared: Arc<Mutex<SharedCanceller>>,
    resampler: Option<Arc<Mutex<ResampleContext>>>,
) -> Result<Stream, cpal::BuildStreamError>
where
    T: Sample + SizedSample,
    i16: FromSample<T>,
{
    let mut convert_buffer = Vec::<i16>::new();
    let mut resampled = Vec::<i16>::new();
    device.build_input_stream(
        config,
        move |data: &[T], info: &InputCallbackInfo| {
            if convert_buffer.len() != data.len() {
                convert_buffer.resize(data.len(), 0);
            }
            for (dst, src) in convert_buffer.iter_mut().zip(data.iter()) {
                *dst = i16::from_sample(*src);
            }

            let capture_timestamp = info.timestamp().capture;

            let produced = if let Some(resampler_handle) = &resampler {
                resampled.clear();
                if let Ok(mut ctx) = resampler_handle.lock() {
                    ctx.process(&convert_buffer, &mut resampled);
                }
                resampled.as_slice()
            } else {
                convert_buffer.as_slice()
            };

            let mut state = shared.lock().unwrap();
            state.process_capture_resampled(produced, Some(capture_timestamp));
        },
        move |err| eprintln!("Input stream error: {err}"),
        None,
    )
}

/// Producer-side sibling to `BufferedCircularProducer`.
/// Provides chunked, mostly zero-copy write access to a `HeapProd`.
/// Call `chunk_mut()` to obtain a contiguous region and `commit()` afterwards
/// to advance the underlying write index (or copy scratch data in).
struct BufferedCircularProducer<T: Copy> {
    producer: HeapProd<T>,
    scratch: Vec<T>
}

impl<T: Copy> BufferedCircularProducer<T> {
    fn new(producer: HeapProd<T>) -> Self {
        Self {
            producer,
            scratch: Vec::new()
        }
    }

    fn finish_write(&mut self, need_to_write_outputs: bool, num_written: usize) {
        if need_to_write_outputs {
            // wrote to scratch, need to add it to producer
            let _appended = self.producer.push_slice(&self.scratch[..num_written]);
        } else {
            // wrote directly to producer, simply advance write index
            unsafe { self.producer.advance_write_index(num_written) };
        }
    }
}

impl<T: Copy + Default> BufferedCircularProducer<T> {
    fn get_chunk_to_write(&mut self, size: usize) -> (bool, &mut [T]) {
        let (first, second) = self.producer.vacant_slices_mut();
        // we can simply 
        if first.len() >= size {
            let buf = unsafe { assume_init_slice_mut(first) };
            (true, &mut buf[..size])
        } else if first.is_empty() && second.len() >= size {
            let buf = unsafe { assume_init_slice_mut(second) };
            (true, &mut buf[..size])
        }
        else {
            if self.scratch.len() < size {
                self.scratch.resize_with(size, Default::default);
            }
            (false, &mut self.scratch[..size])
        }
    }
}

/// Helper that makes the consumer half of a ring buffer feel like a stream of contiguous slices.
/// It tries to return zero-copy slices when the occupied region is already contiguous,
/// and otherwise falls back to copying into a scratch buffer.
/// `StreamAligner` can hold one of these alongside its producer half and call `chunk()` /
/// `consume()` whenever it needs to feed the SpeexDSP resampler.
struct BufferedCircularConsumer<T: Copy> {
    consumer: HeapCons<T>,
    scratch: Vec<T>,
}

impl<T: Copy> BufferedCircularConsumer<T> {
    fn new(consumer: HeapCons<T>) -> Self {
        Self {
            consumer,
            scratch: Vec::new(),
        }
    }

    fn finish_read(&mut self, num_read: usize) -> usize {
        self.consumer.skip(num_read)
    }
}

impl<T: Copy> BufferedCircularConsumer<T> {
    fn get_chunk_to_read(&mut self, size: usize) -> &[T] {
        if size == 0 {
            return &[];
        }

        let (head, tail) = self.consumer.as_slices();
        let head_len = head.len();
        let tail_len = tail.len();
        let available = head_len + tail_len;
        if available == 0 {
            eprintln!("Consumer is saturated, this is bad, please increase buffer sizes");
            return &[];
        }

        let take = size.min(available);
        // all fits in head, just return slice of that
        if head_len >= take {
            &head[..take]
        // head is empty so all fits in tail, return that
        } else if head_len == 0 {
            &tail[..take]
        // we need intermediate buffer to join head and tail, use scratch
        } else {
            if self.scratch.capacity() < take {
                self.scratch.reserve(take - self.scratch.capacity());
            }
            self.scratch.clear(); // this empties it but does not remove allocations

            let from_head = head_len.min(take);
            if from_head > 0 {
                self.scratch.extend_from_slice(&head[..from_head]);
            }
            let remaining = take - from_head;
            if remaining > 0 {
                self.scratch.extend_from_slice(&tail[..remaining]);
            }

            &self.scratch[..take]
        }
    }
}

#[derive(Debug)]
struct AudioBufferMetadata {
    num_frames: u64,
    target_emitted_frames: i128
}

struct StreamAligner {
    start_time: Option<std::time::Instant>,
    input_sample_rate: u32,
    output_sample_rate: u32,
    dynamic_output_sample_rate: u32,
    input_audio_buffer_producer: HeapProd<f32>,
    input_audio_buffer_consumer: BufferedCircularConsumer<f32>,
    input_audio_buffer_metadata_producer: HeapProd<AudioBufferMetadata>,
    input_audio_buffer_metadata_consumer: HeapCons<AudioBufferMetadata>,
    output_audio_buffer_producer: BufferedCircularProducer<f32>,
    total_emitted_frames: i128,
    total_input_samples_remaining: i128,
    chunk_sizes: LocalRb<Heap<usize>>, // only accessed by the audio push thread
    system_time_in_frames_when_chunk_ended: LocalRb<Heap<i128>>, // only accessed by the audio input thread
    resampler: Resampler
}

fn input_to_output_frames(input_frames: i128, in_rate: u32, out_rate: u32) -> i128 {
    // u128 to avoid overflow
    (((input_frames as i128) * (out_rate as i128)) / (in_rate as i128)) as i128
}

fn output_to_input_frames(output_frames: i128, in_rate: u32, out_rate: u32) -> i128 {
    // u128 to avoid overflow
    (((output_frames as i128) * (in_rate as i128)) / (out_rate as i128)) as i128
}

fn micros_to_frames(microseconds: i128, sample_rate: i128) -> i128 {
    // There are sample_rate samples per second
    // there are sample_rate / 1_000_000 samples per microsecond
    // now that we have samples_per_microsecond, we simply multiply by microseconds to get total samples
    // rearranging:
    microseconds * sample_rate / 1000000
}

fn frames_to_micros(frames: i128, sample_rate: i128) -> i128 {
    // frames = (microseconds * sample_rate / 1 000 000)
    // frames * 1_000_000 = microseconds * sample_rate
    frames * 1000000 / sample_rate // = microseconds
}


impl StreamAligner {
    // Takes input audio and resamples it to the target rate
    // May slightly stretch or squeeze the audio (via resampling)
    // to ensure the outputs stay aligned with system clock
    fn new(input_sample_rate: u32, output_sample_rate: u32, history_len: usize, audio_buffer_seconds: u32, resampler_quality: i32, output_audio_buffer_producer: HeapProd<f32>) -> Result<Self, Box<dyn Error>>  {
        let (input_audio_buffer_producer, input_audio_buffer_consumer) = HeapRb::<f32>::new((audio_buffer_seconds * input_sample_rate) as usize).split();
        let (input_audio_buffer_metadata_producer, input_audio_buffer_metadata_consumer) = HeapRb::<AudioBufferMetadata>::new(1000).split(); // should be plenty for any practical use
        Ok(Self {
            input_sample_rate: input_sample_rate,
            output_sample_rate: output_sample_rate,
            dynamic_output_sample_rate: output_sample_rate,
            input_audio_buffer_producer: input_audio_buffer_producer,
            // we need buffered because this interfaces with speex which expects continuous buffers
            input_audio_buffer_consumer: BufferedCircularConsumer::<f32>::new(input_audio_buffer_consumer),
            input_audio_buffer_metadata_producer: input_audio_buffer_metadata_producer,
            input_audio_buffer_metadata_consumer: input_audio_buffer_metadata_consumer,
            // these are also buffered because speex also needs continuous buffers to output to
            output_audio_buffer_producer: BufferedCircularProducer::<f32>::new(output_audio_buffer_producer),
            total_input_samples_remaining: 0, // variable to accumulate bc speex goes in chunks
            // alignment data, these are used to adjust resample rate so output stays aligned with true timings (according to sytem clock)
            start_time: None,
            total_emitted_frames: 0,
            chunk_sizes: LocalRb::<Heap<usize>>::new(history_len),
            system_time_in_frames_when_chunk_ended: LocalRb::<Heap<i128>>::new(history_len),
            resampler: Resampler::new(
                1, // channels, we have one of these for each channel
                input_sample_rate,
                output_sample_rate,
                resampler_quality
            )?
        })
    }

    fn estimate_when_most_recent_ended(&self) -> i128 {
        // Take minimum over estimates for all previous recieved
        // Some may be delayed due to cpu being busy, but none can ever arrive too early
        // so this should be a decent estimate
        // it does not account for hardware latency, but we cannot account for that without manual calibration
        // (btw, CPAL timestamps do not work because they may be different for different devices)
        // (wheras this synchronizes us to global system time)
        let mut best_estimate_of_when_most_recent_ended = if let Some(first_time) = self.system_time_in_frames_when_chunk_ended.first() {
            *first_time
        }
        else {
            0 as i128
        };
        for (chunk_size, frames_when_chunk_ended) in self.chunk_sizes.iter().zip(self.system_time_in_frames_when_chunk_ended.iter()) {
            best_estimate_of_when_most_recent_ended = (*frames_when_chunk_ended).min(best_estimate_of_when_most_recent_ended+(*chunk_size as i128));
        }
        best_estimate_of_when_most_recent_ended
    }

    fn process_chunk(&mut self, chunk: &[f32]) {
        let recieved_timestamp = Instant::now(); // store this first so we are as precise as possible
        if let None = self.start_time {
            // choose a start time so that frames_when_chunk_ended starts out equal to chunk len
            self.start_time = Some(recieved_timestamp - Duration::from_micros(frames_to_micros(chunk.len() as i128, self.input_sample_rate as i128) as u64));
        }
        
        // now we have assigned start_time, get it
        let frames_when_chunk_ended = if let Some(start_time) = self.start_time {
            micros_to_frames(recieved_timestamp.duration_since(start_time).as_micros() as i128, i128::from(self.input_sample_rate))
        }
        else {
            0
        };
       
        let appended_count = self.input_audio_buffer_producer.push_slice(chunk);
        if appended_count < chunk.len() { // todo: auto resize
            eprintln!("Error: cannot keep up with audio, buffer is full, try increasing audio_buffer_seconds")
        }
        // delibrately overwrite once we pass history len, we keep a rolling buffer of last 100 or so
        self.chunk_sizes.push_overwrite(chunk.len());
        self.system_time_in_frames_when_chunk_ended.push_overwrite(frames_when_chunk_ended);

        // use our estimate to suggest how many frames we should have emitted
        // this is used to dynamically adjust sample rate until we actually emit that many frames
        // that ensures that we stay synchronized to the system clock and do not drift
        let most_recent_ended_estimate = self.estimate_when_most_recent_ended();
        let metadata = AudioBufferMetadata {
            num_frames: chunk.len() as u64,
            target_emitted_frames: input_to_output_frames(most_recent_ended_estimate, self.input_sample_rate, self.output_sample_rate)
        };
        if let Err(metadata) = self.input_audio_buffer_metadata_producer.try_push(metadata) {
            eprintln!("Error: metadata ring buffer full; dropping {:?}, this is very bad what happened", metadata);
        }
    }

    // do it very slowly
    fn decrease_dynamic_sample_rate(&mut self) {
        self.dynamic_output_sample_rate = (((self.output_sample_rate as f32) * 0.95) as i128).max((self.dynamic_output_sample_rate-1) as i128) as u32;
    }

    fn increase_dynamic_sample_rate(&mut self) {
        self.dynamic_output_sample_rate = (((self.output_sample_rate as f32) * 1.05) as i128).min((self.dynamic_output_sample_rate+1) as i128) as u32;
    }

    fn handle_metadata(&mut self, metadata: AudioBufferMetadata) -> Result<(), Box<dyn std::error::Error>> {
        let target_emitted_frames = metadata.target_emitted_frames;
        let num_available_frames = metadata.num_frames;
        let estimated_emitted_frames = input_to_output_frames(num_available_frames as i128, self.input_sample_rate, self.dynamic_output_sample_rate);
        let updated_total_frames_emitted = self.total_emitted_frames + estimated_emitted_frames;
        // not enough frames, we need to increase dynamic sample rate (to get more samples)
        if updated_total_frames_emitted < target_emitted_frames {
            self.increase_dynamic_sample_rate();
        }
        // too many frames, we need to decrease dynamic sample rate (to get less samples)
        else if updated_total_frames_emitted > target_emitted_frames {
            self.decrease_dynamic_sample_rate();
        }

        //// do resampling ////
        self.resampler.set_rate(self.input_sample_rate, self.dynamic_output_sample_rate)?;

        // there might be some leftover from last call, so use global state
        self.total_input_samples_remaining += num_available_frames as i128;

        let input_buf = self.input_audio_buffer_consumer.get_chunk_to_read(self.total_input_samples_remaining as usize);
        let target_output_samples_count = input_to_output_frames(self.total_input_samples_remaining, self.input_sample_rate, self.dynamic_output_sample_rate) + 10; // add a few extra for rounding
        let (need_to_write_outputs, output_buf) = self.output_audio_buffer_producer.get_chunk_to_write(target_output_samples_count as usize);
        let (consumed, produced) = self.resampler.process_interleaved_f32(input_buf, output_buf)?;
        // it may return less consumed and produced than the sizes of stuff we gave it
        // so use actual processed sizes here instead of our lengths from above
        // (worst case this is like 0.6 ms or so, so it's okay to have them slightly delayed like this)
        self.input_audio_buffer_consumer.finish_read(consumed);
        self.output_audio_buffer_producer.finish_write(need_to_write_outputs, produced);

        // update our total
        self.total_input_samples_remaining -= consumed as i128;
        // the main downside of this is that it'll be persistently behind by 0.6ms or so (the resample frame size), but we'll quickly adjust for that so this shouldn't be a major issue
        // todo: think about how to fix this better (maybe current solution is as good as we can do, and it should average out to correct since past ones accumulated will result in more for this one, still, it's likely to stay behind by this amount)
        self.total_emitted_frames += produced as i128;

        Ok(())
    }

    fn output_chunks(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        while let Some(meta) = self.input_audio_buffer_metadata_consumer.try_pop() {
            self.handle_metadata(meta);
        }
        Ok(())
    }
}

/*
struct SharedAecStream {
    shared: Arc<Mutex<SharedCanceller>>,
    playback: Arc<Mutex<PlaybackBuffer>>,
    playback_resamplers: Mutex<HashMap<u32, ResampleContext>>,
    _output_stream: Stream,
    _input_stream: Stream,
    output_channels: usize,
    start_time: std::time::Instant,
    target_sample_rate: u32,
    input_resampler: Option<Arc<Mutex<ResampleContext>>>,
    output_resampler: Option<Arc<Mutex<ResampleContext>>>,
}

impl SharedAecStream {
    fn new(
        input_device: &Device,
        output_device: &Device,
        input_config: &StreamConfig,
        input_format: SampleFormat,
    output_config: &StreamConfig,
    output_format: SampleFormat,
    target_sample_rate: u32,
    frame_size: usize,
    filter_length: usize,
) -> Result<Self, Box<dyn Error>> {
        let input_channels = input_config.channels as usize;
        let output_channels = output_config.channels as usize;
        let input_rate = input_config.sample_rate.0;
        let output_rate = output_config.sample_rate.0;


        let mut input_resampler = if input_rate != target_sample_rate {
            println!("Resampling input stream to match AEC rate.");
            Some(
                Resampler::new(
                    input_channels as u32,
                    input_rate,
                    target_sample_rate,
                    RESAMPLER_QUALITY,
                )
                .map_err(|e| format!("Failed to create input resampler: {e}"))?,
            )
        } else {
            None
        };

        let mut output_resampler = if output_rate != target_sample_rate {
            println!("Resampling output reference stream to match AEC rate.");
            Some(
                Resampler::new(
                    output_channels as u32,
                    output_rate,
                    target_sample_rate,
                    RESAMPLER_QUALITY,
                )
                .map_err(|e| format!("Failed to create output resampler: {e}"))?,
            )
        } else {
            None
        };

        let canceller =
            EchoCanceller::new_multichannel(frame_size, filter_length, input_channels, output_channels)
                .ok_or("Failed to allocate echo canceller state")?;

        unsafe {
            let mut rate = target_sample_rate as i32;
            speex_echo_ctl(
                canceller.as_ptr(),
                SPEEX_ECHO_SET_SAMPLING_RATE,
                &mut rate as *mut _ as *mut c_void,
            );
        }

        let input_resampler_ctx = input_resampler
            .map(|resampler| Arc::new(Mutex::new(ResampleContext::new(resampler))));
        let output_resampler_ctx = output_resampler
            .map(|resampler| Arc::new(Mutex::new(ResampleContext::new(resampler))));

        let shared = Arc::new(Mutex::new(SharedCanceller::new(
            canceller,
            input_channels,
            output_channels,
            frame_size,
        )));
        let playback = Arc::new(Mutex::new(PlaybackBuffer::new(output_channels)));

        let output_stream = build_output_stream(
            output_device,
            output_config,
            Arc::clone(&shared),
            Arc::clone(&playback),
            output_resampler_ctx.clone(),
            output_format,
        )?;

        let input_stream = build_input_stream(
            input_device,
            input_config,
            Arc::clone(&shared),
            input_resampler_ctx.clone(),
            input_format,
        )?;

        output_stream
            .play()
            .map_err(|e| format!("Failed to start output stream: {e}"))?;
        input_stream
            .play()
            .map_err(|e| format!("Failed to start input stream: {e}"))?;

        Ok(Self {
            shared,
            playback,
            playback_resamplers: Mutex::new(HashMap::new()),
            _output_stream: output_stream,
            _input_stream: input_stream,
            output_channels,
            target_sample_rate,
            input_resampler: input_resampler_ctx,
            output_resampler: output_resampler_ctx,
        })
    }

    fn add_audio(&self, channel: usize, samples: &[f32], sample_rate: u32) {
        if samples.is_empty() {
            return;
        }
        if self.output_channels == 0 {
            eprintln!("add_audio called but no output channels are configured.");
            return;
        }
        if channel >= self.output_channels {
            eprintln!(
                "add_audio requested channel {} but only {} output channel(s) configured.",
                channel, self.output_channels
            );
            return;
        }

        let channel_count = self.output_channels;
        let use_interleaved = channel_count > 1
            && samples.len() >= channel_count
            && samples.len() % channel_count == 0;
        let frames = if use_interleaved {
            samples.len() / channel_count
        } else {
            samples.len()
        };
        if frames == 0 {
            return;
        }

        let mut interleaved = vec![0i16; frames.saturating_mul(channel_count)];

        if use_interleaved {
            for frame_idx in 0..frames {
                let idx = frame_idx * channel_count + channel;
                let value = samples[idx];
                let finite = if value.is_finite() { value } else { 0.0 };
                let clamped = finite.clamp(-1.0, 1.0);
                interleaved[frame_idx * channel_count + channel] =
                    (clamped * i16::MAX as f32) as i16;
            }
        } else {
            for (frame_idx, &value) in samples.iter().enumerate() {
                let finite = if value.is_finite() { value } else { 0.0 };
                let clamped = finite.clamp(-1.0, 1.0);
                interleaved[frame_idx * channel_count + channel] =
                    (clamped * i16::MAX as f32) as i16;
            }
        }

        if sample_rate == self.target_sample_rate {
            if let Ok(mut playback) = self.playback.lock() {
                playback.push_samples(&interleaved);
            }
            return;
        }

        let resampled = {
            let mut resamplers = match self.playback_resamplers.lock() {
                Ok(guard) => guard,
                Err(err) => {
                    eprintln!("Failed to lock playback resamplers: {err}");
                    return;
                }
            };

            let ctx = match resamplers.entry(sample_rate) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    let resampler = match Resampler::new(
                        channel_count as u32,
                        sample_rate,
                        self.target_sample_rate,
                        RESAMPLER_QUALITY,
                    ) {
                        Ok(resampler) => resampler,
                        Err(err) => {
                            eprintln!(
                                "Unable to resample playback from {sample_rate} Hz to {} Hz: {err}",
                                self.target_sample_rate
                            );
                            return;
                        }
                    };
                    entry.insert(ResampleContext::new(resampler))
                }
            };

            let mut produced = Vec::<i16>::new();
            ctx.process(&interleaved, &mut produced);
            produced
        };

        if resampled.is_empty() {
            return;
        }

        if let Ok(mut playback) = self.playback.lock() {
            playback.push_samples(&resampled);
        }
    }

    fn register_callback<F>(&self, callback: F)
    where
        F: FnMut(&[i16]) + Send + 'static,
    {
        let mut guard = self.shared.lock().unwrap();
        guard.add_callback(callback);
    }

    fn shared_state(&self) -> Arc<Mutex<SharedCanceller>> {
        Arc::clone(&self.shared)
    }

    fn flush_resamplers(&self) {
        let mut produced = Vec::<i16>::new();

        if let Some(resampler_handle) = &self.output_resampler {
            produced.clear();
            if let Ok(mut ctx) = resampler_handle.lock() {
                ctx.flush_into(&mut produced);
            }
            if !produced.is_empty() {
                if let Ok(mut state) = self.shared.lock() {
                    state.push_far_end_resampled(&produced);
                }
            }
        }

        if let Some(resampler_handle) = &self.input_resampler {
            produced.clear();
            if let Ok(mut ctx) = resampler_handle.lock() {
                ctx.flush_into(&mut produced);
            }
            if !produced.is_empty() {
                if let Ok(mut state) = self.shared.lock() {
                    state.process_capture_resampled(&produced, None);
                }
            }
        }

        let mut playback_leftover = Vec::<i16>::new();
        if let Ok(mut resamplers) = self.playback_resamplers.lock() {
            for ctx in resamplers.values_mut() {
                ctx.flush_into(&mut playback_leftover);
            }
        }
        if !playback_leftover.is_empty() {
            if let Ok(mut state) = self.shared.lock() {
                state.push_far_end_resampled(&playback_leftover);
            }
        }
    }
}

impl Drop for SharedAecStream {
    fn drop(&mut self) {
        self.flush_resamplers();
    }
}
*/

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
