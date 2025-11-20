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
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{self, TryRecvError},
        Arc,
        Mutex,
    },
    time::Duration,
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Device, FromSample, Host, InputCallbackInfo, Sample, SampleFormat, SampleRate, SizedSample,
    Stream, StreamConfig, StreamInstant, SupportedStreamConfig, SupportedStreamConfigRange,
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

use std::time::{Instant, UNIX_EPOCH};


const DEFAULT_AEC_RATE: u32 = 48_000;
const RESAMPLER_QUALITY: i32 = 5;
const STREAM_ALIGNER_HISTORY_LEN: usize = 128;
const STREAM_ALIGNER_INPUT_BUFFER_SECONDS: u32 = 2;
const STREAM_ALIGNER_OUTPUT_BUFFER_SECONDS: u32 = 2;

type AecCallback = Box<dyn FnMut(&[i16]) + Send + 'static>;



#[inline]
unsafe fn assume_init_slice_mut<T>(slice: &mut [MaybeUninit<T>]) -> &mut [T] {
    &mut *(slice as *mut [MaybeUninit<T>] as *mut [T])
}

/*
struct ResampleContext {
    resampler: Resampler,
    channels: usize,
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
    let input_descriptors = [InputDeviceDescriptor {
        device_name: input_device_name.clone(),
        config: input_stream_config.clone(),
        sample_format: input_config.sample_format(),
    }];
    let output_descriptors = [OutputDeviceDescriptor {
        device_name: output_device_name.clone(),
        config: output_stream_config.clone(),
        sample_format: output_config.sample_format(),
    }];

    let shared_stream = SharedAecStream::new(
        &host,
        &input_descriptors,
        &output_descriptors,
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


*/




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

fn frames_to_micros(frames: u128, sample_rate: u128) -> u128 {
    // frames = (microseconds * sample_rate / 1 000 000)
    // frames * 1_000_000 = microseconds * sample_rate
    frames * 1000000 / sample_rate // = microseconds
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
            if _appended < num_written {
                eprintln!("Warning: Producer cannot keep up, increase buffer size or decrease latency")
            }
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
            (false, &mut buf[..size])
        } else if first.is_empty() && second.len() >= size {
            let buf = unsafe { assume_init_slice_mut(second) };
            (false, &mut buf[..size])
        }
        else {
            if self.scratch.len() < size {
                self.scratch.resize_with(size, Default::default);
            }
            (true, &mut self.scratch[..size])
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

    fn available(&self) -> usize {
        let (head, tail) = self.consumer.as_slices();
        let head_len = head.len();
        let tail_len = tail.len();
        head_len + tail_len
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


// a wrapper around BufferedCircularConsumer that resamples the stream before outputting
// you must call .resample(...)? before calling get_chunk_to_read() or there will be nothing available
// a safe choice is .resample(consumer.available_to_resample())
struct ResampledBufferedCircularProducer {
    consumer: BufferedCircularConsumer<f32>,
    resampled_producer: BufferedCircularProducer<f32>,
    input_sample_rate: u32,
    output_sample_rate: u32,
    total_input_samples_remaining: i128,
    resampler: Resampler
}

impl ResampledBufferedCircularProducer {
    fn new(
        input_sample_rate: u32,
        output_sample_rate : u32,
        resampler_quality: i32,
        audio_buffer_seconds: u32,
        consumer: BufferedCircularConsumer<f32>,
        resampled_producer: BufferedCircularProducer<f32>) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            consumer: consumer,
            resampled_producer: resampled_producer,
            total_input_samples_remaining: 0,
            input_sample_rate: input_sample_rate,
            output_sample_rate: output_sample_rate,
            resampler: Resampler::new(
                1, // channels, we have one of these StreamAligner each channel
                input_sample_rate,
                output_sample_rate,
                resampler_quality
            )?
        })
    }

    fn set_sample_rate(&mut self, input_sample_rate: u32, output_sample_rate: u32) -> Result<(), Box<dyn std::error::Error>> {
        self.resampler.set_rate(input_sample_rate, output_sample_rate)?;
        self.input_sample_rate = input_sample_rate;
        self.output_sample_rate = output_sample_rate;
        Ok(())
    }

    fn available_to_resample(&self) -> usize {
        self.consumer.available()
    }

    fn available(&self) -> usize {
        self.resampled_consumer.available()
    }

    fn finish_read(&mut self, num_read: usize) -> usize {
        self.resampled_consumer.finish_read(num_read)
    }
}

impl ResampledBufferedCircularProducer {
    // resample all data available
    fn resample_all(&mut self) -> Result<(usize, usize), Box<dyn std::error::Error>>  {
        self.resample(self.available_to_resample() as u32)
    }

    fn resample(&mut self, num_available_frames: u32) -> Result<(usize, usize), Box<dyn std::error::Error>> {
        if num_available_frames == 0 {
            return Ok((0,0));
        }
        // there might be some leftover from last call, so use state
        self.total_input_samples_remaining += num_available_frames as i128;

        let input_buf = self.consumer.get_chunk_to_read(self.total_input_samples_remaining as usize);
        let target_output_samples_count = input_to_output_frames(self.total_input_samples_remaining, self.input_sample_rate, self.output_sample_rate) + 10; // add a few extra for rounding
        let (need_to_write_outputs, output_buf) = self.resampled_producer.get_chunk_to_write(target_output_samples_count as usize);
        let (consumed, produced) = self.resampler.process_interleaved_f32(input_buf, output_buf)?;
        // it may return less consumed and produced than the sizes of stuff we gave it
        // so use actual processed sizes here instead of our lengths from above
        // (worst case this is like 0.6 ms or so, so it's okay to have them slightly delayed like this)
        self.consumer.finish_read(consumed);
        self.resampled_producer.finish_write(need_to_write_outputs, produced);

        self.total_input_samples_remaining -= consumed as i128;
        Ok((consumed, produced))
    }
}

enum AudioBufferMetadata {
    Arrive(u64, u128, u128, bool),
    Teardown(),
}


struct InputStreamAlignerProducer {
    start_time_micros: Option<u128>,
    input_sample_rate: u32,
    output_sample_rate: u32,
    input_audio_buffer_producer: HeapProd<f32>,
    input_audio_buffer_metadata_producer: mpsc::Sender<AudioBufferMetadata>,
    chunk_sizes: LocalRb<Heap<usize>>,
    system_time_micros_when_chunk_ended: LocalRb<Heap<u128>>,
    num_calibration_packets: u32,
    num_packets_recieved: u64,
    num_emitted_frames: u128,
}

impl InputStreamAlignerProducer {
    fn new(input_sample_rate: u32, output_sample_rate: u32, history_len: usize, num_calibration_packets: u32, input_audio_buffer_producer: HeapProd<f32>, input_audio_buffer_metadata_producer: mpsc::Sender<AudioBufferMetadata>) -> Result<Self, Box<dyn Error>>  {
        Ok(Self {
            input_sample_rate: input_sample_rate,
            output_sample_rate: output_sample_rate,
            input_audio_buffer_producer: input_audio_buffer_producer,
            input_audio_buffer_metadata_producer: input_audio_buffer_metadata_producer,
            // alignment data, these are used to adjust resample rate so output stays aligned with true timings (according to sytem clock)
            chunk_sizes: LocalRb::<Heap<usize>>::new(history_len),
            system_time_micros_when_chunk_ended: LocalRb::<Heap<u128>>::new(history_len),
            num_calibration_packets: num_calibration_packets,
            num_packets_recieved: 0,
            num_emitted_frames: 0
        })
    }

    fn estimate_micros_when_most_recent_ended(&self) -> u128 {
        // Take minimum over estimates for all previous recieved
        // Some may be delayed due to cpu being busy, but none can ever arrive too early
        // so this should be a decent estimate
        // it does not account for hardware latency, but we cannot account for that without manual calibration
        // (btw, CPAL timestamps do not work because they may be different for different devices)
        // (wheras this synchronizes us to global system time)
        let mut best_estimate_of_when_most_recent_ended = if let Some(most_recent_time) = self.system_time_in_frames_when_chunk_ended.last() {
            *most_recent_time
        }
        else {
            u128::MAX
        };
        let mut frames_until_most_recent = 0 as u128;
        // iterate from most recent backwards (that's what .rev() does)
        for (chunk_size, micros_when_chunk_ended) in self.chunk_sizes.iter().zip(self.system_time_micros_when_chunk_ended.iter()).rev() {
            let micros_until_most_recent_ended = frames_to_micros(frames_until_most_recent as u128, self.input_sample_rate);
            let estimate_of_micros_most_recent_ended = *micros_when_chunk_ended + micros_until_most_recent_ended;
            best_estimate_of_when_most_recent_ended = (estimate_of_micros_most_recent_ended).min(best_estimate_of_when_most_recent_ended);
            // timestamps are at end, not at start, so only increment this after
            frames_until_most_recent += *chunk_size as u128;
        }
        best_estimate_of_when_most_recent_ended
    }

    fn process_chunk(&mut self, chunk: &[f32]) -> Result<(), Box<dyn Error>> {\
        let micros_when_chunk_received = Instant::now().duration_since(UNIX_EPOCH).as_micros();

       
        let appended_count = self.input_audio_buffer_producer.push_slice(chunk);
        if appended_count < chunk.len() { // todo: auto resize
            eprintln!("Error: cannot keep up with audio, buffer is full, try increasing audio_buffer_seconds")
        }
        if appended_count > 0 {
            // delibrately overwrite once we pass history len, we keep a rolling buffer of last 100 or so
            self.chunk_sizes.push_overwrite(appended_count);
            self.system_time_in_frames_when_chunk_ended.push_overwrite(micros_when_chunk_received);

            // use our estimate to suggest how many frames we should have emitted
            // this is used to dynamically adjust sample rate until we actually emit that many frames
            // that ensures that we stay synchronized to the system clock and do not drift
            let micros_when_chunk_ended = self.estimate_micros_when_most_recent_ended();

            self.num_emitted_frames += appended_count;

            let target_emitted_frames, calibrated = if self.num_packets_recieved < self.num_calibration_packets {
                // until we've recieved enough calibration packets, we don't have good enough time estimate
                // thus, simply request num_emitted_frames emitted
                // this avoids large amounts of distortion if we get an initial burst of packets on device init
                self.num_emitted_frames, false
            } else {
                // calibration finished, setup start_time_micros
                if self.num_packets_recieved == self.num_calibration_packets {
                    // now we can actually make a good estimate of our current time,
                    // which allows us to make a good estimate of start time (just convert number packets emitted into an offset)
                    // this isn't ideal when calibration involved a dropped packet
                    // but is about as good as we can do
                    self.start_time_micros = micros_when_chunk_ended - frames_to_micros(self.num_emitted_frames, self.input_sample_rate);
                }
                // look at actual elapsed time, and use that to say how many frames we would have preferred to emitted
                // this can be used later to adjust sample rate slightly to keep us in line with system time
                let elapsed_micros = micros_when_chunk_ended - self.start_time_micros;
                micros_to_frames(elapsed_micros, self.input_sample_rate), true
            };

            // increment afterwards incase num_calibration_packets = 0
            self.num_packets_recieved += 1 as u64;

            let elapsed_frames = micros_to_frames(micros_when_chunk_ended)
            let metadata = AudioBufferMetadata::Arrive(
                // num available frames
                appended_count as u64,
                // estimated timestamp after this sample
                most_recent_ended_estimate,
                // target emitted frames
                target_emitted_frames,
                // calibrated
                calibrated

            );
            self.input_audio_buffer_metadata_producer.send(metadata)?;
        }
    }

}

enum ResamplingMetadata {
    Arrive(usize, u128, bool),
}

struct InputStreamAlignerResampler {
    input_sample_rate: u32,
    output_sample_rate: u32,
    dynamic_output_sample_rate: u32,
    input_audio_buffer_consumer: BufferedCircularConsumer,
    input_audio_buffer_metadata_consumer: mpsc::Receiver<AudioBufferMetadata>,
    total_emitted_frames: i128,
    total_received_frames: i128,
    total_processed_input_frames: i128,
    start_time_micros: u128,
    finished_resampling_producer: mpsc::Sender<ResamplingMetadata>
}

impl InputStreamAlignerResampler {
    // Takes input audio and resamples it to the target rate
    // May slightly stretch or squeeze the audio (via resampling)
    // to ensure the outputs stay aligned with system clock
    fn new(
        input_sample_rate: u32,
        output_sample_rate: u32,
        audio_buffer_seconds: u32,
        resampler_quality: i32,
        input_audio_buffer_consumer: HeapCons<f32>,
        input_audio_buffer_metadata_consumer: mpsc::Receiver<AudioBufferMetadata>
        output_audio_buffer_producer: HeapProd<f32>,
        finished_resampling_producer: mpsc::Sender<ResamplingMetadata>
    ) -> Result<Self, Box<dyn Error>>  {
        Ok(Self {
            input_sample_rate: input_sample_rate,
            output_sample_rate: output_sample_rate,
            dynamic_output_sample_rate: output_sample_rate,
            // we need buffered because this interfaces with speex which expects continuous buffers
            input_audio_buffer_consumer: ResampledBufferedCircularProducer::new(
                input_sample_rate,
                output_sample_rate,
                resampler_quality,
                audio_buffer_seconds, 
                BufferedCircularConsumer::<f32>::new(input_audio_buffer_consumer),
                BufferedCircularProducer::<f32>::new(output_audio_buffer_producer)
            )?,
            input_audio_buffer_metadata_consumer: input_audio_buffer_metadata_consumer,
            // alignment data, these are used to adjust resample rate so output stays aligned with true timings (according to sytem clock)
            total_emitted_frames: 0,
            total_received_frames: 0,
            total_processed_input_frames: 0,
            finished_resampling_producer: finished_resampling_producer,
            start_time_micros: None,
        })
    }

    // do it very slowly
    fn decrease_dynamic_sample_rate(&mut self)  -> Result<(), Box<dyn std::error::Error>>  {
        self.dynamic_output_sample_rate = (((self.output_sample_rate as f32) * 0.95) as i128).max((self.dynamic_output_sample_rate-1) as i128) as u32;
        self.input_audio_buffer_consumer.set_sample_rate(self.input_sample_rate, self.dynamic_output_sample_rate)?;
        Ok(())
    }

    fn increase_dynamic_sample_rate(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.dynamic_output_sample_rate = (((self.output_sample_rate as f32) * 1.05) as i128).min((self.dynamic_output_sample_rate+1) as i128) as u32;
        self.input_audio_buffer_consumer.set_sample_rate(self.input_sample_rate, self.dynamic_output_sample_rate)?;
        Ok(())
    }

    fn handle_metadata(&mut self, num_available_frames : u64, target_emitted_input_frames : u128) -> Result<(usize, usize), Box<dyn std::error::Error>> {
        let estimated_emitted_frames = input_to_output_frames(num_available_frames as u128, self.input_sample_rate, self.dynamic_output_sample_rate);
        let updated_total_frames_emitted = self.total_emitted_frames + estimated_emitted_frames;
        let target_emitted_output_frames = input_to_output_frames(target_emitted_input_frames, self.input_sample_rate, self.output_sample_rate);
        // dynamic adjustment to synchronize input devices to global clock:
        // not enough frames, we need to increase dynamic sample rate (to get more samples)
        if updated_total_frames_emitted < target_emitted_output_frames {
            self.increase_dynamic_sample_rate()?;
        }
        // too many frames, we need to decrease dynamic sample rate (to get less samples)
        else if updated_total_frames_emitted > target_emitted_output_frames {
            self.decrease_dynamic_sample_rate()?;
        }

        //// do resampling ////
        let (consumed, produced) = self.input_audio_buffer_consumer.resample(num_available_frames as u32)?;

        // the main downside of this is that it'll be persistently behind by 0.6ms or so (the resample frame size), but we'll quickly adjust for that so this shouldn't be a major issue
        // todo: think about how to fix this better (maybe current solution is as good as we can do, and it should average out to correct since past ones accumulated will result in more for this one, still, it's likely to stay behind by this amount)
        self.total_emitted_frames += produced as i128;

        Ok((consumed, produced))
    }

    fn resample(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        // process all recieved audio chunks
        match self.heap_cons_reciever.recv() {
            Ok(msg) => match msg {
                AudioBufferMetadata::Arrive(num_available_frames, system_micros_after_packet_finishes, target_emitted_frames, calibrated) => {
                    let num_leftovers_from_prev = self.total_received_frames - self.total_processed_input_frames;
                    self.total_received_frames += num_available_frames;
                    // if it hypothetically consumed all frames every time,
                    // then we would know that current time in system_micros_after_packet_finishes
                    // however, there are a few things that happen:
                    // 1. There are some "leftover" samples from previously
                    // 2. There are some "ignored" samples from cur (that are later leftover)


                    let consumed, produced = self.handle_metadata(num_available_frames, target_emitted_frames)?;
                    self.total_processed_input_frames += consumed;
                    let num_of_ours_consumed = consumed - num_leftovers_from_prev;
                    let num_of_ours_leftover = num_available_frames - num_of_ours_consumed;
                    let micros_earlier = frames_to_micros(num_of_ours_leftover, self.input_sample_rate);
                    let system_micros_after_resampled_packet_finishes = system_micros_after_packet_finishes - micros_earlier;
                    self.finished_resampling_producer.send(ResamplingMetadata::Arrive(produced, system_micros_after_packet_finishes, calibrated))?;
                    true
                },
                AudioBufferMetadata::Teardown() => {
                    false
                }
            },
            Err(TryRecvError::Empty) => true,          // nothing waiting; continue processing
            Err(TryRecvError::Disconnected) => false,   // sender dropped; bail out or log
        }
    }
}

struct InputStreamAlignerConsumer {
    sample_rate: u32,
    final_audio_buffer_consumer: BufferedCircularConsumer<f32>,
    thread_message_sender: mpsc::Sender<AudioBufferMetadata>,
    finished_message_reciever: mpsc::Reciever<ResamplingMetadata>,
    initial_metadata: Vec<ResamplingMetadata>,
    samples_recieved: u128,
    calibrated: bool,
}

impl InputStreamAlignerConsumer {
    fn new(sample_rate: u32, final_audio_buffer_consumer: BufferedCircularConsumer<f32>, thread_message_sender: mpsc::Sender<AudioBufferMetadata>, finished_message_reciever: mpsc::Receiver<ResamplingMetadata>) {
        Self {
            sample_rate: sample_rate,
            final_audio_buffer_consumer: final_audio_buffer_consumer,
            thread_message_sender: thread_message_sender,
            finished_message_reciever: finished_message_reciever,
            calibrated: false,
        }
    }

    fn drop(&self) {
        if let Err(err) = self.thread_message_sender.send(AudioBufferMetadata::Teardown()) {
            eprintln!("failed to send shutdown signal: {}", err);
        }
    }

    // used to poll for when an input stream is actually ready to output data
    // we allow some initial calibration time to synchronize the clocks
    // (it needs some extra time because packets can be delayed sometimes 
    // so waiting and min over a history lets us get better estimate)
    fn is_ready_to_read(&mut self, micros_packet_finished: u128, size: usize) -> bool {
        // non blocking cause maybe it's just not ready (initialized) yet
        loop {
            match self.finished_message_reciever.try_recv() {
                Ok(msg) => {
                    match msg {
                        ResamplingMetadata::Arrive(frames_recieved, system_micros_after_packet_finishes, calibrated) => {
                            self.calibrated = calibrated;
                            self.initial_metadata.push(msg.clone());
                            self.samples_recieved += frames_recieved;
                        }
                    }
                }
                Err(TryRecvError::Empty) => {
                    // no message available right now
                    break;
                }
                Err(TryRecvError::Disconnected) => {
                    // sender dropped; receiver will never get more messages
                    break;
                }
            }
        }

        // we need to skip ahead to be frame aligned
        if self.calibrated {
            let num_samples_that_are_behind_current_packet = self.num_samples_that_are_behind_current_packet(micros_packet_finished, size);
            let available_samples = self.samples_recieved - num_samples_behind_current_packet;
            
            if available_samples < size {
                // we will be able to get all samples for this packet, block until we get them
                if num_samples_behind_current_packet > 0 {
                    // skip ahead so we are only getting samples for this packet
                    self.final_audio_buffer_consumer.finish_read(num_samples_that_are_behind_current_packet);
                    let additional_samples_needed = size - available_samples;
                    read_success, samples = get_chunk_to_read(additional_samples_needed);
                    true // we will read them again later
                }
                // we started in the middle of this packet, we can't get enough, wait until next packet
                else {
                    false
                }
            }
            else {
                // enough samples! ignore the ones we need to ignore and then let the sampling happen elsewhere
                self.final_audio_buffer_consumer.finish_read(num_samples_that_are_behind_current_packet);
                true
            }
        } else {
            false
        }
    }

    fn num_samples_that_are_behind_current_packet(&self, micros_packet_finished: u128, size: usize) -> {
        let micros_packet_started = micros_packet_finished - frames_to_micros(size, self.sample_rate);
        let mut samples_to_ignore = 0;
        for metadata in self.initial_metadata.iter() {
            match metadata {
                ResamplingMetadata::Arrive(frames_recieved, micros_metadata_finished, calibrated) => {
                    let micros_metadata_started = micros_metadata_finished - frames_to_micros(frames_recieved, self.sample_rate);
                    // whole packet is behind, ignore entire thing
                    if micros_metadata_finished < micros_packet_started {
                        samples_to_ignore += frames_recieved;
                    }
                    // keep all data
                    else if micros_metadata_started >= micros_packet_started{
                        
                    } 
                    // it overlaps, only keep stuff before this packet
                    else {
                        let micros_ignoring = micros_packet_started - micros_metadata_started;
                        samples_to_ignore += micros_to_frames(micros_ignoring, self.sample_rate) - 1; // 1 for rounding, to avoid always throwing stuff out
                    }

                }
            }
        }
        samples_to_ignore
    }

    // waits until we have at least that much data
    // (or something errors)
    // returns (success, audio_buffer)
    fn get_chunk_to_read(&self, size: usize) -> (bool, &[f32]) {
        while self.final_audio_buffer_consumer.available() <= size {
            // wait for data to arrive
            match self.finished_message_reciever.recv() {
                Ok(data) => {

                }
                Err(err) => {
                    eprintln!("channel closed: {err}");
                    return (false, false, &[]),
                }
            }
        }
        (true, self.final_audio_buffer_consumer.get_chunk_to_read(size))
    }

    fn finish_read(&mut self, size: usize) -> usize {
        self.final_audio_buffer_consumer.finish_read(size)
    }
}

// make (producer (recieves audio data from device), resampler (resamples input audio to target rate), consumer (contains resampled data)) for input audio alignment
fn create_input_stream_aligner(input_sample_rate: u32, output_sample_rate: u32, history_len: usize, audio_buffer_seconds: u32, resampler_quality: i32) -> Result<(InputStreamAlignerProducer, InputStreamAlignerResampler, InputStreamAlignerConsumer), Box<dyn Error>> {
    let (input_audio_buffer_producer, input_audio_buffer_consumer) = HeapRb::<f32>::new((audio_buffer_seconds * input_sample_rate) as usize).split();
    let (input_audio_buffer_metadata_producer, input_audio_buffer_metadata_consumer) = mpsc::channel::<HeapConsSendMsg>();
    let additional_input_audio_buffer_metadata_producer = input_audio_buffer_metadata_producer.clone(); // make another one, this is ok because it is multiple producer single consumer 
    // this recieves data from audio buffer
    let producer = InputStreamAlignerProducer::new(
        input_sample_rate,
        output_sample_rate,
        history_len,
        input_audio_buffer_producer,
        input_audio_buffer_metadata_producer
    )?;

    let (finished_resampling_producer, finished_resampling_consumer) = mpsc::channel::<()>();

    let (output_audio_buffer_producer, output_audio_buffer_consumer) = HeapRb::<f32>::new((audio_buffer_seconds * input_sample_rate) as usize).split();
    // resampled_consumer: BufferedCircularConsumer::<f32>::new(resampled_consumer))
    // this resamples, designed to run on a seperate thread
    let resampler = InputStreamAlignerResampler::new(
        input_sample_rate,
        output_sample_rate,
        audio_buffer_seconds,
        resampler_quality,
        input_audio_buffer_consumer,
        input_audio_buffer_metadata_consumer,
        output_audio_buffer_producer,
        finished_resampling_producer,
    )?;

    let consumer = InputStreamAlignerConsumer::new(
        BufferedCircularConsumer::new(output_audio_buffer_consumer),
        additional_input_audio_buffer_metadata_producer, // give it ability to send shutdown signal to thread
        finished_resampling_consumer
    );

    Ok((producer, resampler, consumer))
}

type StreamId = u64;

enum HeapConsSendMsg {
    Add(StreamId, u32, u32, i32, ringbuf::HeapCons<f32>),
    Remove(StreamId),
    InterruptAll(),
}

struct OutputStreamAligner {
    device_sample_rate: u32,
    output_sample_rate: u32,
    heap_cons_sender: mpsc::Sender<HeapConsSendMsg>,
    heap_cons_reciever: mpsc::Receiver<HeapConsSendMsg>,
    input_audio_buffer_producer: BufferedCircularProducer<f32>,
    input_audio_buffer_consumer: BufferedCircularConsumer,
    device_audio_producer: BufferedCircularProducer<f32>,
    stream_consumers: HashMap<StreamId, BufferedCircularConsumer>,
    cur_stream_id: Arc<AtomicU64>,
}

// allows for playing audio on top of each other (mixing) or just appending to buffer
impl OutputStreamAligner {
    fn new(device_sample_rate: u32, output_sample_rate: u32, audio_buffer_seconds: u32, resampler_quality: i32, device_audio_producer: HeapProd<f32>) -> Result<Self, Box<dyn Error>>  {

        // used to send across threads
        let (heap_cons_sender, heap_cons_reciever) = mpsc::channel::<HeapConsSendMsg>();
        let (input_audio_buffer_producer, input_audio_buffer_consumer) = HeapRb::<f32>::new((audio_buffer_seconds * device_sample_rate) as usize).split();
        Ok(Self {
            device_sample_rate: device_sample_rate,
            output_sample_rate: output_sample_rate,
            heap_cons_sender: heap_cons_sender,
            heap_cons_reciever: heap_cons_reciever,
            input_audio_buffer_producer: BufferedCircularProducer::new(input_audio_buffer_producer),
            input_audio_buffer_consumer: BufferedCircularConsumer::new(
                BufferedCircularConsumer::new(input_audio_buffer_consumer)
            )?,
            device_audio_producer: BufferedCircularProducer::new(device_audio_producer),
            stream_consumers: HashMap::new(),
            cur_stream_id: Arc::new(AtomicU64::new(0)),
        })
    }

    fn begin_audio_stream(&self, audio_buffer_seconds: u32, sample_rate: u32, resampler_quality: i32) -> Result<(StreamId, HeapProd<f32>), Box<dyn Error>> {
        // this assigns unique ids in a thread-safe way
        let stream_index = self.cur_stream_id.fetch_add(1, Ordering::Relaxed);
        let (producer, consumer) = HeapRb::<f32>::new((audio_buffer_seconds * sample_rate) as usize).split();
        // send the consumer to the consume thread
        self.heap_cons_sender.send(HeapConsSendMsg::Add(stream_index, sample_rate, audio_buffer_seconds, resampler_quality, consumer))?;
        Ok((stream_index, producer))
    }

    fn enqueue_audio(audio_data: &[f32], mut audio_producer: HeapProd<f32>) {
        let num_pushed = audio_producer.push_slice(audio_data);
        if num_pushed < audio_data.len() {
            eprintln!("Error: output audio buffer got behind, try increasing buffer size");
        }
    }

    fn end_audio_stream(&self, stream_index: StreamId) -> Result<(), Box<dyn Error>> {
        self.heap_cons_sender.send(HeapConsSendMsg::Remove(stream_index))?;
        Ok(())
    }

    fn interrupt_all_streams(&self) -> Result<(), Box<dyn Error>> { 
        self.heap_cons_sender.send(HeapConsSendMsg::InterruptAll())?;
        Ok(())
    }
    
    // frame size should be small, on the order of 1-2ms or less.
    // otherwise you may get skipping if you do not provide audio via enqueue_audio fast enough
    // larger frame sizes will also prevent immediate interruption, as interruption can only happen between each frame
    fn get_chunk_to_read(&mut self, input_frame_size: usize, output_chunk_size: usize) -> Result<&[f32], Box<dyn std::error::Error>> {
        while self.input_audio_buffer_consumer.available() < output_chunk_size {
            self.mix_audio_streams(input_frame_size)?;
        }
        Ok(self.input_audio_buffer_consumer.get_chunk_to_read(output_chunk_size))
    }

    fn mix_audio_streams(&mut self, input_chunk_size: usize) -> Result<(), Box<dyn std::error::Error>> {
        // fetch new audio consumers, non-blocking
        loop {
            match self.heap_cons_reciever.try_recv() {
                Ok(msg) => match msg {
                    HeapConsSendMsg::Add(id, sample_rate, audio_buffer_seconds, resampler_quality, cons) => {
                        self.stream_consumers.insert(id, BufferedCircularConsumer::new(cons));
                    }
                    HeapConsSendMsg::Remove(id) => {
                        // remove if present
                        self.stream_consumers.remove(&id);
                    }
                    HeapConsSendMsg::InterruptAll() => {
                        // remove all streams, interrupt requires new streams to be created
                        self.stream_consumers.clear();
                    }
                },
                Err(TryRecvError::Empty) => break,          // nothing waiting; continue processing
                Err(TryRecvError::Disconnected) => break,   // sender dropped; bail out or log
            }
        }

        let (need_to_write_input_values, input_buf_write) = self.input_audio_buffer_producer.get_chunk_to_write(input_chunk_size);
        let actual_input_chunk_size = input_buf_write.len();
        input_buf_write.fill(0.0);
        for (_stream_id, cons) in self.stream_consumers.iter_mut() {
            cons.resample_all()?; // do resampling of any available data
            let buf_from_stream = cons.get_chunk_to_read(actual_input_chunk_size);
            let samples_to_mix = buf_from_stream.len().min(actual_input_chunk_size);
            if samples_to_mix == 0 {
                continue;
            }
            // just add to mix, do not average or clamp. Average results in too quiet, clamp is non-linear (so confuses eac, which only works with linear transformations), 
            // (fyi, resample is a linear operation in speex so it's safe to do while using eac)
            // see this https://dsp.stackexchange.com/a/3603
            for (dst, &src) in input_buf_write[..samples_to_mix]
                .iter_mut()
                .zip(buf_from_stream.iter())
            {
                *dst += src;
            }
            
            cons.finish_read(samples_to_mix);
        }
        // send mixed values to the output device
        let (need_to_write_device_values, device_buf_write) = self.device_audio_producer.get_chunk_to_write(actual_input_chunk_size);
        let samples_to_copy = device_buf_write.len().min(actual_input_chunk_size);
        device_buf_write[..samples_to_copy].copy_from_slice(&input_buf_write[..samples_to_copy]);
        self.device_audio_producer.finish_write(need_to_write_device_values, samples_to_copy);
        
        // finish writing to the input buffer
        self.input_audio_buffer_producer.finish_write(need_to_write_input_values, actual_input_chunk_size);
        
        // resample any available data
        self.input_audio_buffer_consumer.resample_all()?;
        Ok(())
    }
}

struct InputDeviceConfig {
    host: Host,
    device_name: String,
    channels: u16,
    sample_rate: u32,
    sample_format: SampleFormat,
    
    // number of audio chunks to hold in memory, for aligning input devices's values when dropped frames/clock offsets. 100 or so is fine
    history_len: usize,
    // how long buffer of input audio to store, should only really need a few seconds as things are mostly streamed
    audio_buffer_seconds: u32,
    resampler_quality: i32
}

struct OutputDeviceConfig {
    host: Host,
    device_name: String,
    channels: u16,
    sample_rate: u32,
    sample_format: SampleFormat,
    
    // how long buffer of output audio to store, should only really need a few seconds as things are mostly streamed
    audio_buffer_seconds: u32,
    resampler_quality: i32
}

struct AecConfig {
    target_sample_rate: u32,
    frame_size: usize,
    filter_length: usize
}


fn get_input_stream_aligners(device_config: &InputDeviceConfig, aec_config: &AecConfig) -> Result<(Stream, Vec<InputStreamAlignerConsumer>), Box<dyn std::error::Error>>  {

    let device = select_device(
        device_config.host.input_devices(),
        &device_config.device_name,
        "Input",
    )?;

    let supported_config = find_matching_device_config(
        &device,
        &device_config.device_name,
        device_config.channels,
        device_config.sample_rate,
        device_config.sample_format,
        "Input",
    )?;
    
    let mut aligner_producers = Vec::new();
    let mut aligner_consumers = Vec::new();

    for channel in 0..device_config.channels {
        let (producer, resampler, consumer) = create_input_stream_aligner(
            device_config.sample_rate,
            aec_config.target_sample_rate,
            device_config.history_len,
            device_config.audio_buffer_seconds,
            device_config.resampler_quality)?;
        aligner_producers.push(producer);

        // start the resampler thread
        thread::spawn({
            // it returns true when signaled to stop (such as when consumer goes out of scope)
            while resampler.resample() {

            }
        });

        aligner_consumers.push(consumer);
    }

    let stream = build_input_alignment_stream(
        &device,
        device_config,
        supported_config,
        aligner_producers,
    )?;

    Ok((stream, aligner_consumers))
}

fn get_output_stream_aligners(device_config: &OutputDeviceConfig, aec_config: &AecConfig) -> Result<(Stream, Vec<OutputStreamAligner>), Box<dyn std::error::Error>> {

    let device = select_device(
        device_config.host.output_devices(),
        &device_config.device_name,
        "Output",
    )?;

    let supported_config = find_matching_device_config(
        &device,
        &device_config.device_name,
        device_config.channels,
        device_config.sample_rate,
        device_config.sample_format,
        "Output",
    )?;
    
    let mut aligners = Vec::<OutputStreamAligner>::new();
    let mut device_audio_channel_consumers = Vec::new();
    for channel in 0..device_config.channels {
        let (device_audio_producer, device_audio_consumer) = HeapRb::<f32>::new((device_config.audio_buffer_seconds * device_config.sample_rate) as usize).split();
        let aligner = OutputStreamAligner::new(
            device_config.sample_rate,
            aec_config.target_sample_rate,
            device_config.audio_buffer_seconds,
            device_config.resampler_quality,
            device_audio_producer,
        )?;
        aligners.push(aligner);
        device_audio_channel_consumers.push(BufferedCircularConsumer::new(device_audio_consumer));
    }

    let stream = build_output_alignment_stream(
        &device,
        device_config,
        supported_config,
        device_audio_channel_consumers
    )?;

    Ok((stream, aligners))
}

enum DeviceUpdateMessage {
    AddInputDevice(&InputDeviceConfi, Stream, Vec<InputStreamAlignerConsumer>),
    RemoveInputDevice(&InputDeviceConfig),
    AddOutputDevice(&OutputDeviceConfig, Stream, Vec<OutputStreamAligner>),
    RemoveOutputDevice(&OutputDeviceConfig)
}

struct AecStream {
    aec: Option<EchoCanceller>,
    aec_config: AecConfig,
    device_update_sender: mpsc::Sender<DeviceUpdateMessage>,
    device_update_receiver: mpsc::Receiver<DeviceUpdateMessage>,
    input_streams: HashMap<String, Stream>,
    output_streams: HashMap<String, Stream>,
    input_aligners: HashMap<String, Vec<InputStreamAlignerConsumer>>,
    input_aligners_in_progress: HashMap<String, Vec<InputStreamAlignerConsumer>>,
    output_aligners: HashMap<String, Vec<OutputStreamAligner>>,
    sorted_input_aligners: Vec<String>,
    sorted_output_aligners: Vec<String>,
    input_channels: usize,
    output_channels: usize,
    start_micros: Option<u128>,
    total_frames_emitted: u128,
    input_audio_buffer: Vec<i16>,
    output_audio_buffer: Vec<i16>,
    aec_audio_buffer: Vec<i16>,
}

impl AecStream {
    fn new(
        aec_config: AecConfig
    ) -> Result<Self, Box<dyn Error>> {
        if aec_config.target_sample_rate < 0 {
            return Err(format!("Target sample rate is {}, it must be greater than zero.", aec_config.target_sample_rate).into());
        }
        let (device_update_sender, device_update_receiver) = mpsc::channel::<HeapConsSendMsg>();
        Ok(Self {
           aec: None,
           aec_config: aec_config,
           device_update_sender: device_update_sender,
           device_update_receiver: device_update_receiver,
           input_streams: HashMap::new(),
           output_streams: HashMap::new(),
           input_aligners: HashMap::new(),
           input_aligners_in_progress: HashMap::new(),
           output_aligners: HashMap::new(),
           sorted_input_aligners: Vec::new(),
           sorted_output_aligners: Vec::new(),
           input_channels: 0,
           output_channels: 0,
           start_micros: None,
           total_frames_emitted: 0,
           input_audio_buffer: Vec::new(),
           output_audio_buffer: Vec::New(),
           aec_audio_buffer: Vec::new()
        })
    }

    fn num_input_channels(&self) -> usize {
        self.input_aligners.values().flatten().count()
    }

    fn num_output_channels(&self) -> usize {
        self.output_aligners.values().flatten().count()
    }

    fn reinitialize_aec(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.input_channels = self.num_input_channels();
        self.output_channels = self.num_output_channels();

        // store a consistent ordering
        self.sorted_input_aligners = self.input_aligners.keys().cloned().collect();
        self.sorted_input_aligners.sort();

        self.sorted_output_aligners = self.output_aligners.keys().cloned().collect();
        self.sorted_output_aligners.sort();

        self.aec = EchoCanceller::new_multichannel(
            self.aec_config.frame_size,
            self.aec_config.filter_length,
            self.input_channels,
            self.output_channels,
        );

        self.input_audio_buffer.clear();
        self.input_audio_buffer.resize(self.input_channels, 0 as i16);
        self.output_audio_buffer.clear();
        self.output_audio_buffer.resize(self.output_channels, 0 as i16)
        self.aec_audio_buffer.clear();
        self.aec_audio_buffer.resize(self.input_channels, 0 as i16);
        Ok(())
    }

    fn add_input_device(&mut self, config: &InputDeviceConfig) -> Result<(), Box<dyn std::error::Error>> {
        let (stream, aligners) = get_input_stream_aligners(config, &self.aec_config)?;
        self.device_update_sender.send(DeviceUpdateMessage::AddInputDevice(config.clone(), stream, aligners))?;
        Ok(())
    }

    fn add_output_device(&mut self, config: &OutputDeviceConfig) -> Result<(), Box<dyn std::error::Error>> {
        let (stream, aligners) = get_output_stream_aligners(config, &self.aec_config)?;
        self.device_update_sender.send(DeviceUpdateMessage::AddOutputDevice(config.clone(), stream, aligners))?;
        Ok(())
    }

    fn remove_input_device(&mut self, config: &InputDeviceConfig) -> Result<(), Box<dyn std::error::Error>> {
        self.device_update_sender.send(DeviceUpdateMessage::RemoveInputDevice(config.clone()))?;
        Ok(())
    }

    fn remove_output_device(&mut self, config: &OutputDeviceConfig) -> Result<(), Box<dyn std::error::Error>> {
        self.device_update_sender.send(DeviceUpdateMessage::RemoveOutputDevice(config.clone()))?;
        Ok(())
    }

    fn update(&self, chunk_size: usize) {
        let start_micros = if let Some(start_micros_value) = self.start_micros {
            start_micros_value
        } else {
            let start_micros_value = SystemTime::now().duration_since(UNIX_EPOCH).as_micros() - frames_to_micros(chunk_size as u128, self.aec_config.target_sample_rate as u128);
            self.start_micros = Some(start_micros_value);
            start_micros_value
        }
        let chunk_start_micros = frames_to_micros(self.total_frames_emitted, self.aec_config.target_sample_rate as u128) + start_micros;
        let chunk_end_micros = chunk_start_micros + frames_to_micros(chunk_size as u128, self.aec_config.target_sample_rate as u128);
        self.total_frames_emitted += chunk_size as u128;
        // todo: move to crossbeam to avoid mpsc locks
        loop {
            match receiver.try_recv() {
                Ok(msg) => {
                    Ok(msg) => match msg {
                        DeviceUpdateMessage::AddInputDevice(config, stream, aligners) {
                            if let Some(stream) = self.input_streams.get(&config.device_name) {
                                stream.pause()?;
                            }
                            self.input_streams.insert(config.device_name.clone(), stream);
                            self.input_aligners_in_progress.insert(config.device_name.clone(), aligners);
                            self.input_aligners.insert(config.device_name.clone(), Vec::new());

                            self.reinitialize_aec();
                            Ok(())
                        }
                        DeviceUpdateMessage::RemoveInputDevice(config) {
                            if let Some(stream) = self.input_streams.get(&config.device_name) {
                                stream.pause()?;
                            }
                            self.input_streams.remove(&config.device_name);
                            self.input_aligners.remove(&config.device_name);
                            self.input_aligners_in_progress.remove(&config.device_name);

                            self.reinitialize_aec();
                            Ok(())
                        }
                        DeviceUpdateMessage::AddOutputDevice(config, stream, aligners) {
                            if let Some(stream) = self.output_streams.get(&config.device_name) {
                                stream.pause()?;
                            }
                            self.output_streams.insert(config.device_name.clone(), stream);
                            self.output_aligners.insert(config.device_name.clone(), aligners);

                            self.reinitialize_aec();
                            Ok(())
                        }
                        DeviceUpdateMessage::RemoveOutputDevice(config) {
                            if let Some(stream) = self.output_streams.get(&config.device_name) {
                                stream.pause()?;
                            }
                            self.output_streams.remove(&config.device_name);
                            self.output_aligners.remove(&config.device_name);

                            self.reinitialize_aec();
                        }
                    }
                }
                Err(TryRecvError::Empty) => {
                    // no message available right now
                    break;
                }
                Err(TryRecvError::Disconnected) => {
                    // sender dropped; receiver will never get more messages
                    // todo: add error
                }
            } 
        }
        // similarly, if we initialize an output device here
        // we may not get any audio for a little bit
        // won't get 
        if chunk_size == 0 {
            return;
        }
        let Some(aec) = self.aec.as_mut() else { return };
        if self.output_channels == 0 {
            // todo: simply pass through input_channels, no need for aec
            return;
        }

        // initialize any new aligners and align them to our frame step
        let mut modified_aligners = false;
        for key in &self.sorted_input_aligners {
            if let Some(aligners_in_progress) = self.input_aligners_in_progress.get_mut(key) {
                let finished_aligners = Vec::new();
                let remove_indices = Vec::new();
                for (index, aligner) in aligners_in_progress.iter().enumerate() {
                    // returns true and skips ahead once it is calibrated and ready
                    if aligner.is_ready_to_read(chunk_end_micros, chunk_size) {
                        remove_indices.push(index);
                        finished_aligners.push(aligner);
                    }
                }
                if let Some(ready_aligners) = self.input_aligners.get_mut(key) {
                    for remove_index, finished_aligner in remove_indices.iter().zip(finished_aligners.iter()) {
                        aligners_in_progress.remove(remove_index);
                        self.input_aligners.push(ready_aligners);
                        modified_aligners = true;
                    }
                }
            }
        }
        if modified_aligners {
            self.reinitialize_aec();
        }

        let input_channel = 0;
        for key in &self.sorted_input_aligners {
            if let Some(channel_aligners) = self.input_aligners.get_mut(key) {
                for aligner in channel_aligners {
                    let (ok, chunk) = aligner.get_chunk_to_read(chunk_size);
                    if ok {
                        Self::write_channel_from_f32(
                            chunk,
                            input_channel,
                            self.input_channels,
                            chunk_size,
                            &mut self.input_audio_buffer,
                        );
                        aligner.finish_read(chunk.len().min(chunk_size));
                    } else {
                        Self::clear_channel(mic_channel, self.input_channels, chunk_size, &mut mic_frame);
                    }
                    input_channel += 1;
                }
            }
        }

        for (output_i, output_key) in self.sorted_output_aligners.iter().enumerate() {
            if let Some(output_aligner) = self.output_aligners.get(output_key) {
                //let output_channel_i_data = output_aligner.get_chunk_to_read(chunk_size);
                // todo: assign into output_audio_buffer
            } else {
                // todo: set output_audio_buffer values to 0 for this channel
            };
        }
        // todo: send input_audio_buffer and output_audio_buffer in self.aec
    }
    fn write_channel_from_f32(
        src: &[f32],
        channel: usize,
        total_channels: usize,
        frames: usize,
        dst: &mut [i16],
    ) {
        for frame in 0..frames {
            let value = src.get(frame).copied().unwrap_or(0.0);
            dst[frame * total_channels + channel] = Self::f32_to_i16(value);
        }
    }
    fn clear_channel(channel: usize, total_channels: usize, frames: usize, dst: &mut [i16]) {
        for frame in 0..frames {
            dst[frame * total_channels + channel] = 0;
        }
    }
    fn f32_to_i16(sample: f32) -> i16 {
        let clamped = sample.clamp(-1.0, 1.0);
        (clamped * i16::MAX as f32).round() as i16
    }
}
    


fn main() -> Result<(), Box<dyn Error>> {
    Ok(())
}

fn select_device<I>(
    devices: Result<I, cpal::DevicesError>,
    target: &str,
    kind: &str,
) -> Result<Device, Box<dyn Error>>
where
    I: IntoIterator<Item = Device> {
    match devices {
        Ok(device_iter) => {
            let mut available = Vec::new();

            for device in device_iter {
                let name = device.name().unwrap_or_else(|_| "<unknown device>".to_string());
                available.push(name.clone());

                if &name == target {
                    return Ok(device);
                }
            }

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
        Err(err) => Err(format!("Failed to enumerate {kind} devices: {err}").into()),
    }
}

fn find_matching_device_config(
    device: &Device,
    device_name: &String,
    channels: u16,
    sample_rate: u32,
    format: SampleFormat,
    direction: &'static str,
) -> Result<SupportedStreamConfig, Box<dyn Error>> {
    let configs : Vec<_> = match direction {
        "Input" => device.supported_input_configs().map(|configs| configs.collect())
            .map_err(|err| format!("Unable to enumerate input configs for '{device_name}': {err}"))?,
        "Output" => device.supported_output_configs().map(|configs| configs.collect())
            .map_err(|err| format!("Unable to enumerate output configs for '{device_name}': {err}"))?,
        other => {
            return Err(format!("Unknown device direction '{other}' when validating {device_name}., should be input or output").into());
        }
    };

    if configs.is_empty() {
        return Err(format!(
            "{} device '{}' reported no supported stream configurations to validate.",
            direction, device_name
        )
        .into());
    }

    let desired_rate = SampleRate(sample_rate);
    let matching_config = configs
        .iter()
        .filter(|cfg| cfg.channels() == channels && cfg.sample_format() == format)
        .find_map(|cfg| cfg.clone().try_with_sample_rate(desired_rate));
    
    if let Some(config) = matching_config {
        Ok(config)
    } else {
        let supported_list = configs
            .iter()
            .enumerate()
            .map(|(idx, cfg)| {
                let min_rate = cfg.min_sample_rate().0;
                let max_rate = cfg.max_sample_rate().0;
                let rate_desc = if min_rate == max_rate {
                    format!("{min_rate} Hz")
                } else {
                    format!("{min_rate}-{max_rate} Hz")
                };
                format!(
                    "#{idx}: {} channel(s), {:?}, sample rates: {rate_desc}",
                    cfg.channels(),
                    cfg.sample_format()
                )
            })
            .collect::<Vec<_>>()
            .join("; ");

        Err(format!(
            "{} device '{}' does not support {} channel(s), {:?} at {} Hz. Supported configs: {}",
            direction, device_name, channels, format, sample_rate, supported_list
        )
        .into())
    }
}


fn build_input_alignment_stream(
    device: &Device,
    config: &InputDeviceConfig,
    supported_config: SupportedStreamConfig,
    channel_aligners: Vec<InputStreamAlignerProducer>,
) -> Result<Stream, cpal::BuildStreamError> {
    match config.sample_format {
        SampleFormat::I16 => build_input_alignment_stream_typed::<i16>(
            device,
            config,
            supported_config,
            channel_aligners,
        ),
        SampleFormat::F32 => build_input_alignment_stream_typed::<f32>(
            device,
            config,
            supported_config,
            channel_aligners,
        ),
        SampleFormat::U16 => build_input_alignment_stream_typed::<u16>(
            device,
            config,
            supported_config,
            channel_aligners,
        ),
        other => {
            eprintln!(
                "Input device '{0}' uses unsupported sample format {other:?}; cannot build StreamAligner.",
                config.device_name
            );
            Err(cpal::BuildStreamError::StreamConfigNotSupported)
        }
    }
}

fn build_input_alignment_stream_typed<T>(
    device: &Device,
    config: &InputDeviceConfig,
    supported_config: SupportedStreamConfig,
    mut channel_aligners: Vec<InputStreamAlignerProducer>,
) -> Result<Stream, cpal::BuildStreamError>
where
    T: Sample + SizedSample,
    f32: FromSample<T>,
{
    let per_channel_capacity = config.sample_rate
        .saturating_div(20) // ~50 ms of audio per channel
        .max(1024);
    let mut channel_buffers = (0..config.channels)
        .map(|_| Vec::<f32>::with_capacity(per_channel_capacity as usize))
        .collect::<Vec<_>>();
    
    let device_name = config.device_name.clone();
    let channels = config.channels as usize;
    device.build_input_stream(
        &supported_config.config(),
        move |data: &[T], _info: &InputCallbackInfo| {
            if data.is_empty() {
                return;
            }
            for buffer in channel_buffers.iter_mut() {
                buffer.clear();
            }
            // undo the interleaving and convert to f32
            // todo: do this looping per channel in outermost instead of get_mut each time
            // (if it becomes a performance issue)
            for frame in data.chunks(channels) {
                for (channel_idx, sample) in frame.iter().enumerate() {
                    if let Some(buffer) = channel_buffers.get_mut(channel_idx) {
                        buffer.push(f32::from_sample(*sample));
                    }
                }
            }
            for (buffer, channel_aligner) in channel_buffers.iter_mut().zip(channel_aligners.iter_mut()) {
                if buffer.is_empty() {
                    continue;
                }
                channel_aligner.process_chunk(buffer.as_slice());
                buffer.clear();
            }
        },
        move |err| eprintln!("Input stream '{device_name}' error: {err}",),
        None,
    )
}

fn build_output_alignment_stream(
    device: &Device,
    config: &OutputDeviceConfig,
    supported_config: SupportedStreamConfig,
    device_audio_channel_consumers: Vec<BufferedCircularConsumer<f32>>
) -> Result<Stream, cpal::BuildStreamError> {
    match config.sample_format {
        SampleFormat::I16 => build_output_alignment_stream_typed::<i16>(
            device,
            config,
            supported_config,
            device_audio_channel_consumers,
        ),
        SampleFormat::F32 => build_output_alignment_stream_typed::<f32>(
            device,
            config,
            supported_config,
            device_audio_channel_consumers,
        ),
        SampleFormat::U16 => build_output_alignment_stream_typed::<u16>(
            device,
            config,
            supported_config,
            device_audio_channel_consumers,
        ),
        other => {
            eprintln!(
                "Output device '{0}' uses unsupported sample format {other:?}; cannot build StreamAligner.",
                config.device_name
            );
            Err(cpal::BuildStreamError::StreamConfigNotSupported)
        }
    }
}

fn build_output_alignment_stream_typed<T>(
    device: &Device,
    config: &OutputDeviceConfig,
    supported_config: SupportedStreamConfig,
    mut device_audio_channel_consumers: Vec<BufferedCircularConsumer<f32>>
) -> Result<Stream, cpal::BuildStreamError>
where
    T: Sample + SizedSample,
    T: FromSample<f32>,
{
    let device_name = config.device_name.clone();
    let channels = config.channels as usize;
    device.build_output_stream(
        &supported_config.config(),
        move |data: &mut [T], _| {
            let frames = data.len() / channels;
            if frames == 0 {
                return;
            }
            // in case one of the device_audio_channel_consumers doesn't have enough yet
            data.fill(T::from_sample(0.0f32));
            
            // interleave the data which is what cpal expects
            for (channel_idx, consumer) in device_audio_channel_consumers.iter_mut().enumerate() {
                let chunk = consumer.get_chunk_to_read(frames);
                if chunk.is_empty() {
                    continue;
                }

                let samples_to_write = chunk.len().min(frames);
                for (frame_idx, &sample) in chunk.iter().take(samples_to_write).enumerate() {
                    let dst = frame_idx * channels + channel_idx;
                    data[dst] = T::from_sample(sample);
                }

                consumer.finish_read(samples_to_write);
            }
        },
        move |err| eprintln!("Output stream '{device_name}' error: {err}"),
        None,
    )
}

/*
struct InputChannelState {
    device_index: usize,
    channel_index: usize,
    sample_rate: u32,
    aligner: Arc<Mutex<StreamAligner>>,
    output_consumer: BufferedCircularConsumer<f32>,
}

struct DeviceDescriptor {
    device_name: String,
    channels: usize,
    sample_rate: usize,
    sample_format: SampleFormat,
}

struct OutputDeviceDescriptor {
    device_name: String,
    config: StreamConfig,
    sample_format: SampleFormat,
}

struct InputDeviceStream {
    name: String,
    channels: usize,
    sample_rate: u32,
    sample_format: SampleFormat,
    channel_aligners: Vec<Arc<Mutex<StreamAligner>>>,
    _stream: Stream,
}

struct SharedAecStream {
    target_sample_rate: u32,
    input_channels: Vec<InputChannelState>,
    input_streams: Vec<InputDeviceStream>,
    output_device_names: Vec<String>,
}

impl SharedAecStream {
    fn new(
        host: &Host,
        input_devices: &[InputDeviceDescriptor],
        output_devices: &[OutputDeviceDescriptor],
        target_sample_rate: u32,
        frame_size: usize,
        filter_length: usize,
    ) -> Result<Self, Box<dyn Error>> {
        if input_devices.is_empty() {
            return Err("At least one input device is required to build SharedAecStream.".into());
        }
        if target_sample_rate == 0 {
            return Err("Target sample rate must be greater than zero.".into());
        }

        // These parameters will be reintroduced when the canceller plumbing is wired up.
        let _ = (frame_size, filter_length);

        let mut input_streams = Vec::with_capacity(input_devices.len());
        let mut input_channels = Vec::new();

        for (device_index, descriptor) in input_devices.iter().enumerate() {
            let requested_name = descriptor.device_name.as_str();
            let device = select_device(
                host.input_devices(),
                requested_name,
                "Input",
            )?;
            let device_name = device
                .name()
                .unwrap_or_else(|_| requested_name.to_string());

            validate_device_stream_config(
                &device,
                &device_name,
                descriptor.config.channels,
                descriptor.config.sample_rate.0,
                descriptor.sample_format,
                "input",
            )?;

            let stream_config = &descriptor.config;
            let sample_format = descriptor.sample_format;
            let input_rate = stream_config.sample_rate.0;
            let channels = stream_config.channels as usize;

            if channels == 0 {
                return Err(format!(
                    "Input device '{device_name}' reports zero channels; cannot create StreamAligner."
                )
                .into());
            }

            let mut aligners = Vec::with_capacity(channels);
            for channel_index in 0..channels {
                let mut output_capacity = (STREAM_ALIGNER_OUTPUT_BUFFER_SECONDS as usize)
                    .saturating_mul(target_sample_rate as usize);
                if output_capacity == 0 {
                    output_capacity = target_sample_rate as usize;
                }
                if output_capacity == 0 {
                    output_capacity = 1;
                }

                let (output_producer, output_consumer) =
                    HeapRb::<f32>::new(output_capacity).split();
                let aligner = StreamAligner::new(
                    input_rate,
                    target_sample_rate,
                    STREAM_ALIGNER_HISTORY_LEN,
                    STREAM_ALIGNER_INPUT_BUFFER_SECONDS,
                    RESAMPLER_QUALITY,
                    output_producer,
                )
                .map_err(|err| {
                    format!(
                        "Failed to create stream aligner for device '{device_name}' channel {channel_index}: {err}"
                    )
                })?;
                let aligner = Arc::new(Mutex::new(aligner));
                input_channels.push(InputChannelState {
                    device_index,
                    channel_index,
                    sample_rate: input_rate,
                    aligner: Arc::clone(&aligner),
                    output_consumer: BufferedCircularConsumer::new(output_consumer),
                });
                aligners.push(aligner);
            }

            let aligners_for_stream = aligners.iter().map(Arc::clone).collect::<Vec<_>>();
            let stream = build_input_alignment_stream(
                &device,
                &device_name,
                stream_config,
                sample_format,
                aligners_for_stream,
            )?;

            stream
                .play()
                .map_err(|err| format!("Failed to start input stream for '{device_name}': {err}"))?;

            input_streams.push(InputDeviceStream {
                name: device_name,
                channels,
                sample_rate: input_rate,
                sample_format,
                channel_aligners: aligners,
                _stream: stream,
            });
        }

        let mut output_device_names = Vec::with_capacity(output_devices.len());
        for descriptor in output_devices.iter() {
            let requested_name = descriptor.device_name.as_str();
            let device = select_device(
                host.output_devices(),
                requested_name,
                "Output",
            )?;
            let device_name = device
                .name()
                .unwrap_or_else(|_| requested_name.to_string());
            validate_device_stream_config(
                &device,
                &device_name,
                descriptor.config.channels,
                descriptor.config.sample_rate.0,
                descriptor.sample_format,
                "output",
            )?;
            output_device_names.push(device_name);
        }

        Ok(Self {
            target_sample_rate,
            input_channels,
            input_streams,
            output_device_names,
        })
    }
}

fn build_input_alignment_stream(
    device: &Device,
    device_name: &str,
    config: &StreamConfig,
    format: SampleFormat,
    channel_aligners: Vec<Arc<Mutex<StreamAligner>>>,
) -> Result<Stream, cpal::BuildStreamError> {
    let label = device_name.to_string();
    match format {
        SampleFormat::I16 => build_input_alignment_stream_typed::<i16>(
            device,
            label.clone(),
            config,
            channel_aligners.clone(),
        ),
        SampleFormat::F32 => build_input_alignment_stream_typed::<f32>(
            device,
            label.clone(),
            config,
            channel_aligners.clone(),
        ),
        SampleFormat::U16 => build_input_alignment_stream_typed::<u16>(
            device,
            label,
            config,
            channel_aligners,
        ),
        other => {
            eprintln!(
                "Input device '{device_name}' uses unsupported sample format {other:?}; cannot build StreamAligner."
            );
            Err(cpal::BuildStreamError::StreamConfigNotSupported)
        }
    }
}

fn build_input_alignment_stream_typed<T>(
    device: &Device,
    device_label: String,
    config: &StreamConfig,
    channel_aligners: Vec<Arc<Mutex<StreamAligner>>>,
) -> Result<Stream, cpal::BuildStreamError>
where
    T: Sample + SizedSample,
    f32: FromSample<T>,
{
    let channels = config.channels as usize;
    if channels == 0 {
        eprintln!(
            "Input device '{device_label}' reported zero channels; skipping stream creation."
        );
        return Err(cpal::BuildStreamError::StreamConfigNotSupported);
    }

    if channel_aligners.len() != channels {
        eprintln!(
            "Warning: device '{device_label}' channel count ({channels}) differs from aligner count ({}); audio may be dropped.",
            channel_aligners.len()
        );
    }

    let buffer_count = channels.max(1);
    let per_channel_capacity = (config.sample_rate.0 as usize)
        .saturating_div(20) // ~50 ms of audio per channel
        .max(1024);
    let mut channel_buffers = (0..buffer_count)
        .map(|_| Vec::<f32>::with_capacity(per_channel_capacity))
        .collect::<Vec<_>>();

    device.build_input_stream(
        config,
        move |data: &[T], _info: &InputCallbackInfo| {
            if data.is_empty() {
                return;
            }
            for buffer in channel_buffers.iter_mut() {
                buffer.clear();
            }
            for frame in data.chunks(channels) {
                for (channel_idx, sample) in frame.iter().enumerate() {
                    if let Some(buffer) = channel_buffers.get_mut(channel_idx) {
                        buffer.push(f32::from_sample(*sample));
                    }
                }
            }
            let limit = channel_buffers.len().min(channels);
            for channel_idx in 0..limit {
                let buffer = &mut channel_buffers[channel_idx];
                if buffer.is_empty() {
                    continue;
                }
                match channel_aligners.get(channel_idx) {
                    Some(handle) => match handle.lock() {
                        Ok(mut aligner) => aligner.process_chunk(buffer.as_slice()),
                        Err(err) => eprintln!(
                            "Input stream '{device_label}' channel {channel_idx}: failed to lock StreamAligner: {err}"
                        ),
                    },
                    None => eprintln!(
                        "Input stream '{device_label}' channel {channel_idx}: missing StreamAligner; dropping audio."
                    ),
                }
                buffer.clear();
            }
        },
        move |err| eprintln!("Input stream '{device_label}' error: {err}"),
        None,
    )
}

fn check_format_is_valid(
    device: &Device,
    device_name: &str,
    channels: u16,
    sample_rate: u32,
    format: SampleFormat,
    direction: &'static str,
) -> Result<(), Box<dyn Error>> {
    let configs = match direction {
        "input" => gather_supported_input_configs(device)
            .map_err(|err| format!("Unable to enumerate input configs for '{device_name}': {err}"))?,
        "output" => gather_supported_output_configs(device)
            .map_err(|err| format!("Unable to enumerate output configs for '{device_name}': {err}"))?,
        other => {
            return Err(format!("Unknown device direction '{other}' when validating {device_name}.").into());
        }
    };

    if configs.is_empty() {
        return Err(format!(
            "{} device '{}' reported no supported stream configurations to validate.",
            direction, device_name
        )
        .into());
    }

    if supports_rate(&configs, channels, sample_rate, format) {
        Ok(())
    } else {
        Err(format!(
            "{} device '{}' does not support {} channel(s), {:?} at {} Hz.",
            direction, device_name, channels, format, sample_rate
        )
        .into())
    }
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
*/
