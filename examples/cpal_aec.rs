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
    f32::consts::PI,
    ffi::c_void,
    sync::{Arc, Mutex},
    time::Duration,
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Device, Sample, SampleFormat, Stream, StreamConfig,
};
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

    let phase_increment = 440.0f32 * 2.0 * PI / output_rate as f32;
    let output_stream = build_output_stream(
        &output_device,
        &output_stream_config,
        Arc::clone(&shared),
        phase_increment,
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

    // Streams stop and the Speex state is dropped when leaving scope.
    Ok(())
}

struct SharedCanceller {
    aec: EchoCanceller,
    channels: usize,
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
            channels,
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
        if pending.is_empty() {
            return;
        }

        let (in_rate, out_rate) = resampler.get_rate();
        let channels = resampler.channels();

        loop {
            if pending.is_empty() {
                break;
            }

            let available = pending.len();
            let mut expected =
                ((available as u64 * out_rate as u64) / in_rate as u64) as usize + channels;
            expected = expected.max(channels);
            scratch.resize(expected, 0);

            match resampler.process_interleaved_i16(pending.as_slice(), scratch.as_mut_slice()) {
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

    fn process_capture_generic<T: Sample>(&mut self, samples: &[T]) {
        self.input_convert.resize(samples.len(), 0);
        for (dst, src) in self.input_convert.iter_mut().zip(samples.iter()) {
            *dst = src.to_i16();
        }
        self.process_capture(&self.input_convert);
    }
}

fn build_output_stream(
    device: &Device,
    config: &StreamConfig,
    shared: Arc<Mutex<SharedCanceller>>,
    phase_increment: f32,
    channels: usize,
    format: SampleFormat,
) -> Result<Stream, cpal::BuildStreamError> {
    match format {
        SampleFormat::I16 => build_output_stream_for::<i16>(
            device,
            config,
            shared,
            phase_increment,
            channels,
        ),
        SampleFormat::F32 => build_output_stream_for::<f32>(
            device,
            config,
            shared,
            phase_increment,
            channels,
        ),
        SampleFormat::U16 => build_output_stream_for::<u16>(
            device,
            config,
            shared,
            phase_increment,
            channels,
        ),
        format => {
            eprintln!("Unsupported output sample format: {format:?}");
            Err(cpal::BuildStreamError::StreamConfigNotSupported)
        }
    }
}

fn build_output_stream_for<T>(
    device: &Device,
    config: &StreamConfig,
    shared: Arc<Mutex<SharedCanceller>>,
    phase_increment: f32,
    channels: usize,
) -> Result<Stream, cpal::BuildStreamError>
where
    T: Sample,
{
    let mut phase = 0f32;
    let mut far_frame = vec![0i16; channels];
    device.build_output_stream(
        config,
        move |data: &mut [T], _| {
            let mut state = shared.lock().unwrap();
            for frame in data.chunks_mut(channels) {
                phase += phase_increment;
                if phase > 2.0 * PI {
                    phase -= 2.0 * PI;
                }
                let sample_amp = (phase.sin() * 0.2).clamp(-1.0, 1.0);
                let sample_i16 = (sample_amp * i16::MAX as f32) as i16;
                far_frame.fill(sample_i16);
                for out_sample in frame.iter_mut() {
                    *out_sample = Sample::from::<f32>(&sample_amp);
                }
                state.push_far_end(&far_frame);
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
    format: SampleFormat,
) -> Result<Stream, cpal::BuildStreamError> {
    match format {
        SampleFormat::I16 => build_input_stream_for::<i16>(device, config, shared),
        SampleFormat::F32 => build_input_stream_for::<f32>(device, config, shared),
        SampleFormat::U16 => build_input_stream_for::<u16>(device, config, shared),
        format => {
            eprintln!("Unsupported input sample format: {format:?}");
            Err(cpal::BuildStreamError::StreamConfigNotSupported)
        }
    }
}

fn build_input_stream_for<T>(
    device: &Device,
    config: &StreamConfig,
    shared: Arc<Mutex<SharedCanceller>>,
) -> Result<Stream, cpal::BuildStreamError>
where
    T: Sample,
{
    device.build_input_stream(
        config,
        move |data: &[T], _| {
            let mut state = shared.lock().unwrap();
            state.process_capture_generic(data);
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
        Err(format!(
            "{kind} device matching '{target}' not found.\nAvailable:\n  {}",
            available.join("\n  ")
        )
        .into())
    }
}
