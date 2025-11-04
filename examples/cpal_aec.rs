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

use std::{
    collections::VecDeque,
    env,
    error::Error,
    ffi::c_void,
    f32::consts::PI,
    sync::{Arc, Mutex},
    time::Duration,
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Device, SampleFormat, StreamConfig,
};
use speex_rust_aec::{
    speex_echo_cancellation, speex_echo_ctl, EchoCanceller, SPEEX_ECHO_SET_SAMPLING_RATE,
};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    let host = cpal::default_host();

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

    if input_config.sample_format() != SampleFormat::I16
        || output_config.sample_format() != SampleFormat::I16
    {
        eprintln!("This example expects 16-bit audio devices.");
        return Ok(());
    }

    // Use the input configuration for both streams. You may need to adjust
    // this or pick a compatible configuration for your hardware.
    let stream_config: StreamConfig = input_config.config();
    let sample_rate = stream_config.sample_rate.0;
    let channels = stream_config.channels as usize;
    let frame_size = (sample_rate / 100) as usize; // 10 ms frames
    let filter_length = frame_size * 20; // 200 ms echo tail

    let mut canceller = EchoCanceller::new_multichannel(
        frame_size,
        filter_length,
        channels,
        channels,
    )
    .ok_or("Failed to allocate echo canceller state")?;

    unsafe {
        let mut rate = sample_rate as i32;
        speex_echo_ctl(
            canceller.as_ptr(),
            SPEEX_ECHO_SET_SAMPLING_RATE,
            &mut rate as *mut _ as *mut c_void,
        );
    }

    let shared = Arc::new(Mutex::new(SharedCanceller::new(
        canceller, channels, frame_size,
    )));

    let output_shared = Arc::clone(&shared);
    let mut phase = 0f32;
    let output_stream = output_device.build_output_stream(
        &stream_config,
        move |data: &mut [i16], _| {
            let mut state = output_shared.lock().unwrap();
            for frame in data.chunks_mut(state.channels) {
                phase += 440.0f32 * 2.0 * PI / sample_rate as f32;
                let sample = (phase.sin() * 0.2 * i16::MAX as f32) as i16;
                for sample_out in frame.iter_mut() {
                    *sample_out = sample;
                }
                state.push_far_end(frame);
            }
        },
        move |err| eprintln!("Output stream error: {err}"),
        None,
    )?;

    let input_shared = Arc::clone(&shared);
    let input_stream = input_device.build_input_stream(
        &stream_config,
        move |data: &[i16], _| {
            let mut state = input_shared.lock().unwrap();
            state.process_capture(data);
        },
        move |err| eprintln!("Input stream error: {err}"),
        None,
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
    far_queue: VecDeque<i16>,
    mic_queue: VecDeque<i16>,
    mic_frame: Vec<i16>,
    far_frame: Vec<i16>,
    out_frame: Vec<i16>,
    processed_frames: usize,
}

impl SharedCanceller {
    fn new(aec: EchoCanceller, channels: usize, frame_size: usize) -> Self {
        let frame_samples = frame_size * channels;
        Self {
            aec,
            channels,
            frame_samples,
            far_queue: VecDeque::with_capacity(frame_samples * 8),
            mic_queue: VecDeque::with_capacity(frame_samples * 8),
            mic_frame: vec![0; frame_samples],
            far_frame: vec![0; frame_samples],
            out_frame: vec![0; frame_samples],
            processed_frames: 0,
        }
    }

    fn push_far_end(&mut self, samples: &[i16]) {
        self.far_queue.extend(samples.iter().copied());
        let max_len = self.frame_samples * 16;
        while self.far_queue.len() > max_len {
            self.far_queue.pop_front();
        }
    }

    fn process_capture(&mut self, samples: &[i16]) {
        self.mic_queue.extend(samples.iter().copied());
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
        Err(format!(
            "{kind} device matching '{target}' not found (no devices available)"
        )
        .into())
    } else {
        Err(format!(
            "{kind} device matching '{target}' not found.\nAvailable:\n  {}",
            available.join("\n  ")
        )
        .into())
    }
}
