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
    thread,
    collections::{HashMap},
    error::Error,
    mem::{MaybeUninit},
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{self, TryRecvError, RecvError},
        Arc,
    },
};

use std::backtrace::Backtrace;


use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Device, FromSample, Host, InputCallbackInfo, Sample, SampleFormat, SampleRate, SizedSample,
    Stream, SupportedStreamConfig,
};
use ringbuf::{
    traits::{Consumer, Producer, RingBuffer, Split},
    HeapCons, HeapProd, HeapRb, LocalRb,
};
use ringbuf::storage::Heap;
use speex_rust_aec::{
    speex_echo_cancellation, EchoCanceller, Resampler,
};
use std::f32::consts::PI;
use rustfft::{FftPlanner};
use rustfft::num_complex::Complex32;

use std::time::{UNIX_EPOCH, SystemTime};
use hound::{WavReader,SampleFormat as HoundSampleFormat, WavSpec, WavWriter};

#[inline]
unsafe fn assume_init_slice_mut<T>(slice: &mut [MaybeUninit<T>]) -> &mut [T] {
    &mut *(slice as *mut [MaybeUninit<T>] as *mut [T])
}

fn input_to_output_frames(input_frames: u128, in_rate: u32, out_rate: u32) -> u128 {
    // u128 to avoid overflow
    (input_frames * (out_rate as u128)) / (in_rate as u128)
}

fn micros_to_frames(microseconds: u128, sample_rate: u128) -> u128 {
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

/// Generate a short, Hann-windowed multi-tone probe at 16 kHz sample rate.
/// Pass distinct frequency sets per device to keep probes identifiable.
fn generate_probe_tone_16k_with_freqs(duration_ms: f32, freqs: &[f32]) -> Vec<f32> {
    if freqs.is_empty() {
        return Vec::new();
    }
    let sample_rate = 16_000.0f32;
    let samples = (duration_ms * sample_rate / 1000.0).ceil().max(1.0) as usize;
    let mut buf = Vec::with_capacity(samples);
    for n in 0..samples {
        let t = n as f32 / sample_rate;
        let w = 0.5
            * (1.0
                - (2.0 * PI * n as f32 / (samples.saturating_sub(1).max(1) as f32)).cos());
        let sum = freqs
            .iter()
            .fold(0.0f32, |acc, f| acc + (2.0 * PI * f * t).sin());
        buf.push(w * (sum / freqs.len() as f32) * 0.25);
    }
    buf
}

/// Convenience: derive a distinct probe for each device index by nudging the frequencies.
/// Keeps tones in the 1–3.5 kHz band (works at typical 16–48 kHz sample rates).
fn generate_probe_tone_for_device(device_index: usize, duration_ms: f32, sample_rate: f32) -> Vec<f32> {
    let base = [1_000.0f32, 1_800.0f32, 2_600.0f32];
    // Small offset per device to make correlation peaks separable.
    let offset = (device_index as f32 % 5.0) * 120.0;
    let freqs: Vec<f32> = base.iter().map(|f| f + offset).collect();
    generate_probe_tone_with_freqs(duration_ms, sample_rate, &freqs)
}

/// Generate a short, Hann-windowed multi-tone probe at arbitrary sample rate.
/// Pass distinct frequency sets per device to keep probes identifiable.
fn generate_probe_tone_with_freqs(duration_ms: f32, sample_rate: f32, freqs: &[f32]) -> Vec<f32> {
    if freqs.is_empty() || sample_rate <= 0.0 {
        return Vec::new();
    }
    let samples = (duration_ms * sample_rate / 1000.0).ceil().max(1.0) as usize;
    let mut buf = Vec::with_capacity(samples);
    for n in 0..samples {
        let t = n as f32 / sample_rate;
        let w = 0.5
            * (1.0
                - (2.0 * PI * n as f32 / (samples.saturating_sub(1).max(1) as f32)).cos());
        let sum = freqs
            .iter()
            .fold(0.0f32, |acc, f| acc + (2.0 * PI * f * t).sin());
        buf.push(w * (sum / freqs.len() as f32) * 0.25);
    }
    buf
}

/// Probe detection using GCC-PHAT to estimate lag between the captured stream and each probe.
/// Returns best (device_index, start_sample, score) per device that produced a valid match.
fn detect_probe_tones(input_mono: &[f32], num_devices: usize, duration_ms: f32, sample_rate: f32) -> Vec<(usize, i64, f32)> {
    let mut results = Vec::new();
    if num_devices == 0 || input_mono.is_empty() {
        return results;
    }

    for device in 0..num_devices {
        let probe = generate_probe_tone_for_device(device, duration_ms, sample_rate);
        if probe.is_empty() || probe.len() > input_mono.len() {
            continue;
        }
        // Pad probe to match captured length for GCC-PHAT.
        let mut probe_padded = vec![0.0f32; input_mono.len()];
        probe_padded[..probe.len()].copy_from_slice(&probe);
        let margin = input_mono.len().saturating_sub(1);
        if let Some((lag, score)) = gcc_phat_delay(&input_mono, &probe_padded, margin) {
            // positive lag means probe leads capture; convert to start index in capture
            println!("Got lag {lag} {score}");
            results.push((device, -lag, score));
        }
    }
    results
}

/// Cross-correlate `input` with `probe` using FFT convolution.
/// Returns (start_index, normalized_score) for the best match.
fn detect_probe_fft(input: &[f32], probe: &[f32]) -> Option<(usize, f32)> {
    if probe.is_empty() || probe.len() > input.len() {
        return None;
    }

    let probe_energy = probe.iter().map(|v| v * v).sum::<f32>();
    if probe_energy == 0.0 {
        return None;
    }

    // Build reversed probe for correlation.
    let mut b: Vec<Complex32> = probe
        .iter()
        .rev()
        .map(|&v| Complex32::new(v, 0.0))
        .collect();
    let mut a: Vec<Complex32> = input.iter().map(|&v| Complex32::new(v, 0.0)).collect();

    let conv_len = a.len() + b.len() - 1;
    let n_fft = conv_len.next_power_of_two();
    a.resize(n_fft, Complex32::ZERO);
    b.resize(n_fft, Complex32::ZERO);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let ifft = planner.plan_fft_inverse(n_fft);

    fft.process(&mut a);
    fft.process(&mut b);
    for (ai, bi) in a.iter_mut().zip(b.iter()) {
        *ai *= *bi;
    }
    ifft.process(&mut a);

    let scale = 1.0 / (n_fft as f32);

    // Precompute input window energies for normalization.
    let mut prefix_energy = Vec::with_capacity(input.len() + 1);
    prefix_energy.push(0.0f32);
    for &v in input {
        let last = *prefix_energy.last().unwrap();
        prefix_energy.push(last + v * v);
    }

    let mut best: Option<(usize, f32)> = None;
    let valid_starts = input.len() - probe.len() + 1;
    for lag in 0..valid_starts {
        // correlation index aligned to lag
        let idx = lag + probe.len() - 1;
        let corr = a[idx].re * scale;
        let win_energy = prefix_energy[lag + probe.len()] - prefix_energy[lag];
        let denom = (win_energy * probe_energy).sqrt();
        let score = if denom > 0.0 { corr / denom } else { 0.0 };
        if best.map_or(true, |(_, s)| score > s) {
            best = Some((lag, score));
        }
    }
    best
}

/// GCC-PHAT delay estimator between two real signals.
/// Returns (lag_in_samples, score), where positive lag means `sigb` leads `siga`.
fn gcc_phat_delay(siga: &[f32], sigb: &[f32], margin: usize) -> Option<(i64, f32)> {
    let n = siga.len().min(sigb.len());
    if n == 0 {
        return None;
    }
    let n_fft = n.next_power_of_two().max(1);

    let mut a: Vec<Complex32> = vec![Complex32::ZERO; n_fft];
    let mut b: Vec<Complex32> = vec![Complex32::ZERO; n_fft];
    for (i, &v) in siga.iter().take(n).enumerate() {
        a[i].re = v;
    }
    for (i, &v) in sigb.iter().take(n).enumerate() {
        b[i].re = v;
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft_fwd = planner.plan_fft_forward(n_fft);
    let fft_inv = planner.plan_fft_inverse(n_fft);

    fft_fwd.process(&mut a);
    fft_fwd.process(&mut b);

    for (ai, bi) in a.iter_mut().zip(b.iter()) {
        let mut v = *bi * ai.conj();
        let mag = v.norm() + 1e-12;
        v /= mag;
        *ai = v;
    }

    fft_inv.process(&mut a);

    let scale = 1.0 / (n_fft as f32);
    let mut corr: Vec<f32> = a.iter().map(|c| c.re * scale).collect();

    // fftshift so zero-lag is centered
    let mut shifted = vec![0.0f32; n_fft];
    let mid = n_fft / 2;
    for i in 0..n_fft {
        shifted[i] = corr[(i + mid) % n_fft];
    }

    let center = mid;
    let max_margin = center.min(n_fft - center - 1);
    let m = margin.min(max_margin);
    let start = center.saturating_sub(m);
    let end = (center + m + 1).min(n_fft);

    let (rel_idx, &best_val) = shifted[start..end]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))?;
    let lag = (start + rel_idx) as i64 - center as i64;
    Some((lag, best_val))
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
                eprintln!("Warning: Producer cannot keep up, increase buffer size or decrease latency");
                let bt = Backtrace::capture();
                println!("{bt}");
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
    channels: usize,
    consumer: BufferedCircularConsumer<f32>,
    resampled_producer: BufferedCircularProducer<f32>,
    input_sample_rate: u32,
    output_sample_rate: u32,
    total_input_frames_remaining: u128,
    resampler: Resampler
}

impl ResampledBufferedCircularProducer {
    fn new(
        channels: usize,
        input_sample_rate: u32,
        output_sample_rate : u32,
        resampler_quality: i32,
        consumer: BufferedCircularConsumer<f32>,
        resampled_producer: BufferedCircularProducer<f32>) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            channels: channels,
            consumer: consumer,
            resampled_producer: resampled_producer,
            total_input_frames_remaining: 0,
            input_sample_rate: input_sample_rate,
            output_sample_rate: output_sample_rate,
            resampler: Resampler::new(
                channels as u32, // channels, we have one of these StreamAligner each channel
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
}

fn round_to_channels(frames: u32, channels: usize) -> u32 {
    (frames / (channels as u32)) * (channels as u32)
}

impl ResampledBufferedCircularProducer {

    fn resample(&mut self, num_available_frames: u32) -> Result<(usize, usize), Box<dyn std::error::Error>> {
        if num_available_frames == 0 {
            return Ok((0,0));
        }

        let available_frames = self.available_to_resample() / self.channels;
        // there might be some leftover from last call, so use state
        self.total_input_frames_remaining = (self.total_input_frames_remaining + num_available_frames as u128).min(available_frames as u128);
        
        // read in multiples of channels
        let input_buf = self.consumer.get_chunk_to_read((self.total_input_frames_remaining * (self.channels as u128)) as usize);
        let target_output_samples_count = input_to_output_frames(self.total_input_frames_remaining, self.input_sample_rate, self.output_sample_rate)*(self.channels as u128); // add a few extra for rounding
        let (need_to_write_outputs, output_buf) = self.resampled_producer.get_chunk_to_write(target_output_samples_count as usize);
        let (consumed, produced) = self.resampler.process_interleaved_f32(input_buf, output_buf)?;
        // it may return less consumed and produced than the sizes of stuff we gave it
        // so use actual processed sizes here instead of our lengths from above
        // (worst case this is like 0.6 ms or so, so it's okay to have them slightly delayed like this)
        self.consumer.finish_read(consumed);
        self.resampled_producer.finish_write(need_to_write_outputs, produced);

        self.total_input_frames_remaining -= (consumed / self.channels) as u128;
        Ok((consumed, produced))
    }
}

enum AudioBufferMetadata {
    Arrive(u64, u128, u128, bool),
    Teardown(),
}


struct StreamAlignerProducer {
    channels: usize,
    input_sample_rate: u32,
    output_sample_rate: u32,
    input_audio_buffer_producer: HeapProd<f32>,
    input_audio_buffer_metadata_producer: mpsc::Sender<AudioBufferMetadata>,
    chunk_sizes: LocalRb<Heap<usize>>,
    system_time_micros_when_chunk_ended: LocalRb<Heap<u128>>,
    num_calibration_packets: u32,
    num_packets_recieved: u64,
    num_emitted_frames: u128,
    start_time_micros: Option<u128>,
}

impl StreamAlignerProducer {
    fn new(channels: usize, input_sample_rate: u32, output_sample_rate: u32, history_len: usize, num_calibration_packets: u32, input_audio_buffer_producer: HeapProd<f32>, input_audio_buffer_metadata_producer: mpsc::Sender<AudioBufferMetadata>) -> Result<Self, Box<dyn Error>>  {
        Ok(Self {
            channels: channels,
            input_sample_rate: input_sample_rate,
            output_sample_rate: output_sample_rate,
            input_audio_buffer_producer: input_audio_buffer_producer,
            input_audio_buffer_metadata_producer: input_audio_buffer_metadata_producer,
            // alignment data, these are used to adjust resample rate so output stays aligned with true timings (according to sytem clock)
            chunk_sizes: LocalRb::<Heap<usize>>::new(history_len),
            system_time_micros_when_chunk_ended: LocalRb::<Heap<u128>>::new(history_len),
            num_calibration_packets: num_calibration_packets,
            num_packets_recieved: 0,
            num_emitted_frames: 0,
            start_time_micros: None,
        })
    }

    fn estimate_micros_when_most_recent_ended(&self) -> u128 {
        // Take minimum over estimates for all previous recieved
        // Some may be delayed due to cpu being busy, but none can ever arrive too early
        // so this should be a decent estimate
        // it does not account for hardware latency, but we cannot account for that without manual calibration
        // (btw, CPAL timestamps do not work because they may be different for different devices)
        // (wheras this synchronizes us to global system time)
        let mut best_estimate_of_when_most_recent_ended = if let Some(most_recent_time) = self.system_time_micros_when_chunk_ended.last() {
            *most_recent_time
        }
        else {
            u128::MAX
        };
        let mut frames_until_most_recent = 0 as u128;


        // iterate from most recent backwards (that's what .rev() does)
        let mut chunk_iter = self.chunk_sizes.iter().rev();
        let mut time_iter = self.system_time_micros_when_chunk_ended.iter().rev();
        while let (Some(chunk_size), Some(micros_when_chunk_ended)) = (chunk_iter.next(), time_iter.next()) {
            let micros_until_most_recent_ended = frames_to_micros(frames_until_most_recent as u128, self.input_sample_rate as u128);
            let estimate_of_micros_most_recent_ended = *micros_when_chunk_ended + micros_until_most_recent_ended;
            best_estimate_of_when_most_recent_ended = (estimate_of_micros_most_recent_ended).min(best_estimate_of_when_most_recent_ended);
            // timestamps are at end, not at start, so only increment this after
            frames_until_most_recent += *chunk_size as u128;
        }
        best_estimate_of_when_most_recent_ended
    }

    fn process_chunk(&mut self, chunk: &[f32]) -> Result<(), Box<dyn Error>> {
        let micros_when_chunk_received = SystemTime::now().duration_since(UNIX_EPOCH).expect("clock went backwards").as_micros();

       
        let appended_count = self.input_audio_buffer_producer.push_slice(chunk);
        if appended_count < chunk.len() { // todo: auto resize
            eprintln!("Error: cannot keep up with audio, buffer is full, try increasing audio_buffer_seconds")
        }
        if appended_count > 0 {
            let appended_frames = appended_count / self.channels;
            // delibrately overwrite once we pass history len, we keep a rolling buffer of last 100 or so
            self.chunk_sizes.push_overwrite(appended_frames);
            self.system_time_micros_when_chunk_ended.push_overwrite(micros_when_chunk_received);

            // use our estimate to suggest how many frames we should have emitted
            // this is used to dynamically adjust sample rate until we actually emit that many frames
            // that ensures that we stay synchronized to the system clock and do not drift
            let micros_when_chunk_ended = self.estimate_micros_when_most_recent_ended();

            self.num_emitted_frames += appended_frames as u128;

            let (target_emitted_frames, calibrated) = if self.num_packets_recieved < self.num_calibration_packets as u64 {
                // until we've recieved enough calibration packets, we don't have good enough time estimate
                // thus, simply request num_emitted_frames emitted
                // this avoids large amounts of distortion if we get an initial burst of packets on device init
                (self.num_emitted_frames, false)
            } else {
                // calibration finished, setup start_time_micros
                if self.num_packets_recieved == self.num_calibration_packets as u64 {
                    // now we can actually make a good estimate of our current time,
                    // which allows us to make a good estimate of start time (just convert number packets emitted into an offset)
                    // this isn't ideal when calibration involved a dropped packet
                    // but is about as good as we can do
                    self.start_time_micros = Some(micros_when_chunk_ended - frames_to_micros(self.num_emitted_frames, self.input_sample_rate as u128));
                }

                if let Some(start_time_micros_value) = self.start_time_micros {
                    // look at actual elapsed time, and use that to say how many frames we would have preferred to emitted
                    // this can be used later to adjust sample rate slightly to keep us in line with system time
                    let elapsed_micros = micros_when_chunk_ended - start_time_micros_value;
                    (micros_to_frames(elapsed_micros, self.input_sample_rate as u128), true)
                }
                else {
                    (self.num_emitted_frames, false)
                }
            };

            // increment afterwards incase num_calibration_packets = 0
            self.num_packets_recieved += 1;

            let metadata = AudioBufferMetadata::Arrive(
                // num available frames
                appended_frames as u64,
                // estimated timestamp after this sample
                micros_when_chunk_ended,
                // target emitted frames
                target_emitted_frames,
                // calibrated
                calibrated
            );
            self.input_audio_buffer_metadata_producer.send(metadata)?;
        }
        Ok(())
    }

}

#[derive(Clone)]
enum ResamplingMetadata {
    Arrive(usize, u128, u128, bool),
}

struct StreamAlignerResampler {
    channels: usize,
    input_sample_rate: u32,
    output_sample_rate: u32,
    dynamic_output_sample_rate: u32,
    input_audio_buffer_consumer: ResampledBufferedCircularProducer,
    input_audio_buffer_metadata_consumer: mpsc::Receiver<AudioBufferMetadata>,
    total_emitted_frames: u128,
    total_received_frames: u128,
    total_processed_input_frames: u128,
    finished_resampling_producer: mpsc::Sender<ResamplingMetadata>
}

impl StreamAlignerResampler {
    // Takes input audio and resamples it to the target rate
    // May slightly stretch or squeeze the audio (via resampling)
    // to ensure the outputs stay aligned with system clock
    fn new(
        channels: usize,
        input_sample_rate: u32,
        output_sample_rate: u32,
        resampler_quality: i32,
        input_audio_buffer_consumer: HeapCons<f32>,
        input_audio_buffer_metadata_consumer: mpsc::Receiver<AudioBufferMetadata>,
        output_audio_buffer_producer: HeapProd<f32>,
        finished_resampling_producer: mpsc::Sender<ResamplingMetadata>
    ) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            channels: channels,
            input_sample_rate: input_sample_rate,
            output_sample_rate: output_sample_rate,
            dynamic_output_sample_rate: output_sample_rate,
            // we need buffered because this interfaces with speex which expects continuous buffers
            input_audio_buffer_consumer: ResampledBufferedCircularProducer::new(
                channels,
                input_sample_rate,
                output_sample_rate,
                resampler_quality,
                BufferedCircularConsumer::<f32>::new(input_audio_buffer_consumer),
                BufferedCircularProducer::<f32>::new(output_audio_buffer_producer)
            )?,
            input_audio_buffer_metadata_consumer: input_audio_buffer_metadata_consumer,
            // alignment data, these are used to adjust resample rate so output stays aligned with true timings (according to sytem clock)
            total_emitted_frames: 0,
            total_received_frames: 0,
            total_processed_input_frames: 0,
            finished_resampling_producer: finished_resampling_producer,
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

    fn handle_metadata(&mut self, num_available_frames : u64, target_emitted_input_frames : u128, calibrated: bool) -> Result<(usize, usize), Box<dyn std::error::Error>> {
        let estimated_emitted_frames = input_to_output_frames(num_available_frames as u128, self.input_sample_rate, self.dynamic_output_sample_rate);
        let updated_total_frames_emitted = self.total_emitted_frames + estimated_emitted_frames;
        let target_emitted_output_frames = input_to_output_frames(target_emitted_input_frames, self.input_sample_rate, self.output_sample_rate);
        // dynamic adjustment to synchronize input devices to global clock:
        // don't do dynamic adjustment until after calibration, bc it's not gonna drift too much over the course of just a few seconds of calibration data
        // and that simplifies logic/prevents accumulated error during calibration
        // not enough frames, we need to increase dynamic sample rate (to get more samples)
        if updated_total_frames_emitted < target_emitted_output_frames && calibrated {
            self.increase_dynamic_sample_rate()?;
            // println!("Increase to {0} {updated_total_frames_emitted} {target_emitted_output_frames}", self.dynamic_output_sample_rate)
        }
        // too many frames, we need to decrease dynamic sample rate (to get less samples)
        else if updated_total_frames_emitted > target_emitted_output_frames && calibrated {
            self.decrease_dynamic_sample_rate()?;
            // println!("Decrease to {0} {updated_total_frames_emitted} {target_emitted_output_frames}", self.dynamic_output_sample_rate)
        }

        //// do resampling ////
        let (consumed, produced) = self.input_audio_buffer_consumer.resample(num_available_frames as u32)?;

        // the main downside of this is that it'll be persistently behind by 0.6ms or so (the resample frame size), but we'll quickly adjust for that so this shouldn't be a major issue
        // todo: think about how to fix this better (maybe current solution is as good as we can do, and it should average out to correct since past ones accumulated will result in more for this one, still, it's likely to stay behind by this amount)
        self.total_emitted_frames += (produced / self.channels) as u128;

        Ok((consumed, produced))
    }

    fn resample(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        // process all recieved audio chunks
        match self.input_audio_buffer_metadata_consumer.recv() {
            Ok(msg) => match msg {
                AudioBufferMetadata::Arrive(num_available_frames, system_micros_after_packet_finishes, target_emitted_frames, calibrated) => {
                    let num_leftovers_from_prev = self.total_received_frames - self.total_processed_input_frames;
                    self.total_received_frames += num_available_frames as u128;
                    // if it hypothetically consumed all frames every time,
                    // then we would know that current time in system_micros_after_packet_finishes
                    // however, there are a few things that happen:
                    // 1. There are some "leftover" samples from previously
                    // 2. There are some "ignored" samples from cur (that are later leftover)
                    // this logic sets the timestamp to be correct relative to last resampled emitted
                    // which is slightly distinct from system_micros_after_packet_finishes
                    // because resampling may operate at some latency
                    let (consumed, produced) = self.handle_metadata(num_available_frames, target_emitted_frames, calibrated)?;
                    let consumed_frames = consumed / self.channels;
                    self.total_processed_input_frames += consumed_frames as u128;
                    let micros_earlier = if consumed_frames as u128 > num_leftovers_from_prev {
                        let num_of_ours_consumed = (consumed_frames as i128) - (num_leftovers_from_prev as i128);
                        let num_of_ours_leftover = (num_available_frames as i128) - (num_of_ours_consumed as i128);
                        frames_to_micros(num_of_ours_leftover as u128, self.input_sample_rate as u128) as i128
                    } else {
                        // none of ours was consumed, skip back even further
                        let additional_frames_back = (num_leftovers_from_prev as u128) - (consumed_frames as u128);
                        frames_to_micros(num_available_frames as u128 + additional_frames_back, self.input_sample_rate as u128) as i128
                    };
                    // will always be positive because it's relative to 1970
                    let system_micros_after_resampled_packet_finishes = (system_micros_after_packet_finishes as i128) - micros_earlier;
                    let system_micros_at_start_of_packet = (system_micros_after_resampled_packet_finishes as u128) - frames_to_micros(consumed_frames as u128, self.input_sample_rate as u128);
                    self.finished_resampling_producer.send(ResamplingMetadata::Arrive(produced / self.channels, system_micros_at_start_of_packet, system_micros_after_resampled_packet_finishes as u128, calibrated))?;
                    Ok(true)
                },
                AudioBufferMetadata::Teardown() => {
                    Ok(false)
                }
            },
            Err(RecvError) => Err("input metadata channel disconnected".into())   // sender dropped; bail out or log
        }
    }
}

struct StreamAlignerConsumer {
    channels: usize,
    sample_rate: u32,
    final_audio_buffer_consumer: BufferedCircularConsumer<f32>,
    thread_message_sender: mpsc::Sender<AudioBufferMetadata>,
    finished_message_reciever: mpsc::Receiver<ResamplingMetadata>,
    initial_metadata: Vec<ResamplingMetadata>,
    frames_recieved: u128,
    calibrated: bool,
}

impl StreamAlignerConsumer {
    fn new(channels: usize, sample_rate: u32, final_audio_buffer_consumer: BufferedCircularConsumer<f32>, thread_message_sender: mpsc::Sender<AudioBufferMetadata>, finished_message_reciever: mpsc::Receiver<ResamplingMetadata>) -> Self {
        Self {
            channels: channels,
            sample_rate: sample_rate,
            final_audio_buffer_consumer: final_audio_buffer_consumer,
            thread_message_sender: thread_message_sender,
            finished_message_reciever: finished_message_reciever,
            initial_metadata: Vec::new(),
            frames_recieved: 0,
            calibrated: false,
        }
    }

    // used to poll for when an input stream is actually ready to output data
    // we allow some initial calibration time to synchronize the clocks
    // (it needs some extra time because packets can be delayed sometimes 
    // so waiting and min over a history lets us get better estimate)
    fn is_ready_to_read(&mut self, micros_packet_finished: u128, size_in_frames: usize) -> bool {
        // non blocking cause maybe it's just not ready (initialized) yet
        loop {
            match self.finished_message_reciever.try_recv() {
                Ok(msg) => {
                    match msg {
                        ResamplingMetadata::Arrive(frames_recieved, _system_micros_at_start_of_packet, _system_micros_after_packet_finishes, calibrated) => {
                            self.calibrated = calibrated;
                            // this is fine to just accumulate since we don't add any more after we are done with calibration
                            self.initial_metadata.push(msg.clone());
                            self.frames_recieved += frames_recieved as u128;
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
            let num_frames_that_are_behind_current_packet = self.num_frames_that_are_behind_current_packet(micros_packet_finished, size_in_frames);
            let available_frames = self.frames_recieved as i128 - (num_frames_that_are_behind_current_packet as i128);
            
            if available_frames < size_in_frames as i128 {
                if num_frames_that_are_behind_current_packet > 0 {
                    // skip ahead so we are only getting samples for this packet
                    self.final_audio_buffer_consumer.finish_read((num_frames_that_are_behind_current_packet * self.channels as u128) as usize);
                    let additional_frames_needed = (size_in_frames as i128) - available_frames;
                    // we will be able to get all samples for this packet, block until we get them
                    let (_read_success, _samples) = self.get_chunk_to_read((additional_frames_needed * self.channels as i128) as usize);
                    println!("Finished calibrate, ignoring {num_frames_that_are_behind_current_packet} frames");
                    // return _read_success and not true to avoid failed reads clogging up the data
                    _read_success // we will read them again later, at which point we will do finish_read (this is delibrate reading them twice)
                }
                // we started in the middle of this packet, we can't get enough, wait until next packet
                else {
                    false
                }
            }
            else {
                // enough samples! ignore the ones we need to ignore and then let the sampling happen elsewhere
                self.final_audio_buffer_consumer.finish_read((num_frames_that_are_behind_current_packet * self.channels as u128) as usize);
                println!("Finished calibrate (2), ignoring {num_frames_that_are_behind_current_packet} frames");
                true
            }
        } else {
            false
        }
    }

    fn num_frames_that_are_behind_current_packet(&self, micros_packet_finished: u128, size_in_frames: usize) -> u128 {
        let micros_packet_started = micros_packet_finished - frames_to_micros(size_in_frames as u128, self.sample_rate as u128);
        let mut frames_to_ignore = 0 as u128;
        for metadata in self.initial_metadata.iter() {
            match metadata {
                ResamplingMetadata::Arrive(frames_recieved, micros_metadata_started, micros_metadata_finished, _calibrated) => {
                    // whole packet is behind, ignore entire thing
                    if *micros_metadata_finished < micros_packet_started {
                        frames_to_ignore += *frames_recieved as u128;
                    }
                    // keep all data
                    else if *micros_metadata_started >= micros_packet_started{
                        
                    } 
                    // it overlaps, only ignore stuff before this packet
                    else {
                        let micros_ignoring = micros_packet_started - *micros_metadata_started;
                        frames_to_ignore += micros_to_frames(micros_ignoring as u128, self.sample_rate as u128);
                    }

                }
            }
        }
        frames_to_ignore
    }

    // waits until we have at least that much data
    // (or something errors)
    // returns (success, audio_buffer)
    fn get_chunk_to_read(&mut self, size: usize) -> (bool, &[f32]) {
        // drain anything in buffer (non blocking)
        loop {
            match self.finished_message_reciever.try_recv() {
                Ok(_msg) => {},
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    // sender dropped; receiver will never get more messages
                    eprintln!("Error: StreamAlignerConsumer message send disconnected");
                    break;
                }
            }
        }

        while self.final_audio_buffer_consumer.available() < size {
            // wait for data to arrive
            match self.finished_message_reciever.recv() {
                Ok(_data) => {
                    // this is only called after is_ready_to_read returns true (and is no longer used),
                    // so it's fine to ignore this, we don't use samples_recieved anymore
                }
                Err(err) => {
                    eprintln!("channel closed: {err}");
                    return (false, &[])
                }
            }
        }
        (true, self.final_audio_buffer_consumer.get_chunk_to_read(size))
    }

    fn finish_read(&mut self, size: usize) -> usize {
        self.final_audio_buffer_consumer.finish_read(size)
    }
}

impl Drop for StreamAlignerConsumer {
    fn drop(&mut self) {
        if let Err(err) = self.thread_message_sender.send(AudioBufferMetadata::Teardown()) {
            eprintln!("failed to send shutdown signal: {}", err);
        }
    }
}

// make (producer (recieves audio data from device), resampler (resamples input audio to target rate), consumer (contains resampled data)) for input audio alignment
fn create_stream_aligner(channels: usize, input_sample_rate: u32, output_sample_rate: u32, history_len: usize, calibration_packets: u32, audio_buffer_seconds: u32, resampler_quality: i32) -> Result<(StreamAlignerProducer, StreamAlignerResampler, StreamAlignerConsumer), Box<dyn Error>> {
    let (input_audio_buffer_producer, input_audio_buffer_consumer) = HeapRb::<f32>::new((audio_buffer_seconds * input_sample_rate * (channels as u32)) as usize).split();
    let (input_audio_buffer_metadata_producer, input_audio_buffer_metadata_consumer) = mpsc::channel::<AudioBufferMetadata>();
    let additional_input_audio_buffer_metadata_producer = input_audio_buffer_metadata_producer.clone(); // make another one, this is ok because it is multiple producer single consumer 
    // this recieves data from audio buffer
    let producer = StreamAlignerProducer::new(
        channels,
        input_sample_rate,
        output_sample_rate,
        history_len,
        calibration_packets,
        input_audio_buffer_producer,
        input_audio_buffer_metadata_producer
    )?;

    let (finished_resampling_producer, finished_resampling_consumer) = mpsc::channel::<ResamplingMetadata>();

    let (output_audio_buffer_producer, output_audio_buffer_consumer) = HeapRb::<f32>::new((audio_buffer_seconds * output_sample_rate * (channels as u32)) as usize).split();
    // resampled_consumer: BufferedCircularConsumer::<f32>::new(resampled_consumer))
    // this resamples, designed to run on a seperate thread
    
    let resampler = StreamAlignerResampler::new(
        channels,
        input_sample_rate,
        output_sample_rate,
        resampler_quality,
        input_audio_buffer_consumer,
        input_audio_buffer_metadata_consumer,
        output_audio_buffer_producer,
        finished_resampling_producer,
    )?;

   
    let consumer = StreamAlignerConsumer::new(
        channels,
        output_sample_rate,
        BufferedCircularConsumer::new(output_audio_buffer_consumer),
        additional_input_audio_buffer_metadata_producer, // give it ability to send shutdown signal to thread
        finished_resampling_consumer
    );

    Ok((producer, resampler, consumer))
}

type StreamId = u64;

enum OutputStreamMessage {
    Add(StreamId, u32, usize, HashMap<usize, Vec<usize>>, ResampledBufferedCircularProducer, ringbuf::HeapCons<f32>),
    Remove(StreamId),
    InterruptAll(),
}

struct StreamProducer {
    producer: HeapProd<f32>
}

impl StreamProducer {
    fn new(producer: HeapProd<f32>) -> Self {
        Self {
            producer: producer
        }
    }
    fn queue_audio(&mut self, audio_data: &[f32]) {
        let num_pushed = self.producer.push_slice(audio_data);
        if num_pushed < audio_data.len() {
            eprintln!("Error: output audio buffer got behind, try increasing buffer size");
        }
    }
}

// we only have one per device (instead of one per channel)
// because that ensures that multi-channel audio is synchronized properly
// when sent to output device
struct OutputStreamAlignerProducer {
    host_id: cpal::HostId,
    device_name: String,
    channels: usize,
    device_sample_rate: u32,
    output_stream_sender: mpsc::Sender<OutputStreamMessage>,
    cur_stream_id: Arc<AtomicU64>,
}

impl OutputStreamAlignerProducer {

    fn new(host_id: cpal::HostId, device_name: String, channels: usize, device_sample_rate: u32, output_stream_sender: mpsc::Sender<OutputStreamMessage>) -> Self {
        Self {
            host_id: host_id,
            device_name: device_name,
            channels: channels,
            device_sample_rate: device_sample_rate,
            output_stream_sender: output_stream_sender,
            cur_stream_id: Arc::new(AtomicU64::new(0)),
        }
    }

    fn begin_audio_stream(&self, channels: usize, channel_map: HashMap<usize, Vec<usize>>, audio_buffer_seconds: u32, sample_rate: u32, resampler_quality: i32) -> Result<(StreamId, StreamProducer), Box<dyn Error>> {
        // this assigns unique ids in a thread-safe way
        let stream_index = self.cur_stream_id.fetch_add(1, Ordering::Relaxed);
        let (producer, consumer) = HeapRb::<f32>::new((audio_buffer_seconds * sample_rate * (channels as u32)) as usize).split();
        let (resampled_producer, resampled_consumer) = HeapRb::<f32>::new((audio_buffer_seconds * self.device_sample_rate * (channels as u32)) as usize).split();

        // send the consumer to the consume thread
        let resampled_producer = ResampledBufferedCircularProducer::new(
            channels,
            sample_rate,
            self.device_sample_rate,
            resampler_quality,
            BufferedCircularConsumer::<f32>::new(consumer),
            BufferedCircularProducer::<f32>::new(resampled_producer),
        )?;

        self.output_stream_sender.send(OutputStreamMessage::Add(stream_index, sample_rate, channels, channel_map, resampled_producer, resampled_consumer))?;
        Ok((stream_index, StreamProducer::new(producer)))
    }

    fn queue_audio(&self, mut audio_producer: HeapProd<f32>, audio_data: &[f32]) -> HeapProd<f32> {
        let num_pushed = audio_producer.push_slice(audio_data);
        if num_pushed < audio_data.len() {
            eprintln!("Error: output audio buffer got behind, try increasing buffer size");
        }
        audio_producer
    }

    fn end_audio_stream(&self, stream_index: StreamId) -> Result<(), Box<dyn Error>> {
        self.output_stream_sender.send(OutputStreamMessage::Remove(stream_index))?;
        Ok(())
    }

    fn interrupt_all_streams(&self) -> Result<(), Box<dyn Error>> { 
        self.output_stream_sender.send(OutputStreamMessage::InterruptAll())?;
        Ok(())
    }
}

struct OutputStreamAlignerMixer {
    channels: usize,
    device_sample_rate: u32,
    output_sample_rate: u32,
    frame_size: u32,
    device_audio_producer: BufferedCircularProducer<f32>,
    resampled_audio_buffer_producer: StreamAlignerProducer,
    stream_consumers: HashMap<StreamId, (u32, usize, HashMap<usize, Vec<usize>>, ResampledBufferedCircularProducer, BufferedCircularConsumer<f32>)>,
    output_stream_receiver: mpsc::Receiver<OutputStreamMessage>,
}

// allows for playing audio on top of each other (mixing) or just appending to buffer
impl OutputStreamAlignerMixer {
    fn new(channels: usize, device_sample_rate: u32, output_sample_rate: u32, audio_buffer_seconds: u32, resampler_quality: i32, frame_size: u32, output_stream_receiver:  mpsc::Receiver<OutputStreamMessage>, device_audio_producer: HeapProd<f32>, resampled_audio_buffer_producer: StreamAlignerProducer) -> Result<Self, Box<dyn Error>>  {
        // used to send across threads
        Ok(Self {
            channels: channels,
            device_sample_rate: device_sample_rate,
            output_sample_rate: output_sample_rate,
            frame_size: frame_size,
            device_audio_producer: BufferedCircularProducer::new(device_audio_producer),
            output_stream_receiver: output_stream_receiver,
            resampled_audio_buffer_producer: resampled_audio_buffer_producer,
            stream_consumers: HashMap::new(),
        })
    }

    fn mix_audio_streams(&mut self, input_chunk_size: usize) -> Result<(), Box<dyn std::error::Error>> {
        // fetch new audio consumers, non-blocking
        loop {
            match self.output_stream_receiver.try_recv() {
                Ok(msg) => match msg {
                    OutputStreamMessage::Add(id, input_sample_rate, channels, channel_map, resampled_producer, resampled_consumer) => {
                        println!("Got new output device {} with {} channels", id, channels);
                        self.stream_consumers.insert(id, (input_sample_rate, channels, channel_map, resampled_producer, BufferedCircularConsumer::new(resampled_consumer)));
                    }
                    OutputStreamMessage::Remove(id) => {
                        // remove if present
                        self.stream_consumers.remove(&id);
                    }
                    OutputStreamMessage::InterruptAll() => {
                        // remove all streams, interrupt requires new streams to be created
                        self.stream_consumers.clear();
                    }
                },
                Err(TryRecvError::Empty) => break,          // nothing waiting; continue processing
                Err(TryRecvError::Disconnected) => break,   // sender dropped; bail out or log
            }
        }

        let (need_to_write_device_values, device_buf_write) = self.device_audio_producer.get_chunk_to_write(input_chunk_size * self.channels);
        let actual_input_chunk_size = device_buf_write.len();
        let frames_cap = device_buf_write.len() / (self.channels);
        device_buf_write.fill(0.0);
        for (_stream_id, (input_sample_rate, channels, channel_map, resample_producer, resample_consumer)) in self.stream_consumers.iter_mut() {
            let target_input_samples = input_to_output_frames(frames_cap as u128, self.device_sample_rate, *input_sample_rate);
            // this doesn't work because it'll stall for very large audio
            // resample_producer.resample_all()?; // do resampling of any available data
            // instead, do it streaming
            resample_producer.resample((target_input_samples as u32) * 2); // do * 2 so we also grab some leftovers if there are some, this is an upper bound
            let buf_from_stream = resample_consumer.get_chunk_to_read(round_to_channels(actual_input_chunk_size as u32, *channels) as usize);
            let frames = (buf_from_stream.len() / *channels as usize).min(frames_cap);
            if frames == 0 {
                continue;
            }

            let dst_stride = self.channels;
            let src_stride = *channels;
            // map virtual channels to real channels via channel_map
            for (s_idx, dst_chs) in channel_map.iter() {
                for dst_ch in dst_chs.iter() {
                    if *dst_ch >= self.channels { continue; } // guard bad maps
                    let mut dst = *dst_ch as usize;
                    let mut src_idx = *s_idx as usize;
                    for _ in 0..frames {
                        // just add to mix, do not average or clamp. Average results in too quiet, clamp is non-linear (so confuses eac, which only works with linear transformations), 
                        // (fyi, resample is a linear operation in speex so it's safe to do while using eac)
                        // see this https://dsp.stackexchange.com/a/3603
                        device_buf_write[dst] += buf_from_stream[src_idx];
                        dst += dst_stride;
                        src_idx += src_stride;
                    }
                }
            }
            let num_read = frames * (*channels);
            resample_consumer.finish_read(num_read);
        }
        // send output downstream to the eac
        self.resampled_audio_buffer_producer.process_chunk(device_buf_write);
        // finish writing to output device buffer
        self.device_audio_producer.finish_write(need_to_write_device_values, frames_cap * self.channels);
        Ok(())
    }
}

/*
struct OutputStreamAlignerConsumer {
    channels: usize,
    resample_audio_buffer_consumer: BufferedCircularConsumer<f32>,
}

impl OutputStreamAlignerConsumer {
    fn new(channels: usize, resample_audio_buffer_consumer: BufferedCircularConsumer<f32>) -> Self {
        Self {
            channels: channels,
            resample_audio_buffer_consumer: resample_audio_buffer_consumer
        }
    }
    fn get_chunk_to_read(&mut self, size: usize) -> Result<&[f32], Box<dyn std::error::Error>> {
        Ok(self.resample_audio_buffer_consumer.get_chunk_to_read(size))
    }

    fn finish_read(&mut self, size: usize) -> usize {
        self.resample_audio_buffer_consumer.finish_read(size)
    }
}
*/

struct InputDeviceConfig {
    host_id: cpal::HostId,
    device_name: String,
    channels: usize,
    sample_rate: u32,
    sample_format: SampleFormat,
    
    // number of audio chunks to hold in memory, for aligning input devices's values when dropped frames/clock offsets. 100 or so is fine
    history_len: usize,
    // number of packets recieved before we start getting audio data
    // a larger value here will take longer to connect, but result in more accurate timing alignments
    calibration_packets: u32,
    // how long buffer of input audio to store, should only really need a few seconds as things are mostly streamed
    audio_buffer_seconds: u32,
    resampler_quality: i32
}

impl InputDeviceConfig {
    fn new(
        host_id: cpal::HostId,
        device_name: String,
        channels: usize,
        sample_rate: u32,
        sample_format: SampleFormat,
        history_len: usize,
        calibration_packets: u32,
        audio_buffer_seconds: u32,
        resampler_quality: i32,
    ) -> Self {
        Self {
            host_id,
            device_name: device_name.clone(),
            channels,
            sample_rate,
            sample_format,
            history_len,
            calibration_packets,
            audio_buffer_seconds,
            resampler_quality,
        }
    }

    /// Build a config using the device's default input settings plus caller-provided buffer/resampler tuning.
    fn from_default(
        host_id: cpal::HostId,
        device_name: String,
        history_len: usize,
        calibration_packets: u32,
        audio_buffer_seconds: u32,
        resampler_quality: i32,
    ) -> Result<Self, Box<dyn Error>> {
        let host = cpal::host_from_id(host_id)?;
        let input_device = select_device(host.input_devices(), &device_name, "Input")?;
        let default_config = input_device.default_input_config()?;

        Ok(Self::new(
            host_id,
            input_device.name()?,
            default_config.channels() as usize,
            default_config.sample_rate().0,
            default_config.sample_format(),
            history_len,
            calibration_packets,
            audio_buffer_seconds,
            resampler_quality,
        ))
    }
}

struct OutputDeviceConfig {
    host_id: cpal::HostId,
    device_name: String,
    channels: usize,
    sample_rate: u32,
    sample_format: SampleFormat,
    
    // number of audio chunks to hold in memory, for aligning input devices's values when dropped frames/clock offsets. 100 or so is fine
    history_len: usize,
    // number of packets recieved before we start getting audio data
    // a larger value here will take longer to connect, but result in more accurate timing alignments
    calibration_packets: u32,
    // how long buffer of input audio to store, should only really need a few seconds as things are mostly streamed
    audio_buffer_seconds: u32,
    resampler_quality: i32,
    // frame size (in terms of samples) should be small, on the order of 1-2ms or less.
    // otherwise you may get skipping if you do not provide audio via enqueue_audio fast enough
    // larger frame sizes will also prevent immediate interruption, as interruption can only happen between each frame
    frame_size: u32,
}

impl OutputDeviceConfig {
    fn new(
        host_id: cpal::HostId,
        device_name: String,
        channels: usize,
        sample_rate: u32,
        sample_format: SampleFormat,
        history_len: usize,
        calibration_packets: u32,
        audio_buffer_seconds: u32,
        resampler_quality: i32,
        frame_size: u32,
    ) -> Self {
        Self {
            host_id: host_id,
            device_name: device_name.clone(),
            channels: channels,
            sample_rate: sample_rate,
            sample_format: sample_format,
            history_len: history_len,
            calibration_packets: calibration_packets,
            audio_buffer_seconds: audio_buffer_seconds,
            resampler_quality: resampler_quality,
            frame_size: frame_size,
        }
    }

    /// Build a config using the device's default output settings plus caller-provided buffer/resampler tuning.
    fn from_default(
        host_id: cpal::HostId,
        device_name: String,
        history_len: usize,
        calibration_packets: u32,
        audio_buffer_seconds: u32,
        resampler_quality: i32,
        frame_size_millis: u32,
    ) -> Result<Self, Box<dyn Error>> {
        let host = cpal::host_from_id(host_id)?;
        let output_device = select_device(host.output_devices(), &device_name, "Output")?;
        let default_config = output_device.default_output_config()?;
        let sample_rate = default_config.sample_rate().0;
        let frame_size = micros_to_frames((frame_size_millis as u128)*1000, sample_rate as u128);

        Ok(Self::new(
            host_id,
            output_device.name()?,
            default_config.channels() as usize,
            default_config.sample_rate().0,
            default_config.sample_format(),
            history_len,
            calibration_packets,
            audio_buffer_seconds,
            resampler_quality,
            frame_size as u32,
        ))
    }
}

struct AecConfig {
    target_sample_rate: u32,
    frame_size: usize,
    filter_length: usize
}

impl AecConfig {
    fn new(target_sample_rate: u32, frame_size: usize, filter_length: usize) -> Self {
        Self { target_sample_rate, frame_size, filter_length }
    }
}

fn get_input_stream_aligners(device_config: &InputDeviceConfig, aec_config: &AecConfig) -> Result<(Stream, StreamAlignerConsumer), Box<dyn std::error::Error>>  {

    let host = cpal::host_from_id(device_config.host_id)?;
    let device = select_device(
        host.input_devices(),
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
    

    let (producer, mut resampler, consumer) = create_stream_aligner(
        device_config.channels,
        device_config.sample_rate,
        aec_config.target_sample_rate,
        device_config.history_len,
        device_config.calibration_packets,
        device_config.audio_buffer_seconds,
        device_config.resampler_quality)?;

    // start the resampler thread
    thread::spawn(move || {
        // it returns true when signaled to stop (such as when consumer goes out of scope)
        loop {
            match resampler.resample() {
                Ok(true) => continue,
                Ok(false) => break,
                Err(err) => {
                    eprintln!("resampler error: {err}");
                    break;
                }
            }
        }
    });

    let stream = build_input_alignment_stream(
        &device,
        device_config,
        supported_config,
        producer,
    )?;

    // start input stream
    stream.play()?;

    Ok((stream, consumer))
}

fn get_output_stream_aligners(device_config: &OutputDeviceConfig, aec_config: &AecConfig) -> Result<(Stream, OutputStreamAlignerProducer, StreamAlignerConsumer), Box<dyn std::error::Error>> {

    let host = cpal::host_from_id(device_config.host_id)?;

    let device = select_device(
        host.output_devices(),
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

    let (device_audio_producer, device_audio_consumer) = HeapRb::<f32>::new((device_config.audio_buffer_seconds * device_config.sample_rate * (device_config.channels as u32)) as usize).split();
    let (output_stream_sender, output_stream_receiver) = mpsc::channel::<OutputStreamMessage>();

    let output_producer = OutputStreamAlignerProducer::new(
        device_config.host_id,
        device_config.device_name.clone(),
        device_config.channels, // channels
        device_config.sample_rate, // device_sample_rate
        output_stream_sender
    );

    let (producer, mut resampler, consumer) = create_stream_aligner(
        device_config.channels,
        device_config.sample_rate,
        aec_config.target_sample_rate,
        device_config.history_len,
        device_config.calibration_packets,
        device_config.audio_buffer_seconds,
        device_config.resampler_quality)?;
    
    let mixer = OutputStreamAlignerMixer::new(
        device_config.channels,
        device_config.sample_rate,
        aec_config.target_sample_rate,
        device_config.audio_buffer_seconds,
        device_config.resampler_quality,
        device_config.frame_size,
        output_stream_receiver,
        device_audio_producer,
        producer,
    )?;
    

    // start the resampler thread
    thread::spawn(move || {
        // it returns true when signaled to stop (such as when consumer goes out of scope)
        loop {
            match resampler.resample() {
                Ok(true) => continue,
                Ok(false) => break,
                Err(err) => {
                    eprintln!("resampler error: {err}");
                    break;
                }
            }
        }
    });

    let stream = build_output_alignment_stream(
        &device,
        device_config,
        supported_config,
        mixer,
        BufferedCircularConsumer::new(device_audio_consumer)
    )?;

    // start output stream
    stream.play()?;

    Ok((stream, output_producer, consumer))
}


enum DeviceUpdateMessage {
    AddInputDevice(String, Stream, StreamAlignerConsumer),
    RemoveInputDevice(String),
    AddOutputDevice(String, Stream, StreamAlignerConsumer),
    RemoveOutputDevice(String)
}

struct AecStream {
    aec: Option<EchoCanceller>,
    aec_config: AecConfig,
    device_update_sender: mpsc::Sender<DeviceUpdateMessage>,
    device_update_receiver: mpsc::Receiver<DeviceUpdateMessage>,
    input_streams: HashMap<String, Stream>,
    output_streams: HashMap<String, Stream>,
    input_aligners: HashMap<String, StreamAlignerConsumer>,
    input_aligners_in_progress: HashMap<String, StreamAlignerConsumer>,
    output_aligners: HashMap<String, StreamAlignerConsumer>,
    output_aligners_in_progress: HashMap<String, StreamAlignerConsumer>,
    sorted_input_aligners: Vec<String>,
    sorted_output_aligners: Vec<String>,
    input_channels: usize,
    output_channels: usize,
    start_micros: Option<u128>,
    total_frames_emitted: u128,
    input_audio_buffer: Vec<i16>,
    output_audio_buffer: Vec<i16>,
    aec_audio_buffer: Vec<i16>,
    aec_out_audio_buffer: Vec<f32>
}

impl AecStream {
    fn new(
        aec_config: AecConfig
    ) -> Result<Self, Box<dyn Error>> {
        if aec_config.target_sample_rate == 0 {
            return Err(format!("Target sample rate is {}, it must be greater than zero.", aec_config.target_sample_rate).into());
        }
        let (device_update_sender, device_update_receiver) = mpsc::channel::<DeviceUpdateMessage>();
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
           output_aligners_in_progress: HashMap::new(),
           sorted_input_aligners: Vec::new(),
           sorted_output_aligners: Vec::new(),
           input_channels: 0,
           output_channels: 0,
           start_micros: None,
           total_frames_emitted: 0,
           input_audio_buffer: Vec::new(),
           output_audio_buffer: Vec::new(),
           aec_audio_buffer: Vec::new(),
           aec_out_audio_buffer: Vec::new(),
        })
    }

    fn num_input_channels(&self) -> usize {
        self.input_aligners
            .values()
            .map(|aligner| aligner.channels)
            .sum()
    }

    fn num_output_channels(&self) -> usize {
        self.output_aligners
            .values()
            .map(|aligner| aligner.channels)
            .sum()
    }

    fn reinitialize_aec(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.input_channels = self.num_input_channels();
        self.output_channels = self.num_output_channels();

        // store a consistent ordering
        self.sorted_input_aligners = self.input_aligners.keys().cloned().collect();
        self.sorted_input_aligners.sort();

        self.sorted_output_aligners = self.output_aligners.keys().cloned().collect();
        self.sorted_output_aligners.sort();

        self.aec = if self.input_channels > 0 && self.output_channels > 0 {
            EchoCanceller::new_multichannel(
                self.aec_config.frame_size,
                self.aec_config.filter_length,
                self.input_channels,
                self.output_channels,
            )
        } else {
            None
        };
        if let Some(aec) = self.aec.as_mut() {
            aec.set_sampling_rate(self.aec_config.target_sample_rate);
            let sampling_rate = aec.sampling_rate();
            println!("Set sampling rate to {sampling_rate}");
        }

        self.input_audio_buffer.clear();
        self.input_audio_buffer.resize(self.aec_config.frame_size * self.input_channels, 0 as i16);
        self.output_audio_buffer.clear();
        self.output_audio_buffer.resize(self.aec_config.frame_size * self.output_channels, 0 as i16);
        self.aec_audio_buffer.clear();
        self.aec_audio_buffer.resize(self.aec_config.frame_size * self.input_channels, 0 as i16);
        self.aec_out_audio_buffer.clear();
        self.aec_out_audio_buffer.resize(self.aec_config.frame_size * self.input_channels, 0 as f32);
        Ok(())
    }

    fn add_input_device(&mut self, config: &InputDeviceConfig) -> Result<(), Box<dyn std::error::Error>> {
        let (stream, aligners) = get_input_stream_aligners(config, &self.aec_config)?;
        self.device_update_sender.send(DeviceUpdateMessage::AddInputDevice(config.device_name.clone(), stream, aligners))?;
        Ok(())
    }

    fn add_output_device(&mut self, config: &OutputDeviceConfig) -> Result<OutputStreamAlignerProducer, Box<dyn std::error::Error>> {
        let (stream, producer, consumer) = get_output_stream_aligners(config, &self.aec_config)?;
        self.device_update_sender.send(DeviceUpdateMessage::AddOutputDevice(config.device_name.clone(), stream, consumer))?;
        Ok(producer)
    }

    fn remove_input_device(&mut self, config: &InputDeviceConfig) -> Result<(), Box<dyn std::error::Error>> {
        self.device_update_sender.send(DeviceUpdateMessage::RemoveInputDevice(config.device_name.clone()))?;
        Ok(())
    }

    fn remove_output_device(&mut self, config: &OutputDeviceConfig) -> Result<(), Box<dyn std::error::Error>> {
        self.device_update_sender.send(DeviceUpdateMessage::RemoveOutputDevice(config.device_name.clone()))?;
        Ok(())
    }

    fn calibrate(&mut self, output_producers: &mut [OutputStreamAlignerProducer], debug_wav: bool) -> Result<(Vec<usize, Option<i64>>, Vec<usize, Vec<usize, Option<i64>>>), Box<dyn std::error::Error>> {
        let sample_rate = self.aec_config.target_sample_rate as f32;
        // Probe length (~0.1s) to stay quick but audible.
        let tone_ms = 100.0;
        let capture_secs = 3.0;

        // 1) Emit a distinct probe on each output device (all channels), in sorted output order.
        let mut active_streams: Vec<(usize, usize, StreamId)> = Vec::new();
        let mut tones = Vec::new();
        let mut streams = Vec::new();

        for (idx, dev_name) in self.sorted_output_aligners.clone().iter().enumerate() {
            let Some(producer) = output_producers
                .iter_mut()
                .find(|p| p.device_name == *dev_name) else {
                eprintln!("calibrate: no output producer found for '{dev_name}'");
                continue;
            };
            let tone_mono = generate_probe_tone_for_device(idx, tone_ms, sample_rate);
            if tone_mono.is_empty() {
                continue;
            }
            let channels = producer.channels;
            let mut channel_map = HashMap::new();
            let mut out_channels = Vec::new();
            for ch in 0..channels {
                out_channels.push(ch);
            }
            channel_map.insert(0, out_channels); // map first channel to play on all channels
            let (stream_id, stream) = producer.begin_audio_stream(
                1, // 1 channel
                channel_map,
                2, // seconds of buffer for this probe
                self.aec_config.target_sample_rate,
                5, // resampler quality
            )?;
            tones.push(tone_mono);
            active_streams.push((idx, channels, stream_id));
            streams.push(stream);
        }

        for (mut stream, tone) in streams.into_iter().zip(tones.into_iter()) {
            stream.queue_audio(tone.as_slice());
        }

        // Build channel ranges for inputs and outputs (interleaved order).
        let mut input_channel_ranges: Vec<(String, usize, usize)> = Vec::new();
        let mut in_ch_start = 0usize;
        for name in &self.sorted_input_aligners {
            if let Some(aligner) = self.input_aligners.get(name) {
                input_channel_ranges.push((name.clone(), in_ch_start, aligner.channels));
                in_ch_start += aligner.channels;
            }
        }
        let mut output_channel_ranges: Vec<(String, usize, usize)> = Vec::new();
        let mut out_ch_start = 0usize;
        for name in &self.sorted_output_aligners {
            if let Some(aligner) = self.output_aligners.get(name) {
                output_channel_ranges.push((name.clone(), out_ch_start, aligner.channels));
                out_ch_start += aligner.channels;
            }
        }

        // 2) Capture ~3s of aligned input/output data, averaged per device.
        let mut captured_inputs: Vec<Vec<f32>> = vec![Vec::new(); input_channel_ranges.len()];
        let mut captured_outputs: Vec<Vec<f32>> = vec![Vec::new(); output_channel_ranges.len()];
        let mut captured_micros: u128 = 0;
        let target_micros: u128 = ((capture_secs as u128) * 1_000_000) as u128;
        let total_in_ch = self.input_channels.max(1);
        let total_out_ch = self.output_channels.max(1);
        while captured_micros < target_micros {
            let (input_slices, output_slices, _aec_out, start_time, end_time) = self.update_debug()?;
            let chunk_micros = end_time.saturating_sub(start_time);
            if !input_slices.is_empty() && total_in_ch > 0 {
                let frames = input_slices.len() / total_in_ch;
                for frame_idx in 0..frames {
                    let base = frame_idx * total_in_ch;
                    for (dev_idx, (_name, start_ch, ch_count)) in input_channel_ranges.iter().enumerate() {
                        let mut acc = 0.0f32;
                        for ch in 0..*ch_count {
                            let sample = input_slices[base + start_ch + ch];
                            acc += f32::from_sample(sample);
                        }
                        captured_inputs[dev_idx].push(acc / (*ch_count as f32));
                    }
                }
            }
            if !output_slices.is_empty() && total_out_ch > 0 {
                let frames = output_slices.len() / total_out_ch;
                for frame_idx in 0..frames {
                    let base = frame_idx * total_out_ch;
                    for (dev_idx, (_name, start_ch, ch_count)) in output_channel_ranges.iter().enumerate() {
                        let mut acc = 0.0f32;
                        for ch in 0..*ch_count {
                            let sample = output_slices[base + start_ch + ch];
                            acc += f32::from_sample(sample);
                        }
                        captured_outputs[dev_idx].push(acc / (*ch_count as f32));
                    }
                }
            }
            captured_micros += chunk_micros;
        }

        // 3) Stop probe streams now that capture is done.
        for (idx, _channels, stream_id) in active_streams {
            if let Some(producer) = output_producers.get(idx) {
                producer.end_audio_stream(stream_id)?;
            }
        }

        if debug_wav {
            // write captured inputs
            for (i, (name, _, _)) in input_channel_ranges.iter().enumerate() {
                let sanitized = Self::sanitize_filename(name);
                let path = format!("calib_input_{sanitized}.wav");
                let mut writer = WavWriter::create(
                    path,
                    WavSpec {
                        channels: 1,
                        sample_rate: self.aec_config.target_sample_rate,
                        bits_per_sample: 16,
                        sample_format: HoundSampleFormat::Int,
                    },
                )?;
                for &s in &captured_inputs[i] {
                    writer.write_sample(Self::f32_to_i16(s))?;
                }
                writer.finalize()?;
            }
            // write captured outputs
            for (i, (name, _, _)) in output_channel_ranges.iter().enumerate() {
                let sanitized = Self::sanitize_filename(name);
                let path = format!("calib_output_{sanitized}.wav");
                let mut writer = WavWriter::create(
                    path,
                    WavSpec {
                        channels: 1,
                        sample_rate: self.aec_config.target_sample_rate,
                        bits_per_sample: 16,
                        sample_format: HoundSampleFormat::Int,
                    },
                )?;
                for &s in &captured_outputs[i] {
                    writer.write_sample(Self::f32_to_i16(s))?;
                }
                writer.finalize()?;
            }
        }

        // 4) Detect probes on outputs to compute offsets relative to output device 0.
        // mapping of output device -> offset
        let mut output_offsets: Vec<Option<i64>> = vec![None, output_channel_ranges.len()];
        // mapping of input device -> (output device index, output device index offset)
        let mut input_offsets: Vec<usize, Vec<usize, Option<i64>>> = vec![vec![None; output_channel_ranges.len()], input_channel_ranges.len()];

        if !captured_outputs.is_empty() {
            for (dev_idx, buf) in captured_outputs.iter().enumerate() {
                let detections = detect_probe_tones(buf, output_producers.len(), tone_ms, sample_rate);
                let mut start_for_dev: Option<(i64, f32)> = None;
                for (d_idx, start, score) in detections {
                    if d_idx == dev_idx { // just detect this device, no others should show up since it just forwards the data
                        start_for_dev = Some((start, score));
                        break;
                    }
                }
                if let Some((start, score)) = start_for_dev {
                    println!("Output {dev_idx} has offset {start} with score {score}");
                    output_offsets[dev_idx] = start;
                } else {
                    eprintln!("No probe detected for output device index {dev_idx}");
                }
            }
            for (input_idx, buf) in captured_inputs.iter().enumerate() {
                let detections = detect_probe_tones(buf, output_producers.len(), tone_ms, sample_rate);
                let mut start_for_dev: Option<(i64, f32)> = None;
                for (output_idx, start, score) in detections {
                    println!("Output {output_idx} -> Input {input_idx} has offset {start} with score {score}");
                    input_offsets[input_idx][output_idx] = Some(start);
                }
                for output_idx in 0..output_channel_ranges.len() {
                    if let None = input_offsets[input_idx][output_idx] {
                        eprintln!("No probe detected for output {output_idx} -> input {input_idx} (that input could not hear that output, is the volume too low?)");
                    }
                }
            }
        }

        // 5) Detect probes and compute offsets relative to output device 0, per input device.
        
        Ok((output_offsets, input_offsets))
    }

    fn sanitize_filename(name: &str) -> String {
        name.chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect()
    }

    // calls update, but returns all involved audio buffers
    // (if needed for diagnostic reasons, usually .update() (which returns aec'd inputs) should be all you need)
    fn update_debug(&mut self) -> Result<(&[i16], &[i16], &[f32], u128, u128), Box<dyn std::error::Error>> {
        let (start_time, end_time) = {
            let (_, start_time, end_time) = self.update()?;
            (start_time, end_time)
        };
        return Ok((self.input_audio_buffer.as_slice(), self.output_audio_buffer.as_slice(), self.aec_out_audio_buffer.as_slice(), start_time, end_time));
    }

    fn update(&mut self) -> Result<(&[f32], u128, u128), Box<dyn std::error::Error>> {
        let chunk_size = self.aec_config.frame_size;
        let start_micros = if let Some(start_micros_value) = self.start_micros {
            start_micros_value
        } else {
            let start_micros_value = SystemTime::now().duration_since(UNIX_EPOCH).expect("clock went backwards").as_micros() - frames_to_micros(chunk_size as u128, self.aec_config.target_sample_rate as u128);
            self.start_micros = Some(start_micros_value);
            start_micros_value
        };
        let chunk_start_micros = frames_to_micros(self.total_frames_emitted, self.aec_config.target_sample_rate as u128) + start_micros;
        let chunk_end_micros = chunk_start_micros + frames_to_micros(chunk_size as u128, self.aec_config.target_sample_rate as u128);
        self.total_frames_emitted += chunk_size as u128;
        // todo: move to crossbeam to avoid mpsc locks
        // todo: use actual timestamps on mac osx because they are system-wide (on linux they are device-wide so we have to do something else)
        loop {
            match self.device_update_receiver.try_recv() {
                Ok(msg) => match msg {
                    DeviceUpdateMessage::AddInputDevice(device_name, stream, aligner) => {
                        // old stream is stopped by default when it goes out of scope
                        self.input_streams.insert(device_name.clone(), stream);
                        self.input_aligners.remove(&device_name);
                        self.input_aligners_in_progress.insert(device_name.clone(), aligner);

                        self.reinitialize_aec()?;
                    }
                    DeviceUpdateMessage::RemoveInputDevice(device_name) => {
                        // old stream is stopped by default when it goes out of scope
                        self.input_streams.remove(&device_name);
                        self.input_aligners.remove(&device_name);
                        self.input_aligners_in_progress.remove(&device_name);

                        self.reinitialize_aec()?;
                    }
                    DeviceUpdateMessage::AddOutputDevice(device_name, stream, aligner) => {
                        self.output_streams.insert(device_name.clone(), stream);
                        self.output_aligners.remove(&device_name);
                        self.output_aligners_in_progress.insert(device_name.clone(), aligner);

                        self.reinitialize_aec()?;
                    }
                    DeviceUpdateMessage::RemoveOutputDevice(device_name) => {
                        self.output_streams.remove(&device_name);
                        self.output_aligners.remove(&device_name);
                        self.output_aligners_in_progress.remove(&device_name);

                        self.reinitialize_aec()?;
                    }
                }
                Err(TryRecvError::Empty) => {
                    // no message available right now
                    break;
                }
                Err(TryRecvError::Disconnected) => {
                    // sender dropped; receiver will never get more messages
                    eprintln!("Error: Stream message send disconnected");
                    break;
                }
            } 
        }
        // similarly, if we initialize an output device here
        // we may not get any audio for a little bit
        if chunk_size == 0 {
            return Ok((&[], chunk_start_micros, chunk_end_micros));
        }
         // initialize any new aligners and align them to our frame step
        let mut modified_aligners = false;
        for key in self.input_aligners_in_progress.keys().cloned().collect::<Vec<String>>() {
            let ready = self
                .input_aligners_in_progress
                .get_mut(&key)
                .map(|a| a.is_ready_to_read(chunk_end_micros, chunk_size))
                .unwrap_or(false);

            if ready {
                if let Some(aligner) = self.input_aligners_in_progress.remove(&key) {
                    self.input_aligners.insert(key, aligner);
                    modified_aligners = true;
                }
            }
        }
        for key in self.output_aligners_in_progress.keys().cloned().collect::<Vec<String>>() {
            let ready = self
                .output_aligners_in_progress
                .get_mut(&key)
                .map(|a| a.is_ready_to_read(chunk_end_micros, chunk_size))
                .unwrap_or(false);

            if ready {
                if let Some(aligner) = self.output_aligners_in_progress.remove(&key) {
                    self.output_aligners.insert(key, aligner);
                    modified_aligners = true;
                }
            }
        }

        if modified_aligners {
            self.reinitialize_aec()?;
        }

        // recieve audio data and interleave it into our buffers
        let mut input_channel = 0;
        self.input_audio_buffer.fill(0 as i16);
        for key in &self.sorted_input_aligners {
            if let Some(aligner) = self.input_aligners.get_mut(key) {
                let channels = aligner.channels;
                let needed = chunk_size * channels;
                let (ok, chunk) = aligner.get_chunk_to_read(needed);
                let frames = chunk.len() / channels;

                if ok && frames > 0 {
                    for c in 0..channels {
                        let mut src_idx = c;
                        let mut dst = input_channel + c;
                        for _ in 0..frames {
                            self.input_audio_buffer[dst] = Self::f32_to_i16(chunk[src_idx]);
                            dst += self.input_channels;
                            src_idx += channels;
                        }
                    }
                }

                aligner.finish_read(frames * channels);
                input_channel += channels;
            }
        }

        let aec_output = if self.output_channels == 0 {
            // simply pass through input_channels, no need for aec
            &self.input_audio_buffer
        }
        else {
                
            let mut output_channel = 0;
            self.output_audio_buffer.fill(0 as i16);
            for key in &self.sorted_output_aligners {
                if let Some(aligner) = self.output_aligners.get_mut(key) {
                    let channels = aligner.channels;
                    let needed = chunk_size * channels;
                    let (ok, chunk) = aligner.get_chunk_to_read(needed);
                    let frames = chunk.len() / channels;

                    if ok && frames > 0 {
                        for c in 0..channels {
                            let mut src_idx = c;
                            let mut dst = output_channel + c;
                            for _ in 0..frames {
                                self.output_audio_buffer[dst] = Self::f32_to_i16(chunk[src_idx]);
                                dst += self.output_channels;
                                src_idx += channels;
                            }
                        }
                        aligner.finish_read(frames * channels);
                    }

                    output_channel += channels;
                }
            }

            self.aec_audio_buffer.fill(0 as i16);

            if self.input_channels == 0 {
                &self.aec_audio_buffer
            }
            else {
                let Some(aec) = self.aec.as_mut() else { 
                    return Err("no aec".into());
                };

                unsafe {
                    speex_echo_cancellation(
                        aec.as_ptr(),
                        self.input_audio_buffer.as_ptr(),
                        self.output_audio_buffer.as_ptr(),
                        self.aec_audio_buffer.as_mut_ptr(),
                    );
                }
                &self.aec_audio_buffer
            }
        };
        
        for (out, sample) in self.aec_out_audio_buffer.iter_mut().zip(aec_output) {
            *out = f32::from_sample(*sample);
        }

        //let timestamp_value = SystemTime::now().duration_since(UNIX_EPOCH).expect("clock went backwards").as_micros();
        //let diff = (((chunk_end_micros as i128 - timestamp_value as i128) as f32) / 1000.0) / (self.aec_config.target_sample_rate as f32);
        //let frame_size_seconds = (chunk_size as f32) / (self.aec_config.target_sample_rate as f32);
        //println!("{diff} chunk_size {frame_size_seconds}");
     

        Ok((self.aec_out_audio_buffer.as_slice(), chunk_start_micros, chunk_end_micros))
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
    

fn get_host_by_name(target: &str) -> Option<Host> {
    for host_id in cpal::available_hosts() {
        if host_id.name().eq_ignore_ascii_case(target) {
            return cpal::host_from_id(host_id).ok();
        }
    }
    None
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

fn supported_device_configs_to_string(
    device: &Device,
    device_name: &String,
    direction: &'static str
) -> Result<String, Box<dyn Error>> {
    let configs : Vec<_> = match direction {
        "Input" => device.supported_input_configs().map(|configs| configs.collect())
            .map_err(|err| format!("Unable to enumerate input configs for '{device_name}': {err}"))?,
        "Output" => device.supported_output_configs().map(|configs| configs.collect())
            .map_err(|err| format!("Unable to enumerate output configs for '{device_name}': {err}"))?,
        other => {
            return Err(format!("Unknown device direction '{other}' when validating {device_name}., should be input or output").into());
        }
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
        .join("\n      "))
}

fn find_matching_device_config(
    device: &Device,
    device_name: &String,
    channels: usize,
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
        .filter(|cfg| cfg.channels() == (channels as u16) && cfg.sample_format() == format)
        .find_map(|cfg| cfg.clone().try_with_sample_rate(desired_rate));
    
    if let Some(config) = matching_config {
        Ok(config)
    } else {
        let supported_list = configs
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
            .join("\n      ");

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
    channel_aligners: StreamAlignerProducer,
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
    mut channel_aligner: StreamAlignerProducer,
) -> Result<Stream, cpal::BuildStreamError>
where
    T: Sample + SizedSample,
    f32: FromSample<T>,
{
    let per_channel_capacity = config.sample_rate
        .saturating_div(20) // ~50 ms of audio per channel
        .max(1024);
    let mut interleaved_buffer =
        Vec::<f32>::with_capacity((per_channel_capacity as usize) * (config.channels as usize));
    
    let device_name = config.device_name.clone();
    let device_name_inner = config.device_name.clone();
    device.build_input_stream(
        &supported_config.config(),
        move |data: &[T], _info: &InputCallbackInfo| {
            if data.is_empty() {
                return;
            }

            interleaved_buffer.clear();
            interleaved_buffer.reserve(data.len()); // usually already sized, but cheap
            for &s in data {
                interleaved_buffer.push(f32::from_sample(s));
            }
            if let Err(err) = channel_aligner.process_chunk(interleaved_buffer.as_slice()) {
                eprintln!("Input stream '{device_name_inner}' error when process chunk {err}");
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
    mixer: OutputStreamAlignerMixer,
    device_audio_channel_consumer: BufferedCircularConsumer<f32>
) -> Result<Stream, cpal::BuildStreamError> {
    match config.sample_format {
        SampleFormat::I16 => build_output_alignment_stream_typed::<i16>(
            device,
            config,
            supported_config,
            mixer,
            device_audio_channel_consumer,
        ),
        SampleFormat::F32 => build_output_alignment_stream_typed::<f32>(
            device,
            config,
            supported_config,
            mixer,
            device_audio_channel_consumer,
        ),
        SampleFormat::U16 => build_output_alignment_stream_typed::<u16>(
            device,
            config,
            supported_config,
            mixer,
            device_audio_channel_consumer,
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
    mut mixer: OutputStreamAlignerMixer,
    mut device_audio_channel_consumer: BufferedCircularConsumer<f32>
) -> Result<Stream, cpal::BuildStreamError>
where
    T: Sample + SizedSample,
    T: FromSample<f32>,
{
    let device_name = config.device_name.clone();
    device.build_output_stream(
        &supported_config.config(),
        move |data: &mut [T], _| {
            let frames = data.len() / mixer.channels;
            if frames == 0 {
                return;
            }
            // in case we don't have enough data yet
            //data.fill(T::from_sample(0.0f32));
            while device_audio_channel_consumer.available() < frames* mixer.channels {
                mixer.mix_audio_streams(mixer.frame_size as usize);
            }
            let chunk = device_audio_channel_consumer.get_chunk_to_read(frames * mixer.channels);
            
            if chunk.is_empty() {
                return;
            }

            let chunk_frames = chunk.len() / mixer.channels;

            let samples_to_write = chunk_frames*mixer.channels;

            // it arrives already interleaved, so we can just copy
            for (dst, &src) in data
                .iter_mut()
                .zip(chunk.iter().take(samples_to_write))
            {
                *dst = T::from_sample(src);
            }

            device_audio_channel_consumer.finish_read(samples_to_write);

        },
        move |err| eprintln!("Output stream '{device_name}' error: {err}"),
        None,
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let frame_size_ms = 10;
    let filter_length_ms = 300;
    let aec_sample_rate = 16000;
    let aec_config = AecConfig::new(
        aec_sample_rate,
        (aec_sample_rate * frame_size_ms / 1000) as usize,
        (aec_sample_rate * filter_length_ms / 1000) as usize,
    );
    let mut stream = AecStream::new(aec_config)?;

    let host_ids = cpal::available_hosts();
    for host_id in host_ids {
        println!("Host: '{}'", host_id.name());

        // If you want to inspect devices:
        let host = cpal::host_from_id(host_id)?;
        for dev in host.input_devices()? {
            println!("  input: '{}'", dev.name()?);
            match supported_device_configs_to_string(&dev, &dev.name()?, "Input") {
                Ok(cfgs) => println!("      {cfgs}"),
                Err(err) => println!("      {err}"),
            }
        }
        for dev in host.output_devices()? {
            println!("  output: '{}'", dev.name()?);
            match supported_device_configs_to_string(&dev, &dev.name()?, "Output") {
                Ok(cfgs) => println!("      {cfgs}"),
                Err(err) => println!("      {err}"),
            }
        }
    }

    let resampler_quality = 5;

    let host = get_host_by_name("ALSA").unwrap_or_else(cpal::default_host);
    let input_device_config = InputDeviceConfig::from_default(
        host.id(),
        "front:CARD=Beyond,DEV=0".to_string(),
        // number of audio chunks to hold in memory, for aligning input devices's values when dropped frames/clock offsets. 100 or so is fine
        100, // history_len 
        // number of packets recieved before we start getting audio data
        // a larger value here will take longer to connect, but result in more accurate timing alignments
        20, // calibration_packets
        // how long buffer of input audio to store, should only really need a few seconds as things are mostly streamed
        20, // audio_buffer_seconds
        resampler_quality // resampler_quality
    )?;

    let output_device_config = OutputDeviceConfig::from_default(
        host.id(),
        "hdmi:CARD=NVidia,DEV=0".to_string(),
        100, // history_len
        20, // calibration_packets
        20, // audio_buffer_seconds, just for resampling (actual audio buffer determined upon begin_audio_stream creation)
        resampler_quality, // resampler_quality
        3, // frame_size_millis (3 millis of audio per frame)
    )?;

    stream.add_input_device(&input_device_config)?;
    let mut stream_output_creator = stream.add_output_device(&output_device_config)?;

    // output wav files for debugging
    let pcm_spec_input = WavSpec {
        channels: input_device_config.channels as u16,
        sample_rate: aec_sample_rate, // 16_000 in your config
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

    // input wav file to output
    let mut wav = WavReader::open("examples/example_talking.wav")?;
    let spec = wav.spec();
    let wav_channels = spec.channels as usize;
    let wav_rate = spec.sample_rate;
    let wav_samples: Vec<f32> = match spec.sample_format {
        HoundSampleFormat::Int => wav
            .samples::<i16>()
            .map(|s| f32::from_sample(s.unwrap()))
            .collect(),
        HoundSampleFormat::Float => wav.samples::<f32>().map(|s| s.unwrap()).collect(),
    };

    let mut channel_map = HashMap::new();
    for i in 0..wav_channels {
        let mut mapped_to_channels = Vec::new();
        mapped_to_channels.push(0);
        channel_map.insert(i, mapped_to_channels); // map all wav channels to first output channel, for testing
    }

    let (_stream_id, mut stream_output) = stream_output_creator.begin_audio_stream(
        wav_channels as usize,
        channel_map,
        ((wav_samples.len()/(wav_rate as usize) + 1)*2) as u32, // audio_buffer_seconds, needs to be long enough to hold all the audio
        wav_rate,
        resampler_quality
    )?;


    // waits for channels to calibrate
    while stream.num_input_channels() == 0 || stream.num_output_channels() == 0 {
        let (aligned_input, aligned_output, aec_applied, _start_time, _end_time) = stream.update_debug()?;
        // don't write to wav files bc if one device is ready before another,
        // that device will have more samples written
        // which makes it annoying to check alignments in audacity
    }

    println!("Computing calibration");
    let _offsets = stream.calibrate(std::slice::from_mut(&mut stream_output_creator), true)?;
    println!("calibrated");

    // enqueues audio samples to be played after each other
    stream_output.queue_audio(wav_samples.as_slice());
    stream_output.queue_audio(wav_samples.as_slice());


    for _i in 0..1000 {
        let (aligned_input, aligned_output, aec_applied, _start_time, _end_time) = stream.update_debug()?;
        for &s in aligned_input { in_wav.write_sample(s)?; }
        for &s in aligned_output { out_wav.write_sample(s)?; }
        for &s in aec_applied { aec_wav.write_sample(s)?; }
        //println!("Got {} samples", aec_applied.len());
    }
    
    stream_output_creator.end_audio_stream(_stream_id);
    
    stream_output_creator.interrupt_all_streams();
    
    stream.remove_input_device(&input_device_config)?;
    stream.remove_output_device(&output_device_config)?;

    in_wav.finalize()?;
    out_wav.finalize()?;
    aec_wav.finalize()?;

    Ok(())
}
