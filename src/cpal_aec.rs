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
#[cfg(target_arch = "wasm32")]
extern crate js_sys;
#[cfg(target_arch = "wasm32")]
extern crate wasm_bindgen;
#[cfg(target_arch = "wasm32")]
extern crate web_sys;

#[cfg(target_arch = "wasm32")]
use js_sys::Array;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;
#[cfg(target_arch = "wasm32")]
use web_sys::{AudioContext, MediaStream, MediaStreamTrack, MediaStreamConstraints, MediaDevices, Navigator, MediaStreamAudioSourceNode};
#[cfg(target_arch = "wasm32")]
use js_sys::{Float32Array};
#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;
#[cfg(target_arch = "wasm32")]
use std::collections::HashMap;
#[cfg(target_arch = "wasm32")]
use web_sys::AudioContextState;

use futures::channel::mpsc;
use futures::StreamExt;
use futures::executor::block_on;

use std::{
    collections::{HashMap},
    error::Error,
    mem::{MaybeUninit},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::{spawn_local, JsFuture};
use aec3::voip::VoipAec3;

use std::backtrace::Backtrace;

use rustfft::{FftPlanner, num_complex::Complex};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Device, FromSample, Host, Sample, SampleFormat, SampleRate, SizedSample,
    Stream, SupportedStreamConfig,
};

#[cfg(not(target_arch = "wasm32"))]
use cpal::InputCallbackInfo;

#[cfg(not(target_arch = "wasm32"))]
use std::thread;

#[cfg(target_arch = "wasm32")]
use cpal::SupportedBufferSize;
use std::collections::HashSet;

use ringbuf::{
    traits::{Consumer, Producer, RingBuffer, Split, Observer},
    HeapCons, HeapProd, HeapRb, LocalRb,
};
use ringbuf::storage::Heap;
use crate::Resampler;
use std::f32::consts::PI;
use rustfft::num_complex::Complex32;

#[cfg(not(target_arch = "wasm32"))]
use std::time::{UNIX_EPOCH, SystemTime};
#[cfg(target_arch = "wasm32")]
use js_sys::{Date, Promise};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{JsCast, JsValue};
use hound::{SampleFormat as HoundSampleFormat, WavSpec, WavWriter};

use earshot::{VoiceActivityDetector, VoiceActivityProfile};

#[inline]
fn aec_log(msg: impl AsRef<str>) {
    let msg = msg.as_ref();
    #[cfg(target_arch = "wasm32")]
    {
        use js_sys::{Function, Reflect};
        let global = js_sys::global();
        let key = JsValue::from_str("logMessage");
        if let Ok(val) = Reflect::get(&global, &key) {
            if let Some(func) = val.dyn_ref::<Function>() {
                let _ = func.call1(&JsValue::NULL, &JsValue::from_str(msg));
                return;
            }
        }
        // Fallback if the JS helper isn't present.
        web_sys::console::log_1(&JsValue::from_str(msg));
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        println!("{msg}");
    }
}

#[macro_export]
macro_rules! aec_log {
    ($($t:tt)*) => {
        $crate::aec::aec_log(format!($($t)*))
    };
}


#[inline]
unsafe fn assume_init_slice_mut<T>(slice: &mut [MaybeUninit<T>]) -> &mut [T] {
    unsafe {
        &mut *(slice as *mut [MaybeUninit<T>] as *mut [T])
    }
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

#[inline]
fn now_micros() -> u128 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock went backwards")
            .as_micros()
    }
    #[cfg(target_arch = "wasm32")]
    {
        // Date::now returns milliseconds since epoch as f64.
        (Date::now() * 1000.0) as u128
    }
}

#[cfg(target_arch = "wasm32")]
async fn yield_to_event_loop() {
    let _ = JsFuture::from(Promise::resolve(&JsValue::NULL)).await;
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
fn generate_probe_tone_for_device(device_index: usize, empty_seconds: f32, duration_ms: f32, sample_rate: u32) -> Vec<f32> {
    //let base = [1_000.0f32, 1_800.0f32, 2_600.0f32];
    // Small offset per device to make correlation peaks separable.
    //let offset = (device_index as f32 % 5.0) * 120.0;
    //let freqs: Vec<f32> = base.iter().map(|f| f + offset).collect();
    //generate_probe_tone_with_freqs(duration_ms, sample_rate, &freqs);
    //let tone = chirp(1_200.0, 6_500.0, sample_rate, duration_ms);
    indexed_chirp(device_index as u32, sample_rate, empty_seconds, duration_ms / 1000.0)
}

pub fn indexed_chirp(idx: u32, sr: u32, empty_seconds: f32, dur_s: f32) -> Vec<f32> {
    if dur_s <= 0.0 { return Vec::new(); }

    let sr_f = sr as f32;
    let n = (dur_s * sr_f).round().max(1.0) as usize;
    let empty_n = (empty_seconds.max(0.0) * sr_f).round() as usize;

    // NOTE: h.fract() is always 0.0 because h is an integer-valued f32.
    // consider fixing if you actually want per-idx randomness.
    let h = (idx.wrapping_mul(0x9E3779B9) ^ 0x85EBCA6B) as f32;
    let base = 150.0 + (h.fract() * 120.0);       // ~150–270 Hz
    let span = 500.0 + (h.sin().abs() * 300.0);   // +0.5–0.8 kHz
    let nyq_limit = sr_f * 0.2;                   // keep it low
    let f0 = base.min(nyq_limit);
    let f1 = (base + span).min(nyq_limit);

    let k = (f1 / f0).ln() / dur_s;

    let mut out = vec![0.0f32; empty_n];
    out.reserve(n);

    for i in 0..n {
        let t = i as f32 / sr_f;

        // Handle k ~ 0 (f0 ~= f1) to avoid divide-by-zero
        let phase = if k.abs() < 1e-9 {
            2.0 * std::f32::consts::PI * f0 * t
        } else {
            2.0 * std::f32::consts::PI * f0 * ((k * t).exp() - 1.0) / k
        };

        // Hann window (use n-1 so endpoints hit 0 when n>1)
        let w = if n > 1 {
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n as f32 - 1.0)).cos())
        } else {
            1.0
        };

        out.push(phase.sin() * w * 0.3);
    }

    out
}


pub fn indexed_chirp_new(idx: u32, chirp_frames: u32) -> Vec<f32> {
    if dur_s <= 0.0 { return Vec::new(); }

    let sr_f = sr as f32;
    let n = chirp_frames as usize;

    // NOTE: h.fract() is always 0.0 because h is an integer-valued f32.
    // consider fixing if you actually want per-idx randomness.
    let h = (idx.wrapping_mul(0x9E3779B9) ^ 0x85EBCA6B) as f32;
    let base = 150.0 + (h.fract() * 120.0);       // ~150–270 Hz
    let span = 500.0 + (h.sin().abs() * 300.0);   // +0.5–0.8 kHz
    let nyq_limit = sr_f * 0.2;                   // keep it low
    let f0 = base.min(nyq_limit);
    let f1 = (base + span).min(nyq_limit);

    let k = (f1 / f0).ln() / dur_s;

    let mut out = vec![0.0f32; 0];
    out.reserve(n);

    for i in 0..n {
        let t = i as f32 / sr_f;

        // Handle k ~ 0 (f0 ~= f1) to avoid divide-by-zero
        let phase = if k.abs() < 1e-9 {
            2.0 * std::f32::consts::PI * f0 * t
        } else {
            2.0 * std::f32::consts::PI * f0 * ((k * t).exp() - 1.0) / k
        };

        // Hann window (use n-1 so endpoints hit 0 when n>1)
        let w = if n > 1 {
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n as f32 - 1.0)).cos())
        } else {
            1.0
        };

        out.push(phase.sin() * w * 0.3);
    }

    out
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

fn hann(n: usize, i: usize) -> f32 {
    0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n as f32)).cos())
}

fn chirp(f0: f32, f1: f32, sr: f32, dur_s: f32) -> Vec<f32> {
    let n = (dur_s * sr) as usize;
    let k = (f1 / f0).ln() / dur_s; // log sweep rate
    (0..n)
        .map(|i| {
            let t = i as f32 / sr;
            let phase = 2.0 * std::f32::consts::PI * f0 * ( (k * t).exp() - 1.0 ) / k;
            (phase).sin() * hann(n, i)
        })
        .collect()
}


/// Probe detection using GCC-PHAT to estimate lag between the captured stream and each probe.
/// Returns best (device_index, start_sample, score) per device that produced a valid match.
fn detect_probe_tones(input_mono: &[f32], num_devices: usize, empty_seconds: f32, duration_ms: f32, sample_rate: u32) -> Vec<(usize, i64, f32)> {
    let mut results = Vec::new();
    if num_devices == 0 || input_mono.is_empty() {
        return results;
    }

    for device in 0..num_devices {
        let probe = generate_probe_tone_for_device(device, empty_seconds, duration_ms, sample_rate);
        if probe.is_empty() || probe.len() > input_mono.len() {
            continue;
        }
        // Pad probe to match captured length for GCC-PHAT.
        let mut probe_padded = vec![0.0f32; input_mono.len()];
        probe_padded[..probe.len()].copy_from_slice(&probe);
        let (lag, score) = gcc_phat_delay(&input_mono, &probe_padded);
            // positive lag means probe leads capture; lag is the start index in the capture
        results.push((device, lag as i64, score));
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
fn normalize(x: &[f32]) -> Vec<f32> {
    let peak = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if peak == 0.0 { return x.to_vec(); }
    let gain = 1.0 / peak;
    x.iter().map(|&v| v * gain).collect()
}

pub fn gcc_phat_delay(x_in: &[f32], y_in: &[f32]) -> (isize, f32) {
    assert_eq!(x_in.len(), y_in.len());
    let n = x_in.len();
    if n == 0 { return (0, 0.0); }

    let x = normalize(x_in);
    let y = normalize(y_in);

    let nfft = (2 * n).next_power_of_two();

    let mut x_vec = vec![Complex::<f32>::new(0.0, 0.0); nfft];
    let mut y_vec = vec![Complex::<f32>::new(0.0, 0.0); nfft];
    for i in 0..n {
        x_vec[i].re = x[i];
        y_vec[i].re = y[i];
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(nfft);
    fft.process(&mut x_vec);
    fft.process(&mut y_vec);

    let eps = 1e-12f32;
    let mut psi = vec![Complex::<f32>::new(0.0, 0.0); nfft];
    for k in 0..nfft {
        let g = x_vec[k] * y_vec[k].conj();
        psi[k] = g / (g.norm() + eps); // PHAT
    }

    let ifft = planner.plan_fft_inverse(nfft);
    ifft.process(&mut psi);

    // Normalize inverse FFT (rustfft inverse is unnormalized)
    let scale = 1.0 / (nfft as f32);
    for v in &mut psi { *v *= scale; }

    // Search lags only in [-(n-1), +(n-1)]
    let max_lag = (n - 1) as isize;

    // Find best peak
    let mut best_lag = 0isize;
    let mut best_val = -f32::INFINITY;

    for i in 0..nfft {
        let lag = if i <= nfft / 2 { i as isize } else { i as isize - nfft as isize };
        if lag < -max_lag || lag > max_lag { continue; }

        let v = psi[i].re.abs(); // abs is safer than raw re
        if v > best_val {
            best_val = v;
            best_lag = lag;
        }
    }

    let peak = best_val;

    // Estimate "floor" away from the main peak: RMS excluding a guard band
    let guard = 2isize; // exclude +/- 2 samples around peak (tweak as needed)
    let mut sum_sq = 0.0f32;
    let mut count = 0usize;

    for i in 0..nfft {
        let lag = if i <= nfft / 2 { i as isize } else { i as isize - nfft as isize };
        if lag < -max_lag || lag > max_lag { continue; }
        if (lag - best_lag).abs() <= guard { continue; }

        let v = psi[i].re.abs();
        sum_sq += v * v;
        count += 1;
    }

    let rms_floor = if count > 0 { (sum_sq / count as f32).sqrt() } else { 0.0 };
    let score = peak / (rms_floor + 1e-12);

    // Convert FFT index lag to an intuitive offset: positive means y starts after x
    let offset_samples = -best_lag;

    (offset_samples, score)
}

/// Estimates delay (in samples) between x and y using GCC‑PHAT.
/// Assumes x.len() == y.len() and power-of-two length for simplicity.
fn gcc_phat_delay_old_2(x_in: &[f32], y_in: &[f32]) -> isize {
    let x = normalize(x_in);
    let y = normalize(y_in);
    let n = x.len();
    // Prepare complex buffers
    let mut x_vec: Vec<Complex<f32>> = x.iter().map(|&v| Complex::new(v, 0.0)).collect();
    let mut y_vec: Vec<Complex<f32>> = y.iter().map(|&v| Complex::new(v, 0.0)).collect();

    // Forward FFT
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut x_vec);
    fft.process(&mut y_vec);

    // Cross-spectrum with PHAT weighting: G = X * conj(Y); Psi = G / |G|
    let mut psi: Vec<Complex<f32>> = x_vec.iter().zip(y_vec.iter()).map(|(&xk, &yk)| {
        let g = xk * yk.conj();
        let mag = (g.re * g.re + g.im * g.im).sqrt();
        if mag > 1e-12 { g / mag } else { Complex::new(0.0, 0.0) }
    }).collect();

    // Inverse FFT to get correlation-like function
    let ifft = planner.plan_fft_inverse(n);
    ifft.process(&mut psi);

    // Take the peak: unwrap indices so delays near end map to negative lags
    let mut max_idx = 0;
    let mut max_val = psi[0].re;
    for (i, v) in psi.iter().enumerate() {
        if v.re > max_val {
            max_val = v.re;
            max_idx = i;
        }
    }
    let half = n / 2;
    if max_idx > half {
        (max_idx as isize) - n as isize // negative lag
    } else {
        max_idx as isize                 // positive/zero lag
    }
}

/// GCC-PHAT delay estimator between two real signals.
/// Returns (lag_in_samples, score), where positive lag means `sigb` leads `siga`.
fn gcc_phat_delay_old(siga: &[f32], sigb: &[f32], margin: usize) -> Option<(i64, f32)> {

    let len_a = siga.len();
    let len_b = sigb.len();
    if len_a == 0 || len_b == 0 {
        return None;
    }

    // Pad to the linear convolution length to avoid circular wrap-around.
    let conv_len = len_a + len_b - 1;
    let n_fft = conv_len.next_power_of_two().max(1);

    let mut a: Vec<Complex32> = vec![Complex32::ZERO; n_fft];
    let mut b: Vec<Complex32> = vec![Complex32::ZERO; n_fft];
    for (i, &v) in siga.iter().enumerate() {
        a[i].re = v;
    }
    for (i, &v) in sigb.iter().enumerate() {
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
    let corr: Vec<f32> = a.iter().map(|c| c.re * scale).collect();

    // fftshift so zero-lag is centered
    let mut shifted = vec![0.0f32; n_fft];
    let mid = n_fft / 2;
    for i in 0..n_fft {
        shifted[i] = corr[(i + mid) % n_fft];
    }

    let center = mid;
    // Valid lags for linear correlation are roughly ±(max(len_a, len_b) - 1).
    let max_valid_lag = len_a.max(len_b).saturating_sub(1);
    let max_margin = center.min(n_fft.saturating_sub(center + 1)).min(max_valid_lag);
    let m = margin.min(max_margin);
    let start = center.saturating_sub(m);
    let end = (center + m + 1).min(n_fft);

    let (rel_idx, &best_val) = shifted[start..end]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))?;
    let lag = (start + rel_idx) as i64 - center as i64;
    println!("lag {lag} start {start} rel_idx {rel_idx} center {center}");
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
        let micros_when_chunk_received = now_micros();

       
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
            self.input_audio_buffer_metadata_producer.try_send(metadata)?;
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
        if self.dynamic_output_sample_rate >= self.output_sample_rate {
            self.dynamic_output_sample_rate -= 1;
        }
        //self.dynamic_output_sample_rate = (((self.output_sample_rate as f32) * 0.95) as i128).max((self.dynamic_output_sample_rate-1) as i128) as u32;
        self.input_audio_buffer_consumer.set_sample_rate(self.input_sample_rate, self.dynamic_output_sample_rate)?;
        Ok(())
    }

    fn increase_dynamic_sample_rate(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.dynamic_output_sample_rate <= self.output_sample_rate {
            self.dynamic_output_sample_rate += 1;
        }
        //self.dynamic_output_sample_rate = (((self.output_sample_rate as f32) * 1.05) as i128).min((self.dynamic_output_sample_rate+1) as i128) as u32;
        self.input_audio_buffer_consumer.set_sample_rate(self.input_sample_rate, self.dynamic_output_sample_rate)?;
        Ok(())
    }

    fn handle_metadata(&mut self, num_available_frames : u64, target_emitted_input_frames : u128, calibrated: bool) -> Result<(usize, usize), Box<dyn std::error::Error>> {
        let estimated_emitted_frames = input_to_output_frames(num_available_frames as u128, self.input_sample_rate, self.dynamic_output_sample_rate);
        let updated_total_frames_emitted = self.total_emitted_frames + estimated_emitted_frames;
        let target_emitted_output_frames = input_to_output_frames(target_emitted_input_frames, self.input_sample_rate, self.output_sample_rate);
        let margin = 5;
        // dynamic adjustment to synchronize input devices to global clock:
        // don't do dynamic adjustment until after calibration, bc it's not gonna drift too much over the course of just a few seconds of calibration data
        // and that simplifies logic/prevents accumulated error during calibration
        // not enough frames, we need to increase dynamic sample rate (to get more samples)
        if updated_total_frames_emitted < target_emitted_output_frames - margin && calibrated {
            self.increase_dynamic_sample_rate()?;
            println!("Increase to {0} {updated_total_frames_emitted} {target_emitted_output_frames}", self.dynamic_output_sample_rate)
        }
        // too many frames, we need to decrease dynamic sample rate (to get less samples)
        else if updated_total_frames_emitted > target_emitted_output_frames + margin && calibrated {
            self.decrease_dynamic_sample_rate()?;
            println!("Decrease to {0} {updated_total_frames_emitted} {target_emitted_output_frames}", self.dynamic_output_sample_rate)
        }

        //// do resampling ////
        let (consumed, produced) = self.input_audio_buffer_consumer.resample(num_available_frames as u32)?;

        // the main downside of this is that it'll be persistently behind by 0.6ms or so (the resample frame size), but we'll quickly adjust for that so this shouldn't be a major issue
        // todo: think about how to fix this better (maybe current solution is as good as we can do, and it should average out to correct since past ones accumulated will result in more for this one, still, it's likely to stay behind by this amount)
        self.total_emitted_frames += (produced / self.channels) as u128;

        Ok((consumed, produced))
    }

    async fn resample(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        // process all recieved audio chunks
        match self.input_audio_buffer_metadata_consumer.next().await {
            Some(msg) => match msg {
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
                    self.finished_resampling_producer.try_send(ResamplingMetadata::Arrive(produced / self.channels, system_micros_at_start_of_packet, system_micros_after_resampled_packet_finishes as u128, calibrated))?;
                    Ok(true)
                },
                AudioBufferMetadata::Teardown() => {
                    Ok(false)
                }
            },
            None => Err("resample metadata channel disconnected".into())   // sender dropped; bail out or log
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
    async fn is_ready_to_read(&mut self, micros_packet_finished: u128, size_in_frames: usize) -> bool {
        // non blocking cause maybe it's just not ready (initialized) yet
        loop {
            match self.finished_message_reciever.try_next() {
                Ok(Some(msg)) => {
                    match msg {
                        ResamplingMetadata::Arrive(frames_recieved, _system_micros_at_start_of_packet, _system_micros_after_packet_finishes, calibrated) => {
                            self.calibrated = calibrated;
                            // this is fine to just accumulate since we don't add any more after we are done with calibration
                            self.initial_metadata.push(msg.clone());
                            self.frames_recieved += frames_recieved as u128;
                        }
                    }
                }
                // sender dropped; receiver will never get more messages
                Ok(None) => {
                    eprintln!("Error: Stream Aligner Consumer message send disconnected");
                    break;
                }
                Err(_err) => {
                    // no message available right now
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
                    let (_read_success, _samples) = self.get_chunk_to_read((additional_frames_needed * self.channels as i128) as usize).await;
                    //println!("Finished calibrate, ignoring {num_frames_that_are_behind_current_packet} frames");
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
                //println!("Finished calibrate (2), ignoring {num_frames_that_are_behind_current_packet} frames");
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
    async fn get_chunk_to_read(&mut self, size: usize) -> (bool, &[f32]) {
        // drain anything in buffer (non blocking)
        loop {
            match self.finished_message_reciever.try_next() {
                Ok(Some(_msg)) => {},
                // sender dropped; receiver will never get more messages
                Ok(None) => {
                    eprintln!("Error: Stream Aligner Consumer message send disconnected");
                    break;
                }
                Err(_err) => {
                    // no message available right now
                    break;
                }
            }
        }

        while self.final_audio_buffer_consumer.available() < size {
            // wait for data to arrive
            match self.finished_message_reciever.next().await {
                Some(_data) => {
                    // this is only called after is_ready_to_read returns true (and is no longer used),
                    // so it's fine to ignore this, we don't use samples_recieved anymore
                }
                None => {
                    eprintln!("Stream aligner consumer closed closed");
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
        if let Err(err) = self.thread_message_sender.try_send(AudioBufferMetadata::Teardown()) {
            eprintln!("failed to send shutdown signal: {}", err);
        }
    }
}

// some large constant is fine
static CHANNEL_SIZE : usize = 10000;

// make (producer (recieves audio data from device), resampler (resamples input audio to target rate), consumer (contains resampled data)) for input audio alignment
fn create_stream_aligner(channels: usize, input_sample_rate: u32, output_sample_rate: u32, history_len: usize, calibration_packets: u32, audio_buffer_seconds: u32, resampler_quality: i32) -> Result<(StreamAlignerProducer, StreamAlignerResampler, StreamAlignerConsumer), Box<dyn Error>> {
    let (input_audio_buffer_producer, input_audio_buffer_consumer) = HeapRb::<f32>::new((audio_buffer_seconds * input_sample_rate * (channels as u32)) as usize).split();
    let (input_audio_buffer_metadata_producer, input_audio_buffer_metadata_consumer) = mpsc::channel::<AudioBufferMetadata>(CHANNEL_SIZE);
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

    let (finished_resampling_producer, finished_resampling_consumer) = mpsc::channel::<ResamplingMetadata>(CHANNEL_SIZE);

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

#[cfg(not(target_arch = "wasm32"))]
fn spawn_resampler_loop(mut resampler: StreamAlignerResampler) {
    thread::spawn(move || {
        block_on(async move {
            loop {
                match resampler.resample().await {
                    Ok(true) => continue,
                    Ok(false) => break,
                    Err(err) => {
                        eprintln!("resampler error: {err}");
                        break;
                    }
                }
            }
        });
    });
}

#[cfg(target_arch = "wasm32")]
fn spawn_resampler_loop(mut resampler: StreamAlignerResampler) {
    spawn_local(async move {
        loop {
            match resampler.resample().await {
                Ok(true) => {}
                Ok(false) => break,
                Err(err) => {
                    eprintln!("resampler error: {err}");
                    break;
                }
            }
        }
    });
}

type StreamId = u64;

#[cfg(not(target_arch = "wasm32"))]
type InputStream = Stream;

#[cfg(target_arch = "wasm32")]
type InputStream = WasmStream;

#[cfg(not(target_arch = "wasm32"))]
type InputDevice = Device;

#[cfg(target_arch = "wasm32")]
type InputDevice = InputDeviceInfo;

enum OutputStreamMessage {
    Add(StreamId, u32, usize, HashMap<usize, Vec<usize>>, ResampledBufferedCircularProducer, ringbuf::HeapCons<f32>),
    SetStartTime(StreamId, u128),
    Remove(StreamId),
    InterruptAll(),
}

pub struct StreamProducer {
    stream_id: StreamId,
    producer: HeapProd<f32>,
    stream_message_sender: mpsc::Sender<OutputStreamMessage>,
    pub first_queue_time: Option<u128>,
}

impl StreamProducer {
    fn new(stream_id: StreamId, producer: HeapProd<f32>, stream_message_sender: mpsc::Sender<OutputStreamMessage>) -> Self {
        Self {
            stream_id,
            producer: producer,
            stream_message_sender: stream_message_sender,
            first_queue_time: None,
        }
    }

    pub fn stream_id(&self) -> StreamId {
        self.stream_id
    }

    pub fn queue_audio(&mut self, audio_data: &[f32]) -> Result<(), Box<dyn Error>> {
        if let None = self.first_queue_time {
            let start_time = now_micros();
            self.first_queue_time = Some(start_time);
            self.stream_message_sender.try_send(OutputStreamMessage::SetStartTime(self.stream_id,start_time))?;
        }
        let num_pushed = self.producer.push_slice(audio_data);
        if num_pushed < audio_data.len() {
            eprintln!("Error: output audio buffer got behind, try increasing buffer size");
        }
        Ok(())
    }
    pub fn num_queued_samples(&self) -> usize {
        return self.producer.occupied_len();
    }
}

// we only have one per device (instead of one per channel)
// because that ensures that multi-channel audio is synchronized properly
// when sent to output device
pub struct OutputStreamAlignerProducer {
    pub host_id: cpal::HostId,
    pub device_name: String,
    pub channels: usize,
    pub device_sample_rate: u32,
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

    pub fn begin_audio_stream(&mut self, channels: usize, channel_map: HashMap<usize, Vec<usize>>, audio_buffer_seconds: u32, sample_rate: u32, resampler_quality: i32) -> Result<StreamProducer, Box<dyn Error>> {
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

        self.output_stream_sender.try_send(OutputStreamMessage::Add(stream_index, sample_rate, channels, channel_map, resampled_producer, resampled_consumer))?;
        Ok(StreamProducer::new(stream_index, producer, self.output_stream_sender.clone()))
    }

    pub fn end_audio_stream(&mut self, stream: &StreamProducer) -> Result<(), Box<dyn Error>> {
        self.output_stream_sender.try_send(OutputStreamMessage::Remove(stream.stream_id()))?;
        Ok(())
    }

    pub fn interrupt_all_streams(&mut self) -> Result<(), Box<dyn Error>> { 
        self.output_stream_sender.try_send(OutputStreamMessage::InterruptAll())?;
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
    stream_consumers: HashMap<StreamId, (u32, usize, HashMap<usize, Vec<usize>>, ResampledBufferedCircularProducer, BufferedCircularConsumer<f32>, Option<u128>)>,
    output_stream_receiver: mpsc::Receiver<OutputStreamMessage>,
}

// allows for playing audio on top of each other (mixing) or just appending to buffer
impl OutputStreamAlignerMixer {
    fn new(channels: usize, device_sample_rate: u32, output_sample_rate: u32, frame_size: u32, output_stream_receiver:  mpsc::Receiver<OutputStreamMessage>, device_audio_producer: HeapProd<f32>, resampled_audio_buffer_producer: StreamAlignerProducer) -> Result<Self, Box<dyn Error>>  {
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
        let time_at_end_of_chunk = now_micros();
        // fetch new audio consumers, non-blocking
        loop {
            match self.output_stream_receiver.try_next() {
                Ok(Some(msg)) => match msg {
                    OutputStreamMessage::Add(id, input_sample_rate, channels, channel_map, resampled_producer, resampled_consumer) => {
                        self.stream_consumers.insert(id, (input_sample_rate, channels, channel_map, resampled_producer, BufferedCircularConsumer::new(resampled_consumer), None));
                    }
                    OutputStreamMessage::SetStartTime(id, start_time) => {
                        if let Some((_, _, _, _, _, last_start_time)) = self.stream_consumers.get_mut(&id) {
                            *last_start_time = Some(start_time);
                        }
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
                // sender dropped; receiver will never get more messages
                Ok(None) => {
                    eprintln!("Error: Mix audio stream message disconnected");
                    break;
                }
                Err(_err) => {
                    // no message available right now
                    break;
                }
            }
        }

        // sources of latency
        // physical output device latency
        //   - chunk size 512
        //   - say, we send x data at position 490
        //   - we have frame size of 16
        //   - so we don't add to chunk until 496
        //   - then we don't actually play for 496ms
        //   -   new   we won't play for 22ms+device latency
        //   -   + physical device latency
        //   - wheres if we send x data at position 20
        //   - then we don't play for 32
        //   -   + physical device latency
        //   -  then we won't play for 492+device latency
        // device buffer latency
        // 



        let (need_to_write_device_values, device_buf_write) = self.device_audio_producer.get_chunk_to_write(input_chunk_size * self.channels);
        let input_chunk_size_available = device_buf_write.len() / (self.channels);
        device_buf_write.fill(0.0);
        
        for (_stream_id, (input_sample_rate, channels, channel_map, resample_producer, resample_consumer, start_time)) in self.stream_consumers.iter_mut() {
            let skip_ahead_frames = if let Some(start_time) = *start_time {
                // this is the amount of time until it is played (+ system audio)
                let diff = (time_at_end_of_chunk as i128 - start_time as i128).max(0);
                let diff_frames = micros_to_frames(diff as u128, self.device_sample_rate as u128);
                if diff_frames <= input_chunk_size_available as u128 {
                    // we want to delay inital audio to ensure it is always the same latency

                    
                    // sources of latency
                    // physical output device latency
                    //   - chunk size 512
                    //   - say, we send x data at position 490
                    //      it is not recieved until next chunk start
                    //      at which point it plays right at 0
                    //      so latency is 512-490 + device latency
                    //   - say we send x data at position 12
                    //      it is not recieved until next chunk start
                    //      at which point it plays right at 0
                    //      so latency is 512-12 + device latency
                    //   we always want 512 latency (as that is worst case)
                    //   we know 512-490 or 512-12 values
                    //   diff_frames is amount of latency we already have
                    //   so like these ones
                    //   We want 512-490 = 22
                    //           512-12  = 500
                    //   We need to add additional latency until it sums to 512
                    //        so 22 + x = 512
                    //           500 + x = 512
                    //           we just do 512 - diff
                    //   so we need to add 
                    let skip_ahead_frames = input_chunk_size_available - (diff_frames as usize);
                    //      so total latency is 
                    println!("end of chunk {time_at_end_of_chunk} start time {start_time} diff {diff} diff frames {diff_frames}");
                    skip_ahead_frames
                }
                else {
                    0
                }
            } else {
                0
            };
            let requested_frames = input_chunk_size_available - skip_ahead_frames;
            //println!("{requested_frames} {input_chunk_size_available} {skip_ahead_frames}");
            let target_input_samples = input_to_output_frames(requested_frames as u128, self.device_sample_rate, *input_sample_rate);
            // this doesn't work because it'll stall for very large audio
            // resample_producer.resample_all()?; // do resampling of any available data
            // instead, do it streaming
            // todo: there's a bug where the final 5-10ms of audio is cutoff because doesn't have any empty end data for the chunk
            resample_producer.resample((target_input_samples as u32) * 2)?; // do * 2 so we also grab some leftovers if there are some, this is an upper bound
            let buf_from_stream = resample_consumer.get_chunk_to_read((requested_frames * (*channels)) as usize);
            let frames = (buf_from_stream.len() / *channels as usize).min(requested_frames);
            if frames == 0 {
                continue;
            }

            let dst_stride = self.channels;
            let src_stride = *channels;
            // map virtual channels to real channels via channel_map
            for (s_idx, dst_chs) in channel_map.iter() {
                for dst_ch in dst_chs.iter() {
                    if *dst_ch >= self.channels { continue; } // guard bad maps
                    // skip ahead (insert empty silence) to ensure consistent delay
                    let mut dst = *dst_ch as usize + skip_ahead_frames*self.channels;
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
        self.resampled_audio_buffer_producer.process_chunk(device_buf_write)?;
        // finish writing to output device buffer
        self.device_audio_producer.finish_write(need_to_write_device_values, input_chunk_size_available * self.channels);
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputDeviceConfig {
    pub host_id: cpal::HostId,
    pub device_name: String,
    pub channels: usize,
    pub sample_rate: u32,
    pub sample_format: SampleFormat,
    
    // number of audio chunks to hold in memory, for aligning input devices's values when dropped frames/clock offsets. 100 or so is fine
    pub history_len: usize,
    // number of packets recieved before we start getting audio data
    // a larger value here will take longer to connect, but result in more accurate timing alignments
    pub calibration_packets: u32,
    // how long buffer of input audio to store, should only really need a few seconds as things are mostly streamed
    pub audio_buffer_seconds: u32,
    pub resampler_quality: i32
}

impl InputDeviceConfig {
    pub fn new(
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
    pub async fn from_default(
        host_id: cpal::HostId,
        device_name: String,
        history_len: usize,
        calibration_packets: u32,
        audio_buffer_seconds: u32,
        resampler_quality: i32,
    ) -> Result<Self, Box<dyn Error>> {
        let default_config = get_default_input_device_config(&host_id, &device_name).await?;

        Ok(Self::new(
            host_id,
            device_name,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputDeviceConfig {
    pub host_id: cpal::HostId,
    pub device_name: String,
    pub channels: usize,
    pub sample_rate: u32,
    pub sample_format: SampleFormat,
    
    // number of audio chunks to hold in memory, for aligning input devices's values when dropped frames/clock offsets. 100 or so is fine
    pub history_len: usize,
    // number of packets recieved before we start getting audio data
    // a larger value here will take longer to connect, but result in more accurate timing alignments
    pub calibration_packets: u32,
    // how long buffer of input audio to store, should only really need a few seconds as things are mostly streamed
    pub audio_buffer_seconds: u32,
    pub resampler_quality: i32,
    // frame size (in terms of samples) should be small, on the order of 1-2ms or less.
    // otherwise you may get skipping if you do not provide audio via queue_audio fast enough
    // larger frame sizes will also prevent immediate interruption, as interruption can only happen between each frame
    pub frame_size: u32,
}

impl OutputDeviceConfig {
    pub fn new(
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
    pub async fn from_default(
        host_id: cpal::HostId,
        device_name: String,
        history_len: usize,
        calibration_packets: u32,
        audio_buffer_seconds: u32,
        resampler_quality: i32,
        frame_size: u32,
    ) -> Result<Self, Box<dyn Error>> {
        let host = cpal::host_from_id(host_id)?;
        let output_device = select_device(host.output_devices(), &device_name, "Output")?;
        let default_config = output_device.default_output_config()?;
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

pub struct AecConfig {
    target_sample_rate: u32,
    frame_size: usize,
    filter_length: usize
}

impl AecConfig {
    pub fn new(target_sample_rate: u32, frame_size: usize, filter_length: usize) -> Self {
        Self { target_sample_rate, frame_size, filter_length }
    }
}

async fn get_input_stream_aligners(device_config: &InputDeviceConfig, aec_config: &AecConfig) -> Result<(InputStream, StreamAlignerConsumer), Box<dyn std::error::Error>>  {

    // we need to use these methods instead of the more generic select_device because of wasm wrapping to workaround cpal not having webaudio input device support
    let device = select_input_device(
        &device_config.host_id,
        &device_config.device_name
    ).await?;

    let supported_config = find_matching_input_device_config(
        &device,
        &device_config.device_name,
        device_config.channels,
        device_config.sample_rate,
        device_config.sample_format
    ).await?;
    

    let (producer, resampler, consumer) = create_stream_aligner(
        device_config.channels,
        device_config.sample_rate,
        aec_config.target_sample_rate,
        device_config.history_len,
        device_config.calibration_packets,
        device_config.audio_buffer_seconds,
        device_config.resampler_quality)?;

    spawn_resampler_loop(resampler);

    let stream = build_input_alignment_stream(
        &device,
        device_config,
        supported_config,
        producer,
    ).await?;

    // start input stream
    cfg_if::cfg_if! {
        if #[cfg(all(target_arch = "wasm32"))] {
            stream.play().await?; // wasm is async
        } else {
            stream.play()?;
        }
    }

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
    let (output_stream_sender, output_stream_receiver) = mpsc::channel::<OutputStreamMessage>(CHANNEL_SIZE);

    let output_producer = OutputStreamAlignerProducer::new(
        device_config.host_id,
        device_config.device_name.clone(),
        device_config.channels, // channels
        device_config.sample_rate, // device_sample_rate
        output_stream_sender
    );

    let (producer, resampler, consumer) = create_stream_aligner(
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
        device_config.frame_size,
        output_stream_receiver,
        device_audio_producer,
        producer,
    )?;
    

    spawn_resampler_loop(resampler);

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
    AddInputDevice(String, InputStream, StreamAlignerConsumer),
    RemoveInputDevice(String),
    AddOutputDevice(String, Stream, StreamAlignerConsumer),
    RemoveOutputDevice(String)
}

const VAD_FRAME_SIZE : usize = 256;

pub struct AecStream {
    //aec: Option<EchoCanceller>,
    aec_config: AecConfig,
    device_update_sender: mpsc::Sender<DeviceUpdateMessage>,
    device_update_receiver: mpsc::Receiver<DeviceUpdateMessage>,
    input_streams: HashMap<String, InputStream>,
    output_streams: HashMap<String, Stream>,
    input_aligners: HashMap<String, StreamAlignerConsumer>,
    input_aligners_in_progress: HashMap<String, StreamAlignerConsumer>,
    output_aligners: HashMap<String, StreamAlignerConsumer>,
    output_aligners_in_progress: HashMap<String, StreamAlignerConsumer>,
    sorted_input_aligners: Vec<String>,
    sorted_output_aligners: Vec<String>,
    input_channels: usize,
    output_channels: usize,
    input_gain: f32,
    vads: Vec<VoiceActivityDetector>,
    start_micros: Option<u128>,
    total_frames_emitted: u128,
    input_audio_buffer: Vec<f32>,
    output_audio_buffer: Vec<f32>,
    aec_audio_buffer: Vec<f32>,
    vad_buffer: Vec<i16>,
    vad_input_buffer_prod: BufferedCircularProducer::<f32>,
    vad_input_buffer_cons: BufferedCircularConsumer::<f32>,
    vad_output_buffer_prod: BufferedCircularProducer::<f32>,
    vad_output_buffer_cons: BufferedCircularConsumer::<f32>,
    vad_aec_buffer_prod: BufferedCircularProducer::<f32>,
    vad_aec_buffer_cons: BufferedCircularConsumer::<f32>,
    vad_input_buffer: Vec<f32>,
    vad_output_buffer: Vec<f32>,
    vad_aec_buffer: Vec<f32>,

    aec3: Option<VoipAec3>,
}

// AecStream is used behind a Mutex to serialize access; mark it Send so it can cross threads.
unsafe impl Send for AecStream {}

impl AecStream {
    pub fn new(
        aec_config: AecConfig
    ) -> Result<Self, Box<dyn Error>> {
        if aec_config.target_sample_rate == 0 {
            return Err(format!("Target sample rate is {}, it must be greater than zero.", aec_config.target_sample_rate).into());
        }
        let vad_buf_size = (VAD_FRAME_SIZE*128).max(aec_config.frame_size * 128);
        let (vad_input_prod, vad_input_cons) = HeapRb::<f32>::new(vad_buf_size).split();
        let (vad_output_prod, vad_output_cons) = HeapRb::<f32>::new(vad_buf_size).split();
        let (vad_aec_prod, vad_aec_cons) = HeapRb::<f32>::new(vad_buf_size).split();
        let (device_update_sender, device_update_receiver) = mpsc::channel::<DeviceUpdateMessage>(CHANNEL_SIZE);
        Ok(Self {
           //aec: None,
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
           input_gain: 1f32,
           input_channels: 0,
           output_channels: 0,
           start_micros: None,
           total_frames_emitted: 0,
           input_audio_buffer: Vec::new(),
           output_audio_buffer: Vec::new(),
           aec_audio_buffer: Vec::new(),
           vad_buffer: Vec::new(),
           vad_input_buffer: Vec::new(),
           vad_output_buffer: Vec::new(),
           vad_aec_buffer: Vec::new(),
           vad_input_buffer_prod: BufferedCircularProducer::<f32>::new(vad_input_prod),
           vad_input_buffer_cons: BufferedCircularConsumer::<f32>::new(vad_input_cons),
           vad_output_buffer_prod: BufferedCircularProducer::<f32>::new(vad_output_prod),
           vad_output_buffer_cons: BufferedCircularConsumer::<f32>::new(vad_output_cons),
           vad_aec_buffer_prod: BufferedCircularProducer::<f32>::new(vad_aec_prod),
           vad_aec_buffer_cons: BufferedCircularConsumer::<f32>::new(vad_aec_cons),
           aec3: None,
           vads: Vec::new(),
        })
    }
    
    pub fn num_input_channels(&self) -> usize {
        self.input_aligners
            .values()
            .map(|aligner| aligner.channels)
            .sum()
    }

    pub fn num_output_channels(&self) -> usize {
        self.output_aligners
            .values()
            .map(|aligner| aligner.channels)
            .sum()
    }

    pub fn input_gain(&self) -> f32 {
        self.input_gain
    }

    pub fn set_input_gain(&mut self, gain: f32) {
        self.input_gain = gain;
    }

    fn reinitialize_aec(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.input_channels = self.num_input_channels();
        self.output_channels = self.num_output_channels();

        // store a consistent ordering
        self.sorted_input_aligners = self.input_aligners.keys().cloned().collect();
        self.sorted_input_aligners.sort();

        self.sorted_output_aligners = self.output_aligners.keys().cloned().collect();
        self.sorted_output_aligners.sort();

        //self.aec2 = Some(FdafAec::new(1024, 0.02));

        
        /*(self.aec, self.aec3) = if self.input_channels > 0 && self.output_channels > 0 {
            (EchoCanceller::new_multichannel(
                self.aec_config.frame_size,
                self.aec_config.filter_length,
                self.input_channels,
                self.output_channels,
            ), Some(VoipAec3::builder(self.aec_config.target_sample_rate as i32, self.input_channels, self.output_channels)
            .initial_delay_ms((self.aec_config.frame_size/3) as i32)
            .enable_high_pass(true)
            .build()
            .expect("failed to create AEC pipeline")))

            //EchoCanceller::new(
            //    self.aec_config.frame_size,
            //    self.aec_config.filter_length
            //)

        */
        self.aec3 = if self.input_channels > 0 && self.output_channels > 0 {
            println!("Making aec with {} inputs and {} outputs", self.input_channels, self.output_channels);
            Some(VoipAec3::builder(self.aec_config.target_sample_rate as i32, self.output_channels, self.input_channels)
            .initial_delay_ms((self.aec_config.frame_size/3) as i32)
            .enable_high_pass(true)
            .build()
            .expect("failed to create AEC pipeline"))
        } else {
            None
        };

        //if let Some(aec) = self.aec.as_mut() {
        //    aec.set_sampling_rate(self.aec_config.target_sample_rate);
        //    let sampling_rate = aec.sampling_rate();
        //    println!("Set sampling rate to {sampling_rate}");
        //}

        self.input_audio_buffer.clear();
        self.input_audio_buffer.resize(self.aec_config.frame_size * self.input_channels, 0 as f32);
        self.output_audio_buffer.clear();
        self.output_audio_buffer.resize(self.aec_config.frame_size * self.output_channels, 0 as f32);
        self.aec_audio_buffer.clear();
        self.aec_audio_buffer.resize(self.aec_config.frame_size * self.input_channels, 0 as f32);
        self.vads = (0..self.input_channels)
            .map(|_| VoiceActivityDetector::new(VoiceActivityProfile::AGGRESSIVE))
            .collect();
        Ok(())
    }

    pub async fn add_input_device(&mut self, config: &InputDeviceConfig) -> Result<(), Box<dyn std::error::Error>> {
        let (stream, aligners) = get_input_stream_aligners(config, &self.aec_config).await?;
        self.device_update_sender.try_send(DeviceUpdateMessage::AddInputDevice(config.device_name.clone(), stream, aligners))?;
        Ok(())
    }

    pub async fn add_output_device(&mut self, config: &OutputDeviceConfig) -> Result<OutputStreamAlignerProducer, Box<dyn std::error::Error>> {
        let (stream, producer, consumer) = get_output_stream_aligners(config, &self.aec_config)?;
        self.device_update_sender.try_send(DeviceUpdateMessage::AddOutputDevice(config.device_name.clone(), stream, consumer))?;
        Ok(producer)
    }

    pub fn remove_input_device(&mut self, config: &InputDeviceConfig) -> Result<(), Box<dyn std::error::Error>> {
        self.device_update_sender.try_send(DeviceUpdateMessage::RemoveInputDevice(config.device_name.clone()))?;
        Ok(())
    }

    pub fn remove_output_device(&mut self, config: &OutputDeviceConfig) -> Result<(), Box<dyn std::error::Error>> {
        self.device_update_sender.try_send(DeviceUpdateMessage::RemoveOutputDevice(config.device_name.clone()))?;
        Ok(())
    }

    pub async fn calibrate(&mut self, output_producers: &mut [OutputStreamAlignerProducer], debug_wav: bool) -> Result<(), Box<dyn std::error::Error>> {
        self.calibrate_inner(output_producers, debug_wav).await?;
        self.calibrate_inner(output_producers, debug_wav).await?;
        self.calibrate_inner(output_producers, debug_wav).await
    }

    pub async fn calibrate_inner(&mut self, output_producers: &mut [OutputStreamAlignerProducer], debug_wav: bool) -> Result<(), Box<dyn std::error::Error>> {
        let (output_offsets, input_offsets) = self.get_calibration_offsets(output_producers, debug_wav).await?;
        // we need to throw away some samples for each device until we are calibrated
        // each device will have an offset (could be negative)
        
        let mut input_shifts_needed = Vec::new();
        for input_index in 0..input_offsets.len() {
            let mut shifts_needed = Vec::new();
            for output_index in 0..output_offsets.len() {
                if let Some(output_offset) = output_offsets[output_index] {
                    if let Some(input_offset) = input_offsets[input_index][output_index] {
                        let shift_needed = input_offset - output_offset;
                        shifts_needed.push(shift_needed);
                    }
                }
            }
            // take the min of the shift needed for each device (we don't ever want it to occur before the device)
            let shift_needed = if let Some(min_val) = shifts_needed.iter().min() {
                (*min_val as i64) - ((self.aec_config.frame_size/3) as i64) // ime this offset is better than perfectly aligned, and better than /2
            } else {
                0
            };
            input_shifts_needed.push(shift_needed);
        }
        
        let min_input_shift_needed = input_shifts_needed.iter().copied().min();
        // if it is less than zero, it needs to be shifted forwards not backwards (it's currently playing before an output audio device)
        // to do this, we'll need to move all the output devices forward by that much
        // and then since we did that, we'll subtract that amount from our shifts needed
        if let Some(min_input_shift_needed) = min_input_shift_needed {
            if min_input_shift_needed < 0 {
                // if we needed to move the input to the left A amount
                // now, outputs will be moved to the left min_input_shift_needed
                // so we need to move the input a total of A-min_input_shift_needed amount (because min_input_shift_needed is negative)
                // this will also ensure that all input_shifts_needed are positive now
                for s in &mut input_shifts_needed {
                    *s += min_input_shift_needed.unsigned_abs() as i64;  // min_shift is negative, so this adds abs(min_shift)
                }
                println!("Shifting outputs ahead by {}", min_input_shift_needed);
                for output_index in 0..output_offsets.len() {
                    if let Some(aligner) = self.output_aligners.get_mut(&self.sorted_output_aligners[output_index].clone()) {
                        // skip ahead that many samples (* num channels bc it is multi channel)
                        let (_ok, chunk) = aligner.get_chunk_to_read(((-min_input_shift_needed) as usize) * aligner.channels).await;
                        let chunk_len = chunk.len();
                        aligner.finish_read(chunk_len);
                    }
                }
            }
        }
        

        for input_index in 0..input_offsets.len() {
            let shift_needed = input_shifts_needed[input_index];
            println!("Shifting input channel {} by {}",input_index, shift_needed);
            if let Some(aligner) = self.input_aligners.get_mut(&self.sorted_input_aligners[input_index].clone()) {
                // skip ahead that many samples (* num channels bc it is multi channel)
                if shift_needed > 0 {
                    let (_ok, chunk) = aligner.get_chunk_to_read((shift_needed as usize) * aligner.channels).await;
                    let chunk_len = chunk.len();
                    aligner.finish_read(chunk_len);
                }
            }
        }
        
        Ok(())
    }

    async fn get_calibration_offsets(&mut self, output_producers: &mut [OutputStreamAlignerProducer], debug_wav: bool) -> Result<(Vec<Option<i64>>, Vec<Vec<Option<i64>>>), Box<dyn std::error::Error>> {
        let sample_rate = self.aec_config.target_sample_rate as u32;
        // Probe length (~0.1s) to stay quick but audible.
        let tone_ms = 100.0;
        let capture_secs = 3.0;

        // 1) Emit a distinct probe on each output device (all channels), in sorted output order.
        let mut active_streams: Vec<(usize, StreamProducer)> = Vec::new();
        let mut tones = Vec::new();

        for (idx, dev_name) in self.sorted_output_aligners.clone().iter().enumerate() {
            let Some(producer) = output_producers
                .iter_mut()
                .find(|p| p.device_name == *dev_name) else {
                eprintln!("calibrate: no output producer found for '{dev_name}'");
                continue;
            };
            let tone_mono = generate_probe_tone_for_device(idx, capture_secs/2.0, tone_ms, sample_rate);
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
            let stream = producer.begin_audio_stream(
                1, // 1 channel
                channel_map,
                2, // seconds of buffer for this probe
                self.aec_config.target_sample_rate,
                5, // resampler quality
            )?;
            tones.push(tone_mono);
            active_streams.push((idx, stream));
        }

        for ((_, stream), tone) in active_streams.iter_mut().zip(tones.iter()) {
            stream.queue_audio(tone.as_slice())?;
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
            let (input_slices, output_slices, _aec_out, start_time, end_time) =
                self.update_debug().await?;
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
        for (idx, stream) in active_streams.iter() {
            if let Some(producer) = output_producers.get_mut(*idx) {
                producer.end_audio_stream(stream)?;
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
        let mut output_offsets: Vec<Option<i64>> = vec![None; output_channel_ranges.len()];
        // mapping of input device -> (output device index, output device index offset)
        let mut input_offsets: Vec<Vec<Option<i64>>> = vec![vec![None; output_channel_ranges.len()]; input_channel_ranges.len()];

        if !captured_outputs.is_empty() {
            for (dev_idx, buf) in captured_outputs.iter().enumerate() {
                let detections = detect_probe_tones(buf, output_producers.len(), capture_secs/2.0, tone_ms, sample_rate);
                let mut start_for_dev: Option<(i64, f32)> = None;
                for (d_idx, start, score) in detections {
                    if d_idx == dev_idx { // just detect this device, no others should show up since it just forwards the data
                        start_for_dev = Some((start, score));
                        break;
                    }
                }
                if let Some((start, score)) = start_for_dev {
                    let in_seconds = (start as f32) / (sample_rate as f32);
                    println!("Output {dev_idx} has offset {start} {in_seconds} with score {score}");
                    output_offsets[dev_idx] = Some(start);
                } else {
                    eprintln!("No probe detected for output device index {dev_idx}");
                }
            }
            for (input_idx, buf) in captured_inputs.iter().enumerate() {
                let detections = detect_probe_tones(buf, output_producers.len(), capture_secs/2.0, tone_ms, sample_rate);
                for (output_idx, start, score) in detections {
                    let in_seconds = (start as f32) / (sample_rate as f32);
                    println!("Output {output_idx} -> Input {input_idx} has offset {start} {in_seconds} with score {score}");
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

    pub async fn update_debug_vad(&mut self) -> Result<(&[f32], &[f32], &[f32], u128, u128, Vec<bool>), Box<dyn std::error::Error>> {
        let (chunk_start_micros, chunk_end_micros) = self.update_devices().await?; // needs to be done before everything, so num channels doen't change between update_helper calls

        // this has 3 circular buffers of size min(self.aec_config.frame_size*4, VAD_FRAME_SIZE*2) that hold the input_audio_buffer, output_audio_buffer, and aec_outputs from update_debug
        // update_debug is called multiple times until we have at least VAD_FRAME_SIZE samples, at which point the vad is called (one for each channel) and then we send the results
        let mut latest_end_time: u128 = 0;

        let first = true;
        while self.input_channels > 0 && self.output_channels > 0 && (self.vad_input_buffer_cons.available() / self.input_channels) < VAD_FRAME_SIZE {
            let (chunk_start_micros, chunk_end_micros) = if first {
                (chunk_start_micros, chunk_end_micros)
            } else {
                self.get_next_frame()
            };
            let (_start, end) = {
                let (input, start, end) = self.update_helper(chunk_start_micros, chunk_end_micros).await?;
                if input.is_empty() {
                    break;
                }
                else {
                    (start, end)
                }
            };
            latest_end_time = end;
            let (input, output, aec) = (self.input_audio_buffer.as_slice(), self.output_audio_buffer.as_slice(), self.aec_audio_buffer.as_slice());
            let (input_write, input_buf) = self.vad_input_buffer_prod.get_chunk_to_write(input.len());
            let (output_write, output_buf) = self.vad_output_buffer_prod.get_chunk_to_write(output.len());
            let (aec_write, aec_buf) = self.vad_aec_buffer_prod.get_chunk_to_write(aec.len());
            input_buf.copy_from_slice(input);
            output_buf.copy_from_slice(output);
            aec_buf.copy_from_slice(aec);
            self.vad_input_buffer_prod.finish_write(input_write, input.len());
            self.vad_output_buffer_prod.finish_write(output_write, output.len());
            self.vad_aec_buffer_prod.finish_write(aec_write, aec.len());
        }

        // failed, maybe we removed all input devices or something else happened, return nothing
        if self.input_channels == 0 || (self.vad_input_buffer_cons.available() / self.input_channels) < VAD_FRAME_SIZE {
            return Ok((&[], &[], &[], 0, 0, Vec::new()));
        }
        let remaining_frames = self.vad_input_buffer_cons.available() / self.input_channels;
        
        let input_buf = self.vad_input_buffer_cons.get_chunk_to_read(VAD_FRAME_SIZE*self.input_channels);
        let output_buf = self.vad_output_buffer_cons.get_chunk_to_read(VAD_FRAME_SIZE*self.output_channels);
        let aec_buf = self.vad_aec_buffer_cons.get_chunk_to_read(VAD_FRAME_SIZE*self.input_channels);
        let end_time = latest_end_time - frames_to_micros(remaining_frames as u128, self.aec_config.target_sample_rate as u128);
        let start_time = end_time - frames_to_micros(VAD_FRAME_SIZE as u128, self.aec_config.target_sample_rate as u128);

        let mut vad_scores = Vec::new();
        self.vad_buffer.clear();
        self.vad_buffer.resize(VAD_FRAME_SIZE, 0);
        for channel in 0..self.input_channels {
            for frame_idx in 0..VAD_FRAME_SIZE {
                let sample = aec_buf[frame_idx * self.input_channels + channel];
                self.vad_buffer[frame_idx] = Self::f32_to_i16(sample);
            }
            if let Some(detector) = self.vads.get_mut(channel) {
                let is_speech = detector.predict_16khz(&self.vad_buffer)?;
                vad_scores.push(is_speech);
            }
        }

        self.vad_input_buffer.clear();
        self.vad_input_buffer.extend_from_slice(input_buf);
        self.vad_output_buffer.clear();
        self.vad_output_buffer.extend_from_slice(output_buf);
        self.vad_aec_buffer.clear();
        self.vad_aec_buffer.extend_from_slice(aec_buf);
        
        self.vad_input_buffer_cons.finish_read(VAD_FRAME_SIZE*self.input_channels);
        self.vad_output_buffer_cons.finish_read(VAD_FRAME_SIZE*self.output_channels);
        self.vad_aec_buffer_cons.finish_read(VAD_FRAME_SIZE*self.input_channels);
        

        Ok((
            &self.vad_input_buffer.as_slice(),
            &self.vad_output_buffer.as_slice(),
            &self.vad_aec_buffer.as_slice(),
            start_time,
            end_time,
            vad_scores,
        ))
    }

    // calls update, but returns all involved audio buffers
    // (if needed for diagnostic reasons, usually .update() (which returns aec'd inputs) should be all you need)
    pub async fn update_debug(&mut self) -> Result<(&[f32], &[f32], &[f32], u128, u128), Box<dyn std::error::Error>> {
        let (start_time, end_time) = {
            let (_, start_time, end_time) = self.update().await?;
            (start_time, end_time)
        };
        Ok((
            self.input_audio_buffer.as_slice(),
            self.output_audio_buffer.as_slice(),
            self.aec_audio_buffer.as_slice(),
            start_time,
            end_time
        ))
    }

    async fn update_devices(&mut self) -> Result<(u128, u128), Box<dyn std::error::Error>>{
        loop {
            match self.device_update_receiver.try_next() {
                Ok(Some(msg)) => match msg {
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
                // sender dropped; receiver will never get more messages
                Ok(None) => {
                    eprintln!("Error: Aec stream update message send disconnected");
                    break;
                }
                Err(_err) => {
                    // no message available right now
                    break;
                }
            }
        }

        let (chunk_start_micros, chunk_end_micros) = self.get_next_frame();
        let chunk_size = self.aec_config.frame_size;
        
        // similarly, if we initialize an output device here
        // we may not get any audio for a little bit
        if chunk_size == 0 {
            return Ok((chunk_start_micros, chunk_end_micros));
        }
         // initialize any new aligners and align them to our frame step
        let mut modified_aligners = false;
        for key in self.input_aligners_in_progress.keys().cloned().collect::<Vec<String>>() {
            let ready = match self.input_aligners_in_progress.get_mut(&key) {
                Some(a) => a.is_ready_to_read(chunk_end_micros, chunk_size).await,
                None => false,
            };
            if ready {
                if let Some(aligner) = self.input_aligners_in_progress.remove(&key) {
                    self.input_aligners.insert(key, aligner);
                    modified_aligners = true;
                }
            }
        }
        for key in self.output_aligners_in_progress.keys().cloned().collect::<Vec<String>>() {
            let ready = match self.output_aligners_in_progress.get_mut(&key) {
                Some(a) => a.is_ready_to_read(chunk_end_micros, chunk_size).await,
                None => false,
            };
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

        Ok((chunk_start_micros, chunk_end_micros))
    }

    fn get_next_frame(&mut self) -> (u128, u128) {
        let chunk_size = self.aec_config.frame_size;
        let start_micros = if let Some(start_micros_value) = self.start_micros {
            start_micros_value
        } else {
            let start_micros_value = now_micros()
                .saturating_sub(frames_to_micros(chunk_size as u128, self.aec_config.target_sample_rate as u128));
            self.start_micros = Some(start_micros_value);
            start_micros_value
        };
        let chunk_start_micros = frames_to_micros(self.total_frames_emitted, self.aec_config.target_sample_rate as u128) + start_micros;
        let chunk_end_micros = chunk_start_micros + frames_to_micros(chunk_size as u128, self.aec_config.target_sample_rate as u128);
        self.total_frames_emitted += chunk_size as u128;
        (chunk_start_micros, chunk_end_micros)
    }

    pub async fn update(&mut self)-> Result<(&[f32], u128, u128), Box<dyn std::error::Error>> {
        let (chunk_start_micros, chunk_end_micros) = self.update_devices().await?;
        self.update_helper(chunk_start_micros, chunk_end_micros).await
    }

    async fn update_helper(&mut self, chunk_start_micros: u128, chunk_end_micros: u128) -> Result<(&[f32], u128, u128), Box<dyn std::error::Error>> {
        let chunk_size = self.aec_config.frame_size;
        
        // recieve audio data and interleave it into our buffers
        let mut input_channel = 0;
        self.input_audio_buffer.fill(0 as f32);

        for key in &self.sorted_input_aligners {
            let input_gain = self.input_gain;
            if let Some(aligner) = self.input_aligners.get_mut(key) {
                let channels = aligner.channels;
                let needed = chunk_size * channels;
                let (ok, chunk) = aligner.get_chunk_to_read(needed).await;
                let frames = chunk.len() / channels;

                if ok && frames > 0 {
                    for c in 0..channels {
                        let mut src_idx = c;
                        let mut dst = input_channel + c;
                        for _ in 0..frames {
                            self.input_audio_buffer[dst] = chunk[src_idx]*input_gain;
                            dst += self.input_channels;
                            src_idx += channels;
                        }
                    }
                }

                aligner.finish_read(frames * channels);
                input_channel += channels;
            }
        }

        if self.output_channels == 0 {
            // simply pass through input_channels, no need for aec
            self.aec_audio_buffer.copy_from_slice(&self.input_audio_buffer);
        }
        else {
                
            let mut output_channel = 0;
            self.output_audio_buffer.fill(0 as f32);
            for key in &self.sorted_output_aligners {
                if let Some(aligner) = self.output_aligners.get_mut(key) {
                    let channels = aligner.channels;
                    let needed = chunk_size * channels;
                    let (ok, chunk) = aligner.get_chunk_to_read(needed).await;
                    let frames = chunk.len() / channels;

                    if ok && frames > 0 {
                        for c in 0..channels {
                            let mut src_idx = c;
                            let mut dst = output_channel + c;
                            for _ in 0..frames {
                                self.output_audio_buffer[dst] = chunk[src_idx];
                                dst += self.output_channels;
                                src_idx += channels;
                            }
                        }
                        aligner.finish_read(frames * channels);
                    }

                    output_channel += channels;
                }
            }

            self.aec_audio_buffer.fill(0 as f32);

            if self.input_channels != 0 {
                //let Some(aec) = self.aec.as_mut() else { 
                //    return Err("no aec".into());
                //};
                
                // skip ahead if no output, as there's nothing to cancel
                // this helps avoid needing to recalibrate every time we recieve audio
                let output_energy = Self::energy(&self.output_audio_buffer);
                let _input_energy = Self::energy(&self.input_audio_buffer);
                if output_energy < -3.0 {
                    self.aec_audio_buffer.copy_from_slice(&self.input_audio_buffer);
                }
                else {
                    ////// aec3
                    let Some(aec3_value) = self.aec3.as_mut() else {
                        return Err("No aec3".into());
                    };
                    let _metrics = aec3_value.process(&self.input_audio_buffer, Some(&self.output_audio_buffer), false, &mut self.aec_audio_buffer)?;
                }
            }
        };

        Ok((&self.aec_audio_buffer.as_slice(), chunk_start_micros, chunk_end_micros))
    }
    fn energy(buf: &[f32]) -> f64 {
        buf.iter().map(|s| ((*s)*(*s)) as f64).sum::<f64>() / buf.len() as f64
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
    


pub async fn get_supported_output_configs(
    history_len: usize,
    num_calibration_packets: u32,
    audio_buffer_seconds: u32,
    resampler_quality: i32,
    frame_size: u32) -> Result<Vec<Vec<OutputDeviceConfig>>, Box<dyn std::error::Error>>  {
    let mut configs = Vec::new();
    // need to use these wrappers to ensure wasm is treated properly
    for host_id in cpal::available_hosts() {
        for output_device_name in get_output_device_names(&host_id).await? {
            configs.push(get_supported_output_device_configs(&host_id, &output_device_name, history_len, num_calibration_packets, audio_buffer_seconds, resampler_quality, frame_size).await?);
        }
    }
    Ok(configs)
}

pub async fn get_supported_input_configs(
    history_len: usize,
    num_calibration_packets: u32,
    audio_buffer_seconds: u32,
    resampler_quality: i32) -> Result<Vec<Vec<InputDeviceConfig>>, Box<dyn std::error::Error>> {
    let mut configs = Vec::new();
    // need to use these wrappers to ensure wasm is treated properly
    for host_id in cpal::available_hosts() {
        for input_device_name in get_input_device_names(&host_id).await? {
            configs.push(get_supported_input_device_configs(&host_id, &input_device_name, history_len, num_calibration_packets, audio_buffer_seconds, resampler_quality).await?);
        }
    }
    Ok(configs)
}

fn get_host_by_name(target: &str) -> Option<Host> {
    for host_id in cpal::available_hosts() {
        if host_id.name().eq_ignore_ascii_case(target) {
            return cpal::host_from_id(host_id).ok();
        }
    }
    None
}

#[cfg(not(target_arch = "wasm32"))]
async fn select_input_device(
    host_id: &cpal::HostId,
    device_name: &str
) -> Result<InputDevice, Box<dyn Error>>
{
    let host = cpal::host_from_id(*host_id)?;
    let device = select_device(
        host.input_devices(),
        device_name,
        "Input",
    )?;
    Ok(device)
}

#[cfg(target_arch = "wasm32")]
async fn select_input_device(
    _host_id: &cpal::HostId,
    device_name: &str
) -> Result<InputDevice, Box<dyn Error>>
{
    match get_webaudio_input_devices().await {
        Ok(device_iter) => {
            aec_log("Got webaudio input devices");
            let mut available = Vec::new();

            for device in device_iter {
                let name = device.device_id.clone();
                aec_log(format!("Device {name}"));
                available.push(name.clone());

                if &name == device_name {
                    aec_log(format!("Device found {name}"));
                    return Ok(device);
                }
            }
            let quoted = available
                .iter()
                .map(|name| format!("'{name}'"))
                .collect::<Vec<_>>()
                .join(", ");

            aec_log(format!("Failed find device {device_name} available {quoted}"));
            Err(format!(
                "Input device matching '{device_name}' not found. Available: {quoted}"
            )
            .into())
        }
        Err(err) => Err(format!("Failed to enumerate input devices: {err}").into()),
    }
}

fn select_device<I>(
    devices: Result<I, cpal::DevicesError>,
    device_name: &str,
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

                if &name == device_name {
                    return Ok(device);
                }
            }

            let quoted = available
                .iter()
                .map(|name| format!("'{name}'"))
                .collect::<Vec<_>>()
                .join(", ");
            Err(format!(
                "{kind} device matching '{device_name}' not found. Available: {quoted}"
            )
            .into())
        }
        Err(err) => Err(format!("Failed to enumerate {kind} devices: {err}").into()),
    }
}

#[cfg(target_arch = "wasm32")]
async fn get_input_device_names(_host_id: &cpal::HostId) -> Result<Vec<String>, Box<dyn Error>> {
    match get_webaudio_input_devices().await {
        Ok(device_iter) => {
            let mut available = Vec::new();

            for device in device_iter {
                let name = device.device_id;
                available.push(name.clone());
            }
            // don't sort since the order matters, first is default
            Ok(available)
        }
        Err(err) => Err(format!("Failed to enumerate input devices: {err}").into()),
    }
}

#[cfg(not(target_arch = "wasm32"))]
async fn get_input_device_names(host_id: &cpal::HostId) -> Result<Vec<String>, Box<dyn Error>> {
    let host = cpal::host_from_id(*host_id)?;
    let mut available = Vec::new();
    for dev in host.input_devices()? {
        available.push(dev.name()?.clone());
    }
    Ok(available)
}

async fn get_output_device_names(host_id: &cpal::HostId) -> Result<Vec<String>, Box<dyn Error>> {
    let host = cpal::host_from_id(*host_id)?;
    let mut available = Vec::new();
    for dev in host.output_devices()? {
        available.push(dev.name()?.clone());
    }
    Ok(available)
}


static COMMON_SAMPLE_RATES: [u32; 21] = [5512, 8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000, 88200, 96000, 128000, 176400, 192000, 256000, 352800, 384000, 705600, 768000];

fn get_sample_rates_in_range(min_rate: u32, max_rate: u32) -> Vec<u32> {
    let mut rates = HashSet::new();
    rates.insert(min_rate);
    rates.insert(max_rate);

    for rate in COMMON_SAMPLE_RATES {
        if rate >= min_rate && rate <= max_rate {
            rates.insert(rate);
        }
    }
    
    let mut list: Vec<u32> = rates.into_iter().collect();
    list.sort_unstable();
    list
}

#[cfg(target_arch = "wasm32")]
async fn get_supported_input_device_configs(
    host_id: &cpal::HostId,
    device_name: &String,
    history_len: usize,
    num_calibration_packets: u32,
    audio_buffer_seconds: u32,
    resampler_quality: i32) -> Result<Vec<InputDeviceConfig>, Box<dyn Error>> {

    // wasm only has one
    let default_config = InputDeviceConfig::from_default(
        host_id.clone(),
        device_name.clone(),
        // number of audio chunks to hold in memory, for aligning input devices's values when dropped frames/clock offsets. 100 or so is fine
        history_len, // history_len 
        // number of packets recieved before we start getting audio data
        // a larger value here will take longer to connect, but result in more accurate timing alignments
        num_calibration_packets, // calibration_packets
        // how long buffer of input audio to store, should only really need a few seconds as things are mostly streamed
        audio_buffer_seconds, // audio_buffer_seconds
        resampler_quality // resampler_quality
    ).await?;

    let mut result_configs = Vec::new();
    result_configs.push(default_config);
    Ok(result_configs)
}

#[cfg(not(target_arch = "wasm32"))]
async fn get_supported_input_device_configs(
    host_id: &cpal::HostId,
    device_name: &String,
    history_len: usize,
    num_calibration_packets: u32,
    audio_buffer_seconds: u32,
    resampler_quality: i32,
) -> Result<Vec<InputDeviceConfig>, Box<dyn Error>> {
    // put default as first item in list
    let default_config = InputDeviceConfig::from_default(
        host_id.clone(),
        device_name.clone(),
        // number of audio chunks to hold in memory, for aligning input devices's values when dropped frames/clock offsets. 100 or so is fine
        history_len, // history_len 
        // number of packets recieved before we start getting audio data
        // a larger value here will take longer to connect, but result in more accurate timing alignments
        num_calibration_packets, // calibration_packets
        // how long buffer of input audio to store, should only really need a few seconds as things are mostly streamed
        audio_buffer_seconds, // audio_buffer_seconds
        resampler_quality // resampler_quality
    ).await?;

    let mut result_configs = Vec::new();
    result_configs.push(default_config.clone());

    let host = cpal::host_from_id(*host_id)?;
    let device = select_device(host.input_devices(), &device_name, "Input")?;
    let configs : Vec<_> = device.supported_input_configs().map(|configs| configs.collect())
            .map_err(|err| format!("Unable to enumerate input configs for '{device_name}': {err}"))?;

    for cfg in configs {
        let min_rate = cfg.min_sample_rate().0;
        let max_rate = cfg.max_sample_rate().0;
        for sample_rate in get_sample_rates_in_range(min_rate, max_rate).iter() {
            let device_config = InputDeviceConfig::new(
                host_id.clone(),
                device_name.clone(),
                cfg.channels() as usize,
                *sample_rate,
                cfg.sample_format(),
                history_len,
                num_calibration_packets,
                audio_buffer_seconds,
                resampler_quality);
            if device_config != default_config {
                result_configs.push(device_config);
            }
        }
    }
    Ok(result_configs)
}


async fn get_supported_output_device_configs(
    host_id: &cpal::HostId,
    device_name: &String,
    history_len: usize,
    num_calibration_packets: u32,
    audio_buffer_seconds: u32,
    resampler_quality: i32,
    frame_size: u32,
) -> Result<Vec<OutputDeviceConfig>, Box<dyn Error>> {
    // put default as first item in list
    let default_config = OutputDeviceConfig::from_default(
        host_id.clone(),
        device_name.clone(),
        // number of audio chunks to hold in memory, for aligning input devices's values when dropped frames/clock offsets. 100 or so is fine
        history_len, // history_len 
        // number of packets recieved before we start getting audio data
        // a larger value here will take longer to connect, but result in more accurate timing alignments
        num_calibration_packets, // calibration_packets
        // how long buffer of input audio to store, should only really need a few seconds as things are mostly streamed
        audio_buffer_seconds, // audio_buffer_seconds
        resampler_quality, // resampler_quality
        frame_size
    ).await?;

    let mut result_configs = Vec::new();
    result_configs.push(default_config.clone());

    let host = cpal::host_from_id(*host_id)?;
    let device = select_device(host.output_devices(), &device_name, "Output")?;
    let configs : Vec<_> = device.supported_output_configs().map(|configs| configs.collect())
            .map_err(|err| format!("Unable to enumerate output configs for '{device_name}': {err}"))?;

    for cfg in configs {
        let min_rate = cfg.min_sample_rate().0;
        let max_rate = cfg.max_sample_rate().0;
        for sample_rate in get_sample_rates_in_range(min_rate, max_rate).iter()  {
            let device_config = OutputDeviceConfig::new(
                host_id.clone(),
                device_name.clone(),
                cfg.channels() as usize,
                *sample_rate,
                cfg.sample_format(),
                history_len,
                num_calibration_packets,
                audio_buffer_seconds,
                resampler_quality,
                frame_size);
            if device_config != default_config {
                result_configs.push(device_config);
            }
        }
    }
    Ok(result_configs)
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

#[cfg(target_arch = "wasm32")]
async fn get_default_input_device_config(host_id: &cpal::HostId, device_name: &String) -> Result<SupportedStreamConfig, Box<dyn Error>> {
    let device = select_input_device(
        &host_id,
        &device_name
    ).await?;

    // it stores the defaults in the device itself, for wasm
    Ok(SupportedStreamConfig::new(
        device.channels as u16,
        SampleRate(device.sample_rate),
        SupportedBufferSize::Unknown,
        device.sample_format,
    ))
}

#[cfg(not(target_arch = "wasm32"))]
async fn get_default_input_device_config(host_id: &cpal::HostId, device_name: &String) -> Result<SupportedStreamConfig, Box<dyn Error>> {
    let device = select_input_device(
        &host_id,
        &device_name
    ).await?;

    Ok(device.default_input_config()?)
}

#[cfg(target_arch = "wasm32")]
async fn find_matching_input_device_config(
    device: &InputDevice,
    device_name: &String,
    channels: usize,
    sample_rate: u32,
    format: SampleFormat,
) -> Result<SupportedStreamConfig, Box<dyn Error>> {
    // wasm only supports one format (for some browsers), just check against that
    if channels != device.channels || 
        sample_rate != device.sample_rate ||
        format != SampleFormat::F32 {
        let supported_str = format!("{} channel(s), {:?}, sample rate: {}",
            device.channels, SampleFormat::F32, device.sample_rate);
        Err(format!("Input device '{}' does not support {} channel(s), {:?} at {} Hz. Supported configs: {}",
            device_name, channels, format, sample_rate, supported_str
        ).into())
    } else {
        // construct it from the device data
        Ok(SupportedStreamConfig::new(
            device.channels as u16,
            SampleRate(device.sample_rate),
            SupportedBufferSize::Unknown,
            SampleFormat::F32,
        ))
    }
}

#[cfg(not(target_arch = "wasm32"))]
async fn find_matching_input_device_config(
    device: &InputDevice,
    device_name: &String,
    channels: usize,
    sample_rate: u32,
    format: SampleFormat,
) -> Result<SupportedStreamConfig, Box<dyn Error>> {
    find_matching_device_config(
        device,
        device_name,
        channels,
        sample_rate,
        format,
        "Input",
    )
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




async fn build_input_alignment_stream(
    device: &InputDevice,
    config: &InputDeviceConfig,
    supported_config: SupportedStreamConfig,
    channel_aligners: StreamAlignerProducer,
) -> Result<InputStream, Box<dyn Error>> {
    let stream_res = match config.sample_format {
        SampleFormat::I16 => build_input_alignment_stream_typed::<i16>(
            device,
            config,
            supported_config,
            channel_aligners,
        )
        .await,
        SampleFormat::F32 => build_input_alignment_stream_typed::<f32>(
            device,
            config,
            supported_config,
            channel_aligners,
        )
        .await,
        SampleFormat::U16 => build_input_alignment_stream_typed::<u16>(
            device,
            config,
            supported_config,
            channel_aligners,
        )
        .await,
        other => {
            eprintln!(
                "Input device '{0}' uses unsupported sample format {other:?}; cannot build StreamAligner.",
                config.device_name
            );
            Err(cpal::BuildStreamError::StreamConfigNotSupported)
        }?
    };
    Ok(stream_res?)
}

#[cfg(not(target_arch = "wasm32"))]
async fn build_input_alignment_stream_typed<T>(
    device: &InputDevice,
    config: &InputDeviceConfig,
    supported_config: SupportedStreamConfig,
    mut channel_aligner: StreamAlignerProducer,
) -> Result<InputStream, Box<dyn Error>>
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
    Ok(device
        .build_input_stream(
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
        .map_err(|e| -> Box<dyn Error> { Box::new(e) })?)
}

#[cfg(target_arch = "wasm32")]
async fn build_input_alignment_stream_typed<T>(
    device: &InputDevice,
    config: &InputDeviceConfig,
    _supported_config: SupportedStreamConfig,
    mut channel_aligner: StreamAlignerProducer,
) -> Result<InputStream, Box<dyn Error>>
where
    T: Sample + SizedSample,
    f32: FromSample<T>,
{    
    let device_name_inner = config.device_name.clone();
    Ok(build_webaudio_input_stream(device as &InputDeviceInfo, move |data: &[f32]| {
        if let Err(err) = channel_aligner.process_chunk(data) {
                eprintln!("Input stream '{device_name_inner}' error when process chunk {err}");
        }
    }).await?)
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
            let mut frames_needed = frames as i64 - (device_audio_channel_consumer.available() / mixer.channels) as i64;
            while frames_needed > 0 {
                // todo: print mix errors
                let _ = mixer.mix_audio_streams(((frames as usize)) as usize); // a few extra in case of resampling
                frames_needed = frames as i64 - (device_audio_channel_consumer.available() / mixer.channels) as i64;
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


#[cfg(target_arch = "wasm32")]
/// Discovered details for a specific audio input device obtained via `getUserMedia` constraints.
#[derive(Clone, Debug)]
pub struct InputDeviceInfo {
    pub device_id: String,
    pub label: Option<String>,
    pub sample_rate: u32,
    pub channels: usize,
    pub sample_format: cpal::SampleFormat,
}

#[cfg(target_arch = "wasm32")]
#[inline]
fn helper_log(msg: impl AsRef<str>) {
    let msg = msg.as_ref();
    #[cfg(target_arch = "wasm32")]
    {
        use js_sys::{Function, Reflect};
        let global = js_sys::global();
        let key = JsValue::from_str("logMessage");
        if let Ok(val) = Reflect::get(&global, &key) {
            if let Some(func) = val.dyn_ref::<Function>() {
                let _ = func.call1(&JsValue::NULL, &JsValue::from_str(msg));
                return;
            }
        }
        // Fallback if the JS helper isn't present.
        web_sys::console::log_1(&JsValue::from_str(msg));
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        println!("{msg}");
    }
}
#[cfg(target_arch = "wasm32")]
#[macro_export]
macro_rules! helper_log {
    ($($t:tt)*) => {
        $crate::aec::helper_log(format!($($t)*))
    };
}

#[cfg(target_arch = "wasm32")]
pub struct WasmStream {
    device_id: String
}
#[cfg(target_arch = "wasm32")]
impl WasmStream {
    pub fn new(device_id: String) -> Self {
        Self {
            device_id: device_id
        }
    }

    pub async fn play(&self) -> Result<(), JsErr> {
        let window = web_sys::window().ok_or_else(|| JsValue::from_str("window not available"))?;
        let navigator: Navigator = window.navigator();
        let media_devices: MediaDevices = navigator.media_devices()?;
    
        let (device_stream, audio_context, source) = take_cached_stream(&media_devices, &self.device_id).await?;

        JsFuture::from(audio_context.resume()?).await?;
        put_cached_stream(&self.device_id, device_stream, audio_context, source);
        Ok(())
    }

    pub async fn pause(&self) -> Result<(), JsErr> {
        let window = web_sys::window().ok_or_else(|| JsValue::from_str("window not available"))?;
        let navigator: Navigator = window.navigator();
        let media_devices: MediaDevices = navigator.media_devices()?;
    
        let (device_stream, audio_context, source) = take_cached_stream(&media_devices, &self.device_id).await?;
        
        JsFuture::from(audio_context.suspend()?).await?;
        put_cached_stream(&self.device_id, device_stream, audio_context, source);
        Ok(())
    }
}
#[cfg(target_arch = "wasm32")]
// lets us auto convert JsValue to Error
#[derive(Debug, thiserror::Error)]
pub enum JsErr {
    #[error("js error: {0}")]
    Js(String),
}
#[cfg(target_arch = "wasm32")]
impl From<JsValue> for JsErr {
    fn from(v: JsValue) -> Self {
        JsErr::Js(
            v.as_string()
            .or_else(|| js_sys::Error::from(v.clone()).message().as_string())
            .unwrap_or_else(|| format!("{v:?}"))
        )
    }
}
#[cfg(target_arch = "wasm32")]
async fn cleanup_audio_context(context: web_sys::AudioContext) {
    match context.close() {
        Ok(val) => {
            match JsFuture::from(val).await {
                Ok(_) => {

                }
                Err(err) => {
                    let error_error = JsErr::from(err);
                    eprintln!("Cleanup wasm stream failed: {error_error}");
                }
            }
        }
        Err(err) => {
            let error_error = JsErr::from(err);
            eprintln!("Cleanup wasm stream failed: {error_error}");
        }
    }
}
#[cfg(target_arch = "wasm32")]
fn buffer_time_step_secs(buffer_size_frames: usize, sample_rate: u32) -> f64 {
    buffer_size_frames as f64 / (sample_rate as f64)
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    // Cache the MediaDevices handle after permission is granted. Do not retain the stream; keeping
    // it alive can hold the microphone and block future getUserMedia calls.
    static INPUT_ACCESS_CACHE: RefCell<Option<MediaDevices>> = RefCell::new(None);
    static INPUT_DEVICE_CACHE: RefCell<Option<Vec<InputDeviceInfo>>> = RefCell::new(None);
    static DEVICE_PROBE_CACHE: RefCell<HashMap<String, (MediaStream, AudioContext, MediaStreamAudioSourceNode)>> = RefCell::new(HashMap::new());
}
#[cfg(target_arch = "wasm32")]
fn cleanup_stream(stream: web_sys::MediaStream) {
    let tracks = stream.get_tracks();
    let length = tracks.length();
    
    for i in 0..length {
        if let Some(track) = tracks.get(i).dyn_ref::<MediaStreamTrack>() {
            track.stop();
        }
    }
    // Note: remove_track might not be necessary after stop()
}


#[cfg(target_arch = "wasm32")]
pub async fn request_input_access() -> Result<(), JsErr> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("window not available"))?;
    let navigator: Navigator = window.navigator();
    let media_devices: MediaDevices = navigator.media_devices()?;

    let constraints = MediaStreamConstraints::new();
    constraints.set_audio(&JsValue::from_bool(true));
    constraints.set_video(&JsValue::from_bool(false));

    let default_stream = media_devices.get_user_media_with_constraints(&constraints)?;
    let default_stream = JsFuture::from(default_stream)
        .await
        .map_err(|e| {
            helper_log(format!("getUserMedia rejected: {e:?}"));
            e
        })?;
    let default_stream: MediaStream = default_stream.dyn_into()?;
    cleanup_stream(default_stream);
    Ok(())
}
#[cfg(target_arch = "wasm32")]
async fn take_cached_stream(media_devices: &MediaDevices, device_id: &String) -> Result<(MediaStream, AudioContext, MediaStreamAudioSourceNode), JsErr> {


    let entry = DEVICE_PROBE_CACHE.with(|cache| cache.borrow_mut().remove(device_id));
    let (device_stream, audio_context, source) = if let Some(probe) = entry {
        probe
    } else {

        // Probe the specific device so we can grab its channel count and sample rate.
        let constraints = MediaStreamConstraints::new();
        constraints.set_video(&JsValue::from_bool(false));

        let audio_obj = js_sys::Object::new();
        let device_obj = js_sys::Object::new();
        let _ = js_sys::Reflect::set(
            &device_obj,
            &JsValue::from_str("exact"),
            &JsValue::from_str(&device_id),
        );
        let _ = js_sys::Reflect::set(&audio_obj, &JsValue::from_str("deviceId"), &device_obj);
        constraints.set_audio(&audio_obj.into());

        let device_stream = media_devices.get_user_media_with_constraints(&constraints)?;
        let device_stream = JsFuture::from(device_stream).await?;
        let device_stream: MediaStream = device_stream.dyn_into()?;

        let audio_context = AudioContext::new()?;
        // Necessary to read sample rate in some browsers.
        let source = audio_context.create_media_stream_source(&device_stream)?;

        (device_stream, audio_context, source)
    };

    Ok((device_stream, audio_context, source))
}
#[cfg(target_arch = "wasm32")]
fn put_cached_stream(device_name: &String, device_stream: MediaStream, audio_context: AudioContext, source: MediaStreamAudioSourceNode) {
    DEVICE_PROBE_CACHE.with(|cache| {
        cache.borrow_mut().insert(device_name.clone(), (device_stream, audio_context, source));
    });
}
#[cfg(target_arch = "wasm32")]
pub async fn get_webaudio_input_devices() -> Result<Vec<InputDeviceInfo>, JsErr> {
    if let Some(cached_input_devices) = INPUT_DEVICE_CACHE.with(|cell| cell.borrow().clone()) {
        return Ok(cached_input_devices);
    }
    request_input_access().await?;
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("window not available"))?;
    let navigator: Navigator = window.navigator();
    let media_devices: MediaDevices = navigator.media_devices()?;
    // Now enumerate concrete audio input devices and probe each with its deviceId constraint.
    let devices = JsFuture::from(media_devices.enumerate_devices()?).await?;
    let devices: js_sys::Array = devices.dyn_into()?;
    let mut infos = Vec::new();

    for device in devices.iter() {
        let kind = js_sys::Reflect::get(&device, &JsValue::from_str("kind"))
            .ok()
            .and_then(|k| k.as_string());
        if kind.as_deref() != Some("audioinput") {
            continue;
        }

        let device_id = js_sys::Reflect::get(&device, &JsValue::from_str("deviceId"))
            .ok()
            .and_then(|id| id.as_string())
            .unwrap_or_default();
        if device_id.is_empty() {
            continue;
        }

        let label = js_sys::Reflect::get(&device, &JsValue::from_str("label"))
            .ok()
            .and_then(|l| l.as_string())
            .filter(|l| !l.is_empty());

        //let (device_stream, audio_context, source) = take_cached_stream(&media_devices, &device_id).await?;

        //let sample_rate = audio_context.sample_rate() as u32;
        //let channels = source.channel_count() as usize;

        let sample_format = SampleFormat::F32;

        infos.push(InputDeviceInfo {
            device_id: device_id.clone(),
            label,
            sample_rate: 44000,// sample_rate,
            channels: 2, //channels,
            sample_format, // wasm is always f32 sample format
        });

        //put_cached_stream(&device_id, device_stream, audio_context, source);
    }
    INPUT_DEVICE_CACHE.with(|cell| {
        *cell.borrow_mut() = Some(infos.clone()); // clone if you still need `devices` locally
    });
    Ok(infos)
}


#[cfg(target_arch = "wasm32")]
pub async fn build_webaudio_input_stream<D>(
    device_info: &InputDeviceInfo,
    mut data_callback: D,
) -> Result<WasmStream, JsErr>
    where
        D: FnMut(&[f32]) + Send + 'static,
{
    
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("window not available"))?;
    let navigator: Navigator = window.navigator();
    let media_devices: MediaDevices = navigator.media_devices()?;

    let constraints = MediaStreamConstraints::new();
    constraints.set_audio(&JsValue::from_bool(true));
    constraints.set_video(&JsValue::from_bool(false));

    let default_stream = media_devices.get_user_media_with_constraints(&constraints)?;
    let default_stream = JsFuture::from(default_stream)
        .await
        .map_err(|e| {
            helper_log(format!("getUserMedia rejected: {e:?}"));
            e
        })?;
    let default_stream: MediaStream = default_stream.dyn_into()?;
    cleanup_stream(default_stream);
    
    let (device_stream, audio_context, source) = take_cached_stream(&media_devices, &device_info.device_id).await?;
    
    JsFuture::from(audio_context.resume().unwrap()).await.unwrap();

    // must be fetched after call to create_media_stream_source (before that, it will not be populated)
    let _sample_rate = audio_context.sample_rate() as u32;

    // AudioWorklet module is served as a static file via webpack copy plugin.
    let url = "cpal-input-processor.js";
    // need to resume or adding worklet will hang
    let audio_worklet = audio_context.audio_worklet().map_err(|err| {
        err
    })?;

    let processor = audio_worklet.add_module(&url).map_err(|err| {
        err
    })?;
    JsFuture::from(processor).await.map_err(|err| {
        err
    })?;

    let worklet_node = web_sys::AudioWorkletNode::new(audio_context.as_ref(), "cpal-input-processor")
        .expect("Failed to create audio worklet node");

    source.connect_with_audio_node(&worklet_node).unwrap();

    let mut output_buf: Vec<f32> = Vec::new();

    // Float32Array
    let js_closure = Closure::wrap(Box::new(move |msg: wasm_bindgen::JsValue| {
        
        let msg_event = msg.dyn_into::<web_sys::MessageEvent>().unwrap();

        let data = msg_event.data();

        let data : Vec<Vec<f32>> = Array::from(&data).iter()
                    .map(|v| Float32Array::from(v).to_vec())
                    .collect();

        let channels = data.len();

        if channels == 0 {
            return;
        }
        
        let frames = data[0].len();

        if frames == 0 {
            return;
        }

        output_buf.clear();
        output_buf.resize(channels*frames, 0.0f32);

        // interleave the data into output_buf
        for ch in 0..channels {
            for frame in 0..frames {
                output_buf[frame * channels + ch] = data[ch][frame];
            }
        }
        
        (data_callback)(&mut output_buf.as_slice());
    }) as Box<dyn FnMut(wasm_bindgen::JsValue)>);

    let js_func = js_closure.as_ref().unchecked_ref();

    worklet_node
        .port()
        .expect("Failed to get port")
        .set_onmessage(Some(js_func));


    put_cached_stream(&device_info.device_id, device_stream, audio_context, source);
    
    Ok(WasmStream::new(device_info.device_id.clone()))
}
