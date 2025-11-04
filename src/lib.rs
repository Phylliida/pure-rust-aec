#![allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code,
    improper_ctypes,
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::cargo,
    clippy::restriction,
    clippy::complexity,
    clippy::perf,
    clippy::style,
    clippy::suspicious,
    clippy::tests_outside_test_module,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing
)]

pub mod c2rust;

use std::convert::TryFrom;
use std::ffi::{c_int, CStr};
use std::fmt;
use std::ptr::NonNull;

pub use c2rust::speex_echo_h::{
    SPEEX_ECHO_GET_FRAME_SIZE, SPEEX_ECHO_GET_IMPULSE_RESPONSE,
    SPEEX_ECHO_GET_IMPULSE_RESPONSE_SIZE, SPEEX_ECHO_GET_SAMPLING_RATE,
    SPEEX_ECHO_SET_SAMPLING_RATE,
};
pub use c2rust::mdf::{
    speex_echo_cancel, speex_echo_cancellation, speex_echo_capture, speex_echo_ctl,
    speex_echo_get_residual, speex_echo_playback, speex_echo_state_destroy, speex_echo_state_init,
    speex_echo_state_init_mc, speex_echo_state_reset, SpeexEchoState,
};
pub use c2rust::resample::{
    speex_resampler_destroy, speex_resampler_get_input_latency, speex_resampler_get_output_latency,
    speex_resampler_get_quality, speex_resampler_get_rate, speex_resampler_init,
    speex_resampler_init_frac, speex_resampler_process_float, speex_resampler_process_int,
    speex_resampler_process_interleaved_float, speex_resampler_process_interleaved_int,
    speex_resampler_reset_mem, speex_resampler_set_quality, speex_resampler_set_rate,
    speex_resampler_set_rate_frac, speex_resampler_skip_zeros, speex_resampler_strerror,
    SpeexResamplerState, RESAMPLER_ERR_ALLOC_FAILED, RESAMPLER_ERR_BAD_STATE,
    RESAMPLER_ERR_INVALID_ARG, RESAMPLER_ERR_MAX_ERROR, RESAMPLER_ERR_OVERFLOW,
    RESAMPLER_ERR_PTR_OVERLAP, RESAMPLER_ERR_SUCCESS,
};

/// Safe wrapper around the translated Speex echo canceller.
pub struct EchoCanceller {
    state: *mut SpeexEchoState,
    frame_size: usize,
    mic_channels: usize,
    speaker_channels: usize,
}

impl EchoCanceller {
    /// Create a new echo canceller for the given configuration.
    pub fn new(frame_size: usize, filter_length: usize) -> Option<Self> {
        unsafe {
            let state = speex_echo_state_init(frame_size as c_int, filter_length as c_int);
            if state.is_null() {
                None
            } else {
                Some(Self {
                    state,
                    frame_size,
                    mic_channels: 1,
                    speaker_channels: 1,
                })
            }
        }
    }

    /// Create a multi-channel echo canceller. Returns `None` on allocation failure.
    pub fn new_multichannel(
        frame_size: usize,
        filter_length: usize,
        mics: usize,
        speakers: usize,
    ) -> Option<Self> {
        unsafe {
            let state = speex_echo_state_init_mc(
                frame_size as c_int,
                filter_length as c_int,
                mics as c_int,
                speakers as c_int,
            );
            if state.is_null() {
                None
            } else {
                Some(Self {
                    state,
                    frame_size,
                    mic_channels: mics,
                    speaker_channels: speakers,
                })
            }
        }
    }

    /// Reset the internal filter and statistics.
    pub fn reset(&mut self) {
        unsafe {
            speex_echo_state_reset(self.state);
        }
    }

    /// Process a frame of near-end (`mic`) and far-end (`speaker`) audio.
    ///
    /// * `mic` - interleaved microphone samples (`mic_channels * frame_size`)
    /// * `speaker` - interleaved loudspeaker samples (`speaker_channels * frame_size`)
    /// * `out` - buffer receiving echo-cancelled microphone samples (same layout as `mic`)
    pub fn cancel_frame(&mut self, mic: &[i16], speaker: &[i16], out: &mut [i16]) {
        assert_eq!(mic.len(), self.frame_size * self.mic_channels);
        assert_eq!(speaker.len(), self.frame_size * self.speaker_channels);
        assert_eq!(out.len(), mic.len());
        unsafe {
            speex_echo_cancellation(self.state, mic.as_ptr(), speaker.as_ptr(), out.as_mut_ptr());
        }
    }

    /// Capture microphone audio using the internal playback delay buffer.
    pub fn capture(&mut self, mic: &[i16], out: &mut [i16]) {
        assert_eq!(mic.len(), self.frame_size * self.mic_channels);
        assert_eq!(out.len(), mic.len());
        unsafe {
            speex_echo_capture(self.state, mic.as_ptr(), out.as_mut_ptr());
        }
    }

    /// Provide the next loudspeaker frame for use with [`capture`].
    pub fn playback(&mut self, speaker: &[i16]) {
        assert_eq!(speaker.len(), self.frame_size * self.speaker_channels);
        unsafe {
            speex_echo_playback(self.state, speaker.as_ptr());
        }
    }

    /// Raw pointer to the underlying Speex echo state (for advanced use).
    pub fn as_ptr(&self) -> *mut SpeexEchoState {
        self.state
    }
}

unsafe impl Send for EchoCanceller {}

impl Drop for EchoCanceller {
    fn drop(&mut self) {
        unsafe {
            speex_echo_state_destroy(self.state);
        }
    }
}

unsafe impl Send for Resampler {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResamplerError {
    AllocFailed,
    BadState,
    InvalidArg,
    PtrOverlap,
    Overflow,
    Unknown(i32),
}

impl ResamplerError {
    fn from_code(code: c_int) -> Option<Self> {
        if code == 0 {
            return None;
        }
        match code as u32 {
            RESAMPLER_ERR_ALLOC_FAILED => Some(Self::AllocFailed),
            RESAMPLER_ERR_BAD_STATE => Some(Self::BadState),
            RESAMPLER_ERR_INVALID_ARG => Some(Self::InvalidArg),
            RESAMPLER_ERR_PTR_OVERLAP => Some(Self::PtrOverlap),
            RESAMPLER_ERR_OVERFLOW => Some(Self::Overflow),
            _ => Some(Self::Unknown(code)),
        }
    }

    fn code(self) -> c_int {
        match self {
            Self::AllocFailed => RESAMPLER_ERR_ALLOC_FAILED as c_int,
            Self::BadState => RESAMPLER_ERR_BAD_STATE as c_int,
            Self::InvalidArg => RESAMPLER_ERR_INVALID_ARG as c_int,
            Self::PtrOverlap => RESAMPLER_ERR_PTR_OVERLAP as c_int,
            Self::Overflow => RESAMPLER_ERR_OVERFLOW as c_int,
            Self::Unknown(code) => code,
        }
    }

    fn check(code: c_int) -> Result<(), Self> {
        if let Some(err) = Self::from_code(code) {
            Err(err)
        } else {
            Ok(())
        }
    }
}

impl fmt::Display for ResamplerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message =
            unsafe { CStr::from_ptr(speex_resampler_strerror(self.code())).to_string_lossy() };
        write!(f, "{message}")
    }
}

impl std::error::Error for ResamplerError {}

/// Safe wrapper around the translated Speex resampler.
pub struct Resampler {
    state: NonNull<SpeexResamplerState>,
    channels: usize,
}

impl Resampler {
    fn create_with<F>(channels: u32, init: F) -> Result<Self, ResamplerError>
    where
        F: FnOnce(*mut c_int) -> *mut SpeexResamplerState,
    {
        if channels == 0 {
            return Err(ResamplerError::InvalidArg);
        }
        let channels = usize::try_from(channels).map_err(|_| ResamplerError::InvalidArg)?;
        let mut err_code = RESAMPLER_ERR_SUCCESS as c_int;
        let raw = init(&mut err_code);
        let state = NonNull::new(raw).ok_or_else(|| {
            ResamplerError::from_code(err_code).unwrap_or(ResamplerError::AllocFailed)
        })?;
        if let Some(err) = ResamplerError::from_code(err_code) {
            unsafe {
                speex_resampler_destroy(state.as_ptr());
            }
            return Err(err);
        }
        Ok(Self { state, channels })
    }

    /// Create a resampler with integer in/out sample rates.
    pub fn new(
        channels: u32,
        in_rate: u32,
        out_rate: u32,
        quality: i32,
    ) -> Result<Self, ResamplerError> {
        Self::create_with(channels, |err| unsafe {
            speex_resampler_init(channels, in_rate, out_rate, quality, err)
        })
    }

    /// Create a resampler with an explicit rational rate ratio.
    pub fn new_frac(
        channels: u32,
        ratio_num: u32,
        ratio_den: u32,
        in_rate: u32,
        out_rate: u32,
        quality: i32,
    ) -> Result<Self, ResamplerError> {
        if ratio_num == 0 || ratio_den == 0 {
            return Err(ResamplerError::InvalidArg);
        }
        Self::create_with(channels, |err| unsafe {
            speex_resampler_init_frac(
                channels, ratio_num, ratio_den, in_rate, out_rate, quality, err,
            )
        })
    }

    /// Number of interleaved channels processed by the resampler.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Raw pointer to the underlying Speex resampler state.
    pub fn as_ptr(&self) -> *mut SpeexResamplerState {
        self.state.as_ptr()
    }

    fn frames_from_len(&self, len: usize) -> Result<u32, ResamplerError> {
        if len % self.channels != 0 {
            return Err(ResamplerError::InvalidArg);
        }
        let frames = len / self.channels;
        u32::try_from(frames).map_err(|_| ResamplerError::InvalidArg)
    }

    /// Resample interleaved 16-bit audio. Returns (consumed_samples, produced_samples).
    pub fn process_interleaved_i16(
        &mut self,
        input: &[i16],
        output: &mut [i16],
    ) -> Result<(usize, usize), ResamplerError> {
        let mut in_frames = self.frames_from_len(input.len())?;
        let mut out_frames = self.frames_from_len(output.len())?;
        let code = unsafe {
            speex_resampler_process_interleaved_int(
                self.state.as_ptr(),
                input.as_ptr(),
                &mut in_frames,
                output.as_mut_ptr(),
                &mut out_frames,
            )
        };
        ResamplerError::check(code)?;
        Ok((
            in_frames as usize * self.channels,
            out_frames as usize * self.channels,
        ))
    }

    /// Resample interleaved 32-bit float audio. Returns (consumed_samples, produced_samples).
    pub fn process_interleaved_f32(
        &mut self,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(usize, usize), ResamplerError> {
        let mut in_frames = self.frames_from_len(input.len())?;
        let mut out_frames = self.frames_from_len(output.len())?;
        let code = unsafe {
            speex_resampler_process_interleaved_float(
                self.state.as_ptr(),
                input.as_ptr(),
                &mut in_frames,
                output.as_mut_ptr(),
                &mut out_frames,
            )
        };
        ResamplerError::check(code)?;
        Ok((
            in_frames as usize * self.channels,
            out_frames as usize * self.channels,
        ))
    }

    /// Update the input/output sample rates.
    pub fn set_rate(&mut self, in_rate: u32, out_rate: u32) -> Result<(), ResamplerError> {
        let code = unsafe { speex_resampler_set_rate(self.state.as_ptr(), in_rate, out_rate) };
        ResamplerError::check(code)
    }

    /// Update the resampler ratio using explicit numerator/denominator.
    pub fn set_rate_frac(
        &mut self,
        ratio_num: u32,
        ratio_den: u32,
        in_rate: u32,
        out_rate: u32,
    ) -> Result<(), ResamplerError> {
        if ratio_num == 0 || ratio_den == 0 {
            return Err(ResamplerError::InvalidArg);
        }
        let code = unsafe {
            speex_resampler_set_rate_frac(
                self.state.as_ptr(),
                ratio_num,
                ratio_den,
                in_rate,
                out_rate,
            )
        };
        ResamplerError::check(code)
    }

    /// Current input/output sampling rates.
    pub fn get_rate(&self) -> (u32, u32) {
        let mut in_rate = 0;
        let mut out_rate = 0;
        unsafe {
            speex_resampler_get_rate(self.state.as_ptr(), &mut in_rate, &mut out_rate);
        }
        (in_rate, out_rate)
    }

    /// Change the quality preset (0-10).
    pub fn set_quality(&mut self, quality: i32) -> Result<(), ResamplerError> {
        ResamplerError::check(unsafe { speex_resampler_set_quality(self.state.as_ptr(), quality) })
    }

    /// Current quality preset.
    pub fn quality(&self) -> i32 {
        let mut quality = 0;
        unsafe {
            speex_resampler_get_quality(self.state.as_ptr(), &mut quality);
        }
        quality
    }

    /// Reset the internal memory while preserving configuration.
    pub fn reset(&mut self) -> Result<(), ResamplerError> {
        ResamplerError::check(unsafe { speex_resampler_reset_mem(self.state.as_ptr()) })
    }

    /// Skip the startup latency by pre-filling internal buffers.
    pub fn skip_zeros(&mut self) -> Result<(), ResamplerError> {
        ResamplerError::check(unsafe { speex_resampler_skip_zeros(self.state.as_ptr()) })
    }

    /// Number of input samples of latency per channel.
    pub fn input_latency(&self) -> usize {
        unsafe { speex_resampler_get_input_latency(self.state.as_ptr()) as usize }
    }

    /// Number of output samples of latency per channel.
    pub fn output_latency(&self) -> usize {
        unsafe { speex_resampler_get_output_latency(self.state.as_ptr()) as usize }
    }
}

impl Drop for Resampler {
    fn drop(&mut self) {
        unsafe {
            speex_resampler_destroy(self.state.as_ptr());
        }
    }
}
