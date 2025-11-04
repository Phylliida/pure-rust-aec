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

use std::ffi::c_int;

pub use c2rust::mdf::{
    speex_echo_cancel, speex_echo_cancellation, speex_echo_capture, speex_echo_ctl,
    speex_echo_get_residual, speex_echo_playback, speex_echo_state_destroy,
    speex_echo_state_init, speex_echo_state_init_mc, speex_echo_state_reset,
    SpeexEchoState,
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
            speex_echo_cancellation(
                self.state,
                mic.as_ptr(),
                speaker.as_ptr(),
                out.as_mut_ptr(),
            );
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

impl Drop for EchoCanceller {
    fn drop(&mut self) {
        unsafe {
            speex_echo_state_destroy(self.state);
        }
    }
}
