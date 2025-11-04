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

pub mod fftwrap;
pub mod kiss_fft;
pub mod kiss_fftr;
pub mod mdf;
pub mod resample;
pub mod smallft;

pub use self::mdf::speex_echo_h;
