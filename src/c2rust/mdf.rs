#![allow(unused_assignments)]
#![allow(unused_variables)]
#![allow(unused_unsafe)]

pub mod types_h {
    pub type __int16_t = i16;
    pub type __int32_t = i32;
    pub type __off_t = std::ffi::c_long;
    pub type __off64_t = std::ffi::c_long;
}
pub mod stdint_intn_h {
    pub type int16_t = __int16_t;
    pub type int32_t = __int32_t;
    use super::types_h::{__int16_t, __int32_t};
}
pub mod speexdsp_config_types_h {
    pub type spx_int16_t = int16_t;
    pub type spx_int32_t = int32_t;
    use super::stdint_intn_h::{int16_t, int32_t};
}
pub mod arch_h {
    pub type spx_mem_t = std::ffi::c_float;
    pub type spx_word16_t = std::ffi::c_float;
    pub type spx_word32_t = std::ffi::c_float;
}
pub mod speex_echo_h {
    pub type SpeexEchoState = SpeexEchoState_;
    pub const SPEEX_ECHO_GET_FRAME_SIZE: std::ffi::c_int = unsafe { 3 };
    pub const SPEEX_ECHO_SET_SAMPLING_RATE: std::ffi::c_int = unsafe { 24 };
    pub const SPEEX_ECHO_GET_SAMPLING_RATE: std::ffi::c_int = unsafe { 25 };
    pub const SPEEX_ECHO_GET_IMPULSE_RESPONSE_SIZE: std::ffi::c_int = unsafe { 27 };
    pub const SPEEX_ECHO_GET_IMPULSE_RESPONSE: std::ffi::c_int = unsafe { 29 };
    use super::SpeexEchoState_;
}
pub mod stddef_h {
    pub type size_t = usize;
}
pub mod FILE_h {
    pub type FILE = _IO_FILE;
    use super::struct_FILE_h::_IO_FILE;
}
pub mod struct_FILE_h {
    #[derive(Copy, Clone)]
    #[repr(C)]
    pub struct _IO_FILE {
        pub _flags: std::ffi::c_int,
        pub _IO_read_ptr: *mut std::ffi::c_char,
        pub _IO_read_end: *mut std::ffi::c_char,
        pub _IO_read_base: *mut std::ffi::c_char,
        pub _IO_write_base: *mut std::ffi::c_char,
        pub _IO_write_ptr: *mut std::ffi::c_char,
        pub _IO_write_end: *mut std::ffi::c_char,
        pub _IO_buf_base: *mut std::ffi::c_char,
        pub _IO_buf_end: *mut std::ffi::c_char,
        pub _IO_save_base: *mut std::ffi::c_char,
        pub _IO_backup_base: *mut std::ffi::c_char,
        pub _IO_save_end: *mut std::ffi::c_char,
        pub _markers: *mut _IO_marker,
        pub _chain: *mut _IO_FILE,
        pub _fileno: std::ffi::c_int,
        pub _flags2: std::ffi::c_int,
        pub _old_offset: __off_t,
        pub _cur_column: std::ffi::c_ushort,
        pub _vtable_offset: std::ffi::c_schar,
        pub _shortbuf: [std::ffi::c_char; 1],
        pub _lock: *mut std::ffi::c_void,
        pub _offset: __off64_t,
        pub _codecvt: *mut _IO_codecvt,
        pub _wide_data: *mut _IO_wide_data,
        pub _freeres_list: *mut _IO_FILE,
        pub _freeres_buf: *mut std::ffi::c_void,
        pub _prevchain: *mut *mut _IO_FILE,
        pub _mode: std::ffi::c_int,
        pub _unused2: [std::ffi::c_char; 20],
    }
    pub type _IO_lock_t = ();
    use super::types_h::{__off_t, __off64_t};
    #[repr(C)]
    pub struct _IO_wide_data {
        _unused: [u8; 0],
    }
    #[repr(C)]
    pub struct _IO_codecvt {
        _unused: [u8; 0],
    }
    #[repr(C)]
    pub struct _IO_marker {
        _unused: [u8; 0],
    }
}
pub mod fftwrap_h {
    use super::arch_h::spx_word16_t;
    extern "C" {
        pub fn spx_fft_init(size: std::ffi::c_int) -> *mut std::ffi::c_void;
        pub fn spx_fft_destroy(table: *mut std::ffi::c_void);
        pub fn spx_fft(
            table: *mut std::ffi::c_void,
            in_0: *mut spx_word16_t,
            out: *mut spx_word16_t,
        );
        pub fn spx_ifft(
            table: *mut std::ffi::c_void,
            in_0: *mut spx_word16_t,
            out: *mut spx_word16_t,
        );
    }
}
pub mod stdio_h {
    use super::FILE_h::FILE;
    extern "C" {
        pub static mut stderr: *mut FILE;
        pub fn fprintf(
            __stream: *mut FILE,
            __format: *const std::ffi::c_char,
            ...
        ) -> std::ffi::c_int;
    }
}
pub mod stdlib_h {
    use super::stddef_h::size_t;
    extern "C" {
        pub fn calloc(__nmemb: size_t, __size: size_t) -> *mut std::ffi::c_void;
        pub fn free(__ptr: *mut std::ffi::c_void);
    }
}
pub mod os_support_h {
    #[inline]
    pub unsafe extern "C" fn speex_alloc(
        size: std::ffi::c_int,
    ) -> *mut std::ffi::c_void {
        return calloc(size as size_t, 1 as size_t);
    }
    #[inline]
    pub unsafe extern "C" fn speex_free(ptr: *mut std::ffi::c_void) {
        free(ptr);
    }
    #[inline]
    pub unsafe extern "C" fn speex_warning(str: *const std::ffi::c_char) {
        fprintf(stderr, b"warning: %s\n\0" as *const u8 as *const std::ffi::c_char, str);
    }
    #[inline]
    pub unsafe extern "C" fn speex_warning_int(
        str: *const std::ffi::c_char,
        val: std::ffi::c_int,
    ) {
        fprintf(
            stderr,
            b"warning: %s %d\n\0" as *const u8 as *const std::ffi::c_char,
            str,
            val,
        );
    }
    use super::stdlib_h::{calloc, free};
    use super::stddef_h::size_t;
    use super::stdio_h::{fprintf, stderr};
}
pub mod math_approx_h {
    pub const spx_sqrt: unsafe extern "C" fn(std::ffi::c_double) -> std::ffi::c_double = unsafe {
        sqrt
    };
    use super::mathcalls_h::sqrt;
}
pub mod math_h {
    pub const M_PI: std::ffi::c_double = unsafe { 3.14159265358979323846f64 };
}
pub mod mathcalls_h {
    extern "C" {
        pub fn cos(__x: std::ffi::c_double) -> std::ffi::c_double;
        pub fn exp(__x: std::ffi::c_double) -> std::ffi::c_double;
        pub fn sqrt(__x: std::ffi::c_double) -> std::ffi::c_double;
        pub fn floor(__x: std::ffi::c_double) -> std::ffi::c_double;
    }
}
pub mod pseudofloat_h {
    pub const FLOAT_ZERO: std::ffi::c_float = unsafe { 0.0f32 };
    pub const FLOAT_ONE: std::ffi::c_float = unsafe { 1.0f32 };
}
pub use self::types_h::{__int16_t, __int32_t, __off_t, __off64_t};
pub use self::stdint_intn_h::{int16_t, int32_t};
pub use self::speexdsp_config_types_h::{spx_int16_t, spx_int32_t};
pub use self::arch_h::{spx_mem_t, spx_word16_t, spx_word32_t};
pub use self::speex_echo_h::{
    SpeexEchoState, SPEEX_ECHO_GET_FRAME_SIZE, SPEEX_ECHO_SET_SAMPLING_RATE,
    SPEEX_ECHO_GET_SAMPLING_RATE, SPEEX_ECHO_GET_IMPULSE_RESPONSE_SIZE,
    SPEEX_ECHO_GET_IMPULSE_RESPONSE,
};
pub use self::stddef_h::size_t;
pub use self::FILE_h::FILE;
pub use self::struct_FILE_h::{
    _IO_FILE, _IO_lock_t, _IO_wide_data, _IO_codecvt, _IO_marker,
};
use self::fftwrap_h::{spx_fft_init, spx_fft_destroy, spx_fft, spx_ifft};
pub use self::os_support_h::{speex_alloc, speex_free, speex_warning, speex_warning_int};
pub use self::math_approx_h::spx_sqrt;
pub use self::math_h::M_PI;
use self::mathcalls_h::{cos, exp, sqrt, floor};
pub use self::pseudofloat_h::{FLOAT_ZERO, FLOAT_ONE};
#[derive(Copy, Clone)]
#[repr(C)]
pub struct SpeexEchoState_ {
    pub frame_size: std::ffi::c_int,
    pub window_size: std::ffi::c_int,
    pub M: std::ffi::c_int,
    pub cancel_count: std::ffi::c_int,
    pub adapted: std::ffi::c_int,
    pub saturated: std::ffi::c_int,
    pub screwed_up: std::ffi::c_int,
    pub C: std::ffi::c_int,
    pub K: std::ffi::c_int,
    pub sampling_rate: spx_int32_t,
    pub spec_average: spx_word16_t,
    pub beta0: spx_word16_t,
    pub beta_max: spx_word16_t,
    pub sum_adapt: spx_word32_t,
    pub leak_estimate: spx_word16_t,
    pub e: *mut spx_word16_t,
    pub x: *mut spx_word16_t,
    pub X: *mut spx_word16_t,
    pub input: *mut spx_word16_t,
    pub y: *mut spx_word16_t,
    pub last_y: *mut spx_word16_t,
    pub Y: *mut spx_word16_t,
    pub E: *mut spx_word16_t,
    pub PHI: *mut spx_word32_t,
    pub W: *mut spx_word32_t,
    pub foreground: *mut spx_word16_t,
    pub Davg1: spx_word32_t,
    pub Davg2: spx_word32_t,
    pub Dvar1: std::ffi::c_float,
    pub Dvar2: std::ffi::c_float,
    pub power: *mut spx_word32_t,
    pub power_1: *mut std::ffi::c_float,
    pub wtmp: *mut spx_word16_t,
    pub Rf: *mut spx_word32_t,
    pub Yf: *mut spx_word32_t,
    pub Xf: *mut spx_word32_t,
    pub Eh: *mut spx_word32_t,
    pub Yh: *mut spx_word32_t,
    pub Pey: std::ffi::c_float,
    pub Pyy: std::ffi::c_float,
    pub window: *mut spx_word16_t,
    pub prop: *mut spx_word16_t,
    pub fft_table: *mut std::ffi::c_void,
    pub memX: *mut spx_word16_t,
    pub memD: *mut spx_word16_t,
    pub memE: *mut spx_word16_t,
    pub preemph: spx_word16_t,
    pub notch_radius: spx_word16_t,
    pub notch_mem: *mut spx_mem_t,
    pub play_buf: *mut spx_int16_t,
    pub play_buf_pos: std::ffi::c_int,
    pub play_buf_started: std::ffi::c_int,
}
static mut MIN_LEAK: std::ffi::c_float = 0.005f32;
static mut VAR1_SMOOTH: std::ffi::c_float = 0.36f32;
static mut VAR2_SMOOTH: std::ffi::c_float = 0.7225f32;
static mut VAR1_UPDATE: std::ffi::c_float = 0.5f32;
static mut VAR2_UPDATE: std::ffi::c_float = 0.25f32;
static mut VAR_BACKTRACK: std::ffi::c_float = 4.0f32;
pub const PLAYBACK_DELAY: std::ffi::c_int = unsafe { 2 as std::ffi::c_int };
#[inline]
unsafe extern "C" fn filter_dc_notch16(
    in_0: *const spx_int16_t,
    radius: spx_word16_t,
    out: *mut spx_word16_t,
    len: std::ffi::c_int,
    mem: *mut spx_mem_t,
    stride: std::ffi::c_int,
) {
    let mut i: std::ffi::c_int = 0;
    let mut den2: spx_word16_t = 0.;
    den2 = ((radius * radius) as std::ffi::c_double
        + 0.7f64 * (1 as std::ffi::c_int as spx_word16_t - radius) as std::ffi::c_double
            * (1 as std::ffi::c_int as spx_word16_t - radius) as std::ffi::c_double)
        as spx_word16_t;
    i = 0 as std::ffi::c_int;
    while i < len {
        let vin: spx_word16_t = *in_0.offset((i * stride) as isize) as spx_word16_t;
        let vout: spx_word32_t = *mem.offset(0 as std::ffi::c_int as isize)
            + vin as spx_word32_t;
        *mem.offset(0 as std::ffi::c_int as isize) = (*mem
            .offset(1 as std::ffi::c_int as isize)
            + 2 as std::ffi::c_int as spx_word32_t
                * (-(vin as spx_word32_t) + radius as spx_word32_t * vout)) as spx_mem_t;
        *mem.offset(1 as std::ffi::c_int as isize) = (vin as spx_word32_t
            - den2 as spx_word32_t * vout) as spx_mem_t;
        *out.offset(i as isize) = (radius as spx_word32_t * vout) as spx_word16_t;
        i += 1;
    }
}
#[inline]
unsafe extern "C" fn mdf_inner_prod(
    mut x: *const spx_word16_t,
    mut y: *const spx_word16_t,
    mut len: std::ffi::c_int,
) -> spx_word32_t {
    let mut sum: spx_word32_t = 0 as std::ffi::c_int as spx_word32_t;
    len >>= 1 as std::ffi::c_int;
    loop {
        let fresh0 = len;
        len = len - 1;
        if !(fresh0 != 0) {
            break;
        }
        let mut part: spx_word32_t = 0 as std::ffi::c_int as spx_word32_t;
        let fresh1 = x;
        x = x.offset(1);
        let fresh2 = y;
        y = y.offset(1);
        part = part + *fresh1 * *fresh2;
        let fresh3 = x;
        x = x.offset(1);
        let fresh4 = y;
        y = y.offset(1);
        part = part + *fresh3 * *fresh4;
        sum = sum + part;
    }
    return sum;
}
#[inline]
unsafe extern "C" fn power_spectrum(
    X: *const spx_word16_t,
    ps: *mut spx_word32_t,
    N: std::ffi::c_int,
) {
    let mut i: std::ffi::c_int = 0;
    let mut j: std::ffi::c_int = 0;
    *ps.offset(0 as std::ffi::c_int as isize) = *X.offset(0 as std::ffi::c_int as isize)
        * *X.offset(0 as std::ffi::c_int as isize);
    i = 1 as std::ffi::c_int;
    j = 1 as std::ffi::c_int;
    while i < N - 1 as std::ffi::c_int {
        *ps.offset(j as isize) = *X.offset(i as isize) * *X.offset(i as isize)
            + *X.offset((i + 1 as std::ffi::c_int) as isize)
                * *X.offset((i + 1 as std::ffi::c_int) as isize);
        i += 2 as std::ffi::c_int;
        j += 1;
    }
    *ps.offset(j as isize) = *X.offset(i as isize) * *X.offset(i as isize);
}
#[inline]
unsafe extern "C" fn power_spectrum_accum(
    X: *const spx_word16_t,
    ps: *mut spx_word32_t,
    N: std::ffi::c_int,
) {
    let mut i: std::ffi::c_int = 0;
    let mut j: std::ffi::c_int = 0;
    *ps.offset(0 as std::ffi::c_int as isize)
        += *X.offset(0 as std::ffi::c_int as isize)
            * *X.offset(0 as std::ffi::c_int as isize);
    i = 1 as std::ffi::c_int;
    j = 1 as std::ffi::c_int;
    while i < N - 1 as std::ffi::c_int {
        *ps.offset(j as isize)
            += *X.offset(i as isize) * *X.offset(i as isize)
                + *X.offset((i + 1 as std::ffi::c_int) as isize)
                    * *X.offset((i + 1 as std::ffi::c_int) as isize);
        i += 2 as std::ffi::c_int;
        j += 1;
    }
    *ps.offset(j as isize) += *X.offset(i as isize) * *X.offset(i as isize);
}
#[inline]
unsafe extern "C" fn spectral_mul_accum(
    mut X: *const spx_word16_t,
    mut Y: *const spx_word32_t,
    acc: *mut spx_word16_t,
    N: std::ffi::c_int,
    M: std::ffi::c_int,
) {
    let mut i: std::ffi::c_int = 0;
    let mut j: std::ffi::c_int = 0;
    i = 0 as std::ffi::c_int;
    while i < N {
        *acc.offset(i as isize) = 0 as std::ffi::c_int as spx_word16_t;
        i += 1;
    }
    j = 0 as std::ffi::c_int;
    while j < M {
        let ref mut fresh5 = *acc.offset(0 as std::ffi::c_int as isize);
        *fresh5
            += (*X.offset(0 as std::ffi::c_int as isize)
                * *Y.offset(0 as std::ffi::c_int as isize)) as std::ffi::c_float;
        i = 1 as std::ffi::c_int;
        while i < N - 1 as std::ffi::c_int {
            let ref mut fresh6 = *acc.offset(i as isize);
            *fresh6
                += (*X.offset(i as isize) * *Y.offset(i as isize)
                    - *X.offset((i + 1 as std::ffi::c_int) as isize)
                        * *Y.offset((i + 1 as std::ffi::c_int) as isize))
                    as std::ffi::c_float;
            let ref mut fresh7 = *acc.offset((i + 1 as std::ffi::c_int) as isize);
            *fresh7
                += (*X.offset((i + 1 as std::ffi::c_int) as isize)
                    * *Y.offset(i as isize)
                    + *X.offset(i as isize)
                        * *Y.offset((i + 1 as std::ffi::c_int) as isize))
                    as std::ffi::c_float;
            i += 2 as std::ffi::c_int;
        }
        let ref mut fresh8 = *acc.offset(i as isize);
        *fresh8 += (*X.offset(i as isize) * *Y.offset(i as isize)) as std::ffi::c_float;
        X = X.offset(N as isize);
        Y = Y.offset(N as isize);
        j += 1;
    }
}
pub const spectral_mul_accum16: unsafe extern "C" fn(
    *const spx_word16_t,
    *const spx_word32_t,
    *mut spx_word16_t,
    std::ffi::c_int,
    std::ffi::c_int,
) -> () = unsafe { spectral_mul_accum };
#[inline]
unsafe extern "C" fn weighted_spectral_mul_conj(
    w: *const std::ffi::c_float,
    p: std::ffi::c_float,
    X: *const spx_word16_t,
    Y: *const spx_word16_t,
    prod: *mut spx_word32_t,
    N: std::ffi::c_int,
) {
    let mut i: std::ffi::c_int = 0;
    let mut j: std::ffi::c_int = 0;
    let mut W: std::ffi::c_float = 0.;
    W = p * *w.offset(0 as std::ffi::c_int as isize);
    *prod.offset(0 as std::ffi::c_int as isize) = W as spx_word32_t
        * (*X.offset(0 as std::ffi::c_int as isize)
            * *Y.offset(0 as std::ffi::c_int as isize));
    i = 1 as std::ffi::c_int;
    j = 1 as std::ffi::c_int;
    while i < N - 1 as std::ffi::c_int {
        W = p * *w.offset(j as isize);
        *prod.offset(i as isize) = W as spx_word32_t
            * (*X.offset(i as isize) * *Y.offset(i as isize)
                + *X.offset((i + 1 as std::ffi::c_int) as isize)
                    * *Y.offset((i + 1 as std::ffi::c_int) as isize));
        *prod.offset((i + 1 as std::ffi::c_int) as isize) = W as spx_word32_t
            * (-*X.offset((i + 1 as std::ffi::c_int) as isize) * *Y.offset(i as isize)
                + *X.offset(i as isize)
                    * *Y.offset((i + 1 as std::ffi::c_int) as isize));
        i += 2 as std::ffi::c_int;
        j += 1;
    }
    W = p * *w.offset(j as isize);
    *prod.offset(i as isize) = W as spx_word32_t
        * (*X.offset(i as isize) * *Y.offset(i as isize));
}
#[inline]
unsafe extern "C" fn mdf_adjust_prop(
    W: *const spx_word32_t,
    N: std::ffi::c_int,
    M: std::ffi::c_int,
    P: std::ffi::c_int,
    prop: *mut spx_word16_t,
) {
    let mut i: std::ffi::c_int = 0;
    let mut j: std::ffi::c_int = 0;
    let mut p: std::ffi::c_int = 0;
    let mut max_sum: spx_word16_t = 1 as std::ffi::c_int as spx_word16_t;
    let mut prop_sum: spx_word32_t = 1 as std::ffi::c_int as spx_word32_t;
    i = 0 as std::ffi::c_int;
    while i < M {
        let mut tmp: spx_word32_t = 1 as std::ffi::c_int as spx_word32_t;
        p = 0 as std::ffi::c_int;
        while p < P {
            j = 0 as std::ffi::c_int;
            while j < N {
                tmp
                    += *W.offset((p * N * M + i * N + j) as isize)
                        * *W.offset((p * N * M + i * N + j) as isize);
                j += 1;
            }
            p += 1;
        }
        *prop.offset(i as isize) = sqrt(tmp as std::ffi::c_double) as spx_word16_t;
        if *prop.offset(i as isize) > max_sum {
            max_sum = *prop.offset(i as isize);
        }
        i += 1;
    }
    i = 0 as std::ffi::c_int;
    while i < M {
        let ref mut fresh9 = *prop.offset(i as isize);
        *fresh9 += (0.1f32 * max_sum) as std::ffi::c_float;
        prop_sum += *prop.offset(i as isize) as std::ffi::c_float;
        i += 1;
    }
    i = 0 as std::ffi::c_int;
    while i < M {
        *prop.offset(i as isize) = (0.99f32 * *prop.offset(i as isize) / prop_sum)
            as spx_word16_t;
        i += 1;
    }
}
#[no_mangle]
pub unsafe extern "C" fn speex_echo_state_init(
    frame_size: std::ffi::c_int,
    filter_length: std::ffi::c_int,
) -> *mut SpeexEchoState {
    return speex_echo_state_init_mc(
        frame_size,
        filter_length,
        1 as std::ffi::c_int,
        1 as std::ffi::c_int,
    );
}
#[no_mangle]
pub unsafe extern "C" fn speex_echo_state_init_mc(
    frame_size: std::ffi::c_int,
    filter_length: std::ffi::c_int,
    nb_mic: std::ffi::c_int,
    nb_speakers: std::ffi::c_int,
) -> *mut SpeexEchoState {
    let mut i: std::ffi::c_int = 0;
    let mut N: std::ffi::c_int = 0;
    let mut M: std::ffi::c_int = 0;
    let mut C: std::ffi::c_int = 0;
    let mut K: std::ffi::c_int = 0;
    let st: *mut SpeexEchoState = speex_alloc(
        ::core::mem::size_of::<SpeexEchoState>() as std::ffi::c_int,
    ) as *mut SpeexEchoState;
    (*st).K = nb_speakers;
    (*st).C = nb_mic;
    C = (*st).C;
    K = (*st).K;
    (*st).frame_size = frame_size;
    (*st).window_size = 2 as std::ffi::c_int * frame_size;
    N = (*st).window_size;
    (*st).M = (filter_length + (*st).frame_size - 1 as std::ffi::c_int) / frame_size;
    M = (*st).M;
    (*st).cancel_count = 0 as std::ffi::c_int;
    (*st).sum_adapt = 0 as std::ffi::c_int as spx_word32_t;
    (*st).saturated = 0 as std::ffi::c_int;
    (*st).screwed_up = 0 as std::ffi::c_int;
    (*st).sampling_rate = 8000 as std::ffi::c_int as spx_int32_t;
    (*st).spec_average = (*st).frame_size as spx_word16_t
        / (*st).sampling_rate as spx_word16_t;
    (*st).beta0 = (2.0f32 * (*st).frame_size as std::ffi::c_float
        / (*st).sampling_rate as std::ffi::c_float) as spx_word16_t;
    (*st).beta_max = (0.5f32 * (*st).frame_size as std::ffi::c_float
        / (*st).sampling_rate as std::ffi::c_float) as spx_word16_t;
    (*st).leak_estimate = 0 as std::ffi::c_int as spx_word16_t;
    (*st).fft_table = spx_fft_init(N);
    (*st).e = speex_alloc(
        ((C * N) as std::ffi::c_ulong)
            .wrapping_mul(::core::mem::size_of::<spx_word16_t>() as std::ffi::c_ulong)
            as std::ffi::c_int,
    ) as *mut spx_word16_t;
    (*st).x = speex_alloc(
        ((K * N) as std::ffi::c_ulong)
            .wrapping_mul(::core::mem::size_of::<spx_word16_t>() as std::ffi::c_ulong)
            as std::ffi::c_int,
    ) as *mut spx_word16_t;
    (*st).input = speex_alloc(
        ((C * (*st).frame_size) as std::ffi::c_ulong)
            .wrapping_mul(::core::mem::size_of::<spx_word16_t>() as std::ffi::c_ulong)
            as std::ffi::c_int,
    ) as *mut spx_word16_t;
    (*st).y = speex_alloc(
        ((C * N) as std::ffi::c_ulong)
            .wrapping_mul(::core::mem::size_of::<spx_word16_t>() as std::ffi::c_ulong)
            as std::ffi::c_int,
    ) as *mut spx_word16_t;
    (*st).last_y = speex_alloc(
        ((C * N) as std::ffi::c_ulong)
            .wrapping_mul(::core::mem::size_of::<spx_word16_t>() as std::ffi::c_ulong)
            as std::ffi::c_int,
    ) as *mut spx_word16_t;
    (*st).Yf = speex_alloc(
        (((*st).frame_size + 1 as std::ffi::c_int) as std::ffi::c_ulong)
            .wrapping_mul(::core::mem::size_of::<spx_word32_t>() as std::ffi::c_ulong)
            as std::ffi::c_int,
    ) as *mut spx_word32_t;
    (*st).Rf = speex_alloc(
        (((*st).frame_size + 1 as std::ffi::c_int) as std::ffi::c_ulong)
            .wrapping_mul(::core::mem::size_of::<spx_word32_t>() as std::ffi::c_ulong)
            as std::ffi::c_int,
    ) as *mut spx_word32_t;
    (*st).Xf = speex_alloc(
        (((*st).frame_size + 1 as std::ffi::c_int) as std::ffi::c_ulong)
            .wrapping_mul(::core::mem::size_of::<spx_word32_t>() as std::ffi::c_ulong)
            as std::ffi::c_int,
    ) as *mut spx_word32_t;
    (*st).Yh = speex_alloc(
        (((*st).frame_size + 1 as std::ffi::c_int) as std::ffi::c_ulong)
            .wrapping_mul(::core::mem::size_of::<spx_word32_t>() as std::ffi::c_ulong)
            as std::ffi::c_int,
    ) as *mut spx_word32_t;
    (*st).Eh = speex_alloc(
        (((*st).frame_size + 1 as std::ffi::c_int) as std::ffi::c_ulong)
            .wrapping_mul(::core::mem::size_of::<spx_word32_t>() as std::ffi::c_ulong)
            as std::ffi::c_int,
    ) as *mut spx_word32_t;
    (*st).X = speex_alloc(
        ((K * (M + 1 as std::ffi::c_int) * N) as std::ffi::c_ulong)
            .wrapping_mul(::core::mem::size_of::<spx_word16_t>() as std::ffi::c_ulong)
            as std::ffi::c_int,
    ) as *mut spx_word16_t;
    (*st).Y = speex_alloc(
        ((C * N) as std::ffi::c_ulong)
            .wrapping_mul(::core::mem::size_of::<spx_word16_t>() as std::ffi::c_ulong)
            as std::ffi::c_int,
    ) as *mut spx_word16_t;
    (*st).E = speex_alloc(
        ((C * N) as std::ffi::c_ulong)
            .wrapping_mul(::core::mem::size_of::<spx_word16_t>() as std::ffi::c_ulong)
            as std::ffi::c_int,
    ) as *mut spx_word16_t;
    (*st).W = speex_alloc(
        ((C * K * M * N) as std::ffi::c_ulong)
            .wrapping_mul(::core::mem::size_of::<spx_word32_t>() as std::ffi::c_ulong)
            as std::ffi::c_int,
    ) as *mut spx_word32_t;
    (*st).foreground = speex_alloc(
        ((M * N * C * K) as std::ffi::c_ulong)
            .wrapping_mul(::core::mem::size_of::<spx_word16_t>() as std::ffi::c_ulong)
            as std::ffi::c_int,
    ) as *mut spx_word16_t;
    (*st).PHI = speex_alloc(
        (N as std::ffi::c_ulong)
            .wrapping_mul(::core::mem::size_of::<spx_word32_t>() as std::ffi::c_ulong)
            as std::ffi::c_int,
    ) as *mut spx_word32_t;
    (*st).power = speex_alloc(
        ((frame_size + 1 as std::ffi::c_int) as std::ffi::c_ulong)
            .wrapping_mul(::core::mem::size_of::<spx_word32_t>() as std::ffi::c_ulong)
            as std::ffi::c_int,
    ) as *mut spx_word32_t;
    (*st).power_1 = speex_alloc(
        ((frame_size + 1 as std::ffi::c_int) as std::ffi::c_ulong)
            .wrapping_mul(
                ::core::mem::size_of::<std::ffi::c_float>() as std::ffi::c_ulong,
            ) as std::ffi::c_int,
    ) as *mut std::ffi::c_float;
    (*st).window = speex_alloc(
        (N as std::ffi::c_ulong)
            .wrapping_mul(::core::mem::size_of::<spx_word16_t>() as std::ffi::c_ulong)
            as std::ffi::c_int,
    ) as *mut spx_word16_t;
    (*st).prop = speex_alloc(
        (M as std::ffi::c_ulong)
            .wrapping_mul(::core::mem::size_of::<spx_word16_t>() as std::ffi::c_ulong)
            as std::ffi::c_int,
    ) as *mut spx_word16_t;
    (*st).wtmp = speex_alloc(
        (N as std::ffi::c_ulong)
            .wrapping_mul(::core::mem::size_of::<spx_word16_t>() as std::ffi::c_ulong)
            as std::ffi::c_int,
    ) as *mut spx_word16_t;
    i = 0 as std::ffi::c_int;
    while i < N {
        *((*st).window).offset(i as isize) = (0.5f64
            - 0.5f64
                * cos(
                    2 as std::ffi::c_int as std::ffi::c_double * M_PI
                        * i as std::ffi::c_double / N as std::ffi::c_double,
                )) as spx_word16_t;
        i += 1;
    }
    i = 0 as std::ffi::c_int;
    while i <= (*st).frame_size {
        *((*st).power_1).offset(i as isize) = FLOAT_ONE;
        i += 1;
    }
    i = 0 as std::ffi::c_int;
    while i < N * M * K * C {
        *((*st).W).offset(i as isize) = 0 as std::ffi::c_int as spx_word32_t;
        i += 1;
    }
    let mut sum: spx_word32_t = 0 as std::ffi::c_int as spx_word32_t;
    let decay: spx_word16_t = exp(
        -(2.4f64 as spx_word16_t / M as spx_word16_t) as std::ffi::c_double,
    ) as spx_word16_t;
    *((*st).prop).offset(0 as std::ffi::c_int as isize) = 0.7f32;
    sum = *((*st).prop).offset(0 as std::ffi::c_int as isize) as spx_word32_t;
    i = 1 as std::ffi::c_int;
    while i < M {
        *((*st).prop).offset(i as isize) = *((*st).prop)
            .offset((i - 1 as std::ffi::c_int) as isize) * decay;
        sum = (sum as spx_word16_t + *((*st).prop).offset(i as isize)) as spx_word32_t;
        i += 1;
    }
    i = M - 1 as std::ffi::c_int;
    while i >= 0 as std::ffi::c_int {
        *((*st).prop).offset(i as isize) = (0.8f32 * *((*st).prop).offset(i as isize)
            / sum) as spx_word16_t;
        i -= 1;
    }
    (*st).memX = speex_alloc(
        (K as std::ffi::c_ulong)
            .wrapping_mul(::core::mem::size_of::<spx_word16_t>() as std::ffi::c_ulong)
            as std::ffi::c_int,
    ) as *mut spx_word16_t;
    (*st).memD = speex_alloc(
        (C as std::ffi::c_ulong)
            .wrapping_mul(::core::mem::size_of::<spx_word16_t>() as std::ffi::c_ulong)
            as std::ffi::c_int,
    ) as *mut spx_word16_t;
    (*st).memE = speex_alloc(
        (C as std::ffi::c_ulong)
            .wrapping_mul(::core::mem::size_of::<spx_word16_t>() as std::ffi::c_ulong)
            as std::ffi::c_int,
    ) as *mut spx_word16_t;
    (*st).preemph = 0.9f32;
    if (*st).sampling_rate < 12000 as spx_int32_t {
        (*st).notch_radius = 0.9f32;
    } else if (*st).sampling_rate < 24000 as spx_int32_t {
        (*st).notch_radius = 0.982f32;
    } else {
        (*st).notch_radius = 0.992f32;
    }
    (*st).notch_mem = speex_alloc(
        ((2 as std::ffi::c_int * C) as std::ffi::c_ulong)
            .wrapping_mul(::core::mem::size_of::<spx_mem_t>() as std::ffi::c_ulong)
            as std::ffi::c_int,
    ) as *mut spx_mem_t;
    (*st).adapted = 0 as std::ffi::c_int;
    (*st).Pyy = FLOAT_ONE;
    (*st).Pey = (*st).Pyy;
    (*st).Davg2 = 0 as std::ffi::c_int as spx_word32_t;
    (*st).Davg1 = (*st).Davg2;
    (*st).Dvar2 = FLOAT_ZERO;
    (*st).Dvar1 = (*st).Dvar2;
    (*st).play_buf = speex_alloc(
        ((K * (PLAYBACK_DELAY + 1 as std::ffi::c_int) * (*st).frame_size)
            as std::ffi::c_ulong)
            .wrapping_mul(::core::mem::size_of::<spx_int16_t>() as std::ffi::c_ulong)
            as std::ffi::c_int,
    ) as *mut spx_int16_t;
    (*st).play_buf_pos = PLAYBACK_DELAY * (*st).frame_size;
    (*st).play_buf_started = 0 as std::ffi::c_int;
    return st;
}
#[no_mangle]
pub unsafe extern "C" fn speex_echo_state_reset(st: *mut SpeexEchoState) {
    let mut i: std::ffi::c_int = 0;
    let mut M: std::ffi::c_int = 0;
    let mut N: std::ffi::c_int = 0;
    let mut C: std::ffi::c_int = 0;
    let mut K: std::ffi::c_int = 0;
    (*st).cancel_count = 0 as std::ffi::c_int;
    (*st).screwed_up = 0 as std::ffi::c_int;
    N = (*st).window_size;
    M = (*st).M;
    C = (*st).C;
    K = (*st).K;
    i = 0 as std::ffi::c_int;
    while i < N * M {
        *((*st).W).offset(i as isize) = 0 as std::ffi::c_int as spx_word32_t;
        i += 1;
    }
    i = 0 as std::ffi::c_int;
    while i < N * M {
        *((*st).foreground).offset(i as isize) = 0 as std::ffi::c_int as spx_word16_t;
        i += 1;
    }
    i = 0 as std::ffi::c_int;
    while i < N * (M + 1 as std::ffi::c_int) {
        *((*st).X).offset(i as isize) = 0 as std::ffi::c_int as spx_word16_t;
        i += 1;
    }
    i = 0 as std::ffi::c_int;
    while i <= (*st).frame_size {
        *((*st).power).offset(i as isize) = 0 as std::ffi::c_int as spx_word32_t;
        *((*st).power_1).offset(i as isize) = FLOAT_ONE;
        *((*st).Eh).offset(i as isize) = 0 as std::ffi::c_int as spx_word32_t;
        *((*st).Yh).offset(i as isize) = 0 as std::ffi::c_int as spx_word32_t;
        i += 1;
    }
    i = 0 as std::ffi::c_int;
    while i < (*st).frame_size {
        *((*st).last_y).offset(i as isize) = 0 as std::ffi::c_int as spx_word16_t;
        i += 1;
    }
    i = 0 as std::ffi::c_int;
    while i < N * C {
        *((*st).E).offset(i as isize) = 0 as std::ffi::c_int as spx_word16_t;
        i += 1;
    }
    i = 0 as std::ffi::c_int;
    while i < N * K {
        *((*st).x).offset(i as isize) = 0 as std::ffi::c_int as spx_word16_t;
        i += 1;
    }
    i = 0 as std::ffi::c_int;
    while i < 2 as std::ffi::c_int * C {
        *((*st).notch_mem).offset(i as isize) = 0 as std::ffi::c_int as spx_mem_t;
        i += 1;
    }
    i = 0 as std::ffi::c_int;
    while i < C {
        let ref mut fresh10 = *((*st).memE).offset(i as isize);
        *fresh10 = 0 as std::ffi::c_int as spx_word16_t;
        *((*st).memD).offset(i as isize) = *fresh10;
        i += 1;
    }
    i = 0 as std::ffi::c_int;
    while i < K {
        *((*st).memX).offset(i as isize) = 0 as std::ffi::c_int as spx_word16_t;
        i += 1;
    }
    (*st).saturated = 0 as std::ffi::c_int;
    (*st).adapted = 0 as std::ffi::c_int;
    (*st).sum_adapt = 0 as std::ffi::c_int as spx_word32_t;
    (*st).Pyy = FLOAT_ONE;
    (*st).Pey = (*st).Pyy;
    (*st).Davg2 = 0 as std::ffi::c_int as spx_word32_t;
    (*st).Davg1 = (*st).Davg2;
    (*st).Dvar2 = FLOAT_ZERO;
    (*st).Dvar1 = (*st).Dvar2;
    i = 0 as std::ffi::c_int;
    while i < 3 as std::ffi::c_int * (*st).frame_size {
        *((*st).play_buf).offset(i as isize) = 0 as spx_int16_t;
        i += 1;
    }
    (*st).play_buf_pos = PLAYBACK_DELAY * (*st).frame_size;
    (*st).play_buf_started = 0 as std::ffi::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn speex_echo_state_destroy(st: *mut SpeexEchoState) {
    spx_fft_destroy((*st).fft_table);
    speex_free((*st).e as *mut std::ffi::c_void);
    speex_free((*st).x as *mut std::ffi::c_void);
    speex_free((*st).input as *mut std::ffi::c_void);
    speex_free((*st).y as *mut std::ffi::c_void);
    speex_free((*st).last_y as *mut std::ffi::c_void);
    speex_free((*st).Yf as *mut std::ffi::c_void);
    speex_free((*st).Rf as *mut std::ffi::c_void);
    speex_free((*st).Xf as *mut std::ffi::c_void);
    speex_free((*st).Yh as *mut std::ffi::c_void);
    speex_free((*st).Eh as *mut std::ffi::c_void);
    speex_free((*st).X as *mut std::ffi::c_void);
    speex_free((*st).Y as *mut std::ffi::c_void);
    speex_free((*st).E as *mut std::ffi::c_void);
    speex_free((*st).W as *mut std::ffi::c_void);
    speex_free((*st).foreground as *mut std::ffi::c_void);
    speex_free((*st).PHI as *mut std::ffi::c_void);
    speex_free((*st).power as *mut std::ffi::c_void);
    speex_free((*st).power_1 as *mut std::ffi::c_void);
    speex_free((*st).window as *mut std::ffi::c_void);
    speex_free((*st).prop as *mut std::ffi::c_void);
    speex_free((*st).wtmp as *mut std::ffi::c_void);
    speex_free((*st).memX as *mut std::ffi::c_void);
    speex_free((*st).memD as *mut std::ffi::c_void);
    speex_free((*st).memE as *mut std::ffi::c_void);
    speex_free((*st).notch_mem as *mut std::ffi::c_void);
    speex_free((*st).play_buf as *mut std::ffi::c_void);
    speex_free(st as *mut std::ffi::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn speex_echo_capture(
    st: *mut SpeexEchoState,
    rec: *const spx_int16_t,
    out: *mut spx_int16_t,
) {
    let mut i: std::ffi::c_int = 0;
    (*st).play_buf_started = 1 as std::ffi::c_int;
    if (*st).play_buf_pos >= (*st).frame_size {
        speex_echo_cancellation(st, rec, (*st).play_buf, out);
        (*st).play_buf_pos -= (*st).frame_size;
        i = 0 as std::ffi::c_int;
        while i < (*st).play_buf_pos {
            *((*st).play_buf).offset(i as isize) = *((*st).play_buf)
                .offset((i + (*st).frame_size) as isize);
            i += 1;
        }
    } else {
        speex_warning(
            b"No playback frame available (your application is buggy and/or got xruns)\0"
                as *const u8 as *const std::ffi::c_char,
        );
        if (*st).play_buf_pos != 0 as std::ffi::c_int {
            speex_warning(
                b"internal playback buffer corruption?\0" as *const u8
                    as *const std::ffi::c_char,
            );
            (*st).play_buf_pos = 0 as std::ffi::c_int;
        }
        i = 0 as std::ffi::c_int;
        while i < (*st).frame_size {
            *out.offset(i as isize) = *rec.offset(i as isize);
            i += 1;
        }
    };
}
#[no_mangle]
pub unsafe extern "C" fn speex_echo_playback(
    st: *mut SpeexEchoState,
    play: *const spx_int16_t,
) {
    if (*st).play_buf_started == 0 {
        speex_warning(
            b"discarded first playback frame\0" as *const u8 as *const std::ffi::c_char,
        );
        return;
    }
    if (*st).play_buf_pos <= PLAYBACK_DELAY * (*st).frame_size {
        let mut i: std::ffi::c_int = 0;
        i = 0 as std::ffi::c_int;
        while i < (*st).frame_size {
            *((*st).play_buf).offset(((*st).play_buf_pos + i) as isize) = *play
                .offset(i as isize);
            i += 1;
        }
        (*st).play_buf_pos += (*st).frame_size;
        if (*st).play_buf_pos
            <= (PLAYBACK_DELAY - 1 as std::ffi::c_int) * (*st).frame_size
        {
            speex_warning(
                b"Auto-filling the buffer (your application is buggy and/or got xruns)\0"
                    as *const u8 as *const std::ffi::c_char,
            );
            i = 0 as std::ffi::c_int;
            while i < (*st).frame_size {
                *((*st).play_buf).offset(((*st).play_buf_pos + i) as isize) = *play
                    .offset(i as isize);
                i += 1;
            }
            (*st).play_buf_pos += (*st).frame_size;
        }
    } else {
        speex_warning(
            b"Had to discard a playback frame (your application is buggy and/or got xruns)\0"
                as *const u8 as *const std::ffi::c_char,
        );
    };
}
#[no_mangle]
pub unsafe extern "C" fn speex_echo_cancel(
    st: *mut SpeexEchoState,
    in_0: *const spx_int16_t,
    far_end: *const spx_int16_t,
    out: *mut spx_int16_t,
    Yout: *mut spx_int32_t,
) {
    speex_echo_cancellation(st, in_0, far_end, out);
}
#[no_mangle]
pub unsafe extern "C" fn speex_echo_cancellation(
    st: *mut SpeexEchoState,
    in_0: *const spx_int16_t,
    far_end: *const spx_int16_t,
    out: *mut spx_int16_t,
) {
    let mut i: std::ffi::c_int = 0;
    let mut j: std::ffi::c_int = 0;
    let mut chan: std::ffi::c_int = 0;
    let mut speak: std::ffi::c_int = 0;
    let mut N: std::ffi::c_int = 0;
    let mut M: std::ffi::c_int = 0;
    let mut C: std::ffi::c_int = 0;
    let mut K: std::ffi::c_int = 0;
    let mut Syy: spx_word32_t = 0.;
    let mut See: spx_word32_t = 0.;
    let mut Sxx: spx_word32_t = 0.;
    let mut Sdd: spx_word32_t = 0.;
    let mut Sff: spx_word32_t = 0.;
    let mut Dbf: spx_word32_t = 0.;
    let mut update_foreground: std::ffi::c_int = 0;
    let mut Sey: spx_word32_t = 0.;
    let mut ss: spx_word16_t = 0.;
    let mut ss_1: spx_word16_t = 0.;
    let mut Pey: std::ffi::c_float = FLOAT_ONE;
    let mut Pyy: std::ffi::c_float = FLOAT_ONE;
    let mut alpha: std::ffi::c_float = 0.;
    let mut alpha_1: std::ffi::c_float = 0.;
    let mut RER: spx_word16_t = 0.;
    let mut tmp32: spx_word32_t = 0.;
    N = (*st).window_size;
    M = (*st).M;
    C = (*st).C;
    K = (*st).K;
    (*st).cancel_count += 1;
    ss = (0.35f64 / M as std::ffi::c_double) as spx_word16_t;
    ss_1 = 1 as std::ffi::c_int as spx_word16_t - ss;
    chan = 0 as std::ffi::c_int;
    while chan < C {
        filter_dc_notch16(
            in_0.offset(chan as isize),
            (*st).notch_radius,
            ((*st).input).offset((chan * (*st).frame_size) as isize),
            (*st).frame_size,
            ((*st).notch_mem).offset((2 as std::ffi::c_int * chan) as isize),
            C,
        );
        i = 0 as std::ffi::c_int;
        while i < (*st).frame_size {
            let mut tmp32_0: spx_word32_t = 0.;
            tmp32_0 = (*((*st).input).offset((chan * (*st).frame_size + i) as isize)
                - (*st).preemph * *((*st).memD).offset(chan as isize)) as spx_word32_t;
            *((*st).memD).offset(chan as isize) = *((*st).input)
                .offset((chan * (*st).frame_size + i) as isize);
            *((*st).input).offset((chan * (*st).frame_size + i) as isize) = tmp32_0
                as spx_word16_t;
            i += 1;
        }
        chan += 1;
    }
    speak = 0 as std::ffi::c_int;
    while speak < K {
        i = 0 as std::ffi::c_int;
        while i < (*st).frame_size {
            let mut tmp32_1: spx_word32_t = 0.;
            *((*st).x).offset((speak * N + i) as isize) = *((*st).x)
                .offset((speak * N + i + (*st).frame_size) as isize);
            tmp32_1 = (*far_end.offset((i * K + speak) as isize) as std::ffi::c_int
                as spx_word16_t - (*st).preemph * *((*st).memX).offset(speak as isize))
                as spx_word32_t;
            *((*st).x).offset((speak * N + i + (*st).frame_size) as isize) = tmp32_1
                as spx_word16_t;
            *((*st).memX).offset(speak as isize) = *far_end
                .offset((i * K + speak) as isize) as spx_word16_t;
            i += 1;
        }
        speak += 1;
    }
    speak = 0 as std::ffi::c_int;
    while speak < K {
        j = M - 1 as std::ffi::c_int;
        while j >= 0 as std::ffi::c_int {
            i = 0 as std::ffi::c_int;
            while i < N {
                *((*st).X)
                    .offset(
                        ((j + 1 as std::ffi::c_int) * N * K + speak * N + i) as isize,
                    ) = *((*st).X).offset((j * N * K + speak * N + i) as isize);
                i += 1;
            }
            j -= 1;
        }
        spx_fft(
            (*st).fft_table,
            ((*st).x).offset((speak * N) as isize),
            &mut *((*st).X).offset((speak * N) as isize),
        );
        speak += 1;
    }
    Sxx = 0 as std::ffi::c_int as spx_word32_t;
    speak = 0 as std::ffi::c_int;
    while speak < K {
        Sxx
            += mdf_inner_prod(
                ((*st).x).offset((speak * N) as isize).offset((*st).frame_size as isize),
                ((*st).x).offset((speak * N) as isize).offset((*st).frame_size as isize),
                (*st).frame_size,
            );
        power_spectrum_accum(((*st).X).offset((speak * N) as isize), (*st).Xf, N);
        speak += 1;
    }
    Sff = 0 as std::ffi::c_int as spx_word32_t;
    chan = 0 as std::ffi::c_int;
    while chan < C {
        spectral_mul_accum(
            (*st).X,
            ((*st).foreground).offset((chan * N * K * M) as isize),
            ((*st).Y).offset((chan * N) as isize),
            N,
            M * K,
        );
        spx_ifft(
            (*st).fft_table,
            ((*st).Y).offset((chan * N) as isize),
            ((*st).e).offset((chan * N) as isize),
        );
        i = 0 as std::ffi::c_int;
        while i < (*st).frame_size {
            *((*st).e).offset((chan * N + i) as isize) = *((*st).input)
                .offset((chan * (*st).frame_size + i) as isize)
                - *((*st).e).offset((chan * N + i + (*st).frame_size) as isize);
            i += 1;
        }
        Sff
            += mdf_inner_prod(
                ((*st).e).offset((chan * N) as isize),
                ((*st).e).offset((chan * N) as isize),
                (*st).frame_size,
            );
        chan += 1;
    }
    if (*st).adapted != 0 {
        mdf_adjust_prop((*st).W, N, M, C * K, (*st).prop);
    }
    if (*st).saturated == 0 as std::ffi::c_int {
        chan = 0 as std::ffi::c_int;
        while chan < C {
            speak = 0 as std::ffi::c_int;
            while speak < K {
                j = M - 1 as std::ffi::c_int;
                while j >= 0 as std::ffi::c_int {
                    weighted_spectral_mul_conj(
                        (*st).power_1,
                        *((*st).prop).offset(j as isize),
                        &mut *((*st).X)
                            .offset(
                                ((j + 1 as std::ffi::c_int) * N * K + speak * N) as isize,
                            ),
                        ((*st).E).offset((chan * N) as isize),
                        (*st).PHI,
                        N,
                    );
                    i = 0 as std::ffi::c_int;
                    while i < N {
                        *((*st).W)
                            .offset(
                                (chan * N * K * M + j * N * K + speak * N + i) as isize,
                            ) += *((*st).PHI).offset(i as isize);
                        i += 1;
                    }
                    j -= 1;
                }
                speak += 1;
            }
            chan += 1;
        }
    } else {
        (*st).saturated -= 1;
    }
    chan = 0 as std::ffi::c_int;
    while chan < C {
        speak = 0 as std::ffi::c_int;
        while speak < K {
            j = 0 as std::ffi::c_int;
            while j < M {
                if j == 0 as std::ffi::c_int
                    || (*st).cancel_count % (M - 1 as std::ffi::c_int)
                        == j - 1 as std::ffi::c_int
                {
                    spx_ifft(
                        (*st).fft_table,
                        &mut *((*st).W)
                            .offset((chan * N * K * M + j * N * K + speak * N) as isize),
                        (*st).wtmp,
                    );
                    i = (*st).frame_size;
                    while i < N {
                        *((*st).wtmp).offset(i as isize) = 0 as std::ffi::c_int
                            as spx_word16_t;
                        i += 1;
                    }
                    spx_fft(
                        (*st).fft_table,
                        (*st).wtmp,
                        &mut *((*st).W)
                            .offset((chan * N * K * M + j * N * K + speak * N) as isize),
                    );
                }
                j += 1;
            }
            speak += 1;
        }
        chan += 1;
    }
    i = 0 as std::ffi::c_int;
    while i <= (*st).frame_size {
        let ref mut fresh11 = *((*st).Xf).offset(i as isize);
        *fresh11 = 0 as std::ffi::c_int as spx_word32_t;
        let ref mut fresh12 = *((*st).Yf).offset(i as isize);
        *fresh12 = *fresh11;
        *((*st).Rf).offset(i as isize) = *fresh12;
        i += 1;
    }
    Dbf = 0 as std::ffi::c_int as spx_word32_t;
    See = 0 as std::ffi::c_int as spx_word32_t;
    chan = 0 as std::ffi::c_int;
    while chan < C {
        spectral_mul_accum(
            (*st).X,
            ((*st).W).offset((chan * N * K * M) as isize),
            ((*st).Y).offset((chan * N) as isize),
            N,
            M * K,
        );
        spx_ifft(
            (*st).fft_table,
            ((*st).Y).offset((chan * N) as isize),
            ((*st).y).offset((chan * N) as isize),
        );
        i = 0 as std::ffi::c_int;
        while i < (*st).frame_size {
            *((*st).e).offset((chan * N + i) as isize) = *((*st).e)
                .offset((chan * N + i + (*st).frame_size) as isize)
                - *((*st).y).offset((chan * N + i + (*st).frame_size) as isize);
            i += 1;
        }
        Dbf
            += 10 as std::ffi::c_int as spx_word32_t
                + mdf_inner_prod(
                    ((*st).e).offset((chan * N) as isize),
                    ((*st).e).offset((chan * N) as isize),
                    (*st).frame_size,
                );
        i = 0 as std::ffi::c_int;
        while i < (*st).frame_size {
            *((*st).e).offset((chan * N + i) as isize) = *((*st).input)
                .offset((chan * (*st).frame_size + i) as isize)
                - *((*st).y).offset((chan * N + i + (*st).frame_size) as isize);
            i += 1;
        }
        See
            += mdf_inner_prod(
                ((*st).e).offset((chan * N) as isize),
                ((*st).e).offset((chan * N) as isize),
                (*st).frame_size,
            );
        chan += 1;
    }
    (*st).Davg1 = 0.6f32 * (*st).Davg1 + 0.4f32 * (Sff - See);
    (*st).Davg2 = 0.85f32 * (*st).Davg2 + 0.15f32 * (Sff - See);
    (*st).Dvar1 = (VAR1_SMOOTH as spx_word32_t * (*st).Dvar1 as spx_word32_t
        + 0.4f32 * Sff * (0.4f32 * Dbf)) as std::ffi::c_float;
    (*st).Dvar2 = (VAR2_SMOOTH as spx_word32_t * (*st).Dvar2 as spx_word32_t
        + 0.15f32 * Sff * (0.15f32 * Dbf)) as std::ffi::c_float;
    update_foreground = 0 as std::ffi::c_int;
    if (Sff - See)
        * (if Sff - See < 0 as std::ffi::c_int as spx_word32_t {
            -(Sff - See)
        } else {
            Sff - See
        }) > Sff * Dbf
    {
        update_foreground = 1 as std::ffi::c_int;
    } else if (*st).Davg1
        * (if (*st).Davg1 < 0 as std::ffi::c_int as spx_word32_t {
            -(*st).Davg1
        } else {
            (*st).Davg1
        }) > VAR1_UPDATE * (*st).Dvar1
    {
        update_foreground = 1 as std::ffi::c_int;
    } else if (*st).Davg2
        * (if (*st).Davg2 < 0 as std::ffi::c_int as spx_word32_t {
            -(*st).Davg2
        } else {
            (*st).Davg2
        }) > VAR2_UPDATE * (*st).Dvar2
    {
        update_foreground = 1 as std::ffi::c_int;
    }
    if update_foreground != 0 {
        (*st).Davg2 = 0 as std::ffi::c_int as spx_word32_t;
        (*st).Davg1 = (*st).Davg2;
        (*st).Dvar2 = FLOAT_ZERO;
        (*st).Dvar1 = (*st).Dvar2;
        i = 0 as std::ffi::c_int;
        while i < N * M * C * K {
            *((*st).foreground).offset(i as isize) = *((*st).W).offset(i as isize)
                as spx_word16_t;
            i += 1;
        }
        chan = 0 as std::ffi::c_int;
        while chan < C {
            i = 0 as std::ffi::c_int;
            while i < (*st).frame_size {
                *((*st).e).offset((chan * N + i + (*st).frame_size) as isize) = *((*st)
                    .window)
                    .offset((i + (*st).frame_size) as isize)
                    * *((*st).e).offset((chan * N + i + (*st).frame_size) as isize)
                    + *((*st).window).offset(i as isize)
                        * *((*st).y).offset((chan * N + i + (*st).frame_size) as isize);
                i += 1;
            }
            chan += 1;
        }
    } else {
        let mut reset_background: std::ffi::c_int = 0 as std::ffi::c_int;
        if -(Sff - See)
            * (if Sff - See < 0 as std::ffi::c_int as spx_word32_t {
                -(Sff - See)
            } else {
                Sff - See
            }) > VAR_BACKTRACK as spx_word32_t * (Sff * Dbf)
        {
            reset_background = 1 as std::ffi::c_int;
        }
        if -(*st).Davg1
            * (if (*st).Davg1 < 0 as std::ffi::c_int as spx_word32_t {
                -(*st).Davg1
            } else {
                (*st).Davg1
            }) > VAR_BACKTRACK * (*st).Dvar1
        {
            reset_background = 1 as std::ffi::c_int;
        }
        if -(*st).Davg2
            * (if (*st).Davg2 < 0 as std::ffi::c_int as spx_word32_t {
                -(*st).Davg2
            } else {
                (*st).Davg2
            }) > VAR_BACKTRACK * (*st).Dvar2
        {
            reset_background = 1 as std::ffi::c_int;
        }
        if reset_background != 0 {
            i = 0 as std::ffi::c_int;
            while i < N * M * C * K {
                *((*st).W).offset(i as isize) = *((*st).foreground).offset(i as isize)
                    as spx_word32_t;
                i += 1;
            }
            chan = 0 as std::ffi::c_int;
            while chan < C {
                i = 0 as std::ffi::c_int;
                while i < (*st).frame_size {
                    *((*st).y).offset((chan * N + i + (*st).frame_size) as isize) = *((*st)
                        .e)
                        .offset((chan * N + i + (*st).frame_size) as isize);
                    i += 1;
                }
                i = 0 as std::ffi::c_int;
                while i < (*st).frame_size {
                    *((*st).e).offset((chan * N + i) as isize) = *((*st).input)
                        .offset((chan * (*st).frame_size + i) as isize)
                        - *((*st).y).offset((chan * N + i + (*st).frame_size) as isize);
                    i += 1;
                }
                chan += 1;
            }
            See = Sff;
            (*st).Davg2 = 0 as std::ffi::c_int as spx_word32_t;
            (*st).Davg1 = (*st).Davg2;
            (*st).Dvar2 = FLOAT_ZERO;
            (*st).Dvar1 = (*st).Dvar2;
        }
    }
    Sdd = 0 as std::ffi::c_int as spx_word32_t;
    Syy = Sdd;
    Sey = Syy;
    chan = 0 as std::ffi::c_int;
    while chan < C {
        i = 0 as std::ffi::c_int;
        while i < (*st).frame_size {
            let mut tmp_out: spx_word32_t = 0.;
            tmp_out = (*((*st).input).offset((chan * (*st).frame_size + i) as isize)
                - *((*st).e).offset((chan * N + i + (*st).frame_size) as isize))
                as spx_word32_t;
            tmp_out = (tmp_out as spx_word16_t
                + (*st).preemph * *((*st).memE).offset(chan as isize)) as spx_word32_t;
            if *in_0.offset((i * C + chan) as isize) as std::ffi::c_int
                <= -(32000 as std::ffi::c_int)
                || *in_0.offset((i * C + chan) as isize) as std::ffi::c_int
                    >= 32000 as std::ffi::c_int
            {
                if (*st).saturated == 0 as std::ffi::c_int {
                    (*st).saturated = 1 as std::ffi::c_int;
                }
            }
            *out.offset((i * C + chan) as isize) = (if tmp_out < -32767.5f32 {
                -(32768 as std::ffi::c_int)
            } else if tmp_out > 32766.5f32 {
                32767 as std::ffi::c_int
            } else {
                floor(0.5f64 + tmp_out as std::ffi::c_double) as spx_int16_t
                    as std::ffi::c_int
            }) as spx_int16_t;
            *((*st).memE).offset(chan as isize) = tmp_out as spx_word16_t;
            i += 1;
        }
        i = 0 as std::ffi::c_int;
        while i < (*st).frame_size {
            *((*st).e).offset((chan * N + i + (*st).frame_size) as isize) = *((*st).e)
                .offset((chan * N + i) as isize);
            *((*st).e).offset((chan * N + i) as isize) = 0 as std::ffi::c_int
                as spx_word16_t;
            i += 1;
        }
        Sey
            += mdf_inner_prod(
                ((*st).e).offset((chan * N) as isize).offset((*st).frame_size as isize),
                ((*st).y).offset((chan * N) as isize).offset((*st).frame_size as isize),
                (*st).frame_size,
            );
        Syy
            += mdf_inner_prod(
                ((*st).y).offset((chan * N) as isize).offset((*st).frame_size as isize),
                ((*st).y).offset((chan * N) as isize).offset((*st).frame_size as isize),
                (*st).frame_size,
            );
        Sdd
            += mdf_inner_prod(
                ((*st).input).offset((chan * (*st).frame_size) as isize),
                ((*st).input).offset((chan * (*st).frame_size) as isize),
                (*st).frame_size,
            );
        spx_fft(
            (*st).fft_table,
            ((*st).e).offset((chan * N) as isize),
            ((*st).E).offset((chan * N) as isize),
        );
        i = 0 as std::ffi::c_int;
        while i < (*st).frame_size {
            *((*st).y).offset((i + chan * N) as isize) = 0 as std::ffi::c_int
                as spx_word16_t;
            i += 1;
        }
        spx_fft(
            (*st).fft_table,
            ((*st).y).offset((chan * N) as isize),
            ((*st).Y).offset((chan * N) as isize),
        );
        power_spectrum_accum(((*st).E).offset((chan * N) as isize), (*st).Rf, N);
        power_spectrum_accum(((*st).Y).offset((chan * N) as isize), (*st).Yf, N);
        chan += 1;
    }
    if !(Syy >= 0 as std::ffi::c_int as spx_word32_t
        && Sxx >= 0 as std::ffi::c_int as spx_word32_t
        && See >= 0 as std::ffi::c_int as spx_word32_t)
        || !((Sff as std::ffi::c_double) < N as std::ffi::c_double * 1e9f64
            && (Syy as std::ffi::c_double) < N as std::ffi::c_double * 1e9f64
            && (Sxx as std::ffi::c_double) < N as std::ffi::c_double * 1e9f64)
    {
        (*st).screwed_up += 50 as std::ffi::c_int;
        i = 0 as std::ffi::c_int;
        while i < (*st).frame_size * C {
            *out.offset(i as isize) = 0 as spx_int16_t;
            i += 1;
        }
    } else if Sff > Sdd + N as spx_word32_t * 10000 as std::ffi::c_int as spx_word32_t {
        (*st).screwed_up += 1;
    } else {
        (*st).screwed_up = 0 as std::ffi::c_int;
    }
    if (*st).screwed_up >= 50 as std::ffi::c_int {
        speex_warning(
            b"The echo canceller started acting funny and got slapped (reset). It swears it will behave now.\0"
                as *const u8 as *const std::ffi::c_char,
        );
        speex_echo_state_reset(st);
        return;
    }
    See = if See > N as spx_word32_t * 100 as std::ffi::c_int as spx_word32_t {
        See
    } else {
        N as spx_word32_t * 100 as std::ffi::c_int as spx_word32_t
    };
    speak = 0 as std::ffi::c_int;
    while speak < K {
        Sxx
            += mdf_inner_prod(
                ((*st).x).offset((speak * N) as isize).offset((*st).frame_size as isize),
                ((*st).x).offset((speak * N) as isize).offset((*st).frame_size as isize),
                (*st).frame_size,
            );
        power_spectrum_accum(((*st).X).offset((speak * N) as isize), (*st).Xf, N);
        speak += 1;
    }
    j = 0 as std::ffi::c_int;
    while j <= (*st).frame_size {
        *((*st).power).offset(j as isize) = ss_1 as spx_word32_t
            * *((*st).power).offset(j as isize) + 1 as std::ffi::c_int as spx_word32_t
            + ss as spx_word32_t * *((*st).Xf).offset(j as isize);
        j += 1;
    }
    j = (*st).frame_size;
    while j >= 0 as std::ffi::c_int {
        let mut Eh: std::ffi::c_float = 0.;
        let mut Yh: std::ffi::c_float = 0.;
        Eh = (*((*st).Rf).offset(j as isize) - *((*st).Eh).offset(j as isize))
            as std::ffi::c_float;
        Yh = (*((*st).Yf).offset(j as isize) - *((*st).Yh).offset(j as isize))
            as std::ffi::c_float;
        Pey = Pey + Eh * Yh;
        Pyy = Pyy + Yh * Yh;
        *((*st).Eh).offset(j as isize) = (1 as std::ffi::c_int as spx_word32_t
            - (*st).spec_average as spx_word32_t) * *((*st).Eh).offset(j as isize)
            + (*st).spec_average as spx_word32_t * *((*st).Rf).offset(j as isize);
        *((*st).Yh).offset(j as isize) = (1 as std::ffi::c_int as spx_word32_t
            - (*st).spec_average as spx_word32_t) * *((*st).Yh).offset(j as isize)
            + (*st).spec_average as spx_word32_t * *((*st).Yf).offset(j as isize);
        j -= 1;
    }
    Pyy = sqrt(Pyy as std::ffi::c_double) as std::ffi::c_float;
    Pey = Pey / Pyy;
    tmp32 = (*st).beta0 as spx_word32_t * Syy;
    if tmp32 > (*st).beta_max as spx_word32_t * See {
        tmp32 = (*st).beta_max as spx_word32_t * See;
    }
    alpha = (tmp32 / See) as std::ffi::c_float;
    alpha_1 = 1.0f32 - alpha;
    (*st).Pey = alpha_1 * (*st).Pey + alpha * Pey;
    (*st).Pyy = alpha_1 * (*st).Pyy + alpha * Pyy;
    if (*st).Pyy < 1.0f32 {
        (*st).Pyy = FLOAT_ONE;
    }
    if (*st).Pey < MIN_LEAK * (*st).Pyy {
        (*st).Pey = MIN_LEAK * (*st).Pyy;
    }
    if (*st).Pey > (*st).Pyy {
        (*st).Pey = (*st).Pyy;
    }
    (*st).leak_estimate = ((*st).Pey / (*st).Pyy) as spx_word16_t;
    if (*st).leak_estimate > 16383 as std::ffi::c_int as spx_word16_t {
        (*st).leak_estimate = 32767 as std::ffi::c_int as spx_word16_t;
    } else {
        (*st).leak_estimate = (*st).leak_estimate;
    }
    RER = ((0.0001f64 * Sxx as std::ffi::c_double
        + 3.0f64 * ((*st).leak_estimate as spx_word32_t * Syy) as std::ffi::c_double)
        / See as std::ffi::c_double) as spx_word16_t;
    if RER < Sey * Sey / (1 as std::ffi::c_int as spx_word32_t + See * Syy) {
        RER = (Sey * Sey / (1 as std::ffi::c_int as spx_word32_t + See * Syy))
            as spx_word16_t;
    }
    if RER as std::ffi::c_double > 0.5f64 {
        RER = 0.5f32;
    }
    if (*st).adapted == 0 && (*st).sum_adapt > M as spx_word32_t
        && (*st).leak_estimate as spx_word32_t * Syy > 0.03f32 * Syy
    {
        (*st).adapted = 1 as std::ffi::c_int;
    }
    if (*st).adapted != 0 {
        i = 0 as std::ffi::c_int;
        while i <= (*st).frame_size {
            let mut r: spx_word32_t = 0.;
            let mut e: spx_word32_t = 0.;
            r = (*st).leak_estimate as spx_word32_t * *((*st).Yf).offset(i as isize);
            e = *((*st).Rf).offset(i as isize) + 1 as std::ffi::c_int as spx_word32_t;
            if r as std::ffi::c_double > 0.5f64 * e as std::ffi::c_double {
                r = (0.5f64 * e as std::ffi::c_double) as spx_word32_t;
            }
            r = (0.7f64 * r as std::ffi::c_double
                + 0.3f64 * (RER as spx_word32_t * e) as std::ffi::c_double)
                as spx_word32_t;
            *((*st).power_1).offset(i as isize) = (r
                / (e
                    * (*((*st).power).offset(i as isize)
                        + 10 as std::ffi::c_int as spx_word32_t))) as std::ffi::c_float;
            i += 1;
        }
    } else {
        let mut adapt_rate: spx_word16_t = 0 as std::ffi::c_int as spx_word16_t;
        if Sxx > N as spx_word32_t * 1000 as std::ffi::c_int as spx_word32_t {
            tmp32 = 0.25f32 * Sxx;
            if tmp32 as std::ffi::c_double > 0.25f64 * See as std::ffi::c_double {
                tmp32 = (0.25f64 * See as std::ffi::c_double) as spx_word32_t;
            }
            adapt_rate = (tmp32 / See) as spx_word16_t;
        }
        i = 0 as std::ffi::c_int;
        while i <= (*st).frame_size {
            *((*st).power_1).offset(i as isize) = (adapt_rate as spx_word32_t
                / (*((*st).power).offset(i as isize)
                    + 10 as std::ffi::c_int as spx_word32_t)) as std::ffi::c_float;
            i += 1;
        }
        (*st).sum_adapt = ((*st).sum_adapt as spx_word16_t + adapt_rate) as spx_word32_t;
    }
    i = 0 as std::ffi::c_int;
    while i < (*st).frame_size {
        *((*st).last_y).offset(i as isize) = *((*st).last_y)
            .offset(((*st).frame_size + i) as isize);
        i += 1;
    }
    if (*st).adapted != 0 {
        i = 0 as std::ffi::c_int;
        while i < (*st).frame_size {
            *((*st).last_y).offset(((*st).frame_size + i) as isize) = (*in_0
                .offset(i as isize) as std::ffi::c_int
                - *out.offset(i as isize) as std::ffi::c_int) as spx_word16_t;
            i += 1;
        }
    }
}
#[no_mangle]
pub unsafe extern "C" fn speex_echo_get_residual(
    st: *mut SpeexEchoState,
    residual_echo: *mut spx_word32_t,
    len: std::ffi::c_int,
) {
    let mut i: std::ffi::c_int = 0;
    let mut leak2: spx_word16_t = 0.;
    let mut N: std::ffi::c_int = 0;
    N = (*st).window_size;
    i = 0 as std::ffi::c_int;
    while i < N {
        *((*st).y).offset(i as isize) = *((*st).window).offset(i as isize)
            * *((*st).last_y).offset(i as isize);
        i += 1;
    }
    spx_fft((*st).fft_table, (*st).y, (*st).Y);
    power_spectrum((*st).Y, residual_echo, N);
    if (*st).leak_estimate as std::ffi::c_double > 0.5f64 {
        leak2 = 1 as std::ffi::c_int as spx_word16_t;
    } else {
        leak2 = 2 as std::ffi::c_int as spx_word16_t * (*st).leak_estimate;
    }
    i = 0 as std::ffi::c_int;
    while i <= (*st).frame_size {
        *residual_echo.offset(i as isize) = (leak2 as spx_word32_t
            * *residual_echo.offset(i as isize)) as spx_int32_t as spx_word32_t;
        i += 1;
    }
}
#[no_mangle]
pub unsafe extern "C" fn speex_echo_ctl(
    st: *mut SpeexEchoState,
    request: std::ffi::c_int,
    ptr: *mut std::ffi::c_void,
) -> std::ffi::c_int {
    match request {
        SPEEX_ECHO_GET_FRAME_SIZE => {
            *(ptr as *mut std::ffi::c_int) = (*st).frame_size;
        }
        SPEEX_ECHO_SET_SAMPLING_RATE => {
            (*st).sampling_rate = *(ptr as *mut std::ffi::c_int) as spx_int32_t;
            (*st).spec_average = (*st).frame_size as spx_word16_t
                / (*st).sampling_rate as spx_word16_t;
            (*st).beta0 = (2.0f32 * (*st).frame_size as std::ffi::c_float
                / (*st).sampling_rate as std::ffi::c_float) as spx_word16_t;
            (*st).beta_max = (0.5f32 * (*st).frame_size as std::ffi::c_float
                / (*st).sampling_rate as std::ffi::c_float) as spx_word16_t;
            if (*st).sampling_rate < 12000 as spx_int32_t {
                (*st).notch_radius = 0.9f32;
            } else if (*st).sampling_rate < 24000 as spx_int32_t {
                (*st).notch_radius = 0.982f32;
            } else {
                (*st).notch_radius = 0.992f32;
            }
        }
        SPEEX_ECHO_GET_SAMPLING_RATE => {
            *(ptr as *mut std::ffi::c_int) = (*st).sampling_rate as std::ffi::c_int;
        }
        SPEEX_ECHO_GET_IMPULSE_RESPONSE_SIZE => {
            *(ptr as *mut spx_int32_t) = ((*st).M * (*st).frame_size) as spx_int32_t;
        }
        SPEEX_ECHO_GET_IMPULSE_RESPONSE => {
            let M: std::ffi::c_int = (*st).M;
            let N: std::ffi::c_int = (*st).window_size;
            let n: std::ffi::c_int = (*st).frame_size;
            let mut i: std::ffi::c_int = 0;
            let mut j: std::ffi::c_int = 0;
            let filt: *mut spx_int32_t = ptr as *mut spx_int32_t;
            j = 0 as std::ffi::c_int;
            while j < M {
                spx_ifft(
                    (*st).fft_table,
                    &mut *((*st).W).offset((j * N) as isize),
                    (*st).wtmp,
                );
                i = 0 as std::ffi::c_int;
                while i < n {
                    *filt.offset((j * n + i) as isize) = (32767 as std::ffi::c_int
                        as spx_word32_t * *((*st).wtmp).offset(i as isize))
                        as spx_int32_t;
                    i += 1;
                }
                j += 1;
            }
        }
        _ => {
            speex_warning_int(
                b"Unknown speex_echo_ctl request: \0" as *const u8
                    as *const std::ffi::c_char,
                request,
            );
            return -(1 as std::ffi::c_int);
        }
    }
    return 0 as std::ffi::c_int;
}
