#![allow(unused_assignments)]
#![allow(unused_variables)]

extern "C" {
    fn calloc(__nmemb: size_t, __size: size_t) -> *mut std::ffi::c_void;
    fn realloc(__ptr: *mut std::ffi::c_void, __size: size_t) -> *mut std::ffi::c_void;
    fn free(__ptr: *mut std::ffi::c_void);
    fn sin(__x: std::ffi::c_double) -> std::ffi::c_double;
    fn fabs(__x: std::ffi::c_double) -> std::ffi::c_double;
    fn floor(__x: std::ffi::c_double) -> std::ffi::c_double;
}
pub type __int16_t = i16;
pub type __int32_t = i32;
pub type __uint32_t = u32;
pub type int16_t = __int16_t;
pub type int32_t = __int32_t;
pub type uint32_t = __uint32_t;
pub type spx_int16_t = int16_t;
pub type spx_int32_t = int32_t;
pub type spx_uint32_t = uint32_t;
pub type C2RustUnnamed = std::ffi::c_uint;
pub const RESAMPLER_ERR_MAX_ERROR: C2RustUnnamed = 6;
pub const RESAMPLER_ERR_OVERFLOW: C2RustUnnamed = 5;
pub const RESAMPLER_ERR_PTR_OVERLAP: C2RustUnnamed = 4;
pub const RESAMPLER_ERR_INVALID_ARG: C2RustUnnamed = 3;
pub const RESAMPLER_ERR_BAD_STATE: C2RustUnnamed = 2;
pub const RESAMPLER_ERR_ALLOC_FAILED: C2RustUnnamed = 1;
pub const RESAMPLER_ERR_SUCCESS: C2RustUnnamed = 0;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct SpeexResamplerState_ {
    pub in_rate: spx_uint32_t,
    pub out_rate: spx_uint32_t,
    pub num_rate: spx_uint32_t,
    pub den_rate: spx_uint32_t,
    pub quality: std::ffi::c_int,
    pub nb_channels: spx_uint32_t,
    pub filt_len: spx_uint32_t,
    pub mem_alloc_size: spx_uint32_t,
    pub buffer_size: spx_uint32_t,
    pub int_advance: std::ffi::c_int,
    pub frac_advance: std::ffi::c_int,
    pub cutoff: std::ffi::c_float,
    pub oversample: spx_uint32_t,
    pub initialised: std::ffi::c_int,
    pub started: std::ffi::c_int,
    pub last_sample: *mut spx_int32_t,
    pub samp_frac_num: *mut spx_uint32_t,
    pub magic_samples: *mut spx_uint32_t,
    pub mem: *mut spx_word16_t,
    pub sinc_table: *mut spx_word16_t,
    pub sinc_table_length: spx_uint32_t,
    pub resampler_ptr: resampler_basic_func,
    pub in_stride: std::ffi::c_int,
    pub out_stride: std::ffi::c_int,
}
pub type resampler_basic_func = Option<
    unsafe extern "C" fn(
        *mut SpeexResamplerState,
        spx_uint32_t,
        *const spx_word16_t,
        *mut spx_uint32_t,
        *mut spx_word16_t,
        *mut spx_uint32_t,
    ) -> std::ffi::c_int,
>;
pub type spx_word16_t = std::ffi::c_float;
pub type SpeexResamplerState = SpeexResamplerState_;
pub type size_t = usize;
pub type spx_word32_t = std::ffi::c_float;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct FuncDef {
    pub table: *const std::ffi::c_double,
    pub oversample: std::ffi::c_int,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct QualityMapping {
    pub base_length: std::ffi::c_int,
    pub oversample: std::ffi::c_int,
    pub downsample_bandwidth: std::ffi::c_float,
    pub upsample_bandwidth: std::ffi::c_float,
    pub window_func: *const FuncDef,
}
pub const __INT_MAX__: std::ffi::c_int = unsafe { 2147483647 as std::ffi::c_int };
pub const UINT32_MAX: std::ffi::c_uint = unsafe { 4294967295 as std::ffi::c_uint };
pub const NULL: std::ffi::c_int = unsafe { 0 as std::ffi::c_int };
#[inline]
unsafe extern "C" fn speex_alloc(mut size: std::ffi::c_int) -> *mut std::ffi::c_void {
    return calloc(size as size_t, 1 as size_t);
}
#[inline]
unsafe extern "C" fn speex_realloc(
    mut ptr: *mut std::ffi::c_void,
    mut size: std::ffi::c_int,
) -> *mut std::ffi::c_void {
    return realloc(ptr, size as size_t);
}
#[inline]
unsafe extern "C" fn speex_free(mut ptr: *mut std::ffi::c_void) {
    free(ptr);
}
pub const M_PI: std::ffi::c_double = unsafe { 3.14159265358979323846f64 };
pub const INT_MAX: std::ffi::c_int = unsafe { 2147483647 as std::ffi::c_int };
pub const FIXED_STACK_ALLOC: std::ffi::c_int = unsafe { 1024 as std::ffi::c_int };
static mut kaiser12_table: [std::ffi::c_double; 68] = [
    0.99859849f64,
    1.00000000f64,
    0.99859849f64,
    0.99440475f64,
    0.98745105f64,
    0.97779076f64,
    0.96549770f64,
    0.95066529f64,
    0.93340547f64,
    0.91384741f64,
    0.89213598f64,
    0.86843014f64,
    0.84290116f64,
    0.81573067f64,
    0.78710866f64,
    0.75723148f64,
    0.72629970f64,
    0.69451601f64,
    0.66208321f64,
    0.62920216f64,
    0.59606986f64,
    0.56287762f64,
    0.52980938f64,
    0.49704014f64,
    0.46473455f64,
    0.43304576f64,
    0.40211431f64,
    0.37206735f64,
    0.34301800f64,
    0.31506490f64,
    0.28829195f64,
    0.26276832f64,
    0.23854851f64,
    0.21567274f64,
    0.19416736f64,
    0.17404546f64,
    0.15530766f64,
    0.13794294f64,
    0.12192957f64,
    0.10723616f64,
    0.09382272f64,
    0.08164178f64,
    0.07063950f64,
    0.06075685f64,
    0.05193064f64,
    0.04409466f64,
    0.03718069f64,
    0.03111947f64,
    0.02584161f64,
    0.02127838f64,
    0.01736250f64,
    0.01402878f64,
    0.01121463f64,
    0.00886058f64,
    0.00691064f64,
    0.00531256f64,
    0.00401805f64,
    0.00298291f64,
    0.00216702f64,
    0.00153438f64,
    0.00105297f64,
    0.00069463f64,
    0.00043489f64,
    0.00025272f64,
    0.00013031f64,
    0.0000527734f64,
    0.00001000f64,
    0.00000000f64,
];
static mut kaiser10_table: [std::ffi::c_double; 36] = [
    0.99537781f64,
    1.00000000f64,
    0.99537781f64,
    0.98162644f64,
    0.95908712f64,
    0.92831446f64,
    0.89005583f64,
    0.84522401f64,
    0.79486424f64,
    0.74011713f64,
    0.68217934f64,
    0.62226347f64,
    0.56155915f64,
    0.50119680f64,
    0.44221549f64,
    0.38553619f64,
    0.33194107f64,
    0.28205962f64,
    0.23636152f64,
    0.19515633f64,
    0.15859932f64,
    0.12670280f64,
    0.09935205f64,
    0.07632451f64,
    0.05731132f64,
    0.04193980f64,
    0.02979584f64,
    0.02044510f64,
    0.01345224f64,
    0.00839739f64,
    0.00488951f64,
    0.00257636f64,
    0.00115101f64,
    0.00035515f64,
    0.00000000f64,
    0.00000000f64,
];
static mut kaiser8_table: [std::ffi::c_double; 36] = [
    0.99635258f64,
    1.00000000f64,
    0.99635258f64,
    0.98548012f64,
    0.96759014f64,
    0.94302200f64,
    0.91223751f64,
    0.87580811f64,
    0.83439927f64,
    0.78875245f64,
    0.73966538f64,
    0.68797126f64,
    0.63451750f64,
    0.58014482f64,
    0.52566725f64,
    0.47185369f64,
    0.41941150f64,
    0.36897272f64,
    0.32108304f64,
    0.27619388f64,
    0.23465776f64,
    0.19672670f64,
    0.16255380f64,
    0.13219758f64,
    0.10562887f64,
    0.08273982f64,
    0.06335451f64,
    0.04724088f64,
    0.03412321f64,
    0.02369490f64,
    0.01563093f64,
    0.00959968f64,
    0.00527363f64,
    0.00233883f64,
    0.00050000f64,
    0.00000000f64,
];
static mut kaiser6_table: [std::ffi::c_double; 36] = [
    0.99733006f64,
    1.00000000f64,
    0.99733006f64,
    0.98935595f64,
    0.97618418f64,
    0.95799003f64,
    0.93501423f64,
    0.90755855f64,
    0.87598009f64,
    0.84068475f64,
    0.80211977f64,
    0.76076565f64,
    0.71712752f64,
    0.67172623f64,
    0.62508937f64,
    0.57774224f64,
    0.53019925f64,
    0.48295561f64,
    0.43647969f64,
    0.39120616f64,
    0.34752997f64,
    0.30580127f64,
    0.26632152f64,
    0.22934058f64,
    0.19505503f64,
    0.16360756f64,
    0.13508755f64,
    0.10953262f64,
    0.08693120f64,
    0.06722600f64,
    0.05031820f64,
    0.03607231f64,
    0.02432151f64,
    0.01487334f64,
    0.00752000f64,
    0.00000000f64,
];
static mut kaiser12_funcdef: FuncDef = unsafe {
    {
        let mut init = FuncDef {
            table: kaiser12_table.as_ptr(),
            oversample: 64 as std::ffi::c_int,
        };
        init
    }
};
static mut kaiser10_funcdef: FuncDef = unsafe {
    {
        let mut init = FuncDef {
            table: kaiser10_table.as_ptr(),
            oversample: 32 as std::ffi::c_int,
        };
        init
    }
};
static mut kaiser8_funcdef: FuncDef = unsafe {
    {
        let mut init = FuncDef {
            table: kaiser8_table.as_ptr(),
            oversample: 32 as std::ffi::c_int,
        };
        init
    }
};
static mut kaiser6_funcdef: FuncDef = unsafe {
    {
        let mut init = FuncDef {
            table: kaiser6_table.as_ptr(),
            oversample: 32 as std::ffi::c_int,
        };
        init
    }
};
static mut quality_map: [QualityMapping; 11] = unsafe {
    [
        {
            let mut init = QualityMapping {
                base_length: 8 as std::ffi::c_int,
                oversample: 4 as std::ffi::c_int,
                downsample_bandwidth: 0.830f32,
                upsample_bandwidth: 0.860f32,
                window_func: &kaiser6_funcdef as *const FuncDef,
            };
            init
        },
        {
            let mut init = QualityMapping {
                base_length: 16 as std::ffi::c_int,
                oversample: 4 as std::ffi::c_int,
                downsample_bandwidth: 0.850f32,
                upsample_bandwidth: 0.880f32,
                window_func: &kaiser6_funcdef as *const FuncDef,
            };
            init
        },
        {
            let mut init = QualityMapping {
                base_length: 32 as std::ffi::c_int,
                oversample: 4 as std::ffi::c_int,
                downsample_bandwidth: 0.882f32,
                upsample_bandwidth: 0.910f32,
                window_func: &kaiser6_funcdef as *const FuncDef,
            };
            init
        },
        {
            let mut init = QualityMapping {
                base_length: 48 as std::ffi::c_int,
                oversample: 8 as std::ffi::c_int,
                downsample_bandwidth: 0.895f32,
                upsample_bandwidth: 0.917f32,
                window_func: &kaiser8_funcdef as *const FuncDef,
            };
            init
        },
        {
            let mut init = QualityMapping {
                base_length: 64 as std::ffi::c_int,
                oversample: 8 as std::ffi::c_int,
                downsample_bandwidth: 0.921f32,
                upsample_bandwidth: 0.940f32,
                window_func: &kaiser8_funcdef as *const FuncDef,
            };
            init
        },
        {
            let mut init = QualityMapping {
                base_length: 80 as std::ffi::c_int,
                oversample: 16 as std::ffi::c_int,
                downsample_bandwidth: 0.922f32,
                upsample_bandwidth: 0.940f32,
                window_func: &kaiser10_funcdef as *const FuncDef,
            };
            init
        },
        {
            let mut init = QualityMapping {
                base_length: 96 as std::ffi::c_int,
                oversample: 16 as std::ffi::c_int,
                downsample_bandwidth: 0.940f32,
                upsample_bandwidth: 0.945f32,
                window_func: &kaiser10_funcdef as *const FuncDef,
            };
            init
        },
        {
            let mut init = QualityMapping {
                base_length: 128 as std::ffi::c_int,
                oversample: 16 as std::ffi::c_int,
                downsample_bandwidth: 0.950f32,
                upsample_bandwidth: 0.950f32,
                window_func: &kaiser10_funcdef as *const FuncDef,
            };
            init
        },
        {
            let mut init = QualityMapping {
                base_length: 160 as std::ffi::c_int,
                oversample: 16 as std::ffi::c_int,
                downsample_bandwidth: 0.960f32,
                upsample_bandwidth: 0.960f32,
                window_func: &kaiser10_funcdef as *const FuncDef,
            };
            init
        },
        {
            let mut init = QualityMapping {
                base_length: 192 as std::ffi::c_int,
                oversample: 32 as std::ffi::c_int,
                downsample_bandwidth: 0.968f32,
                upsample_bandwidth: 0.968f32,
                window_func: &kaiser12_funcdef as *const FuncDef,
            };
            init
        },
        {
            let mut init = QualityMapping {
                base_length: 256 as std::ffi::c_int,
                oversample: 32 as std::ffi::c_int,
                downsample_bandwidth: 0.975f32,
                upsample_bandwidth: 0.975f32,
                window_func: &kaiser12_funcdef as *const FuncDef,
            };
            init
        },
    ]
};
unsafe extern "C" fn compute_func(
    mut x: std::ffi::c_float,
    mut func: *const FuncDef,
) -> std::ffi::c_double {
    let mut y: std::ffi::c_float = 0.;
    let mut frac: std::ffi::c_float = 0.;
    let mut interp: [std::ffi::c_double; 4] = [0.; 4];
    let mut ind: std::ffi::c_int = 0;
    y = x * (*func).oversample as std::ffi::c_float;
    ind = floor(y as std::ffi::c_double) as std::ffi::c_int;
    frac = y - ind as std::ffi::c_float;
    interp[3 as std::ffi::c_int as usize] = -0.1666666667f64 * frac as std::ffi::c_double
        + 0.1666666667f64 * (frac * frac * frac) as std::ffi::c_double;
    interp[2 as std::ffi::c_int as usize] = frac as std::ffi::c_double
        + 0.5f64 * (frac * frac) as std::ffi::c_double
        - 0.5f64 * (frac * frac * frac) as std::ffi::c_double;
    interp[0 as std::ffi::c_int as usize] = -0.3333333333f64 * frac as std::ffi::c_double
        + 0.5f64 * (frac * frac) as std::ffi::c_double
        - 0.1666666667f64 * (frac * frac * frac) as std::ffi::c_double;
    interp[1 as std::ffi::c_int as usize] = 1.0f64
        - interp[3 as std::ffi::c_int as usize]
        - interp[2 as std::ffi::c_int as usize]
        - interp[0 as std::ffi::c_int as usize];
    return interp[0 as std::ffi::c_int as usize] * *((*func).table).offset(ind as isize)
        + interp[1 as std::ffi::c_int as usize]
            * *((*func).table).offset((ind + 1 as std::ffi::c_int) as isize)
        + interp[2 as std::ffi::c_int as usize]
            * *((*func).table).offset((ind + 2 as std::ffi::c_int) as isize)
        + interp[3 as std::ffi::c_int as usize]
            * *((*func).table).offset((ind + 3 as std::ffi::c_int) as isize);
}
unsafe extern "C" fn sinc(
    mut cutoff: std::ffi::c_float,
    mut x: std::ffi::c_float,
    mut N: std::ffi::c_int,
    mut window_func: *const FuncDef,
) -> spx_word16_t {
    let mut xx: std::ffi::c_float = x * cutoff;
    if fabs(x as std::ffi::c_double) < 1e-6f64 {
        return cutoff as spx_word16_t;
    } else if fabs(x as std::ffi::c_double) > 0.5f64 * N as std::ffi::c_double {
        return 0 as std::ffi::c_int as spx_word16_t;
    }
    return (cutoff as std::ffi::c_double * sin(M_PI * xx as std::ffi::c_double)
        / (M_PI * xx as std::ffi::c_double)
        * compute_func(
            fabs(2.0f64 * x as std::ffi::c_double / N as std::ffi::c_double) as std::ffi::c_float,
            window_func,
        )) as spx_word16_t;
}
unsafe extern "C" fn cubic_coef(mut frac: spx_word16_t, mut interp: *mut spx_word16_t) {
    *interp.offset(0 as std::ffi::c_int as isize) =
        -0.16667f32 * frac + 0.16667f32 * frac * frac * frac;
    *interp.offset(1 as std::ffi::c_int as isize) =
        frac + 0.5f32 * frac * frac - 0.5f32 * frac * frac * frac;
    *interp.offset(3 as std::ffi::c_int as isize) =
        -0.33333f32 * frac + 0.5f32 * frac * frac - 0.16667f32 * frac * frac * frac;
    *interp.offset(2 as std::ffi::c_int as isize) = (1.0f64
        - *interp.offset(0 as std::ffi::c_int as isize) as std::ffi::c_double
        - *interp.offset(1 as std::ffi::c_int as isize) as std::ffi::c_double
        - *interp.offset(3 as std::ffi::c_int as isize) as std::ffi::c_double)
        as spx_word16_t;
}
unsafe extern "C" fn resampler_basic_direct_single(
    mut st: *mut SpeexResamplerState,
    mut channel_index: spx_uint32_t,
    mut in_0: *const spx_word16_t,
    mut in_len: *mut spx_uint32_t,
    mut out: *mut spx_word16_t,
    mut out_len: *mut spx_uint32_t,
) -> std::ffi::c_int {
    let N: std::ffi::c_int = (*st).filt_len as std::ffi::c_int;
    let mut out_sample: std::ffi::c_int = 0 as std::ffi::c_int;
    let mut last_sample: std::ffi::c_int = *((*st).last_sample).offset(channel_index as isize);
    let mut samp_frac_num: spx_uint32_t = *((*st).samp_frac_num).offset(channel_index as isize);
    let mut sinc_table: *const spx_word16_t = (*st).sinc_table;
    let out_stride: std::ffi::c_int = (*st).out_stride;
    let int_advance: std::ffi::c_int = (*st).int_advance;
    let frac_advance: std::ffi::c_int = (*st).frac_advance;
    let den_rate: spx_uint32_t = (*st).den_rate;
    let mut sum: spx_word32_t = 0.;
    while !(last_sample as spx_int32_t >= *in_len as spx_int32_t
        || out_sample as spx_int32_t >= *out_len as spx_int32_t)
    {
        let mut sinct: *const spx_word16_t = &*sinc_table
            .offset(samp_frac_num.wrapping_mul(N as spx_uint32_t) as isize)
            as *const spx_word16_t;
        let mut iptr: *const spx_word16_t =
            &*in_0.offset(last_sample as isize) as *const spx_word16_t;
        let mut j: std::ffi::c_int = 0;
        sum = 0 as std::ffi::c_int as spx_word32_t;
        j = 0 as std::ffi::c_int;
        while j < N {
            sum += *sinct.offset(j as isize) * *iptr.offset(j as isize);
            j += 1;
        }
        sum = sum;
        let fresh0 = out_sample;
        out_sample = out_sample + 1;
        *out.offset((out_stride * fresh0) as isize) = sum as spx_word16_t;
        last_sample += int_advance;
        samp_frac_num = samp_frac_num.wrapping_add(frac_advance as spx_uint32_t);
        if samp_frac_num >= den_rate {
            samp_frac_num = samp_frac_num.wrapping_sub(den_rate);
            last_sample += 1;
        }
    }
    *((*st).last_sample).offset(channel_index as isize) = last_sample as spx_int32_t;
    *((*st).samp_frac_num).offset(channel_index as isize) = samp_frac_num;
    return out_sample;
}
unsafe extern "C" fn resampler_basic_direct_double(
    mut st: *mut SpeexResamplerState,
    mut channel_index: spx_uint32_t,
    mut in_0: *const spx_word16_t,
    mut in_len: *mut spx_uint32_t,
    mut out: *mut spx_word16_t,
    mut out_len: *mut spx_uint32_t,
) -> std::ffi::c_int {
    let N: std::ffi::c_int = (*st).filt_len as std::ffi::c_int;
    let mut out_sample: std::ffi::c_int = 0 as std::ffi::c_int;
    let mut last_sample: std::ffi::c_int = *((*st).last_sample).offset(channel_index as isize);
    let mut samp_frac_num: spx_uint32_t = *((*st).samp_frac_num).offset(channel_index as isize);
    let mut sinc_table: *const spx_word16_t = (*st).sinc_table;
    let out_stride: std::ffi::c_int = (*st).out_stride;
    let int_advance: std::ffi::c_int = (*st).int_advance;
    let frac_advance: std::ffi::c_int = (*st).frac_advance;
    let den_rate: spx_uint32_t = (*st).den_rate;
    let mut sum: std::ffi::c_double = 0.;
    while !(last_sample as spx_int32_t >= *in_len as spx_int32_t
        || out_sample as spx_int32_t >= *out_len as spx_int32_t)
    {
        let mut sinct: *const spx_word16_t = &*sinc_table
            .offset(samp_frac_num.wrapping_mul(N as spx_uint32_t) as isize)
            as *const spx_word16_t;
        let mut iptr: *const spx_word16_t =
            &*in_0.offset(last_sample as isize) as *const spx_word16_t;
        let mut j: std::ffi::c_int = 0;
        let mut accum: [std::ffi::c_double; 4] = [
            0 as std::ffi::c_int as std::ffi::c_double,
            0 as std::ffi::c_int as std::ffi::c_double,
            0 as std::ffi::c_int as std::ffi::c_double,
            0 as std::ffi::c_int as std::ffi::c_double,
        ];
        j = 0 as std::ffi::c_int;
        while j < N {
            accum[0 as std::ffi::c_int as usize] +=
                (*sinct.offset(j as isize) * *iptr.offset(j as isize)) as std::ffi::c_double;
            accum[1 as std::ffi::c_int as usize] += (*sinct
                .offset((j + 1 as std::ffi::c_int) as isize)
                * *iptr.offset((j + 1 as std::ffi::c_int) as isize))
                as std::ffi::c_double;
            accum[2 as std::ffi::c_int as usize] += (*sinct
                .offset((j + 2 as std::ffi::c_int) as isize)
                * *iptr.offset((j + 2 as std::ffi::c_int) as isize))
                as std::ffi::c_double;
            accum[3 as std::ffi::c_int as usize] += (*sinct
                .offset((j + 3 as std::ffi::c_int) as isize)
                * *iptr.offset((j + 3 as std::ffi::c_int) as isize))
                as std::ffi::c_double;
            j += 4 as std::ffi::c_int;
        }
        sum = accum[0 as std::ffi::c_int as usize]
            + accum[1 as std::ffi::c_int as usize]
            + accum[2 as std::ffi::c_int as usize]
            + accum[3 as std::ffi::c_int as usize];
        let fresh1 = out_sample;
        out_sample = out_sample + 1;
        *out.offset((out_stride * fresh1) as isize) = sum as spx_word16_t;
        last_sample += int_advance;
        samp_frac_num = samp_frac_num.wrapping_add(frac_advance as spx_uint32_t);
        if samp_frac_num >= den_rate {
            samp_frac_num = samp_frac_num.wrapping_sub(den_rate);
            last_sample += 1;
        }
    }
    *((*st).last_sample).offset(channel_index as isize) = last_sample as spx_int32_t;
    *((*st).samp_frac_num).offset(channel_index as isize) = samp_frac_num;
    return out_sample;
}
unsafe extern "C" fn resampler_basic_interpolate_single(
    mut st: *mut SpeexResamplerState,
    mut channel_index: spx_uint32_t,
    mut in_0: *const spx_word16_t,
    mut in_len: *mut spx_uint32_t,
    mut out: *mut spx_word16_t,
    mut out_len: *mut spx_uint32_t,
) -> std::ffi::c_int {
    let N: std::ffi::c_int = (*st).filt_len as std::ffi::c_int;
    let mut out_sample: std::ffi::c_int = 0 as std::ffi::c_int;
    let mut last_sample: std::ffi::c_int = *((*st).last_sample).offset(channel_index as isize);
    let mut samp_frac_num: spx_uint32_t = *((*st).samp_frac_num).offset(channel_index as isize);
    let out_stride: std::ffi::c_int = (*st).out_stride;
    let int_advance: std::ffi::c_int = (*st).int_advance;
    let frac_advance: std::ffi::c_int = (*st).frac_advance;
    let den_rate: spx_uint32_t = (*st).den_rate;
    let mut sum: spx_word32_t = 0.;
    while !(last_sample as spx_int32_t >= *in_len as spx_int32_t
        || out_sample as spx_int32_t >= *out_len as spx_int32_t)
    {
        let mut iptr: *const spx_word16_t =
            &*in_0.offset(last_sample as isize) as *const spx_word16_t;
        let offset: std::ffi::c_int = samp_frac_num
            .wrapping_mul((*st).oversample)
            .wrapping_div((*st).den_rate) as std::ffi::c_int;
        let frac: spx_word16_t = samp_frac_num
            .wrapping_mul((*st).oversample)
            .wrapping_rem((*st).den_rate) as spx_word16_t
            / (*st).den_rate as spx_word16_t;
        let mut interp: [spx_word16_t; 4] = [0.; 4];
        let mut j: std::ffi::c_int = 0;
        let mut accum: [spx_word32_t; 4] = [
            0 as std::ffi::c_int as spx_word32_t,
            0 as std::ffi::c_int as spx_word32_t,
            0 as std::ffi::c_int as spx_word32_t,
            0 as std::ffi::c_int as spx_word32_t,
        ];
        j = 0 as std::ffi::c_int;
        while j < N {
            let curr_in: spx_word16_t = *iptr.offset(j as isize);
            accum[0 as std::ffi::c_int as usize] += curr_in
                * *((*st).sinc_table).offset(
                    (4 as spx_uint32_t)
                        .wrapping_add(
                            ((j + 1 as std::ffi::c_int) as spx_uint32_t)
                                .wrapping_mul((*st).oversample),
                        )
                        .wrapping_sub(offset as spx_uint32_t)
                        .wrapping_sub(2 as spx_uint32_t) as isize,
                );
            accum[1 as std::ffi::c_int as usize] += curr_in
                * *((*st).sinc_table).offset(
                    (4 as spx_uint32_t)
                        .wrapping_add(
                            ((j + 1 as std::ffi::c_int) as spx_uint32_t)
                                .wrapping_mul((*st).oversample),
                        )
                        .wrapping_sub(offset as spx_uint32_t)
                        .wrapping_sub(1 as spx_uint32_t) as isize,
                );
            accum[2 as std::ffi::c_int as usize] += curr_in
                * *((*st).sinc_table).offset(
                    (4 as spx_uint32_t)
                        .wrapping_add(
                            ((j + 1 as std::ffi::c_int) as spx_uint32_t)
                                .wrapping_mul((*st).oversample),
                        )
                        .wrapping_sub(offset as spx_uint32_t) as isize,
                );
            accum[3 as std::ffi::c_int as usize] += curr_in
                * *((*st).sinc_table).offset(
                    (4 as spx_uint32_t)
                        .wrapping_add(
                            ((j + 1 as std::ffi::c_int) as spx_uint32_t)
                                .wrapping_mul((*st).oversample),
                        )
                        .wrapping_sub(offset as spx_uint32_t)
                        .wrapping_add(1 as spx_uint32_t) as isize,
                );
            j += 1;
        }
        cubic_coef(frac, interp.as_mut_ptr());
        sum = interp[0 as std::ffi::c_int as usize] * accum[0 as std::ffi::c_int as usize]
            + interp[1 as std::ffi::c_int as usize] * accum[1 as std::ffi::c_int as usize]
            + interp[2 as std::ffi::c_int as usize] * accum[2 as std::ffi::c_int as usize]
            + interp[3 as std::ffi::c_int as usize] * accum[3 as std::ffi::c_int as usize];
        sum = sum;
        let fresh2 = out_sample;
        out_sample = out_sample + 1;
        *out.offset((out_stride * fresh2) as isize) = sum as spx_word16_t;
        last_sample += int_advance;
        samp_frac_num = samp_frac_num.wrapping_add(frac_advance as spx_uint32_t);
        if samp_frac_num >= den_rate {
            samp_frac_num = samp_frac_num.wrapping_sub(den_rate);
            last_sample += 1;
        }
    }
    *((*st).last_sample).offset(channel_index as isize) = last_sample as spx_int32_t;
    *((*st).samp_frac_num).offset(channel_index as isize) = samp_frac_num;
    return out_sample;
}
unsafe extern "C" fn resampler_basic_interpolate_double(
    mut st: *mut SpeexResamplerState,
    mut channel_index: spx_uint32_t,
    mut in_0: *const spx_word16_t,
    mut in_len: *mut spx_uint32_t,
    mut out: *mut spx_word16_t,
    mut out_len: *mut spx_uint32_t,
) -> std::ffi::c_int {
    let N: std::ffi::c_int = (*st).filt_len as std::ffi::c_int;
    let mut out_sample: std::ffi::c_int = 0 as std::ffi::c_int;
    let mut last_sample: std::ffi::c_int = *((*st).last_sample).offset(channel_index as isize);
    let mut samp_frac_num: spx_uint32_t = *((*st).samp_frac_num).offset(channel_index as isize);
    let out_stride: std::ffi::c_int = (*st).out_stride;
    let int_advance: std::ffi::c_int = (*st).int_advance;
    let frac_advance: std::ffi::c_int = (*st).frac_advance;
    let den_rate: spx_uint32_t = (*st).den_rate;
    let mut sum: spx_word32_t = 0.;
    while !(last_sample as spx_int32_t >= *in_len as spx_int32_t
        || out_sample as spx_int32_t >= *out_len as spx_int32_t)
    {
        let mut iptr: *const spx_word16_t =
            &*in_0.offset(last_sample as isize) as *const spx_word16_t;
        let offset: std::ffi::c_int = samp_frac_num
            .wrapping_mul((*st).oversample)
            .wrapping_div((*st).den_rate) as std::ffi::c_int;
        let frac: spx_word16_t = samp_frac_num
            .wrapping_mul((*st).oversample)
            .wrapping_rem((*st).den_rate) as spx_word16_t
            / (*st).den_rate as spx_word16_t;
        let mut interp: [spx_word16_t; 4] = [0.; 4];
        let mut j: std::ffi::c_int = 0;
        let mut accum: [std::ffi::c_double; 4] = [
            0 as std::ffi::c_int as std::ffi::c_double,
            0 as std::ffi::c_int as std::ffi::c_double,
            0 as std::ffi::c_int as std::ffi::c_double,
            0 as std::ffi::c_int as std::ffi::c_double,
        ];
        j = 0 as std::ffi::c_int;
        while j < N {
            let curr_in: std::ffi::c_double = *iptr.offset(j as isize) as std::ffi::c_double;
            accum[0 as std::ffi::c_int as usize] += (curr_in as spx_word32_t
                * *((*st).sinc_table).offset(
                    (4 as spx_uint32_t)
                        .wrapping_add(
                            ((j + 1 as std::ffi::c_int) as spx_uint32_t)
                                .wrapping_mul((*st).oversample),
                        )
                        .wrapping_sub(offset as spx_uint32_t)
                        .wrapping_sub(2 as spx_uint32_t) as isize,
                )) as std::ffi::c_double;
            accum[1 as std::ffi::c_int as usize] += (curr_in as spx_word32_t
                * *((*st).sinc_table).offset(
                    (4 as spx_uint32_t)
                        .wrapping_add(
                            ((j + 1 as std::ffi::c_int) as spx_uint32_t)
                                .wrapping_mul((*st).oversample),
                        )
                        .wrapping_sub(offset as spx_uint32_t)
                        .wrapping_sub(1 as spx_uint32_t) as isize,
                )) as std::ffi::c_double;
            accum[2 as std::ffi::c_int as usize] += (curr_in as spx_word32_t
                * *((*st).sinc_table).offset(
                    (4 as spx_uint32_t)
                        .wrapping_add(
                            ((j + 1 as std::ffi::c_int) as spx_uint32_t)
                                .wrapping_mul((*st).oversample),
                        )
                        .wrapping_sub(offset as spx_uint32_t) as isize,
                )) as std::ffi::c_double;
            accum[3 as std::ffi::c_int as usize] += (curr_in as spx_word32_t
                * *((*st).sinc_table).offset(
                    (4 as spx_uint32_t)
                        .wrapping_add(
                            ((j + 1 as std::ffi::c_int) as spx_uint32_t)
                                .wrapping_mul((*st).oversample),
                        )
                        .wrapping_sub(offset as spx_uint32_t)
                        .wrapping_add(1 as spx_uint32_t) as isize,
                )) as std::ffi::c_double;
            j += 1;
        }
        cubic_coef(frac, interp.as_mut_ptr());
        sum = (interp[0 as std::ffi::c_int as usize] as std::ffi::c_double
            * accum[0 as std::ffi::c_int as usize]
            + interp[1 as std::ffi::c_int as usize] as std::ffi::c_double
                * accum[1 as std::ffi::c_int as usize]
            + interp[2 as std::ffi::c_int as usize] as std::ffi::c_double
                * accum[2 as std::ffi::c_int as usize]
            + interp[3 as std::ffi::c_int as usize] as std::ffi::c_double
                * accum[3 as std::ffi::c_int as usize]) as spx_word32_t;
        let fresh3 = out_sample;
        out_sample = out_sample + 1;
        *out.offset((out_stride * fresh3) as isize) = sum as spx_word16_t;
        last_sample += int_advance;
        samp_frac_num = samp_frac_num.wrapping_add(frac_advance as spx_uint32_t);
        if samp_frac_num >= den_rate {
            samp_frac_num = samp_frac_num.wrapping_sub(den_rate);
            last_sample += 1;
        }
    }
    *((*st).last_sample).offset(channel_index as isize) = last_sample as spx_int32_t;
    *((*st).samp_frac_num).offset(channel_index as isize) = samp_frac_num;
    return out_sample;
}
unsafe extern "C" fn resampler_basic_zero(
    mut st: *mut SpeexResamplerState,
    mut channel_index: spx_uint32_t,
    mut in_0: *const spx_word16_t,
    mut in_len: *mut spx_uint32_t,
    mut out: *mut spx_word16_t,
    mut out_len: *mut spx_uint32_t,
) -> std::ffi::c_int {
    let mut out_sample: std::ffi::c_int = 0 as std::ffi::c_int;
    let mut last_sample: std::ffi::c_int = *((*st).last_sample).offset(channel_index as isize);
    let mut samp_frac_num: spx_uint32_t = *((*st).samp_frac_num).offset(channel_index as isize);
    let out_stride: std::ffi::c_int = (*st).out_stride;
    let int_advance: std::ffi::c_int = (*st).int_advance;
    let frac_advance: std::ffi::c_int = (*st).frac_advance;
    let den_rate: spx_uint32_t = (*st).den_rate;
    while !(last_sample as spx_int32_t >= *in_len as spx_int32_t
        || out_sample as spx_int32_t >= *out_len as spx_int32_t)
    {
        let fresh4 = out_sample;
        out_sample = out_sample + 1;
        *out.offset((out_stride * fresh4) as isize) = 0 as std::ffi::c_int as spx_word16_t;
        last_sample += int_advance;
        samp_frac_num = samp_frac_num.wrapping_add(frac_advance as spx_uint32_t);
        if samp_frac_num >= den_rate {
            samp_frac_num = samp_frac_num.wrapping_sub(den_rate);
            last_sample += 1;
        }
    }
    *((*st).last_sample).offset(channel_index as isize) = last_sample as spx_int32_t;
    *((*st).samp_frac_num).offset(channel_index as isize) = samp_frac_num;
    return out_sample;
}
unsafe extern "C" fn multiply_frac(
    mut result: *mut spx_uint32_t,
    mut value: spx_uint32_t,
    mut num: spx_uint32_t,
    mut den: spx_uint32_t,
) -> std::ffi::c_int {
    let mut major: spx_uint32_t = value.wrapping_div(den);
    let mut remain: spx_uint32_t = value.wrapping_rem(den);
    if remain > (UINT32_MAX as spx_uint32_t).wrapping_div(num)
        || major > (UINT32_MAX as spx_uint32_t).wrapping_div(num)
        || major.wrapping_mul(num)
            > (UINT32_MAX as spx_uint32_t).wrapping_sub(remain.wrapping_mul(num).wrapping_div(den))
    {
        return RESAMPLER_ERR_OVERFLOW as std::ffi::c_int;
    }
    *result = remain
        .wrapping_mul(num)
        .wrapping_div(den)
        .wrapping_add(major.wrapping_mul(num));
    return RESAMPLER_ERR_SUCCESS as std::ffi::c_int;
}
unsafe extern "C" fn update_filter(mut st: *mut SpeexResamplerState) -> std::ffi::c_int {
    let mut current_block: u64;
    let mut old_length: spx_uint32_t = (*st).filt_len;
    let mut old_alloc_size: spx_uint32_t = (*st).mem_alloc_size;
    let mut use_direct: std::ffi::c_int = 0;
    let mut min_sinc_table_length: spx_uint32_t = 0;
    let mut min_alloc_size: spx_uint32_t = 0;
    (*st).int_advance = ((*st).num_rate).wrapping_div((*st).den_rate) as std::ffi::c_int;
    (*st).frac_advance = ((*st).num_rate).wrapping_rem((*st).den_rate) as std::ffi::c_int;
    (*st).oversample = quality_map[(*st).quality as usize].oversample as spx_uint32_t;
    (*st).filt_len = quality_map[(*st).quality as usize].base_length as spx_uint32_t;
    if (*st).num_rate > (*st).den_rate {
        (*st).cutoff = quality_map[(*st).quality as usize].downsample_bandwidth
            * (*st).den_rate as std::ffi::c_float
            / (*st).num_rate as std::ffi::c_float;
        if multiply_frac(
            &mut (*st).filt_len,
            (*st).filt_len,
            (*st).num_rate,
            (*st).den_rate,
        ) != RESAMPLER_ERR_SUCCESS as std::ffi::c_int
        {
            current_block = 13719021656572692010;
        } else {
            (*st).filt_len = (((*st).filt_len).wrapping_sub(1 as spx_uint32_t)
                & !(0x7 as std::ffi::c_int) as spx_uint32_t)
                .wrapping_add(8 as spx_uint32_t);
            if (2 as spx_uint32_t).wrapping_mul((*st).den_rate) < (*st).num_rate {
                (*st).oversample >>= 1 as std::ffi::c_int;
            }
            if (4 as spx_uint32_t).wrapping_mul((*st).den_rate) < (*st).num_rate {
                (*st).oversample >>= 1 as std::ffi::c_int;
            }
            if (8 as spx_uint32_t).wrapping_mul((*st).den_rate) < (*st).num_rate {
                (*st).oversample >>= 1 as std::ffi::c_int;
            }
            if (16 as spx_uint32_t).wrapping_mul((*st).den_rate) < (*st).num_rate {
                (*st).oversample >>= 1 as std::ffi::c_int;
            }
            if (*st).oversample < 1 as spx_uint32_t {
                (*st).oversample = 1 as spx_uint32_t;
            }
            current_block = 15652330335145281839;
        }
    } else {
        (*st).cutoff = quality_map[(*st).quality as usize].upsample_bandwidth;
        current_block = 15652330335145281839;
    }
    match current_block {
        15652330335145281839 => {
            use_direct = (((*st).filt_len).wrapping_mul((*st).den_rate)
                <= ((*st).filt_len)
                    .wrapping_mul((*st).oversample)
                    .wrapping_add(8 as spx_uint32_t)
                && (INT_MAX as std::ffi::c_ulong)
                    .wrapping_div(::core::mem::size_of::<spx_word16_t>() as std::ffi::c_ulong)
                    .wrapping_div((*st).den_rate as std::ffi::c_ulong)
                    >= (*st).filt_len as std::ffi::c_ulong)
                as std::ffi::c_int;
            if use_direct != 0 {
                min_sinc_table_length = ((*st).filt_len).wrapping_mul((*st).den_rate);
                current_block = 14576567515993809846;
            } else if (INT_MAX as std::ffi::c_ulong)
                .wrapping_div(::core::mem::size_of::<spx_word16_t>() as std::ffi::c_ulong)
                .wrapping_sub(8 as std::ffi::c_ulong)
                .wrapping_div((*st).oversample as std::ffi::c_ulong)
                < (*st).filt_len as std::ffi::c_ulong
            {
                current_block = 13719021656572692010;
            } else {
                min_sinc_table_length = ((*st).filt_len)
                    .wrapping_mul((*st).oversample)
                    .wrapping_add(8 as spx_uint32_t);
                current_block = 14576567515993809846;
            }
            match current_block {
                13719021656572692010 => {}
                _ => {
                    if (*st).sinc_table_length < min_sinc_table_length {
                        let mut sinc_table: *mut spx_word16_t = speex_realloc(
                            (*st).sinc_table as *mut std::ffi::c_void,
                            (min_sinc_table_length as std::ffi::c_ulong).wrapping_mul(
                                ::core::mem::size_of::<spx_word16_t>() as std::ffi::c_ulong,
                            ) as std::ffi::c_int,
                        )
                            as *mut spx_word16_t;
                        if sinc_table.is_null() {
                            current_block = 13719021656572692010;
                        } else {
                            (*st).sinc_table = sinc_table;
                            (*st).sinc_table_length = min_sinc_table_length;
                            current_block = 7056779235015430508;
                        }
                    } else {
                        current_block = 7056779235015430508;
                    }
                    match current_block {
                        13719021656572692010 => {}
                        _ => {
                            if use_direct != 0 {
                                let mut i: spx_uint32_t = 0;
                                i = 0 as spx_uint32_t;
                                while i < (*st).den_rate {
                                    let mut j: spx_int32_t = 0;
                                    j = 0 as std::ffi::c_int as spx_int32_t;
                                    while (j as spx_uint32_t) < (*st).filt_len {
                                        *((*st).sinc_table).offset(
                                            i.wrapping_mul((*st).filt_len)
                                                .wrapping_add(j as spx_uint32_t)
                                                as isize,
                                        ) = sinc(
                                            (*st).cutoff,
                                            (j - (*st).filt_len as spx_int32_t / 2 as spx_int32_t
                                                + 1 as spx_int32_t)
                                                as std::ffi::c_float
                                                - i as std::ffi::c_float
                                                    / (*st).den_rate as std::ffi::c_float,
                                            (*st).filt_len as std::ffi::c_int,
                                            quality_map[(*st).quality as usize].window_func,
                                        );
                                        j += 1;
                                    }
                                    i = i.wrapping_add(1);
                                }
                                if (*st).quality > 8 as std::ffi::c_int {
                                    (*st).resampler_ptr = Some(
                                        resampler_basic_direct_double
                                            as unsafe extern "C" fn(
                                                *mut SpeexResamplerState,
                                                spx_uint32_t,
                                                *const spx_word16_t,
                                                *mut spx_uint32_t,
                                                *mut spx_word16_t,
                                                *mut spx_uint32_t,
                                            )
                                                -> std::ffi::c_int,
                                    )
                                        as resampler_basic_func;
                                } else {
                                    (*st).resampler_ptr = Some(
                                        resampler_basic_direct_single
                                            as unsafe extern "C" fn(
                                                *mut SpeexResamplerState,
                                                spx_uint32_t,
                                                *const spx_word16_t,
                                                *mut spx_uint32_t,
                                                *mut spx_word16_t,
                                                *mut spx_uint32_t,
                                            )
                                                -> std::ffi::c_int,
                                    )
                                        as resampler_basic_func;
                                }
                            } else {
                                let mut i_0: spx_int32_t = 0;
                                i_0 = -(4 as std::ffi::c_int) as spx_int32_t;
                                while i_0
                                    < ((*st).oversample)
                                        .wrapping_mul((*st).filt_len)
                                        .wrapping_add(4 as spx_uint32_t)
                                        as spx_int32_t
                                {
                                    *((*st).sinc_table).offset((i_0 + 4 as spx_int32_t) as isize) =
                                        sinc(
                                            (*st).cutoff,
                                            i_0 as std::ffi::c_float
                                                / (*st).oversample as std::ffi::c_float
                                                - ((*st).filt_len).wrapping_div(2 as spx_uint32_t)
                                                    as std::ffi::c_float,
                                            (*st).filt_len as std::ffi::c_int,
                                            quality_map[(*st).quality as usize].window_func,
                                        );
                                    i_0 += 1;
                                }
                                if (*st).quality > 8 as std::ffi::c_int {
                                    (*st).resampler_ptr = Some(
                                        resampler_basic_interpolate_double
                                            as unsafe extern "C" fn(
                                                *mut SpeexResamplerState,
                                                spx_uint32_t,
                                                *const spx_word16_t,
                                                *mut spx_uint32_t,
                                                *mut spx_word16_t,
                                                *mut spx_uint32_t,
                                            )
                                                -> std::ffi::c_int,
                                    )
                                        as resampler_basic_func;
                                } else {
                                    (*st).resampler_ptr = Some(
                                        resampler_basic_interpolate_single
                                            as unsafe extern "C" fn(
                                                *mut SpeexResamplerState,
                                                spx_uint32_t,
                                                *const spx_word16_t,
                                                *mut spx_uint32_t,
                                                *mut spx_word16_t,
                                                *mut spx_uint32_t,
                                            )
                                                -> std::ffi::c_int,
                                    )
                                        as resampler_basic_func;
                                }
                            }
                            min_alloc_size = ((*st).filt_len)
                                .wrapping_sub(1 as spx_uint32_t)
                                .wrapping_add((*st).buffer_size);
                            if min_alloc_size > (*st).mem_alloc_size {
                                let mut mem: *mut spx_word16_t = 0 as *mut spx_word16_t;
                                if (INT_MAX as std::ffi::c_ulong)
                                    .wrapping_div(
                                        ::core::mem::size_of::<spx_word16_t>() as std::ffi::c_ulong
                                    )
                                    .wrapping_div((*st).nb_channels as std::ffi::c_ulong)
                                    < min_alloc_size as std::ffi::c_ulong
                                {
                                    current_block = 13719021656572692010;
                                } else {
                                    mem = speex_realloc(
                                        (*st).mem as *mut std::ffi::c_void,
                                        (((*st).nb_channels).wrapping_mul(min_alloc_size)
                                            as std::ffi::c_ulong)
                                            .wrapping_mul(::core::mem::size_of::<spx_word16_t>()
                                                as std::ffi::c_ulong)
                                            as std::ffi::c_int,
                                    )
                                        as *mut spx_word16_t;
                                    if mem.is_null() {
                                        current_block = 13719021656572692010;
                                    } else {
                                        (*st).mem = mem;
                                        (*st).mem_alloc_size = min_alloc_size;
                                        current_block = 5891011138178424807;
                                    }
                                }
                            } else {
                                current_block = 5891011138178424807;
                            }
                            match current_block {
                                13719021656572692010 => {}
                                _ => {
                                    if (*st).started == 0 {
                                        let mut i_1: spx_uint32_t = 0;
                                        i_1 = 0 as spx_uint32_t;
                                        while i_1
                                            < ((*st).nb_channels).wrapping_mul((*st).mem_alloc_size)
                                        {
                                            *((*st).mem).offset(i_1 as isize) =
                                                0 as std::ffi::c_int as spx_word16_t;
                                            i_1 = i_1.wrapping_add(1);
                                        }
                                    } else if (*st).filt_len > old_length {
                                        let mut i_2: spx_uint32_t = 0;
                                        i_2 = (*st).nb_channels;
                                        loop {
                                            let fresh5 = i_2;
                                            i_2 = i_2.wrapping_sub(1);
                                            if !(fresh5 != 0) {
                                                break;
                                            }
                                            let mut j_0: spx_uint32_t = 0;
                                            let mut olen: spx_uint32_t = old_length;
                                            let mut start: spx_uint32_t =
                                                i_2.wrapping_mul((*st).mem_alloc_size);
                                            let mut magic_samples: spx_uint32_t =
                                                *((*st).magic_samples).offset(i_2 as isize);
                                            olen = old_length.wrapping_add(
                                                (2 as spx_uint32_t).wrapping_mul(magic_samples),
                                            );
                                            j_0 = old_length
                                                .wrapping_sub(1 as spx_uint32_t)
                                                .wrapping_add(magic_samples);
                                            loop {
                                                let fresh6 = j_0;
                                                j_0 = j_0.wrapping_sub(1);
                                                if !(fresh6 != 0) {
                                                    break;
                                                }
                                                *((*st).mem).offset(
                                                    start
                                                        .wrapping_add(j_0)
                                                        .wrapping_add(magic_samples)
                                                        as isize,
                                                ) = *((*st).mem).offset(
                                                    i_2.wrapping_mul(old_alloc_size)
                                                        .wrapping_add(j_0)
                                                        as isize,
                                                );
                                            }
                                            j_0 = 0 as spx_uint32_t;
                                            while j_0 < magic_samples {
                                                *((*st).mem)
                                                    .offset(start.wrapping_add(j_0) as isize) =
                                                    0 as std::ffi::c_int as spx_word16_t;
                                                j_0 = j_0.wrapping_add(1);
                                            }
                                            *((*st).magic_samples).offset(i_2 as isize) =
                                                0 as spx_uint32_t;
                                            if (*st).filt_len > olen {
                                                j_0 = 0 as spx_uint32_t;
                                                while j_0 < olen.wrapping_sub(1 as spx_uint32_t) {
                                                    *((*st).mem).offset(
                                                        start.wrapping_add(
                                                            ((*st).filt_len)
                                                                .wrapping_sub(2 as spx_uint32_t)
                                                                .wrapping_sub(j_0),
                                                        )
                                                            as isize,
                                                    ) = *((*st).mem).offset(
                                                        start.wrapping_add(
                                                            olen.wrapping_sub(2 as spx_uint32_t)
                                                                .wrapping_sub(j_0),
                                                        )
                                                            as isize,
                                                    );
                                                    j_0 = j_0.wrapping_add(1);
                                                }
                                                while j_0
                                                    < ((*st).filt_len)
                                                        .wrapping_sub(1 as spx_uint32_t)
                                                {
                                                    *((*st).mem).offset(
                                                        start.wrapping_add(
                                                            ((*st).filt_len)
                                                                .wrapping_sub(2 as spx_uint32_t)
                                                                .wrapping_sub(j_0),
                                                        )
                                                            as isize,
                                                    ) = 0 as std::ffi::c_int as spx_word16_t;
                                                    j_0 = j_0.wrapping_add(1);
                                                }
                                                let ref mut fresh7 =
                                                    *((*st).last_sample).offset(i_2 as isize);
                                                *fresh7 = (*fresh7 as spx_uint32_t).wrapping_add(
                                                    ((*st).filt_len)
                                                        .wrapping_sub(olen)
                                                        .wrapping_div(2 as spx_uint32_t),
                                                )
                                                    as spx_int32_t
                                                    as spx_int32_t;
                                            } else {
                                                magic_samples = olen
                                                    .wrapping_sub((*st).filt_len)
                                                    .wrapping_div(2 as spx_uint32_t);
                                                j_0 = 0 as spx_uint32_t;
                                                while j_0
                                                    < ((*st).filt_len)
                                                        .wrapping_sub(1 as spx_uint32_t)
                                                        .wrapping_add(magic_samples)
                                                {
                                                    *((*st).mem)
                                                        .offset(start.wrapping_add(j_0) as isize) =
                                                        *((*st).mem).offset(
                                                            start
                                                                .wrapping_add(j_0)
                                                                .wrapping_add(magic_samples)
                                                                as isize,
                                                        );
                                                    j_0 = j_0.wrapping_add(1);
                                                }
                                                *((*st).magic_samples).offset(i_2 as isize) =
                                                    magic_samples;
                                            }
                                        }
                                    } else if (*st).filt_len < old_length {
                                        let mut i_3: spx_uint32_t = 0;
                                        i_3 = 0 as spx_uint32_t;
                                        while i_3 < (*st).nb_channels {
                                            let mut j_1: spx_uint32_t = 0;
                                            let mut old_magic: spx_uint32_t =
                                                *((*st).magic_samples).offset(i_3 as isize);
                                            *((*st).magic_samples).offset(i_3 as isize) =
                                                old_length
                                                    .wrapping_sub((*st).filt_len)
                                                    .wrapping_div(2 as spx_uint32_t);
                                            j_1 = 0 as spx_uint32_t;
                                            while j_1
                                                < ((*st).filt_len)
                                                    .wrapping_sub(1 as spx_uint32_t)
                                                    .wrapping_add(
                                                        *((*st).magic_samples).offset(i_3 as isize),
                                                    )
                                                    .wrapping_add(old_magic)
                                            {
                                                *((*st).mem).offset(
                                                    i_3.wrapping_mul((*st).mem_alloc_size)
                                                        .wrapping_add(j_1)
                                                        as isize,
                                                ) = *((*st).mem).offset(
                                                    i_3.wrapping_mul((*st).mem_alloc_size)
                                                        .wrapping_add(j_1)
                                                        .wrapping_add(
                                                            *((*st).magic_samples)
                                                                .offset(i_3 as isize),
                                                        )
                                                        as isize,
                                                );
                                                j_1 = j_1.wrapping_add(1);
                                            }
                                            let ref mut fresh8 =
                                                *((*st).magic_samples).offset(i_3 as isize);
                                            *fresh8 = (*fresh8).wrapping_add(old_magic);
                                            i_3 = i_3.wrapping_add(1);
                                        }
                                    }
                                    return RESAMPLER_ERR_SUCCESS as std::ffi::c_int;
                                }
                            }
                        }
                    }
                }
            }
        }
        _ => {}
    }
    (*st).resampler_ptr = Some(
        resampler_basic_zero
            as unsafe extern "C" fn(
                *mut SpeexResamplerState,
                spx_uint32_t,
                *const spx_word16_t,
                *mut spx_uint32_t,
                *mut spx_word16_t,
                *mut spx_uint32_t,
            ) -> std::ffi::c_int,
    ) as resampler_basic_func;
    (*st).filt_len = old_length;
    return RESAMPLER_ERR_ALLOC_FAILED as std::ffi::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn speex_resampler_init(
    mut nb_channels: spx_uint32_t,
    mut in_rate: spx_uint32_t,
    mut out_rate: spx_uint32_t,
    mut quality: std::ffi::c_int,
    mut err: *mut std::ffi::c_int,
) -> *mut SpeexResamplerState {
    return speex_resampler_init_frac(
        nb_channels,
        in_rate,
        out_rate,
        in_rate,
        out_rate,
        quality,
        err,
    );
}
#[no_mangle]
pub unsafe extern "C" fn speex_resampler_init_frac(
    mut nb_channels: spx_uint32_t,
    mut ratio_num: spx_uint32_t,
    mut ratio_den: spx_uint32_t,
    mut in_rate: spx_uint32_t,
    mut out_rate: spx_uint32_t,
    mut quality: std::ffi::c_int,
    mut err: *mut std::ffi::c_int,
) -> *mut SpeexResamplerState {
    let mut st: *mut SpeexResamplerState = 0 as *mut SpeexResamplerState;
    let mut filter_err: std::ffi::c_int = 0;
    if nb_channels == 0 as spx_uint32_t
        || ratio_num == 0 as spx_uint32_t
        || ratio_den == 0 as spx_uint32_t
        || quality > 10 as std::ffi::c_int
        || quality < 0 as std::ffi::c_int
    {
        if !err.is_null() {
            *err = RESAMPLER_ERR_INVALID_ARG as std::ffi::c_int;
        }
        return NULL as *mut SpeexResamplerState;
    }
    st = speex_alloc(::core::mem::size_of::<SpeexResamplerState>() as std::ffi::c_int)
        as *mut SpeexResamplerState;
    if st.is_null() {
        if !err.is_null() {
            *err = RESAMPLER_ERR_ALLOC_FAILED as std::ffi::c_int;
        }
        return NULL as *mut SpeexResamplerState;
    }
    (*st).initialised = 0 as std::ffi::c_int;
    (*st).started = 0 as std::ffi::c_int;
    (*st).in_rate = 0 as spx_uint32_t;
    (*st).out_rate = 0 as spx_uint32_t;
    (*st).num_rate = 0 as spx_uint32_t;
    (*st).den_rate = 0 as spx_uint32_t;
    (*st).quality = -(1 as std::ffi::c_int);
    (*st).sinc_table_length = 0 as spx_uint32_t;
    (*st).mem_alloc_size = 0 as spx_uint32_t;
    (*st).filt_len = 0 as spx_uint32_t;
    (*st).mem = 0 as *mut spx_word16_t;
    (*st).resampler_ptr = None;
    (*st).cutoff = 1.0f32;
    (*st).nb_channels = nb_channels;
    (*st).in_stride = 1 as std::ffi::c_int;
    (*st).out_stride = 1 as std::ffi::c_int;
    (*st).buffer_size = 160 as spx_uint32_t;
    (*st).last_sample = speex_alloc(
        (nb_channels as std::ffi::c_ulong)
            .wrapping_mul(::core::mem::size_of::<spx_int32_t>() as std::ffi::c_ulong)
            as std::ffi::c_int,
    ) as *mut spx_int32_t;
    if !((*st).last_sample).is_null() {
        (*st).magic_samples = speex_alloc(
            (nb_channels as std::ffi::c_ulong)
                .wrapping_mul(::core::mem::size_of::<spx_uint32_t>() as std::ffi::c_ulong)
                as std::ffi::c_int,
        ) as *mut spx_uint32_t;
        if !((*st).magic_samples).is_null() {
            (*st).samp_frac_num = speex_alloc(
                (nb_channels as std::ffi::c_ulong)
                    .wrapping_mul(::core::mem::size_of::<spx_uint32_t>() as std::ffi::c_ulong)
                    as std::ffi::c_int,
            ) as *mut spx_uint32_t;
            if !((*st).samp_frac_num).is_null() {
                speex_resampler_set_quality(st, quality);
                speex_resampler_set_rate_frac(st, ratio_num, ratio_den, in_rate, out_rate);
                filter_err = update_filter(st);
                if filter_err == RESAMPLER_ERR_SUCCESS as std::ffi::c_int {
                    (*st).initialised = 1 as std::ffi::c_int;
                } else {
                    speex_resampler_destroy(st);
                    st = NULL as *mut SpeexResamplerState;
                }
                if !err.is_null() {
                    *err = filter_err;
                }
                return st;
            }
        }
    }
    if !err.is_null() {
        *err = RESAMPLER_ERR_ALLOC_FAILED as std::ffi::c_int;
    }
    speex_resampler_destroy(st);
    return NULL as *mut SpeexResamplerState;
}
#[no_mangle]
pub unsafe extern "C" fn speex_resampler_destroy(mut st: *mut SpeexResamplerState) {
    speex_free((*st).mem as *mut std::ffi::c_void);
    speex_free((*st).sinc_table as *mut std::ffi::c_void);
    speex_free((*st).last_sample as *mut std::ffi::c_void);
    speex_free((*st).magic_samples as *mut std::ffi::c_void);
    speex_free((*st).samp_frac_num as *mut std::ffi::c_void);
    speex_free(st as *mut std::ffi::c_void);
}
unsafe extern "C" fn speex_resampler_process_native(
    mut st: *mut SpeexResamplerState,
    mut channel_index: spx_uint32_t,
    mut in_len: *mut spx_uint32_t,
    mut out: *mut spx_word16_t,
    mut out_len: *mut spx_uint32_t,
) -> std::ffi::c_int {
    let mut j: std::ffi::c_int = 0 as std::ffi::c_int;
    let N: std::ffi::c_int = (*st).filt_len as std::ffi::c_int;
    let mut out_sample: std::ffi::c_int = 0 as std::ffi::c_int;
    let mut mem: *mut spx_word16_t =
        ((*st).mem).offset(channel_index.wrapping_mul((*st).mem_alloc_size) as isize);
    let mut ilen: spx_uint32_t = 0;
    (*st).started = 1 as std::ffi::c_int;
    out_sample = ((*st).resampler_ptr).expect("non-null function pointer")(
        st,
        channel_index,
        mem,
        in_len,
        out,
        out_len,
    );
    if *((*st).last_sample).offset(channel_index as isize) < *in_len as spx_int32_t {
        *in_len = *((*st).last_sample).offset(channel_index as isize) as spx_uint32_t;
    }
    *out_len = out_sample as spx_uint32_t;
    let ref mut fresh9 = *((*st).last_sample).offset(channel_index as isize);
    *fresh9 = (*fresh9 as spx_uint32_t).wrapping_sub(*in_len) as spx_int32_t as spx_int32_t;
    ilen = *in_len;
    j = 0 as std::ffi::c_int;
    while j < N - 1 as std::ffi::c_int {
        *mem.offset(j as isize) = *mem.offset((j as spx_uint32_t).wrapping_add(ilen) as isize);
        j += 1;
    }
    return RESAMPLER_ERR_SUCCESS as std::ffi::c_int;
}
unsafe extern "C" fn speex_resampler_magic(
    mut st: *mut SpeexResamplerState,
    mut channel_index: spx_uint32_t,
    mut out: *mut *mut spx_word16_t,
    mut out_len: spx_uint32_t,
) -> std::ffi::c_int {
    let mut tmp_in_len: spx_uint32_t = *((*st).magic_samples).offset(channel_index as isize);
    let mut mem: *mut spx_word16_t =
        ((*st).mem).offset(channel_index.wrapping_mul((*st).mem_alloc_size) as isize);
    let N: std::ffi::c_int = (*st).filt_len as std::ffi::c_int;
    speex_resampler_process_native(st, channel_index, &mut tmp_in_len, *out, &mut out_len);
    let ref mut fresh10 = *((*st).magic_samples).offset(channel_index as isize);
    *fresh10 = (*fresh10).wrapping_sub(tmp_in_len);
    if *((*st).magic_samples).offset(channel_index as isize) != 0 {
        let mut i: spx_uint32_t = 0;
        i = 0 as spx_uint32_t;
        while i < *((*st).magic_samples).offset(channel_index as isize) {
            *mem.offset(((N - 1 as std::ffi::c_int) as spx_uint32_t).wrapping_add(i) as isize) =
                *mem.offset(
                    ((N - 1 as std::ffi::c_int) as spx_uint32_t)
                        .wrapping_add(i)
                        .wrapping_add(tmp_in_len) as isize,
                );
            i = i.wrapping_add(1);
        }
    }
    *out = (*out).offset(out_len.wrapping_mul((*st).out_stride as spx_uint32_t) as isize);
    return out_len as std::ffi::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn speex_resampler_process_float(
    mut st: *mut SpeexResamplerState,
    mut channel_index: spx_uint32_t,
    mut in_0: *const std::ffi::c_float,
    mut in_len: *mut spx_uint32_t,
    mut out: *mut std::ffi::c_float,
    mut out_len: *mut spx_uint32_t,
) -> std::ffi::c_int {
    let mut j: std::ffi::c_int = 0;
    let mut ilen: spx_uint32_t = *in_len;
    let mut olen: spx_uint32_t = *out_len;
    let mut x: *mut spx_word16_t =
        ((*st).mem).offset(channel_index.wrapping_mul((*st).mem_alloc_size) as isize);
    let filt_offs: std::ffi::c_int =
        ((*st).filt_len).wrapping_sub(1 as spx_uint32_t) as std::ffi::c_int;
    let xlen: spx_uint32_t = ((*st).mem_alloc_size).wrapping_sub(filt_offs as spx_uint32_t);
    let istride: std::ffi::c_int = (*st).in_stride;
    if *((*st).magic_samples).offset(channel_index as isize) != 0 {
        olen = olen
            .wrapping_sub(speex_resampler_magic(st, channel_index, &mut out, olen) as spx_uint32_t);
    }
    if *((*st).magic_samples).offset(channel_index as isize) == 0 {
        while ilen != 0 && olen != 0 {
            let mut ichunk: spx_uint32_t = if ilen > xlen { xlen } else { ilen };
            let mut ochunk: spx_uint32_t = olen;
            if !in_0.is_null() {
                j = 0 as std::ffi::c_int;
                while (j as spx_uint32_t) < ichunk {
                    *x.offset((j + filt_offs) as isize) =
                        *in_0.offset((j * istride) as isize) as spx_word16_t;
                    j += 1;
                }
            } else {
                j = 0 as std::ffi::c_int;
                while (j as spx_uint32_t) < ichunk {
                    *x.offset((j + filt_offs) as isize) = 0 as std::ffi::c_int as spx_word16_t;
                    j += 1;
                }
            }
            speex_resampler_process_native(
                st,
                channel_index,
                &mut ichunk,
                out as *mut spx_word16_t,
                &mut ochunk,
            );
            ilen = ilen.wrapping_sub(ichunk);
            olen = olen.wrapping_sub(ochunk);
            out = out.offset(ochunk.wrapping_mul((*st).out_stride as spx_uint32_t) as isize);
            if !in_0.is_null() {
                in_0 = in_0.offset(ichunk.wrapping_mul(istride as spx_uint32_t) as isize);
            }
        }
    }
    *in_len = (*in_len).wrapping_sub(ilen);
    *out_len = (*out_len).wrapping_sub(olen);
    return if (*st).resampler_ptr
        == Some(
            resampler_basic_zero
                as unsafe extern "C" fn(
                    *mut SpeexResamplerState,
                    spx_uint32_t,
                    *const spx_word16_t,
                    *mut spx_uint32_t,
                    *mut spx_word16_t,
                    *mut spx_uint32_t,
                ) -> std::ffi::c_int,
        ) {
        RESAMPLER_ERR_ALLOC_FAILED as std::ffi::c_int
    } else {
        RESAMPLER_ERR_SUCCESS as std::ffi::c_int
    };
}
#[no_mangle]
pub unsafe extern "C" fn speex_resampler_process_int(
    mut st: *mut SpeexResamplerState,
    mut channel_index: spx_uint32_t,
    mut in_0: *const spx_int16_t,
    mut in_len: *mut spx_uint32_t,
    mut out: *mut spx_int16_t,
    mut out_len: *mut spx_uint32_t,
) -> std::ffi::c_int {
    let mut j: std::ffi::c_int = 0;
    let istride_save: std::ffi::c_int = (*st).in_stride;
    let ostride_save: std::ffi::c_int = (*st).out_stride;
    let mut ilen: spx_uint32_t = *in_len;
    let mut olen: spx_uint32_t = *out_len;
    let mut x: *mut spx_word16_t =
        ((*st).mem).offset(channel_index.wrapping_mul((*st).mem_alloc_size) as isize);
    let xlen: spx_uint32_t =
        ((*st).mem_alloc_size).wrapping_sub(((*st).filt_len).wrapping_sub(1 as spx_uint32_t));
    let ylen: std::ffi::c_uint = FIXED_STACK_ALLOC as std::ffi::c_uint;
    let mut ystack: [spx_word16_t; 1024] = [0.; 1024];
    (*st).out_stride = 1 as std::ffi::c_int;
    while ilen != 0 && olen != 0 {
        let mut y: *mut spx_word16_t = ystack.as_mut_ptr();
        let mut ichunk: spx_uint32_t = if ilen > xlen { xlen } else { ilen };
        let mut ochunk: spx_uint32_t = if olen > ylen as spx_uint32_t {
            ylen as spx_uint32_t
        } else {
            olen
        };
        let mut omagic: spx_uint32_t = 0 as spx_uint32_t;
        if *((*st).magic_samples).offset(channel_index as isize) != 0 {
            omagic = speex_resampler_magic(st, channel_index, &mut y, ochunk) as spx_uint32_t;
            ochunk = ochunk.wrapping_sub(omagic);
            olen = olen.wrapping_sub(omagic);
        }
        if *((*st).magic_samples).offset(channel_index as isize) == 0 {
            if !in_0.is_null() {
                j = 0 as std::ffi::c_int;
                while (j as spx_uint32_t) < ichunk {
                    *x.offset(
                        (j as spx_uint32_t)
                            .wrapping_add((*st).filt_len)
                            .wrapping_sub(1 as spx_uint32_t) as isize,
                    ) = *in_0.offset((j * istride_save) as isize) as spx_word16_t;
                    j += 1;
                }
            } else {
                j = 0 as std::ffi::c_int;
                while (j as spx_uint32_t) < ichunk {
                    *x.offset(
                        (j as spx_uint32_t)
                            .wrapping_add((*st).filt_len)
                            .wrapping_sub(1 as spx_uint32_t) as isize,
                    ) = 0 as std::ffi::c_int as spx_word16_t;
                    j += 1;
                }
            }
            speex_resampler_process_native(st, channel_index, &mut ichunk, y, &mut ochunk);
        } else {
            ichunk = 0 as spx_uint32_t;
            ochunk = 0 as spx_uint32_t;
        }
        j = 0 as std::ffi::c_int;
        while (j as spx_uint32_t) < ochunk.wrapping_add(omagic) {
            *out.offset((j * ostride_save) as isize) = (if ystack[j as usize] < -32767.5f32 {
                -(32768 as std::ffi::c_int)
            } else if ystack[j as usize] > 32766.5f32 {
                32767 as std::ffi::c_int
            } else {
                floor(0.5f64 + ystack[j as usize] as std::ffi::c_double) as spx_int16_t
                    as std::ffi::c_int
            }) as spx_int16_t;
            j += 1;
        }
        ilen = ilen.wrapping_sub(ichunk);
        olen = olen.wrapping_sub(ochunk);
        out = out.offset(
            ochunk
                .wrapping_add(omagic)
                .wrapping_mul(ostride_save as spx_uint32_t) as isize,
        );
        if !in_0.is_null() {
            in_0 = in_0.offset(ichunk.wrapping_mul(istride_save as spx_uint32_t) as isize);
        }
    }
    (*st).out_stride = ostride_save;
    *in_len = (*in_len).wrapping_sub(ilen);
    *out_len = (*out_len).wrapping_sub(olen);
    return if (*st).resampler_ptr
        == Some(
            resampler_basic_zero
                as unsafe extern "C" fn(
                    *mut SpeexResamplerState,
                    spx_uint32_t,
                    *const spx_word16_t,
                    *mut spx_uint32_t,
                    *mut spx_word16_t,
                    *mut spx_uint32_t,
                ) -> std::ffi::c_int,
        ) {
        RESAMPLER_ERR_ALLOC_FAILED as std::ffi::c_int
    } else {
        RESAMPLER_ERR_SUCCESS as std::ffi::c_int
    };
}
#[no_mangle]
pub unsafe extern "C" fn speex_resampler_process_interleaved_float(
    mut st: *mut SpeexResamplerState,
    mut in_0: *const std::ffi::c_float,
    mut in_len: *mut spx_uint32_t,
    mut out: *mut std::ffi::c_float,
    mut out_len: *mut spx_uint32_t,
) -> std::ffi::c_int {
    let mut i: spx_uint32_t = 0;
    let mut istride_save: std::ffi::c_int = 0;
    let mut ostride_save: std::ffi::c_int = 0;
    let mut bak_out_len: spx_uint32_t = *out_len;
    let mut bak_in_len: spx_uint32_t = *in_len;
    istride_save = (*st).in_stride;
    ostride_save = (*st).out_stride;
    (*st).out_stride = (*st).nb_channels as std::ffi::c_int;
    (*st).in_stride = (*st).out_stride;
    i = 0 as spx_uint32_t;
    while i < (*st).nb_channels {
        *out_len = bak_out_len;
        *in_len = bak_in_len;
        if !in_0.is_null() {
            speex_resampler_process_float(
                st,
                i,
                in_0.offset(i as isize),
                in_len,
                out.offset(i as isize),
                out_len,
            );
        } else {
            speex_resampler_process_float(
                st,
                i,
                NULL as *const std::ffi::c_float,
                in_len,
                out.offset(i as isize),
                out_len,
            );
        }
        i = i.wrapping_add(1);
    }
    (*st).in_stride = istride_save;
    (*st).out_stride = ostride_save;
    return if (*st).resampler_ptr
        == Some(
            resampler_basic_zero
                as unsafe extern "C" fn(
                    *mut SpeexResamplerState,
                    spx_uint32_t,
                    *const spx_word16_t,
                    *mut spx_uint32_t,
                    *mut spx_word16_t,
                    *mut spx_uint32_t,
                ) -> std::ffi::c_int,
        ) {
        RESAMPLER_ERR_ALLOC_FAILED as std::ffi::c_int
    } else {
        RESAMPLER_ERR_SUCCESS as std::ffi::c_int
    };
}
#[no_mangle]
pub unsafe extern "C" fn speex_resampler_process_interleaved_int(
    mut st: *mut SpeexResamplerState,
    mut in_0: *const spx_int16_t,
    mut in_len: *mut spx_uint32_t,
    mut out: *mut spx_int16_t,
    mut out_len: *mut spx_uint32_t,
) -> std::ffi::c_int {
    let mut i: spx_uint32_t = 0;
    let mut istride_save: std::ffi::c_int = 0;
    let mut ostride_save: std::ffi::c_int = 0;
    let mut bak_out_len: spx_uint32_t = *out_len;
    let mut bak_in_len: spx_uint32_t = *in_len;
    istride_save = (*st).in_stride;
    ostride_save = (*st).out_stride;
    (*st).out_stride = (*st).nb_channels as std::ffi::c_int;
    (*st).in_stride = (*st).out_stride;
    i = 0 as spx_uint32_t;
    while i < (*st).nb_channels {
        *out_len = bak_out_len;
        *in_len = bak_in_len;
        if !in_0.is_null() {
            speex_resampler_process_int(
                st,
                i,
                in_0.offset(i as isize),
                in_len,
                out.offset(i as isize),
                out_len,
            );
        } else {
            speex_resampler_process_int(
                st,
                i,
                NULL as *const spx_int16_t,
                in_len,
                out.offset(i as isize),
                out_len,
            );
        }
        i = i.wrapping_add(1);
    }
    (*st).in_stride = istride_save;
    (*st).out_stride = ostride_save;
    return if (*st).resampler_ptr
        == Some(
            resampler_basic_zero
                as unsafe extern "C" fn(
                    *mut SpeexResamplerState,
                    spx_uint32_t,
                    *const spx_word16_t,
                    *mut spx_uint32_t,
                    *mut spx_word16_t,
                    *mut spx_uint32_t,
                ) -> std::ffi::c_int,
        ) {
        RESAMPLER_ERR_ALLOC_FAILED as std::ffi::c_int
    } else {
        RESAMPLER_ERR_SUCCESS as std::ffi::c_int
    };
}
#[no_mangle]
pub unsafe extern "C" fn speex_resampler_set_rate(
    mut st: *mut SpeexResamplerState,
    mut in_rate: spx_uint32_t,
    mut out_rate: spx_uint32_t,
) -> std::ffi::c_int {
    return speex_resampler_set_rate_frac(st, in_rate, out_rate, in_rate, out_rate);
}
#[no_mangle]
pub unsafe extern "C" fn speex_resampler_get_rate(
    mut st: *mut SpeexResamplerState,
    mut in_rate: *mut spx_uint32_t,
    mut out_rate: *mut spx_uint32_t,
) {
    *in_rate = (*st).in_rate;
    *out_rate = (*st).out_rate;
}
#[inline]
unsafe extern "C" fn compute_gcd(mut a: spx_uint32_t, mut b: spx_uint32_t) -> spx_uint32_t {
    while b != 0 as spx_uint32_t {
        let mut temp: spx_uint32_t = a;
        a = b;
        b = temp.wrapping_rem(b);
    }
    return a;
}
#[no_mangle]
pub unsafe extern "C" fn speex_resampler_set_rate_frac(
    mut st: *mut SpeexResamplerState,
    mut ratio_num: spx_uint32_t,
    mut ratio_den: spx_uint32_t,
    mut in_rate: spx_uint32_t,
    mut out_rate: spx_uint32_t,
) -> std::ffi::c_int {
    let mut fact: spx_uint32_t = 0;
    let mut old_den: spx_uint32_t = 0;
    let mut i: spx_uint32_t = 0;
    if ratio_num == 0 as spx_uint32_t || ratio_den == 0 as spx_uint32_t {
        return RESAMPLER_ERR_INVALID_ARG as std::ffi::c_int;
    }
    if (*st).in_rate == in_rate
        && (*st).out_rate == out_rate
        && (*st).num_rate == ratio_num
        && (*st).den_rate == ratio_den
    {
        return RESAMPLER_ERR_SUCCESS as std::ffi::c_int;
    }
    old_den = (*st).den_rate;
    (*st).in_rate = in_rate;
    (*st).out_rate = out_rate;
    (*st).num_rate = ratio_num;
    (*st).den_rate = ratio_den;
    fact = compute_gcd((*st).num_rate, (*st).den_rate);
    (*st).num_rate = ((*st).num_rate).wrapping_div(fact);
    (*st).den_rate = ((*st).den_rate).wrapping_div(fact);
    if old_den > 0 as spx_uint32_t {
        i = 0 as spx_uint32_t;
        while i < (*st).nb_channels {
            if multiply_frac(
                &mut *((*st).samp_frac_num).offset(i as isize),
                *((*st).samp_frac_num).offset(i as isize),
                (*st).den_rate,
                old_den,
            ) != RESAMPLER_ERR_SUCCESS as std::ffi::c_int
            {
                return RESAMPLER_ERR_OVERFLOW as std::ffi::c_int;
            }
            if *((*st).samp_frac_num).offset(i as isize) >= (*st).den_rate {
                *((*st).samp_frac_num).offset(i as isize) =
                    ((*st).den_rate).wrapping_sub(1 as spx_uint32_t);
            }
            i = i.wrapping_add(1);
        }
    }
    if (*st).initialised != 0 {
        return update_filter(st);
    }
    return RESAMPLER_ERR_SUCCESS as std::ffi::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn speex_resampler_get_ratio(
    mut st: *mut SpeexResamplerState,
    mut ratio_num: *mut spx_uint32_t,
    mut ratio_den: *mut spx_uint32_t,
) {
    *ratio_num = (*st).num_rate;
    *ratio_den = (*st).den_rate;
}
#[no_mangle]
pub unsafe extern "C" fn speex_resampler_set_quality(
    mut st: *mut SpeexResamplerState,
    mut quality: std::ffi::c_int,
) -> std::ffi::c_int {
    if quality > 10 as std::ffi::c_int || quality < 0 as std::ffi::c_int {
        return RESAMPLER_ERR_INVALID_ARG as std::ffi::c_int;
    }
    if (*st).quality == quality {
        return RESAMPLER_ERR_SUCCESS as std::ffi::c_int;
    }
    (*st).quality = quality;
    if (*st).initialised != 0 {
        return update_filter(st);
    }
    return RESAMPLER_ERR_SUCCESS as std::ffi::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn speex_resampler_get_quality(
    mut st: *mut SpeexResamplerState,
    mut quality: *mut std::ffi::c_int,
) {
    *quality = (*st).quality;
}
#[no_mangle]
pub unsafe extern "C" fn speex_resampler_set_input_stride(
    mut st: *mut SpeexResamplerState,
    mut stride: spx_uint32_t,
) {
    (*st).in_stride = stride as std::ffi::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn speex_resampler_get_input_stride(
    mut st: *mut SpeexResamplerState,
    mut stride: *mut spx_uint32_t,
) {
    *stride = (*st).in_stride as spx_uint32_t;
}
#[no_mangle]
pub unsafe extern "C" fn speex_resampler_set_output_stride(
    mut st: *mut SpeexResamplerState,
    mut stride: spx_uint32_t,
) {
    (*st).out_stride = stride as std::ffi::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn speex_resampler_get_output_stride(
    mut st: *mut SpeexResamplerState,
    mut stride: *mut spx_uint32_t,
) {
    *stride = (*st).out_stride as spx_uint32_t;
}
#[no_mangle]
pub unsafe extern "C" fn speex_resampler_get_input_latency(
    mut st: *mut SpeexResamplerState,
) -> std::ffi::c_int {
    return ((*st).filt_len).wrapping_div(2 as spx_uint32_t) as std::ffi::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn speex_resampler_get_output_latency(
    mut st: *mut SpeexResamplerState,
) -> std::ffi::c_int {
    return ((*st).filt_len)
        .wrapping_div(2 as spx_uint32_t)
        .wrapping_mul((*st).den_rate)
        .wrapping_add((*st).num_rate >> 1 as std::ffi::c_int)
        .wrapping_div((*st).num_rate) as std::ffi::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn speex_resampler_skip_zeros(
    mut st: *mut SpeexResamplerState,
) -> std::ffi::c_int {
    let mut i: spx_uint32_t = 0;
    i = 0 as spx_uint32_t;
    while i < (*st).nb_channels {
        *((*st).last_sample).offset(i as isize) =
            ((*st).filt_len).wrapping_div(2 as spx_uint32_t) as spx_int32_t;
        i = i.wrapping_add(1);
    }
    return RESAMPLER_ERR_SUCCESS as std::ffi::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn speex_resampler_reset_mem(
    mut st: *mut SpeexResamplerState,
) -> std::ffi::c_int {
    let mut i: spx_uint32_t = 0;
    i = 0 as spx_uint32_t;
    while i < (*st).nb_channels {
        *((*st).last_sample).offset(i as isize) = 0 as std::ffi::c_int as spx_int32_t;
        *((*st).magic_samples).offset(i as isize) = 0 as spx_uint32_t;
        *((*st).samp_frac_num).offset(i as isize) = 0 as spx_uint32_t;
        i = i.wrapping_add(1);
    }
    i = 0 as spx_uint32_t;
    while i < ((*st).nb_channels).wrapping_mul(((*st).filt_len).wrapping_sub(1 as spx_uint32_t)) {
        *((*st).mem).offset(i as isize) = 0 as std::ffi::c_int as spx_word16_t;
        i = i.wrapping_add(1);
    }
    return RESAMPLER_ERR_SUCCESS as std::ffi::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn speex_resampler_strerror(
    mut err: std::ffi::c_int,
) -> *const std::ffi::c_char {
    match err {
        0 => return b"Success.\0" as *const u8 as *const std::ffi::c_char,
        1 => {
            return b"Memory allocation failed.\0" as *const u8 as *const std::ffi::c_char;
        }
        2 => return b"Bad resampler state.\0" as *const u8 as *const std::ffi::c_char,
        3 => return b"Invalid argument.\0" as *const u8 as *const std::ffi::c_char,
        4 => {
            return b"Input and output buffers overlap.\0" as *const u8 as *const std::ffi::c_char;
        }
        _ => {
            return b"Unknown error. Bad error code or strange version mismatch.\0" as *const u8
                as *const std::ffi::c_char;
        }
    };
}
