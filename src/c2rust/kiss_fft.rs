#![allow(unused_assignments)]
#![allow(unused_variables)]
#![allow(unused_unsafe)]

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

extern "C" {
    fn calloc(__nmemb: size_t, __size: size_t) -> *mut std::ffi::c_void;
    fn exit(__status: std::ffi::c_int) -> !;
    fn cos(__x: std::ffi::c_double) -> std::ffi::c_double;
    fn sin(__x: std::ffi::c_double) -> std::ffi::c_double;
    static mut stderr: *mut FILE;
    fn fprintf(
        __stream: *mut FILE,
        __format: *const std::ffi::c_char,
        ...
    ) -> std::ffi::c_int;
}
pub type size_t = usize;
pub type int32_t = std::ffi::c_int;
pub type __off_t = std::ffi::c_long;
pub type __off64_t = std::ffi::c_long;
pub type spx_int32_t = int32_t;
pub type spx_word32_t = std::ffi::c_float;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct kiss_fft_cpx {
    pub r: std::ffi::c_float,
    pub i: std::ffi::c_float,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct kiss_fft_state {
    pub nfft: std::ffi::c_int,
    pub inverse: std::ffi::c_int,
    pub factors: [std::ffi::c_int; 64],
    pub twiddles: [kiss_fft_cpx; 1],
}
pub type kiss_fft_cfg = *mut kiss_fft_state;
pub type FILE = _IO_FILE;
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
pub const NULL: std::ffi::c_int = unsafe { 0 as std::ffi::c_int };
pub const KISS_FFT_MALLOC: unsafe extern "C" fn(
    std::ffi::c_int,
) -> *mut std::ffi::c_void = unsafe { speex_alloc };
#[inline]
unsafe extern "C" fn speex_alloc(size: std::ffi::c_int) -> *mut std::ffi::c_void {
    return calloc(size as size_t, 1 as size_t);
}
#[inline]
unsafe extern "C" fn _speex_fatal(
    str: *const std::ffi::c_char,
    file: *const std::ffi::c_char,
    line: std::ffi::c_int,
) {
    fprintf(
        stderr,
        b"Fatal (internal) error in %s, line %d: %s\n\0" as *const u8
            as *const std::ffi::c_char,
        file,
        line,
        str,
    );
    exit(1 as std::ffi::c_int);
}
unsafe extern "C" fn kf_bfly2(
    mut Fout: *mut kiss_fft_cpx,
    fstride: size_t,
    st: kiss_fft_cfg,
    m: std::ffi::c_int,
    N: std::ffi::c_int,
    mm: std::ffi::c_int,
) {
    let mut Fout2: *mut kiss_fft_cpx = 0 as *mut kiss_fft_cpx;
    let mut tw1: *mut kiss_fft_cpx = 0 as *mut kiss_fft_cpx;
    let mut t: kiss_fft_cpx = kiss_fft_cpx { r: 0., i: 0. };
    if (*st).inverse == 0 {
        let mut i: std::ffi::c_int = 0;
        let mut j: std::ffi::c_int = 0;
        let Fout_beg: *mut kiss_fft_cpx = Fout;
        i = 0 as std::ffi::c_int;
        while i < N {
            Fout = Fout_beg.offset((i * mm) as isize);
            Fout2 = Fout.offset(m as isize);
            tw1 = ((*st).twiddles).as_mut_ptr();
            j = 0 as std::ffi::c_int;
            while j < m {
                let mut tr: spx_word32_t = 0.;
                let mut ti: spx_word32_t = 0.;
                tr = (*Fout2).r * (*tw1).r - (*Fout2).i * (*tw1).i;
                ti = (*Fout2).i * (*tw1).r + (*Fout2).r * (*tw1).i;
                tw1 = tw1.offset(fstride as isize);
                (*Fout2).r = ((*Fout).r as spx_word32_t - tr) as std::ffi::c_float;
                (*Fout2).i = ((*Fout).i as spx_word32_t - ti) as std::ffi::c_float;
                (*Fout).r = ((*Fout).r as spx_word32_t + tr) as std::ffi::c_float;
                (*Fout).i = ((*Fout).i as spx_word32_t + ti) as std::ffi::c_float;
                Fout2 = Fout2.offset(1);
                Fout = Fout.offset(1);
                j += 1;
            }
            i += 1;
        }
    } else {
        let mut i_0: std::ffi::c_int = 0;
        let mut j_0: std::ffi::c_int = 0;
        let Fout_beg_0: *mut kiss_fft_cpx = Fout;
        i_0 = 0 as std::ffi::c_int;
        while i_0 < N {
            Fout = Fout_beg_0.offset((i_0 * mm) as isize);
            Fout2 = Fout.offset(m as isize);
            tw1 = ((*st).twiddles).as_mut_ptr();
            j_0 = 0 as std::ffi::c_int;
            while j_0 < m {
                t.r = (*Fout2).r * (*tw1).r - (*Fout2).i * (*tw1).i;
                t.i = (*Fout2).r * (*tw1).i + (*Fout2).i * (*tw1).r;
                tw1 = tw1.offset(fstride as isize);
                (*Fout2).r = (*Fout).r - t.r;
                (*Fout2).i = (*Fout).i - t.i;
                (*Fout).r += t.r;
                (*Fout).i += t.i;
                Fout2 = Fout2.offset(1);
                Fout = Fout.offset(1);
                j_0 += 1;
            }
            i_0 += 1;
        }
    };
}
unsafe extern "C" fn kf_bfly4(
    mut Fout: *mut kiss_fft_cpx,
    fstride: size_t,
    st: kiss_fft_cfg,
    m: std::ffi::c_int,
    N: std::ffi::c_int,
    mm: std::ffi::c_int,
) {
    let mut tw1: *mut kiss_fft_cpx = 0 as *mut kiss_fft_cpx;
    let mut tw2: *mut kiss_fft_cpx = 0 as *mut kiss_fft_cpx;
    let mut tw3: *mut kiss_fft_cpx = 0 as *mut kiss_fft_cpx;
    let mut scratch: [kiss_fft_cpx; 6] = [kiss_fft_cpx { r: 0., i: 0. }; 6];
    let m2: size_t = (2 as std::ffi::c_int * m) as size_t;
    let m3: size_t = (3 as std::ffi::c_int * m) as size_t;
    let mut i: std::ffi::c_int = 0;
    let mut j: std::ffi::c_int = 0;
    if (*st).inverse != 0 {
        let Fout_beg: *mut kiss_fft_cpx = Fout;
        i = 0 as std::ffi::c_int;
        while i < N {
            Fout = Fout_beg.offset((i * mm) as isize);
            tw1 = ((*st).twiddles).as_mut_ptr();
            tw2 = tw1;
            tw3 = tw2;
            j = 0 as std::ffi::c_int;
            while j < m {
                scratch[0 as std::ffi::c_int as usize].r = (*Fout.offset(m as isize)).r
                    * (*tw1).r - (*Fout.offset(m as isize)).i * (*tw1).i;
                scratch[0 as std::ffi::c_int as usize].i = (*Fout.offset(m as isize)).r
                    * (*tw1).i + (*Fout.offset(m as isize)).i * (*tw1).r;
                scratch[1 as std::ffi::c_int as usize].r = (*Fout.offset(m2 as isize)).r
                    * (*tw2).r - (*Fout.offset(m2 as isize)).i * (*tw2).i;
                scratch[1 as std::ffi::c_int as usize].i = (*Fout.offset(m2 as isize)).r
                    * (*tw2).i + (*Fout.offset(m2 as isize)).i * (*tw2).r;
                scratch[2 as std::ffi::c_int as usize].r = (*Fout.offset(m3 as isize)).r
                    * (*tw3).r - (*Fout.offset(m3 as isize)).i * (*tw3).i;
                scratch[2 as std::ffi::c_int as usize].i = (*Fout.offset(m3 as isize)).r
                    * (*tw3).i + (*Fout.offset(m3 as isize)).i * (*tw3).r;
                scratch[5 as std::ffi::c_int as usize].r = (*Fout).r
                    - scratch[1 as std::ffi::c_int as usize].r;
                scratch[5 as std::ffi::c_int as usize].i = (*Fout).i
                    - scratch[1 as std::ffi::c_int as usize].i;
                (*Fout).r += scratch[1 as std::ffi::c_int as usize].r;
                (*Fout).i += scratch[1 as std::ffi::c_int as usize].i;
                scratch[3 as std::ffi::c_int as usize].r = scratch[0 as std::ffi::c_int
                        as usize]
                    .r + scratch[2 as std::ffi::c_int as usize].r;
                scratch[3 as std::ffi::c_int as usize].i = scratch[0 as std::ffi::c_int
                        as usize]
                    .i + scratch[2 as std::ffi::c_int as usize].i;
                scratch[4 as std::ffi::c_int as usize].r = scratch[0 as std::ffi::c_int
                        as usize]
                    .r - scratch[2 as std::ffi::c_int as usize].r;
                scratch[4 as std::ffi::c_int as usize].i = scratch[0 as std::ffi::c_int
                        as usize]
                    .i - scratch[2 as std::ffi::c_int as usize].i;
                (*Fout.offset(m2 as isize)).r = (*Fout).r
                    - scratch[3 as std::ffi::c_int as usize].r;
                (*Fout.offset(m2 as isize)).i = (*Fout).i
                    - scratch[3 as std::ffi::c_int as usize].i;
                tw1 = tw1.offset(fstride as isize);
                tw2 = tw2.offset(fstride.wrapping_mul(2 as size_t) as isize);
                tw3 = tw3.offset(fstride.wrapping_mul(3 as size_t) as isize);
                (*Fout).r += scratch[3 as std::ffi::c_int as usize].r;
                (*Fout).i += scratch[3 as std::ffi::c_int as usize].i;
                (*Fout.offset(m as isize)).r = scratch[5 as std::ffi::c_int as usize].r
                    - scratch[4 as std::ffi::c_int as usize].i;
                (*Fout.offset(m as isize)).i = scratch[5 as std::ffi::c_int as usize].i
                    + scratch[4 as std::ffi::c_int as usize].r;
                (*Fout.offset(m3 as isize)).r = scratch[5 as std::ffi::c_int as usize].r
                    + scratch[4 as std::ffi::c_int as usize].i;
                (*Fout.offset(m3 as isize)).i = scratch[5 as std::ffi::c_int as usize].i
                    - scratch[4 as std::ffi::c_int as usize].r;
                Fout = Fout.offset(1);
                j += 1;
            }
            i += 1;
        }
    } else {
        let Fout_beg_0: *mut kiss_fft_cpx = Fout;
        i = 0 as std::ffi::c_int;
        while i < N {
            Fout = Fout_beg_0.offset((i * mm) as isize);
            tw1 = ((*st).twiddles).as_mut_ptr();
            tw2 = tw1;
            tw3 = tw2;
            j = 0 as std::ffi::c_int;
            while j < m {
                scratch[0 as std::ffi::c_int as usize].r = (*Fout.offset(m as isize)).r
                    * (*tw1).r - (*Fout.offset(m as isize)).i * (*tw1).i;
                scratch[0 as std::ffi::c_int as usize].i = (*Fout.offset(m as isize)).r
                    * (*tw1).i + (*Fout.offset(m as isize)).i * (*tw1).r;
                scratch[1 as std::ffi::c_int as usize].r = (*Fout.offset(m2 as isize)).r
                    * (*tw2).r - (*Fout.offset(m2 as isize)).i * (*tw2).i;
                scratch[1 as std::ffi::c_int as usize].i = (*Fout.offset(m2 as isize)).r
                    * (*tw2).i + (*Fout.offset(m2 as isize)).i * (*tw2).r;
                scratch[2 as std::ffi::c_int as usize].r = (*Fout.offset(m3 as isize)).r
                    * (*tw3).r - (*Fout.offset(m3 as isize)).i * (*tw3).i;
                scratch[2 as std::ffi::c_int as usize].i = (*Fout.offset(m3 as isize)).r
                    * (*tw3).i + (*Fout.offset(m3 as isize)).i * (*tw3).r;
                (*Fout).r = (*Fout).r;
                (*Fout).i = (*Fout).i;
                scratch[5 as std::ffi::c_int as usize].r = (*Fout).r
                    - scratch[1 as std::ffi::c_int as usize].r;
                scratch[5 as std::ffi::c_int as usize].i = (*Fout).i
                    - scratch[1 as std::ffi::c_int as usize].i;
                (*Fout).r += scratch[1 as std::ffi::c_int as usize].r;
                (*Fout).i += scratch[1 as std::ffi::c_int as usize].i;
                scratch[3 as std::ffi::c_int as usize].r = scratch[0 as std::ffi::c_int
                        as usize]
                    .r + scratch[2 as std::ffi::c_int as usize].r;
                scratch[3 as std::ffi::c_int as usize].i = scratch[0 as std::ffi::c_int
                        as usize]
                    .i + scratch[2 as std::ffi::c_int as usize].i;
                scratch[4 as std::ffi::c_int as usize].r = scratch[0 as std::ffi::c_int
                        as usize]
                    .r - scratch[2 as std::ffi::c_int as usize].r;
                scratch[4 as std::ffi::c_int as usize].i = scratch[0 as std::ffi::c_int
                        as usize]
                    .i - scratch[2 as std::ffi::c_int as usize].i;
                (*Fout.offset(m2 as isize)).r = (*Fout.offset(m2 as isize)).r;
                (*Fout.offset(m2 as isize)).i = (*Fout.offset(m2 as isize)).i;
                (*Fout.offset(m2 as isize)).r = (*Fout).r
                    - scratch[3 as std::ffi::c_int as usize].r;
                (*Fout.offset(m2 as isize)).i = (*Fout).i
                    - scratch[3 as std::ffi::c_int as usize].i;
                tw1 = tw1.offset(fstride as isize);
                tw2 = tw2.offset(fstride.wrapping_mul(2 as size_t) as isize);
                tw3 = tw3.offset(fstride.wrapping_mul(3 as size_t) as isize);
                (*Fout).r += scratch[3 as std::ffi::c_int as usize].r;
                (*Fout).i += scratch[3 as std::ffi::c_int as usize].i;
                (*Fout.offset(m as isize)).r = scratch[5 as std::ffi::c_int as usize].r
                    + scratch[4 as std::ffi::c_int as usize].i;
                (*Fout.offset(m as isize)).i = scratch[5 as std::ffi::c_int as usize].i
                    - scratch[4 as std::ffi::c_int as usize].r;
                (*Fout.offset(m3 as isize)).r = scratch[5 as std::ffi::c_int as usize].r
                    - scratch[4 as std::ffi::c_int as usize].i;
                (*Fout.offset(m3 as isize)).i = scratch[5 as std::ffi::c_int as usize].i
                    + scratch[4 as std::ffi::c_int as usize].r;
                Fout = Fout.offset(1);
                j += 1;
            }
            i += 1;
        }
    };
}
unsafe extern "C" fn kf_bfly3(
    mut Fout: *mut kiss_fft_cpx,
    fstride: size_t,
    st: kiss_fft_cfg,
    m: size_t,
) {
    let mut k: size_t = m;
    let m2: size_t = (2 as size_t).wrapping_mul(m);
    let mut tw1: *mut kiss_fft_cpx = 0 as *mut kiss_fft_cpx;
    let mut tw2: *mut kiss_fft_cpx = 0 as *mut kiss_fft_cpx;
    let mut scratch: [kiss_fft_cpx; 5] = [kiss_fft_cpx { r: 0., i: 0. }; 5];
    let mut epi3: kiss_fft_cpx = kiss_fft_cpx { r: 0., i: 0. };
    epi3 = *((*st).twiddles).as_mut_ptr().offset(fstride.wrapping_mul(m) as isize);
    tw2 = ((*st).twiddles).as_mut_ptr();
    tw1 = tw2;
    loop {
        (*st).inverse == 0;
        scratch[1 as std::ffi::c_int as usize].r = (*Fout.offset(m as isize)).r
            * (*tw1).r - (*Fout.offset(m as isize)).i * (*tw1).i;
        scratch[1 as std::ffi::c_int as usize].i = (*Fout.offset(m as isize)).r
            * (*tw1).i + (*Fout.offset(m as isize)).i * (*tw1).r;
        scratch[2 as std::ffi::c_int as usize].r = (*Fout.offset(m2 as isize)).r
            * (*tw2).r - (*Fout.offset(m2 as isize)).i * (*tw2).i;
        scratch[2 as std::ffi::c_int as usize].i = (*Fout.offset(m2 as isize)).r
            * (*tw2).i + (*Fout.offset(m2 as isize)).i * (*tw2).r;
        scratch[3 as std::ffi::c_int as usize].r = scratch[1 as std::ffi::c_int as usize]
            .r + scratch[2 as std::ffi::c_int as usize].r;
        scratch[3 as std::ffi::c_int as usize].i = scratch[1 as std::ffi::c_int as usize]
            .i + scratch[2 as std::ffi::c_int as usize].i;
        scratch[0 as std::ffi::c_int as usize].r = scratch[1 as std::ffi::c_int as usize]
            .r - scratch[2 as std::ffi::c_int as usize].r;
        scratch[0 as std::ffi::c_int as usize].i = scratch[1 as std::ffi::c_int as usize]
            .i - scratch[2 as std::ffi::c_int as usize].i;
        tw1 = tw1.offset(fstride as isize);
        tw2 = tw2.offset(fstride.wrapping_mul(2 as size_t) as isize);
        (*Fout.offset(m as isize)).r = ((*Fout).r as std::ffi::c_double
            - scratch[3 as std::ffi::c_int as usize].r as std::ffi::c_double * 0.5f64)
            as std::ffi::c_float;
        (*Fout.offset(m as isize)).i = ((*Fout).i as std::ffi::c_double
            - scratch[3 as std::ffi::c_int as usize].i as std::ffi::c_double * 0.5f64)
            as std::ffi::c_float;
        scratch[0 as std::ffi::c_int as usize].r *= epi3.i;
        scratch[0 as std::ffi::c_int as usize].i *= epi3.i;
        (*Fout).r += scratch[3 as std::ffi::c_int as usize].r;
        (*Fout).i += scratch[3 as std::ffi::c_int as usize].i;
        (*Fout.offset(m2 as isize)).r = (*Fout.offset(m as isize)).r
            + scratch[0 as std::ffi::c_int as usize].i;
        (*Fout.offset(m2 as isize)).i = (*Fout.offset(m as isize)).i
            - scratch[0 as std::ffi::c_int as usize].r;
        (*Fout.offset(m as isize)).r -= scratch[0 as std::ffi::c_int as usize].i;
        (*Fout.offset(m as isize)).i += scratch[0 as std::ffi::c_int as usize].r;
        Fout = Fout.offset(1);
        k = k.wrapping_sub(1);
        if !(k != 0) {
            break;
        }
    };
}
unsafe extern "C" fn kf_bfly5(
    Fout: *mut kiss_fft_cpx,
    fstride: size_t,
    st: kiss_fft_cfg,
    m: std::ffi::c_int,
) {
    let mut Fout0: *mut kiss_fft_cpx = 0 as *mut kiss_fft_cpx;
    let mut Fout1: *mut kiss_fft_cpx = 0 as *mut kiss_fft_cpx;
    let mut Fout2: *mut kiss_fft_cpx = 0 as *mut kiss_fft_cpx;
    let mut Fout3: *mut kiss_fft_cpx = 0 as *mut kiss_fft_cpx;
    let mut Fout4: *mut kiss_fft_cpx = 0 as *mut kiss_fft_cpx;
    let mut u: std::ffi::c_int = 0;
    let mut scratch: [kiss_fft_cpx; 13] = [kiss_fft_cpx { r: 0., i: 0. }; 13];
    let twiddles: *mut kiss_fft_cpx = ((*st).twiddles).as_mut_ptr();
    let mut tw: *mut kiss_fft_cpx = 0 as *mut kiss_fft_cpx;
    let mut ya: kiss_fft_cpx = kiss_fft_cpx { r: 0., i: 0. };
    let mut yb: kiss_fft_cpx = kiss_fft_cpx { r: 0., i: 0. };
    ya = *twiddles.offset(fstride.wrapping_mul(m as size_t) as isize);
    yb = *twiddles
        .offset(fstride.wrapping_mul(2 as size_t).wrapping_mul(m as size_t) as isize);
    Fout0 = Fout;
    Fout1 = Fout0.offset(m as isize);
    Fout2 = Fout0.offset((2 as std::ffi::c_int * m) as isize);
    Fout3 = Fout0.offset((3 as std::ffi::c_int * m) as isize);
    Fout4 = Fout0.offset((4 as std::ffi::c_int * m) as isize);
    tw = ((*st).twiddles).as_mut_ptr();
    u = 0 as std::ffi::c_int;
    while u < m {
        (*st).inverse == 0;
        scratch[0 as std::ffi::c_int as usize] = *Fout0;
        scratch[1 as std::ffi::c_int as usize].r = (*Fout1).r
            * (*tw.offset((u as size_t).wrapping_mul(fstride) as isize)).r
            - (*Fout1).i * (*tw.offset((u as size_t).wrapping_mul(fstride) as isize)).i;
        scratch[1 as std::ffi::c_int as usize].i = (*Fout1).r
            * (*tw.offset((u as size_t).wrapping_mul(fstride) as isize)).i
            + (*Fout1).i * (*tw.offset((u as size_t).wrapping_mul(fstride) as isize)).r;
        scratch[2 as std::ffi::c_int as usize].r = (*Fout2).r
            * (*tw
                .offset(
                    ((2 as std::ffi::c_int * u) as size_t).wrapping_mul(fstride) as isize,
                ))
                .r
            - (*Fout2).i
                * (*tw
                    .offset(
                        ((2 as std::ffi::c_int * u) as size_t).wrapping_mul(fstride)
                            as isize,
                    ))
                    .i;
        scratch[2 as std::ffi::c_int as usize].i = (*Fout2).r
            * (*tw
                .offset(
                    ((2 as std::ffi::c_int * u) as size_t).wrapping_mul(fstride) as isize,
                ))
                .i
            + (*Fout2).i
                * (*tw
                    .offset(
                        ((2 as std::ffi::c_int * u) as size_t).wrapping_mul(fstride)
                            as isize,
                    ))
                    .r;
        scratch[3 as std::ffi::c_int as usize].r = (*Fout3).r
            * (*tw
                .offset(
                    ((3 as std::ffi::c_int * u) as size_t).wrapping_mul(fstride) as isize,
                ))
                .r
            - (*Fout3).i
                * (*tw
                    .offset(
                        ((3 as std::ffi::c_int * u) as size_t).wrapping_mul(fstride)
                            as isize,
                    ))
                    .i;
        scratch[3 as std::ffi::c_int as usize].i = (*Fout3).r
            * (*tw
                .offset(
                    ((3 as std::ffi::c_int * u) as size_t).wrapping_mul(fstride) as isize,
                ))
                .i
            + (*Fout3).i
                * (*tw
                    .offset(
                        ((3 as std::ffi::c_int * u) as size_t).wrapping_mul(fstride)
                            as isize,
                    ))
                    .r;
        scratch[4 as std::ffi::c_int as usize].r = (*Fout4).r
            * (*tw
                .offset(
                    ((4 as std::ffi::c_int * u) as size_t).wrapping_mul(fstride) as isize,
                ))
                .r
            - (*Fout4).i
                * (*tw
                    .offset(
                        ((4 as std::ffi::c_int * u) as size_t).wrapping_mul(fstride)
                            as isize,
                    ))
                    .i;
        scratch[4 as std::ffi::c_int as usize].i = (*Fout4).r
            * (*tw
                .offset(
                    ((4 as std::ffi::c_int * u) as size_t).wrapping_mul(fstride) as isize,
                ))
                .i
            + (*Fout4).i
                * (*tw
                    .offset(
                        ((4 as std::ffi::c_int * u) as size_t).wrapping_mul(fstride)
                            as isize,
                    ))
                    .r;
        scratch[7 as std::ffi::c_int as usize].r = scratch[1 as std::ffi::c_int as usize]
            .r + scratch[4 as std::ffi::c_int as usize].r;
        scratch[7 as std::ffi::c_int as usize].i = scratch[1 as std::ffi::c_int as usize]
            .i + scratch[4 as std::ffi::c_int as usize].i;
        scratch[10 as std::ffi::c_int as usize].r = scratch[1 as std::ffi::c_int
                as usize]
            .r - scratch[4 as std::ffi::c_int as usize].r;
        scratch[10 as std::ffi::c_int as usize].i = scratch[1 as std::ffi::c_int
                as usize]
            .i - scratch[4 as std::ffi::c_int as usize].i;
        scratch[8 as std::ffi::c_int as usize].r = scratch[2 as std::ffi::c_int as usize]
            .r + scratch[3 as std::ffi::c_int as usize].r;
        scratch[8 as std::ffi::c_int as usize].i = scratch[2 as std::ffi::c_int as usize]
            .i + scratch[3 as std::ffi::c_int as usize].i;
        scratch[9 as std::ffi::c_int as usize].r = scratch[2 as std::ffi::c_int as usize]
            .r - scratch[3 as std::ffi::c_int as usize].r;
        scratch[9 as std::ffi::c_int as usize].i = scratch[2 as std::ffi::c_int as usize]
            .i - scratch[3 as std::ffi::c_int as usize].i;
        (*Fout0).r
            += scratch[7 as std::ffi::c_int as usize].r
                + scratch[8 as std::ffi::c_int as usize].r;
        (*Fout0).i
            += scratch[7 as std::ffi::c_int as usize].i
                + scratch[8 as std::ffi::c_int as usize].i;
        scratch[5 as std::ffi::c_int as usize].r = scratch[0 as std::ffi::c_int as usize]
            .r + scratch[7 as std::ffi::c_int as usize].r * ya.r
            + scratch[8 as std::ffi::c_int as usize].r * yb.r;
        scratch[5 as std::ffi::c_int as usize].i = scratch[0 as std::ffi::c_int as usize]
            .i + scratch[7 as std::ffi::c_int as usize].i * ya.r
            + scratch[8 as std::ffi::c_int as usize].i * yb.r;
        scratch[6 as std::ffi::c_int as usize].r = scratch[10 as std::ffi::c_int
                as usize]
            .i * ya.i + scratch[9 as std::ffi::c_int as usize].i * yb.i;
        scratch[6 as std::ffi::c_int as usize].i = -(scratch[10 as std::ffi::c_int
                as usize]
            .r * ya.i) - scratch[9 as std::ffi::c_int as usize].r * yb.i;
        (*Fout1).r = scratch[5 as std::ffi::c_int as usize].r
            - scratch[6 as std::ffi::c_int as usize].r;
        (*Fout1).i = scratch[5 as std::ffi::c_int as usize].i
            - scratch[6 as std::ffi::c_int as usize].i;
        (*Fout4).r = scratch[5 as std::ffi::c_int as usize].r
            + scratch[6 as std::ffi::c_int as usize].r;
        (*Fout4).i = scratch[5 as std::ffi::c_int as usize].i
            + scratch[6 as std::ffi::c_int as usize].i;
        scratch[11 as std::ffi::c_int as usize].r = scratch[0 as std::ffi::c_int
                as usize]
            .r + scratch[7 as std::ffi::c_int as usize].r * yb.r
            + scratch[8 as std::ffi::c_int as usize].r * ya.r;
        scratch[11 as std::ffi::c_int as usize].i = scratch[0 as std::ffi::c_int
                as usize]
            .i + scratch[7 as std::ffi::c_int as usize].i * yb.r
            + scratch[8 as std::ffi::c_int as usize].i * ya.r;
        scratch[12 as std::ffi::c_int as usize].r = -(scratch[10 as std::ffi::c_int
                as usize]
            .i * yb.i) + scratch[9 as std::ffi::c_int as usize].i * ya.i;
        scratch[12 as std::ffi::c_int as usize].i = scratch[10 as std::ffi::c_int
                as usize]
            .r * yb.i - scratch[9 as std::ffi::c_int as usize].r * ya.i;
        (*Fout2).r = scratch[11 as std::ffi::c_int as usize].r
            + scratch[12 as std::ffi::c_int as usize].r;
        (*Fout2).i = scratch[11 as std::ffi::c_int as usize].i
            + scratch[12 as std::ffi::c_int as usize].i;
        (*Fout3).r = scratch[11 as std::ffi::c_int as usize].r
            - scratch[12 as std::ffi::c_int as usize].r;
        (*Fout3).i = scratch[11 as std::ffi::c_int as usize].i
            - scratch[12 as std::ffi::c_int as usize].i;
        Fout0 = Fout0.offset(1);
        Fout1 = Fout1.offset(1);
        Fout2 = Fout2.offset(1);
        Fout3 = Fout3.offset(1);
        Fout4 = Fout4.offset(1);
        u += 1;
    }
}
unsafe extern "C" fn kf_bfly_generic(
    Fout: *mut kiss_fft_cpx,
    fstride: size_t,
    st: kiss_fft_cfg,
    m: std::ffi::c_int,
    p: std::ffi::c_int,
) {
    let mut u: std::ffi::c_int = 0;
    let mut k: std::ffi::c_int = 0;
    let mut q1: std::ffi::c_int = 0;
    let mut q: std::ffi::c_int = 0;
    let twiddles: *mut kiss_fft_cpx = ((*st).twiddles).as_mut_ptr();
    let mut t: kiss_fft_cpx = kiss_fft_cpx { r: 0., i: 0. };
    let mut scratchbuf: [kiss_fft_cpx; 17] = [kiss_fft_cpx { r: 0., i: 0. }; 17];
    let Norig: std::ffi::c_int = (*st).nfft;
    if p > 17 as std::ffi::c_int {
        _speex_fatal(
            b"KissFFT: max radix supported is 17\0" as *const u8
                as *const std::ffi::c_char,
            b"/home/bepis/Documents/cyborgism/speexdsp/libspeexdsp/kiss_fft.c\0"
                as *const u8 as *const std::ffi::c_char,
            294 as std::ffi::c_int,
        );
    }
    u = 0 as std::ffi::c_int;
    while u < m {
        k = u;
        q1 = 0 as std::ffi::c_int;
        while q1 < p {
            scratchbuf[q1 as usize] = *Fout.offset(k as isize);
            (*st).inverse == 0;
            k += m;
            q1 += 1;
        }
        k = u;
        q1 = 0 as std::ffi::c_int;
        while q1 < p {
            let mut twidx: std::ffi::c_int = 0 as std::ffi::c_int;
            *Fout.offset(k as isize) = scratchbuf[0 as std::ffi::c_int as usize];
            q = 1 as std::ffi::c_int;
            while q < p {
                twidx = (twidx as size_t).wrapping_add(fstride.wrapping_mul(k as size_t))
                    as std::ffi::c_int as std::ffi::c_int;
                if twidx >= Norig {
                    twidx -= Norig;
                }
                t.r = scratchbuf[q as usize].r * (*twiddles.offset(twidx as isize)).r
                    - scratchbuf[q as usize].i * (*twiddles.offset(twidx as isize)).i;
                t.i = scratchbuf[q as usize].r * (*twiddles.offset(twidx as isize)).i
                    + scratchbuf[q as usize].i * (*twiddles.offset(twidx as isize)).r;
                (*Fout.offset(k as isize)).r += t.r;
                (*Fout.offset(k as isize)).i += t.i;
                q += 1;
            }
            k += m;
            q1 += 1;
        }
        u += 1;
    }
}
unsafe extern "C" fn kf_shuffle(
    mut Fout: *mut kiss_fft_cpx,
    mut f: *const kiss_fft_cpx,
    fstride: size_t,
    in_stride: std::ffi::c_int,
    mut factors: *mut std::ffi::c_int,
    st: kiss_fft_cfg,
) {
    let fresh0 = factors;
    factors = factors.offset(1);
    let p: std::ffi::c_int = *fresh0;
    let fresh1 = factors;
    factors = factors.offset(1);
    let m: std::ffi::c_int = *fresh1;
    if m == 1 as std::ffi::c_int {
        let mut j: std::ffi::c_int = 0;
        j = 0 as std::ffi::c_int;
        while j < p {
            *Fout.offset(j as isize) = *f;
            f = f.offset(fstride.wrapping_mul(in_stride as size_t) as isize);
            j += 1;
        }
    } else {
        let mut j_0: std::ffi::c_int = 0;
        j_0 = 0 as std::ffi::c_int;
        while j_0 < p {
            kf_shuffle(
                Fout,
                f,
                fstride.wrapping_mul(p as size_t),
                in_stride,
                factors,
                st,
            );
            f = f.offset(fstride.wrapping_mul(in_stride as size_t) as isize);
            Fout = Fout.offset(m as isize);
            j_0 += 1;
        }
    };
}
unsafe extern "C" fn kf_work(
    mut Fout: *mut kiss_fft_cpx,
    f: *const kiss_fft_cpx,
    fstride: size_t,
    in_stride: std::ffi::c_int,
    mut factors: *mut std::ffi::c_int,
    st: kiss_fft_cfg,
    N: std::ffi::c_int,
    s2: std::ffi::c_int,
    m2: std::ffi::c_int,
) {
    let mut i: std::ffi::c_int = 0;
    let Fout_beg: *mut kiss_fft_cpx = Fout;
    let fresh2 = factors;
    factors = factors.offset(1);
    let p: std::ffi::c_int = *fresh2;
    let fresh3 = factors;
    factors = factors.offset(1);
    let m: std::ffi::c_int = *fresh3;
    if !(m == 1 as std::ffi::c_int) {
        kf_work(
            Fout,
            f,
            fstride.wrapping_mul(p as size_t),
            in_stride,
            factors,
            st,
            N * p,
            fstride.wrapping_mul(in_stride as size_t) as std::ffi::c_int,
            m,
        );
    }
    match p {
        2 => {
            kf_bfly2(Fout, fstride, st, m, N, m2);
        }
        3 => {
            i = 0 as std::ffi::c_int;
            while i < N {
                Fout = Fout_beg.offset((i * m2) as isize);
                kf_bfly3(Fout, fstride, st, m as size_t);
                i += 1;
            }
        }
        4 => {
            kf_bfly4(Fout, fstride, st, m, N, m2);
        }
        5 => {
            i = 0 as std::ffi::c_int;
            while i < N {
                Fout = Fout_beg.offset((i * m2) as isize);
                kf_bfly5(Fout, fstride, st, m);
                i += 1;
            }
        }
        _ => {
            i = 0 as std::ffi::c_int;
            while i < N {
                Fout = Fout_beg.offset((i * m2) as isize);
                kf_bfly_generic(Fout, fstride, st, m, p);
                i += 1;
            }
        }
    };
}
unsafe extern "C" fn kf_factor(
    mut n: std::ffi::c_int,
    mut facbuf: *mut std::ffi::c_int,
) {
    let mut p: std::ffi::c_int = 4 as std::ffi::c_int;
    loop {
        while n % p != 0 {
            match p {
                4 => {
                    p = 2 as std::ffi::c_int;
                }
                2 => {
                    p = 3 as std::ffi::c_int;
                }
                _ => {
                    p += 2 as std::ffi::c_int;
                }
            }
            if p > 32000 as std::ffi::c_int || p * p > n {
                p = n;
            }
        }
        n /= p;
        let fresh4 = facbuf;
        facbuf = facbuf.offset(1);
        *fresh4 = p;
        let fresh5 = facbuf;
        facbuf = facbuf.offset(1);
        *fresh5 = n;
        if !(n > 1 as std::ffi::c_int) {
            break;
        }
    };
}
#[no_mangle]
pub unsafe extern "C" fn kiss_fft_alloc(
    nfft: std::ffi::c_int,
    inverse_fft: std::ffi::c_int,
    mem: *mut std::ffi::c_void,
    lenmem: *mut size_t,
) -> kiss_fft_cfg {
    let mut st: kiss_fft_cfg = NULL as kiss_fft_cfg;
    let memneeded: size_t = (::core::mem::size_of::<kiss_fft_state>() as size_t)
        .wrapping_add(
            (::core::mem::size_of::<kiss_fft_cpx>() as size_t)
                .wrapping_mul((nfft - 1 as std::ffi::c_int) as size_t),
        );
    if lenmem.is_null() {
        st = speex_alloc(memneeded as std::ffi::c_int) as kiss_fft_cfg;
    } else {
        if !mem.is_null() && *lenmem >= memneeded {
            st = mem as kiss_fft_cfg;
        }
        *lenmem = memneeded;
    }
    if !st.is_null() {
        let mut i: std::ffi::c_int = 0;
        (*st).nfft = nfft;
        (*st).inverse = inverse_fft;
        i = 0 as std::ffi::c_int;
        while i < nfft {
            let pi: std::ffi::c_double = 3.14159265358979323846264338327f64;
            let mut phase: std::ffi::c_double = -(2 as std::ffi::c_int)
                as std::ffi::c_double * pi / nfft as std::ffi::c_double
                * i as std::ffi::c_double;
            if (*st).inverse != 0 {
                phase *= -(1 as std::ffi::c_int) as std::ffi::c_double;
            }
            (*((*st).twiddles).as_mut_ptr().offset(i as isize)).r = cos(phase)
                as std::ffi::c_float;
            (*((*st).twiddles).as_mut_ptr().offset(i as isize)).i = sin(phase)
                as std::ffi::c_float;
            i += 1;
        }
        kf_factor(nfft, ((*st).factors).as_mut_ptr());
    }
    return st;
}
#[no_mangle]
pub unsafe extern "C" fn kiss_fft_stride(
    st: kiss_fft_cfg,
    fin: *const kiss_fft_cpx,
    fout: *mut kiss_fft_cpx,
    in_stride: std::ffi::c_int,
) {
    if fin == fout as *const kiss_fft_cpx {
        _speex_fatal(
            b"In-place FFT not supported\0" as *const u8 as *const std::ffi::c_char,
            b"/home/bepis/Documents/cyborgism/speexdsp/libspeexdsp/kiss_fft.c\0"
                as *const u8 as *const std::ffi::c_char,
            509 as std::ffi::c_int,
        );
    } else {
        kf_shuffle(fout, fin, 1 as size_t, in_stride, ((*st).factors).as_mut_ptr(), st);
        kf_work(
            fout,
            fin,
            1 as size_t,
            in_stride,
            ((*st).factors).as_mut_ptr(),
            st,
            1 as std::ffi::c_int,
            in_stride,
            1 as std::ffi::c_int,
        );
    };
}
#[no_mangle]
pub unsafe extern "C" fn kiss_fft(
    cfg: kiss_fft_cfg,
    fin: *const kiss_fft_cpx,
    fout: *mut kiss_fft_cpx,
) {
    kiss_fft_stride(cfg, fin, fout, 1 as std::ffi::c_int);
}
