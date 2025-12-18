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
    #[cfg_attr(target_os = "macos", link_name = "__stderrp")]
    static mut stderr: *mut FILE;
    fn fprintf(
        __stream: *mut FILE,
        __format: *const std::ffi::c_char,
        ...
    ) -> std::ffi::c_int;
    fn calloc(__nmemb: size_t, __size: size_t) -> *mut std::ffi::c_void;
    fn exit(__status: std::ffi::c_int) -> !;
    fn cos(__x: std::ffi::c_double) -> std::ffi::c_double;
    fn sin(__x: std::ffi::c_double) -> std::ffi::c_double;
    fn kiss_fft_alloc(
        nfft: std::ffi::c_int,
        inverse_fft: std::ffi::c_int,
        mem: *mut std::ffi::c_void,
        lenmem: *mut size_t,
    ) -> kiss_fft_cfg;
    fn kiss_fft(cfg: kiss_fft_cfg, fin: *const kiss_fft_cpx, fout: *mut kiss_fft_cpx);
}
pub type size_t = usize;
pub type __off_t = std::ffi::c_long;
pub type __off64_t = std::ffi::c_long;
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
pub type FILE = _IO_FILE;
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
#[derive(Copy, Clone)]
#[repr(C)]
pub struct kiss_fftr_state {
    pub substate: kiss_fft_cfg,
    pub tmpbuf: *mut kiss_fft_cpx,
    pub super_twiddles: *mut kiss_fft_cpx,
}
pub type kiss_fftr_cfg = *mut kiss_fftr_state;
pub const NULL: std::ffi::c_int = unsafe { 0 as std::ffi::c_int };
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
#[inline]
unsafe extern "C" fn speex_warning(str: *const std::ffi::c_char) {
    fprintf(stderr, b"warning: %s\n\0" as *const u8 as *const std::ffi::c_char, str);
}
pub const KISS_FFT_MALLOC: unsafe extern "C" fn(
    std::ffi::c_int,
) -> *mut std::ffi::c_void = unsafe { speex_alloc };
#[no_mangle]
pub unsafe extern "C" fn kiss_fftr_alloc(
    mut nfft: std::ffi::c_int,
    inverse_fft: std::ffi::c_int,
    mem: *mut std::ffi::c_void,
    lenmem: *mut size_t,
) -> kiss_fftr_cfg {
    let mut i: std::ffi::c_int = 0;
    let mut st: kiss_fftr_cfg = NULL as kiss_fftr_cfg;
    let mut subsize: size_t = 0;
    let mut memneeded: size_t = 0;
    if nfft & 1 as std::ffi::c_int != 0 {
        speex_warning(
            b"Real FFT optimization must be even.\n\0" as *const u8
                as *const std::ffi::c_char,
        );
        return NULL as kiss_fftr_cfg;
    }
    nfft >>= 1 as std::ffi::c_int;
    kiss_fft_alloc(nfft, inverse_fft, NULL as *mut std::ffi::c_void, &mut subsize);
    memneeded = (::core::mem::size_of::<kiss_fftr_state>() as size_t)
        .wrapping_add(subsize)
        .wrapping_add(
            (::core::mem::size_of::<kiss_fft_cpx>() as size_t)
                .wrapping_mul((nfft * 2 as std::ffi::c_int) as size_t),
        );
    if lenmem.is_null() {
        st = speex_alloc(memneeded as std::ffi::c_int) as kiss_fftr_cfg;
    } else {
        if *lenmem >= memneeded {
            st = mem as kiss_fftr_cfg;
        }
        *lenmem = memneeded;
    }
    if st.is_null() {
        return NULL as kiss_fftr_cfg;
    }
    (*st).substate = st.offset(1 as std::ffi::c_int as isize) as kiss_fft_cfg;
    (*st).tmpbuf = ((*st).substate as *mut std::ffi::c_char).offset(subsize as isize)
        as *mut kiss_fft_cpx;
    (*st).super_twiddles = ((*st).tmpbuf).offset(nfft as isize);
    kiss_fft_alloc(
        nfft,
        inverse_fft,
        (*st).substate as *mut std::ffi::c_void,
        &mut subsize,
    );
    i = 0 as std::ffi::c_int;
    while i < nfft {
        let pi: std::ffi::c_double = 3.14159265358979323846264338327f64;
        let mut phase: std::ffi::c_double = pi
            * (i as std::ffi::c_double / nfft as std::ffi::c_double + 0.5f64);
        if inverse_fft == 0 {
            phase = -phase;
        }
        (*((*st).super_twiddles).offset(i as isize)).r = cos(phase) as std::ffi::c_float;
        (*((*st).super_twiddles).offset(i as isize)).i = sin(phase) as std::ffi::c_float;
        i += 1;
    }
    return st;
}
#[no_mangle]
pub unsafe extern "C" fn kiss_fftr(
    st: kiss_fftr_cfg,
    timedata: *const std::ffi::c_float,
    freqdata: *mut kiss_fft_cpx,
) {
    let mut k: std::ffi::c_int = 0;
    let mut ncfft: std::ffi::c_int = 0;
    let mut fpnk: kiss_fft_cpx = kiss_fft_cpx { r: 0., i: 0. };
    let mut fpk: kiss_fft_cpx = kiss_fft_cpx { r: 0., i: 0. };
    let mut f1k: kiss_fft_cpx = kiss_fft_cpx { r: 0., i: 0. };
    let mut f2k: kiss_fft_cpx = kiss_fft_cpx { r: 0., i: 0. };
    let mut tw: kiss_fft_cpx = kiss_fft_cpx { r: 0., i: 0. };
    let mut tdc: kiss_fft_cpx = kiss_fft_cpx { r: 0., i: 0. };
    if (*(*st).substate).inverse != 0 {
        _speex_fatal(
            b"kiss fft usage error: improper alloc\n\0" as *const u8
                as *const std::ffi::c_char,
            b"/home/bepis/Documents/cyborgism/speexdsp/libspeexdsp/kiss_fftr.c\0"
                as *const u8 as *const std::ffi::c_char,
            88 as std::ffi::c_int,
        );
    }
    ncfft = (*(*st).substate).nfft;
    kiss_fft((*st).substate, timedata as *const kiss_fft_cpx, (*st).tmpbuf);
    tdc.r = (*((*st).tmpbuf).offset(0 as std::ffi::c_int as isize)).r;
    tdc.i = (*((*st).tmpbuf).offset(0 as std::ffi::c_int as isize)).i;
    (*freqdata.offset(0 as std::ffi::c_int as isize)).r = tdc.r + tdc.i;
    (*freqdata.offset(ncfft as isize)).r = tdc.r - tdc.i;
    let ref mut fresh0 = (*freqdata.offset(0 as std::ffi::c_int as isize)).i;
    *fresh0 = 0 as std::ffi::c_int as std::ffi::c_float;
    (*freqdata.offset(ncfft as isize)).i = *fresh0;
    k = 1 as std::ffi::c_int;
    while k <= ncfft / 2 as std::ffi::c_int {
        fpk = *((*st).tmpbuf).offset(k as isize);
        fpnk.r = (*((*st).tmpbuf).offset((ncfft - k) as isize)).r;
        fpnk.i = -(*((*st).tmpbuf).offset((ncfft - k) as isize)).i;
        f1k.r = fpk.r + fpnk.r;
        f1k.i = fpk.i + fpnk.i;
        f2k.r = fpk.r - fpnk.r;
        f2k.i = fpk.i - fpnk.i;
        tw.r = f2k.r * (*((*st).super_twiddles).offset(k as isize)).r
            - f2k.i * (*((*st).super_twiddles).offset(k as isize)).i;
        tw.i = f2k.r * (*((*st).super_twiddles).offset(k as isize)).i
            + f2k.i * (*((*st).super_twiddles).offset(k as isize)).r;
        (*freqdata.offset(k as isize)).r = ((f1k.r + tw.r) as std::ffi::c_double
            * 0.5f64) as std::ffi::c_float;
        (*freqdata.offset(k as isize)).i = ((f1k.i + tw.i) as std::ffi::c_double
            * 0.5f64) as std::ffi::c_float;
        (*freqdata.offset((ncfft - k) as isize)).r = ((f1k.r - tw.r)
            as std::ffi::c_double * 0.5f64) as std::ffi::c_float;
        (*freqdata.offset((ncfft - k) as isize)).i = ((tw.i - f1k.i)
            as std::ffi::c_double * 0.5f64) as std::ffi::c_float;
        k += 1;
    }
}
#[no_mangle]
pub unsafe extern "C" fn kiss_fftri(
    st: kiss_fftr_cfg,
    freqdata: *const kiss_fft_cpx,
    timedata: *mut std::ffi::c_float,
) {
    let mut k: std::ffi::c_int = 0;
    let mut ncfft: std::ffi::c_int = 0;
    if (*(*st).substate).inverse == 0 as std::ffi::c_int {
        _speex_fatal(
            b"kiss fft usage error: improper alloc\n\0" as *const u8
                as *const std::ffi::c_char,
            b"/home/bepis/Documents/cyborgism/speexdsp/libspeexdsp/kiss_fftr.c\0"
                as *const u8 as *const std::ffi::c_char,
            142 as std::ffi::c_int,
        );
    }
    ncfft = (*(*st).substate).nfft;
    (*((*st).tmpbuf).offset(0 as std::ffi::c_int as isize)).r = (*freqdata
        .offset(0 as std::ffi::c_int as isize))
        .r + (*freqdata.offset(ncfft as isize)).r;
    (*((*st).tmpbuf).offset(0 as std::ffi::c_int as isize)).i = (*freqdata
        .offset(0 as std::ffi::c_int as isize))
        .r - (*freqdata.offset(ncfft as isize)).r;
    k = 1 as std::ffi::c_int;
    while k <= ncfft / 2 as std::ffi::c_int {
        let mut fk: kiss_fft_cpx = kiss_fft_cpx { r: 0., i: 0. };
        let mut fnkc: kiss_fft_cpx = kiss_fft_cpx { r: 0., i: 0. };
        let mut fek: kiss_fft_cpx = kiss_fft_cpx { r: 0., i: 0. };
        let mut fok: kiss_fft_cpx = kiss_fft_cpx { r: 0., i: 0. };
        let mut tmp: kiss_fft_cpx = kiss_fft_cpx { r: 0., i: 0. };
        fk = *freqdata.offset(k as isize);
        fnkc.r = (*freqdata.offset((ncfft - k) as isize)).r;
        fnkc.i = -(*freqdata.offset((ncfft - k) as isize)).i;
        fek.r = fk.r + fnkc.r;
        fek.i = fk.i + fnkc.i;
        tmp.r = fk.r - fnkc.r;
        tmp.i = fk.i - fnkc.i;
        fok.r = tmp.r * (*((*st).super_twiddles).offset(k as isize)).r
            - tmp.i * (*((*st).super_twiddles).offset(k as isize)).i;
        fok.i = tmp.r * (*((*st).super_twiddles).offset(k as isize)).i
            + tmp.i * (*((*st).super_twiddles).offset(k as isize)).r;
        (*((*st).tmpbuf).offset(k as isize)).r = fek.r + fok.r;
        (*((*st).tmpbuf).offset(k as isize)).i = fek.i + fok.i;
        (*((*st).tmpbuf).offset((ncfft - k) as isize)).r = fek.r - fok.r;
        (*((*st).tmpbuf).offset((ncfft - k) as isize)).i = fek.i - fok.i;
        (*((*st).tmpbuf).offset((ncfft - k) as isize)).i
            *= -(1 as std::ffi::c_int) as std::ffi::c_float;
        k += 1;
    }
    kiss_fft((*st).substate, (*st).tmpbuf, timedata as *mut kiss_fft_cpx);
}
#[no_mangle]
pub unsafe extern "C" fn kiss_fftr2(
    st: kiss_fftr_cfg,
    timedata: *const std::ffi::c_float,
    freqdata: *mut std::ffi::c_float,
) {
    let mut k: std::ffi::c_int = 0;
    let mut ncfft: std::ffi::c_int = 0;
    let mut f2k: kiss_fft_cpx = kiss_fft_cpx { r: 0., i: 0. };
    let mut tdc: kiss_fft_cpx = kiss_fft_cpx { r: 0., i: 0. };
    let mut f1kr: spx_word32_t = 0.;
    let mut f1ki: spx_word32_t = 0.;
    let mut twr: spx_word32_t = 0.;
    let mut twi: spx_word32_t = 0.;
    if (*(*st).substate).inverse != 0 {
        _speex_fatal(
            b"kiss fft usage error: improper alloc\n\0" as *const u8
                as *const std::ffi::c_char,
            b"/home/bepis/Documents/cyborgism/speexdsp/libspeexdsp/kiss_fftr.c\0"
                as *const u8 as *const std::ffi::c_char,
            181 as std::ffi::c_int,
        );
    }
    ncfft = (*(*st).substate).nfft;
    kiss_fft((*st).substate, timedata as *const kiss_fft_cpx, (*st).tmpbuf);
    tdc.r = (*((*st).tmpbuf).offset(0 as std::ffi::c_int as isize)).r;
    tdc.i = (*((*st).tmpbuf).offset(0 as std::ffi::c_int as isize)).i;
    *freqdata.offset(0 as std::ffi::c_int as isize) = tdc.r + tdc.i;
    *freqdata.offset((2 as std::ffi::c_int * ncfft - 1 as std::ffi::c_int) as isize) = tdc
        .r - tdc.i;
    k = 1 as std::ffi::c_int;
    while k <= ncfft / 2 as std::ffi::c_int {
        f2k.r = (*((*st).tmpbuf).offset(k as isize)).r
            - (*((*st).tmpbuf).offset((ncfft - k) as isize)).r;
        f2k.i = (*((*st).tmpbuf).offset(k as isize)).i
            + (*((*st).tmpbuf).offset((ncfft - k) as isize)).i;
        f1kr = ((*((*st).tmpbuf).offset(k as isize)).r
            + (*((*st).tmpbuf).offset((ncfft - k) as isize)).r) as spx_word32_t;
        f1ki = ((*((*st).tmpbuf).offset(k as isize)).i
            - (*((*st).tmpbuf).offset((ncfft - k) as isize)).i) as spx_word32_t;
        twr = f2k.r * (*((*st).super_twiddles).offset(k as isize)).r
            - f2k.i * (*((*st).super_twiddles).offset(k as isize)).i;
        twi = f2k.i * (*((*st).super_twiddles).offset(k as isize)).r
            + f2k.r * (*((*st).super_twiddles).offset(k as isize)).i;
        *freqdata.offset((2 as std::ffi::c_int * k - 1 as std::ffi::c_int) as isize) = (0.5f32
            * (f1kr + twr)) as std::ffi::c_float;
        *freqdata.offset((2 as std::ffi::c_int * k) as isize) = (0.5f32 * (f1ki + twi))
            as std::ffi::c_float;
        *freqdata
            .offset(
                (2 as std::ffi::c_int * (ncfft - k) - 1 as std::ffi::c_int) as isize,
            ) = (0.5f32 * (f1kr - twr)) as std::ffi::c_float;
        *freqdata.offset((2 as std::ffi::c_int * (ncfft - k)) as isize) = (0.5f32
            * (twi - f1ki)) as std::ffi::c_float;
        k += 1;
    }
}
#[no_mangle]
pub unsafe extern "C" fn kiss_fftri2(
    st: kiss_fftr_cfg,
    freqdata: *const std::ffi::c_float,
    timedata: *mut std::ffi::c_float,
) {
    let mut k: std::ffi::c_int = 0;
    let mut ncfft: std::ffi::c_int = 0;
    if (*(*st).substate).inverse == 0 as std::ffi::c_int {
        _speex_fatal(
            b"kiss fft usage error: improper alloc\n\0" as *const u8
                as *const std::ffi::c_char,
            b"/home/bepis/Documents/cyborgism/speexdsp/libspeexdsp/kiss_fftr.c\0"
                as *const u8 as *const std::ffi::c_char,
            267 as std::ffi::c_int,
        );
    }
    ncfft = (*(*st).substate).nfft;
    (*((*st).tmpbuf).offset(0 as std::ffi::c_int as isize)).r = *freqdata
        .offset(0 as std::ffi::c_int as isize)
        + *freqdata
            .offset((2 as std::ffi::c_int * ncfft - 1 as std::ffi::c_int) as isize);
    (*((*st).tmpbuf).offset(0 as std::ffi::c_int as isize)).i = *freqdata
        .offset(0 as std::ffi::c_int as isize)
        - *freqdata
            .offset((2 as std::ffi::c_int * ncfft - 1 as std::ffi::c_int) as isize);
    k = 1 as std::ffi::c_int;
    while k <= ncfft / 2 as std::ffi::c_int {
        let mut fk: kiss_fft_cpx = kiss_fft_cpx { r: 0., i: 0. };
        let mut fnkc: kiss_fft_cpx = kiss_fft_cpx { r: 0., i: 0. };
        let mut fek: kiss_fft_cpx = kiss_fft_cpx { r: 0., i: 0. };
        let mut fok: kiss_fft_cpx = kiss_fft_cpx { r: 0., i: 0. };
        let mut tmp: kiss_fft_cpx = kiss_fft_cpx { r: 0., i: 0. };
        fk.r = *freqdata
            .offset((2 as std::ffi::c_int * k - 1 as std::ffi::c_int) as isize);
        fk.i = *freqdata.offset((2 as std::ffi::c_int * k) as isize);
        fnkc.r = *freqdata
            .offset(
                (2 as std::ffi::c_int * (ncfft - k) - 1 as std::ffi::c_int) as isize,
            );
        fnkc.i = -*freqdata.offset((2 as std::ffi::c_int * (ncfft - k)) as isize);
        fek.r = fk.r + fnkc.r;
        fek.i = fk.i + fnkc.i;
        tmp.r = fk.r - fnkc.r;
        tmp.i = fk.i - fnkc.i;
        fok.r = tmp.r * (*((*st).super_twiddles).offset(k as isize)).r
            - tmp.i * (*((*st).super_twiddles).offset(k as isize)).i;
        fok.i = tmp.r * (*((*st).super_twiddles).offset(k as isize)).i
            + tmp.i * (*((*st).super_twiddles).offset(k as isize)).r;
        (*((*st).tmpbuf).offset(k as isize)).r = fek.r + fok.r;
        (*((*st).tmpbuf).offset(k as isize)).i = fek.i + fok.i;
        (*((*st).tmpbuf).offset((ncfft - k) as isize)).r = fek.r - fok.r;
        (*((*st).tmpbuf).offset((ncfft - k) as isize)).i = fek.i - fok.i;
        (*((*st).tmpbuf).offset((ncfft - k) as isize)).i
            *= -(1 as std::ffi::c_int) as std::ffi::c_float;
        k += 1;
    }
    kiss_fft((*st).substate, (*st).tmpbuf, timedata as *mut kiss_fft_cpx);
}
