#![allow(unused_assignments)]
#![allow(unused_variables)]


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
    static mut stderr: *mut FILE;
    fn fprintf(
        __stream: *mut FILE,
        __format: *const std::ffi::c_char,
        ...
    ) -> std::ffi::c_int;
    fn calloc(__nmemb: size_t, __size: size_t) -> *mut std::ffi::c_void;
    fn free(__ptr: *mut std::ffi::c_void);
    fn spx_drft_forward(l: *mut drft_lookup, data: *mut std::ffi::c_float);
    fn spx_drft_backward(l: *mut drft_lookup, data: *mut std::ffi::c_float);
    fn spx_drft_init(l: *mut drft_lookup, n: std::ffi::c_int);
    fn spx_drft_clear(l: *mut drft_lookup);
}
pub type __off_t = std::ffi::c_long;
pub type __off64_t = std::ffi::c_long;
pub type size_t = usize;
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
#[derive(Copy, Clone)]
#[repr(C)]
pub struct drft_lookup {
    pub n: std::ffi::c_int,
    pub trigcache: *mut std::ffi::c_float,
    pub splitcache: *mut std::ffi::c_int,
}
#[inline]
unsafe extern "C" fn speex_alloc(size: std::ffi::c_int) -> *mut std::ffi::c_void {
    return calloc(size as size_t, 1 as size_t);
}
#[inline]
unsafe extern "C" fn speex_free(ptr: *mut std::ffi::c_void) {
    free(ptr);
}
#[inline]
unsafe extern "C" fn speex_warning(str: *const std::ffi::c_char) {
    fprintf(stderr, b"warning: %s\n\0" as *const u8 as *const std::ffi::c_char, str);
}
#[no_mangle]
pub unsafe extern "C" fn spx_fft_init(
    size: std::ffi::c_int,
) -> *mut std::ffi::c_void {
    let mut table: *mut drft_lookup = 0 as *mut drft_lookup;
    table = speex_alloc(::core::mem::size_of::<drft_lookup>() as std::ffi::c_int)
        as *mut drft_lookup;
    spx_drft_init(table, size);
    return table as *mut std::ffi::c_void;
}
#[no_mangle]
pub unsafe extern "C" fn spx_fft_destroy(table: *mut std::ffi::c_void) {
    spx_drft_clear(table as *mut drft_lookup);
    speex_free(table);
}
#[no_mangle]
pub unsafe extern "C" fn spx_fft(
    table: *mut std::ffi::c_void,
    in_0: *mut std::ffi::c_float,
    out: *mut std::ffi::c_float,
) {
    if in_0 == out {
        let mut i: std::ffi::c_int = 0;
        let scale: std::ffi::c_float = (1.0f64
            / (*(table as *mut drft_lookup)).n as std::ffi::c_double)
            as std::ffi::c_float;
        speex_warning(
            b"FFT should not be done in-place\0" as *const u8 as *const std::ffi::c_char,
        );
        i = 0 as std::ffi::c_int;
        while i < (*(table as *mut drft_lookup)).n {
            *out.offset(i as isize) = scale * *in_0.offset(i as isize);
            i += 1;
        }
    } else {
        let mut i_0: std::ffi::c_int = 0;
        let scale_0: std::ffi::c_float = (1.0f64
            / (*(table as *mut drft_lookup)).n as std::ffi::c_double)
            as std::ffi::c_float;
        i_0 = 0 as std::ffi::c_int;
        while i_0 < (*(table as *mut drft_lookup)).n {
            *out.offset(i_0 as isize) = scale_0 * *in_0.offset(i_0 as isize);
            i_0 += 1;
        }
    }
    spx_drft_forward(table as *mut drft_lookup, out);
}
#[no_mangle]
pub unsafe extern "C" fn spx_ifft(
    table: *mut std::ffi::c_void,
    in_0: *mut std::ffi::c_float,
    out: *mut std::ffi::c_float,
) {
    if in_0 == out {
        speex_warning(
            b"FFT should not be done in-place\0" as *const u8 as *const std::ffi::c_char,
        );
    } else {
        let mut i: std::ffi::c_int = 0;
        i = 0 as std::ffi::c_int;
        while i < (*(table as *mut drft_lookup)).n {
            *out.offset(i as isize) = *in_0.offset(i as isize);
            i += 1;
        }
    }
    spx_drft_backward(table as *mut drft_lookup, out);
}
#[no_mangle]
pub unsafe extern "C" fn spx_fft_float(
    table: *mut std::ffi::c_void,
    in_0: *mut std::ffi::c_float,
    out: *mut std::ffi::c_float,
) {
    spx_fft(table, in_0, out);
}
#[no_mangle]
pub unsafe extern "C" fn spx_ifft_float(
    table: *mut std::ffi::c_void,
    in_0: *mut std::ffi::c_float,
    out: *mut std::ffi::c_float,
) {
    spx_ifft(table, in_0, out);
}
