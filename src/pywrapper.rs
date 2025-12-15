#![cfg(feature = "cpal-example")]
#![allow(dead_code)]

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, Weak,
    },
    thread,
    time::Duration,
};

use cpal::SampleFormat;
use futures::executor::block_on;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyModule},
    wrap_pyfunction,
};

use crate::cpal_aec::{
    get_supported_input_configs as inner_get_supported_input_configs,
    get_supported_output_configs as inner_get_supported_output_configs, AecConfig as InnerAecConfig,
    AecStream as InnerAecStream, InputDeviceConfig, OutputDeviceConfig,
    OutputStreamAlignerProducer as InnerOutputStreamAlignerProducer,
    StreamProducer as InnerStreamProducer,
};

fn to_py_err(err: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

fn parse_sample_format(fmt: Option<&str>) -> PyResult<SampleFormat> {
    match fmt {
        None => Ok(SampleFormat::F32),
        Some(raw) => match raw.to_ascii_lowercase().as_str() {
            "f32" | "float32" => Ok(SampleFormat::F32),
            "i16" | "s16" | "int16" => Ok(SampleFormat::I16),
            "u16" | "uint16" => Ok(SampleFormat::U16),
            other => Err(PyValueError::new_err(format!(
                "Unknown sample format '{other}', expected f32/i16/u16"
            ))),
        },
    }
}

fn sample_format_to_string(fmt: SampleFormat) -> &'static str {
    match fmt {
        SampleFormat::F32 => "f32",
        SampleFormat::I16 => "i16",
        SampleFormat::U16 => "u16",
        _ => "unknown",
    }
}

fn resolve_host_id(name: Option<&str>) -> PyResult<cpal::HostId> {
    if let Some(name) = name {
        for host in cpal::available_hosts() {
            if host.name().eq_ignore_ascii_case(name) {
                return Ok(host);
            }
        }
        let available = cpal::available_hosts()
            .into_iter()
            .map(|h| h.name().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        Err(PyValueError::new_err(format!(
            "Unknown host '{name}'. Available hosts: {available}"
        )))
    } else {
        Ok(cpal::default_host().id())
    }
}

fn slice_to_pybytes<'py, T>(py: Python<'py>, data: &[T]) -> Py<PyBytes> {
    let raw = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<T>(),
        )
    };
    PyBytes::new(py, raw).into()
}

#[pyclass(name = "InputDeviceConfig")]
#[derive(Clone)]
pub struct PyInputDeviceConfig {
    inner: InputDeviceConfig,
}

impl PyInputDeviceConfig {
    fn from_inner(inner: InputDeviceConfig) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyInputDeviceConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        host_id: Option<String>,
        device_name: String,
        channels: usize,
        sample_rate: u32,
        sample_format: Option<String>,
        history_len: usize,
        calibration_packets: u32,
        audio_buffer_seconds: u32,
        resampler_quality: i32,
    ) -> PyResult<Self> {
        let host_id = resolve_host_id(host_id.as_deref())?;
        let sample_format = parse_sample_format(sample_format.as_deref())?;
        Ok(Self {
            inner: InputDeviceConfig::new(
                host_id,
                device_name,
                channels,
                sample_rate,
                sample_format,
                history_len,
                calibration_packets,
                audio_buffer_seconds,
                resampler_quality,
            ),
        })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn from_default(
        py: Python<'_>,
        host: Option<String>,
        device_name: String,
        history_len: usize,
        calibration_packets: u32,
        audio_buffer_seconds: u32,
        resampler_quality: i32,
    ) -> PyResult<Self> {
        let host_id = resolve_host_id(host.as_deref())?;
        let inner = py
            .allow_threads(move || {
                block_on(InputDeviceConfig::from_default(
                    host_id,
                    device_name,
                    history_len,
                    calibration_packets,
                    audio_buffer_seconds,
                    resampler_quality,
                ))
                .map_err(|e| e.to_string())
            })
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn host_name(&self) -> String {
        self.inner.host_id.name().to_string()
    }

    #[setter]
    fn set_host_name(&mut self, host: String) -> PyResult<()> {
        self.inner.host_id = resolve_host_id(Some(&host))?;
        Ok(())
    }

    #[getter]
    fn device_name(&self) -> String {
        self.inner.device_name.clone()
    }

    #[setter]
    fn set_device_name(&mut self, name: String) {
        self.inner.device_name = name;
    }

    #[getter]
    fn channels(&self) -> usize {
        self.inner.channels
    }

    #[setter]
    fn set_channels(&mut self, channels: usize) {
        self.inner.channels = channels;
    }

    #[getter]
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate
    }

    #[setter]
    fn set_sample_rate(&mut self, rate: u32) {
        self.inner.sample_rate = rate;
    }

    #[getter]
    fn sample_format(&self) -> String {
        sample_format_to_string(self.inner.sample_format).to_string()
    }

    #[setter]
    fn set_sample_format(&mut self, fmt: String) -> PyResult<()> {
        self.inner.sample_format = parse_sample_format(Some(&fmt))?;
        Ok(())
    }

    #[getter]
    fn history_len(&self) -> usize {
        self.inner.history_len
    }

    #[setter]
    fn set_history_len(&mut self, len: usize) {
        self.inner.history_len = len;
    }

    #[getter]
    fn calibration_packets(&self) -> u32 {
        self.inner.calibration_packets
    }

    #[setter]
    fn set_calibration_packets(&mut self, packets: u32) {
        self.inner.calibration_packets = packets;
    }

    #[getter]
    fn audio_buffer_seconds(&self) -> u32 {
        self.inner.audio_buffer_seconds
    }

    #[setter]
    fn set_audio_buffer_seconds(&mut self, secs: u32) {
        self.inner.audio_buffer_seconds = secs;
    }

    #[getter]
    fn resampler_quality(&self) -> i32 {
        self.inner.resampler_quality
    }

    #[setter]
    fn set_resampler_quality(&mut self, quality: i32) {
        self.inner.resampler_quality = quality;
    }

    pub fn clone_config(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

#[pyclass(name = "OutputDeviceConfig")]
#[derive(Clone)]
pub struct PyOutputDeviceConfig {
    inner: OutputDeviceConfig,
}

impl PyOutputDeviceConfig {
    fn from_inner(inner: OutputDeviceConfig) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOutputDeviceConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        host: Option<String>,
        device_name: String,
        channels: usize,
        sample_rate: u32,
        sample_format: Option<String>,
        history_len: usize,
        calibration_packets: u32,
        audio_buffer_seconds: u32,
        resampler_quality: i32,
        frame_size: u32,
    ) -> PyResult<Self> {
        let host_id = resolve_host_id(host.as_deref())?;
        let sample_format = parse_sample_format(sample_format.as_deref())?;
        Ok(Self {
            inner: OutputDeviceConfig::new(
                host_id,
                device_name,
                channels,
                sample_rate,
                sample_format,
                history_len,
                calibration_packets,
                audio_buffer_seconds,
                resampler_quality,
                frame_size,
            ),
        })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn from_default(
        py: Python<'_>,
        host: Option<String>,
        device_name: String,
        history_len: usize,
        calibration_packets: u32,
        audio_buffer_seconds: u32,
        resampler_quality: i32,
        frame_size_millis: u32,
    ) -> PyResult<Self> {
        let host_id = resolve_host_id(host.as_deref())?;
        let inner = py
            .allow_threads(move || {
                block_on(OutputDeviceConfig::from_default(
                    host_id,
                    device_name,
                    history_len,
                    calibration_packets,
                    audio_buffer_seconds,
                    resampler_quality,
                    frame_size_millis,
                ))
                .map_err(|e| e.to_string())
            })
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn host_name(&self) -> String {
        self.inner.host_id.name().to_string()
    }

    #[setter]
    fn set_host_name(&mut self, host: String) -> PyResult<()> {
        self.inner.host_id = resolve_host_id(Some(&host))?;
        Ok(())
    }

    #[getter]
    fn device_name(&self) -> String {
        self.inner.device_name.clone()
    }

    #[setter]
    fn set_device_name(&mut self, name: String) {
        self.inner.device_name = name;
    }

    #[getter]
    fn channels(&self) -> usize {
        self.inner.channels
    }

    #[setter]
    fn set_channels(&mut self, channels: usize) {
        self.inner.channels = channels;
    }

    #[getter]
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate
    }

    #[setter]
    fn set_sample_rate(&mut self, rate: u32) {
        self.inner.sample_rate = rate;
    }

    #[getter]
    fn sample_format(&self) -> String {
        sample_format_to_string(self.inner.sample_format).to_string()
    }

    #[setter]
    fn set_sample_format(&mut self, fmt: String) -> PyResult<()> {
        self.inner.sample_format = parse_sample_format(Some(&fmt))?;
        Ok(())
    }

    #[getter]
    fn history_len(&self) -> usize {
        self.inner.history_len
    }

    #[setter]
    fn set_history_len(&mut self, len: usize) {
        self.inner.history_len = len;
    }

    #[getter]
    fn calibration_packets(&self) -> u32 {
        self.inner.calibration_packets
    }

    #[setter]
    fn set_calibration_packets(&mut self, packets: u32) {
        self.inner.calibration_packets = packets;
    }

    #[getter]
    fn audio_buffer_seconds(&self) -> u32 {
        self.inner.audio_buffer_seconds
    }

    #[setter]
    fn set_audio_buffer_seconds(&mut self, secs: u32) {
        self.inner.audio_buffer_seconds = secs;
    }

    #[getter]
    fn resampler_quality(&self) -> i32 {
        self.inner.resampler_quality
    }

    #[setter]
    fn set_resampler_quality(&mut self, quality: i32) {
        self.inner.resampler_quality = quality;
    }

    #[getter]
    fn frame_size(&self) -> u32 {
        self.inner.frame_size
    }

    #[setter]
    fn set_frame_size(&mut self, size: u32) {
        self.inner.frame_size = size;
    }

    pub fn clone_config(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

#[pyclass(name = "AecConfig")]
#[derive(Clone)]
pub struct PyAecConfig {
    target_sample_rate: u32,
    frame_size: usize,
    filter_length: usize,
}

impl PyAecConfig {
    fn to_inner(&self) -> InnerAecConfig {
        InnerAecConfig::new(self.target_sample_rate, self.frame_size, self.filter_length)
    }
}

#[pymethods]
impl PyAecConfig {
    #[new]
    pub fn new(target_sample_rate: u32, frame_size: usize, filter_length: usize) -> Self {
        Self {
            target_sample_rate,
            frame_size,
            filter_length,
        }
    }

    #[getter]
    fn target_sample_rate(&self) -> u32 {
        self.target_sample_rate
    }

    #[setter]
    fn set_target_sample_rate(&mut self, rate: u32) {
        self.target_sample_rate = rate;
    }

    #[getter]
    fn frame_size(&self) -> usize {
        self.frame_size
    }

    #[setter]
    fn set_frame_size(&mut self, size: usize) {
        self.frame_size = size;
    }

    #[getter]
    fn filter_length(&self) -> usize {
        self.filter_length
    }

    #[setter]
    fn set_filter_length(&mut self, len: usize) {
        self.filter_length = len;
    }
}

#[pyclass(name = "StreamProducer")]
pub struct PyStreamProducer {
    inner: Arc<Mutex<InnerStreamProducer>>,
}

#[pymethods]
impl PyStreamProducer {
    pub fn queue_audio(&mut self, audio: Vec<f32>) {
        self.inner.lock().unwrap().queue_audio(&audio);
    }

    #[getter]
    fn stream_id(&self) -> u64 {
        self.inner.lock().unwrap().stream_id()
    }

    /// Convenience helper to close the stream via its owning output producer.
    pub fn close(&self, producer: &mut PyOutputStreamAlignerProducer) -> PyResult<()> {
        producer.end_audio_stream(self)
    }
}

#[pyclass(name = "OutputStreamAlignerProducer")]
pub struct PyOutputStreamAlignerProducer {
    inner: Option<InnerOutputStreamAlignerProducer>,
}

impl PyOutputStreamAlignerProducer {
    fn inner_mut(&mut self) -> PyResult<&mut InnerOutputStreamAlignerProducer> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Output stream producer no longer available"))
    }
}

#[pymethods]
impl PyOutputStreamAlignerProducer {
    #[getter]
    fn device_name(&self) -> PyResult<String> {
        Ok(self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Output stream producer no longer available"))?
            .device_name
            .clone())
    }

    #[getter]
    fn channels(&self) -> PyResult<usize> {
        Ok(self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Output stream producer no longer available"))?
            .channels)
    }

    pub fn begin_audio_stream(
        &mut self,
        channels: usize,
        channel_map: Option<HashMap<usize, Vec<usize>>>,
        audio_buffer_seconds: u32,
        sample_rate: u32,
        resampler_quality: i32,
    ) -> PyResult<PyStreamProducer> {
        let map = channel_map.unwrap_or_else(|| {
            let mut default = HashMap::new();
            for ch in 0..channels {
                default.insert(ch, vec![ch]);
            }
            default
        });
        let producer = self
            .inner_mut()?
            .begin_audio_stream(
                channels,
                map,
                audio_buffer_seconds,
                sample_rate,
                resampler_quality,
            )
            .map_err(to_py_err)?;
        Ok(PyStreamProducer {
            inner: Arc::new(Mutex::new(producer)),
        })
    }

    pub fn end_audio_stream(&mut self, stream: &PyStreamProducer) -> PyResult<()> {
        let guard = stream
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Stream producer lock poisoned"))?;
        self.inner_mut()?
            .end_audio_stream(&guard)
            .map_err(to_py_err)
    }

    pub fn interrupt_all_streams(&mut self) -> PyResult<()> {
        self.inner_mut()?.interrupt_all_streams().map_err(to_py_err)
    }
}

#[pyclass(name = "AecStream")]
pub struct PyAecStream {
    inner: Arc<Mutex<InnerAecStream>>,
    config: PyAecConfig,
    callback: Arc<Mutex<Option<Py<PyAny>>>>,
    callback_thread: Option<thread::JoinHandle<()>>,
    callback_shutdown: Arc<AtomicBool>,
}

#[pymethods]
impl PyAecStream {
    #[new]
    pub fn new(config: PyAecConfig) -> PyResult<Self> {
        let inner = InnerAecStream::new(config.to_inner()).map_err(to_py_err)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
            config,
            callback: Arc::new(Mutex::new(None)),
            callback_thread: None,
            callback_shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    #[getter]
    fn config(&self) -> PyAecConfig {
        self.config.clone()
    }

    #[getter]
    fn num_input_channels(&self) -> usize {
        self.inner.lock().unwrap().num_input_channels()
    }

    #[getter]
    fn num_output_channels(&self) -> usize {
        self.inner.lock().unwrap().num_output_channels()
    }

    pub fn add_input_device(
        &mut self,
        py: Python<'_>,
        config: &PyInputDeviceConfig,
    ) -> PyResult<()> {
        let inner = &self.inner;
        let cfg = config.inner.clone();
        py.allow_threads(move || {
            let mut guard = inner.lock().unwrap();
            block_on(guard.add_input_device(&cfg)).map_err(|e| e.to_string())
        })
        .map_err(to_py_err)
    }

    pub fn add_output_device(
        &mut self,
        py: Python<'_>,
        config: &PyOutputDeviceConfig,
    ) -> PyResult<PyOutputStreamAlignerProducer> {
        let inner = &self.inner;
        let cfg = config.inner.clone();
        let producer = py
            .allow_threads(move || {
                let mut guard = inner.lock().unwrap();
                block_on(guard.add_output_device(&cfg)).map_err(|e| e.to_string())
            })
            .map_err(to_py_err)?;
        Ok(PyOutputStreamAlignerProducer {
            inner: Some(producer),
        })
    }

    pub fn remove_input_device(&mut self, config: &PyInputDeviceConfig) -> PyResult<()> {
        self.inner
            .lock()
            .unwrap()
            .remove_input_device(&config.inner)
            .map_err(to_py_err)
    }

    pub fn remove_output_device(&mut self, config: &PyOutputDeviceConfig) -> PyResult<()> {
        self.inner
            .lock()
            .unwrap()
            .remove_output_device(&config.inner)
            .map_err(to_py_err)
    }

    pub fn calibrate(
        &mut self,
        py: Python<'_>,
        mut producers: Vec<PyRefMut<PyOutputStreamAlignerProducer>>,
        debug_wav: bool,
    ) -> PyResult<()> {
        let mut owned: Vec<InnerOutputStreamAlignerProducer> = Vec::with_capacity(producers.len());
        for producer in producers.iter_mut() {
            let inner = producer
                .inner
                .take()
                .ok_or_else(|| PyRuntimeError::new_err("Output stream producer no longer available"))?;
            owned.push(inner);
        }

        let inner = &self.inner;
        let (result, owned) = py.allow_threads(move || {
            let mut owned = owned;
            let res = {
                let mut guard = inner.lock().unwrap();
                block_on(guard.calibrate(owned.as_mut_slice(), debug_wav)).map_err(|e| e.to_string())
            };
            (res, owned)
        });

        for (mut wrapper, inner) in producers.into_iter().zip(owned.into_iter()) {
            wrapper.inner = Some(inner);
        }

        result.map_err(to_py_err)
    }

    pub fn update<'py>(&'py mut self, py: Python<'py>) -> PyResult<(Py<PyBytes>, u128, u128)> {
        let inner = &self.inner;
        let (samples, start, end) = py
            .allow_threads(move || {
                let mut guard = inner.lock().unwrap();
                block_on(guard.update())
                    .map(|(buf, s, e)| (buf.to_vec(), s, e))
                    .map_err(|e| e.to_string())
            })
            .map_err(to_py_err)?;
        Ok((slice_to_pybytes(py, samples.as_slice()), start, end))
    }

    pub fn update_debug<'py>(
        &'py mut self,
        py: Python<'py>,
    ) -> PyResult<(Py<PyBytes>, Py<PyBytes>, Py<PyBytes>, u128, u128)> {
        let inner = &self.inner;
        let (aligned_in, aligned_out, aec_applied, start, end) = py
            .allow_threads(move || {
                let mut guard = inner.lock().unwrap();
                block_on(guard.update_debug())
                    .map(|(a, b, c, s, e)| (a.to_vec(), b.to_vec(), c.to_vec(), s, e))
                    .map_err(|e| e.to_string())
            })
            .map_err(to_py_err)?;

        Ok((
            slice_to_pybytes(py, aligned_in.as_slice()),
            slice_to_pybytes(py, aligned_out.as_slice()),
            slice_to_pybytes(py, aec_applied.as_slice()),
            start,
            end,
        ))
    }

    /// Set a Python callback receiving `(bytes, start_micros, end_micros)`.
    pub fn set_callback(&mut self, callback: Py<PyAny>) -> PyResult<()> {
        *self.callback.lock().unwrap() = Some(callback);
        self.maybe_start_callback_thread();
        Ok(())
    }
}

impl PyAecStream {
    fn stop_thread_if_running(&mut self) {
        self.callback_shutdown.store(true, Ordering::SeqCst);
        if let Some(handle) = self.callback_thread.take() {
            let _ = handle.join();
        }
    }

    fn maybe_start_callback_thread(&mut self) {
        if self.callback_thread.is_some() {
            return;
        }
        if self.callback.lock().unwrap().is_none() {
            return;
        }

        let inner = Arc::downgrade(&self.inner);
        let cb_ref = Arc::clone(&self.callback);
        let shutdown = Arc::clone(&self.callback_shutdown);

        self.callback_shutdown.store(false, Ordering::SeqCst);
        self.callback_thread = Some(thread::spawn(move || loop {
            if shutdown.load(Ordering::SeqCst) {
                break;
            }

            let (samples, start, end) = {
                let Some(inner_arc) = inner.upgrade() else {
                    break;
                };
                let mut guard = inner_arc.lock().unwrap();
                match block_on(guard.update())
                    .map(|(buf, s, e)| (buf.to_vec(), s, e))
                    .map_err(|e| e.to_string())
                {
                    Ok(res) => res,
                    Err(err) => {
                        eprintln!("Aec callback update error: {err}");
                        (Vec::new(), 0, 0)
                    }
                }
            };

            if samples.is_empty() {
                thread::sleep(Duration::from_millis(1));
                continue;
            }

            if let Some(callback) = cb_ref.lock().unwrap().as_ref() {
                Python::with_gil(|py| {
                    let py_bytes = slice_to_pybytes(py, samples.as_slice());
                    let _ = callback.call1(py, (py_bytes, start, end));
                });
            } else {
                thread::sleep(Duration::from_millis(5));
            }
        }));
    }
}

impl Drop for PyAecStream {
    fn drop(&mut self) {
        self.stop_thread_if_running();
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn get_supported_input_configs(
    py: Python<'_>,
    history_len: usize,
    num_calibration_packets: u32,
    audio_buffer_seconds: u32,
    resampler_quality: i32,
) -> PyResult<Vec<Vec<PyInputDeviceConfig>>> {
    let configs = py
        .allow_threads(|| {
            block_on(inner_get_supported_input_configs(
                history_len,
                num_calibration_packets,
                audio_buffer_seconds,
                resampler_quality,
            ))
            .map_err(|e| e.to_string())
        })
        .map_err(to_py_err)?;

    Ok(configs
        .into_iter()
        .map(|cfgs| cfgs.into_iter().map(PyInputDeviceConfig::from_inner).collect())
        .collect())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn get_supported_output_configs(
    py: Python<'_>,
    history_len: usize,
    num_calibration_packets: u32,
    audio_buffer_seconds: u32,
    resampler_quality: i32,
    frame_size: u32,
) -> PyResult<Vec<Vec<PyOutputDeviceConfig>>> {
    let configs = py
        .allow_threads(|| {
            block_on(inner_get_supported_output_configs(
                history_len,
                num_calibration_packets,
                audio_buffer_seconds,
                resampler_quality,
                frame_size,
            ))
            .map_err(|e| e.to_string())
        })
        .map_err(to_py_err)?;

    Ok(configs
        .into_iter()
        .map(|cfgs| cfgs.into_iter().map(PyOutputDeviceConfig::from_inner).collect())
        .collect())
}

#[pymodule]
fn melaec3(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyInputDeviceConfig>()?;
    m.add_class::<PyOutputDeviceConfig>()?;
    m.add_class::<PyAecConfig>()?;
    m.add_class::<PyStreamProducer>()?;
    m.add_class::<PyOutputStreamAlignerProducer>()?;
    m.add_class::<PyAecStream>()?;

    m.add_function(wrap_pyfunction!(get_supported_input_configs, m)?)?;
    m.add_function(wrap_pyfunction!(get_supported_output_configs, m)?)?;

    // Keep the module importable even if no classes are used.
    let _ = py;
    Ok(())
}
