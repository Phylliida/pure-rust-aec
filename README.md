# melaec3 Python module

Python bindings live in `src/pywrapper.rs` and are meant to be published as the `melaec3` module. This doc summarizes how to configure, build, and consume the extension.

## Prerequisites
- Rust toolchain (nightly not required).
- Python 3.8+ with `pip`.
- `maturin` for building wheels: `pip install maturin`.
- System audio backends that `cpal` can find (ALSA/Pulse/JACK/CoreAudio/Wasapi/WebAudio).

## Cargo setup
Add PyO3; the `cpal-example` feature is on by default to build the wrapper:

```toml
[dependencies]
pyo3 = { version = "0.21", features = ["extension-module"] }
# futures is already used by the wrapper; keep whatever version is in Cargo.lock.
# cpal is optional; the Py wrapper expects the "cpal-example" feature to be on.

[lib]
crate-type = ["cdylib", "rlib"]

[package.metadata.maturin]
name = "melaec3"
```

The wrapper is included by default; a normal release build works:

```
cargo build --release
```

## Build the Python extension

Editable/dev install:
```
cd pure-rust-aec
maturin develop --release
```

Wheel (for distribution):
```
maturin build --release
# wheel will be under target/wheels/
```

After `maturin develop`, you can import `melaec3` directly in your venv.

## Minimal usage example

```python
import melaec3

# Build configs (you can call get_supported_* to enumerate)
aec = melaec3.AecConfig(target_sample_rate=16000, frame_size=160, filter_length=1600)
stream = melaec3.AecStream(aec)

# Device configs must match your system; use get_supported_* helpers.
# Example (replace names with your devices):
inp = melaec3.InputDeviceConfig.from_default(None, "default", 100, 20, 20, 5)
out = melaec3.OutputDeviceConfig.from_default(None, "default", 100, 20, 20, 5, 3)

stream.add_input_device(inp)
prod = stream.add_output_device(out)

# Create an output stream and queue audio
ch_map = {0: [0]}  # map virtual channel 0 -> hardware channel 0
out_stream = prod.begin_audio_stream(1, ch_map, 5, 16000, 5)
out_stream.queue_audio([0.0] * 16000)  # send silence

# Optional: background callback receives (bytes, start_us, end_us)
def on_chunk(buf, start_us, end_us):
    # buf is raw f32 PCM bytes; len(buf)/4 samples interleaved
    pass
stream.set_callback(on_chunk)

# Manual polling is also possible:
pcm, start, end = stream.update()

# Cleanup
prod.end_audio_stream(out_stream)
stream.clear_callback()
stream.remove_input_device(inp)
stream.remove_output_device(out)
```

Notes:
- `get_supported_input_configs(...)` and `get_supported_output_configs(...)` return per-host device configs; pick one and feed into the add/remove APIs.
- `OutputStreamAlignerProducer.begin_audio_stream` returns a `StreamProducer` holding its stream id; pass that to `end_audio_stream` or call `StreamProducer.close(producer)`.
- The callback thread continuously calls `update()` under the hood; prefer it when you want streaming delivery without manual polling.
