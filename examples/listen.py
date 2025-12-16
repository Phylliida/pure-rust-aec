
import melaec3
import time

# Build configs (you can call get_supported_* to enumerate)
aec = melaec3.AecConfig(target_sample_rate=16000, frame_size=160, filter_length=1600)
stream = melaec3.AecStream(aec)

# Device configs must match your system; use get_supported_* helpers.
# Example (replace names with your devices):
inp = melaec3.InputDeviceConfig.from_default(None, "front:CARD=Beyond,DEV=0", 100, 20, 20, 5)
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
stream.set_callback(on_chunk)

time.sleep(100000)

# Cleanup
prod.end_audio_stream(out_stream)
stream.clear_callback()
stream.remove_input_device(inp)
stream.remove_output_device(out)