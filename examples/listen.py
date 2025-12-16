
import melaec3
import time

MAX_DISPLAY = 10  # how many configs per host/device to show before eliding
# Common tuning knobs for both input and output configs
HISTORY_LEN = 100            # number of audio chunks to hold for alignment (dropped frames/clock offsets)
CALIBRATION_PACKETS = 20     # packets to gather before emitting audio; improves timing alignment
AUDIO_BUFFER_SECONDS = 20    # seconds of audio to buffer (streamed; a few seconds is usually enough)
RESAMPLER_QUALITY = 5        # Speex resampler quality level
FRAME_SIZE_MILLIS = 3        # target output frame duration in milliseconds (small to reduce latency)
FRAME_SIZE_SAMPLES = 160     # target output frame size in samples (~10ms at 16 kHz)

# Build configs (you can call get_supported_* to enumerate)
aec = melaec3.AecConfig(target_sample_rate=16000, frame_size=160, filter_length=1600)
stream = melaec3.AecStream(aec)


# Dump available devices/configs to help pick names and defaults.
inputs = melaec3.get_supported_input_configs(
    history_len=HISTORY_LEN,
    num_calibration_packets=CALIBRATION_PACKETS,
    audio_buffer_seconds=AUDIO_BUFFER_SECONDS,
    resampler_quality=RESAMPLER_QUALITY,
)
print("Supported input configs:")
for device_configs in inputs:
    for idx, cfg in enumerate(device_configs):
        print(
            f"  - {cfg.host_name} / {cfg.device_name} "
            f"{cfg.channels}ch @ {cfg.sample_rate} Hz ({cfg.sample_format})"
        )
        if idx + 1 >= MAX_DISPLAY and len(device_configs) > MAX_DISPLAY:
            print(f"  ... (+{len(device_configs) - MAX_DISPLAY} more)")
            break

outputs = melaec3.get_supported_output_configs(
    history_len=HISTORY_LEN,
    num_calibration_packets=CALIBRATION_PACKETS,
    audio_buffer_seconds=AUDIO_BUFFER_SECONDS,
    resampler_quality=RESAMPLER_QUALITY,
    frame_size=FRAME_SIZE_SAMPLES,
)
print("Supported output configs:")
for device_configs in outputs:
    for idx, cfg in enumerate(device_configs):
        print(
            f"  - {cfg.host_name} / {cfg.device_name} "
            f"{cfg.channels}ch @ {cfg.sample_rate} Hz ({cfg.sample_format}) "
        )
        if idx + 1 >= MAX_DISPLAY and len(device_configs) > MAX_DISPLAY:
            print(f"  ... (+{len(device_configs) - MAX_DISPLAY} more)")
            break

# Device configs must match your system; use get_supported_* helpers.
# Example (replace names with your devices):
inp = melaec3.InputDeviceConfig.from_default(
    host=None,
    device_name="front:CARD=Beyond,DEV=0",
    history_len=HISTORY_LEN,
    calibration_packets=CALIBRATION_PACKETS,
    audio_buffer_seconds=AUDIO_BUFFER_SECONDS,
    resampler_quality=RESAMPLER_QUALITY,
)
out = melaec3.OutputDeviceConfig.from_default(
    host=None,
    device_name="default",
    history_len=HISTORY_LEN,
    calibration_packets=CALIBRATION_PACKETS,
    audio_buffer_seconds=AUDIO_BUFFER_SECONDS,
    resampler_quality=RESAMPLER_QUALITY,
    frame_size_millis=FRAME_SIZE_MILLIS,
)

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

time.sleep(1)

prod.interrupt_all_streams()

time.sleep(1000000)

# Cleanup
prod.end_audio_stream(out_stream)
stream.clear_callback()
stream.remove_input_device(inp)
stream.remove_output_device(out)
