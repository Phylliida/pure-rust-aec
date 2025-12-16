
import wave
from pathlib import Path
import asyncio

import numpy as np
import melaec3

MAX_DISPLAY = 10  # how many configs per host/device to show before eliding
# Common tuning knobs for both input and output configs
HISTORY_LEN = 100            # number of audio chunks to hold for alignment (dropped frames/clock offsets)
CALIBRATION_PACKETS = 20     # packets to gather before emitting audio; improves timing alignment
AUDIO_BUFFER_SECONDS = 20    # seconds of audio to buffer (streamed; a few seconds is usually enough)
RESAMPLER_QUALITY = 5        # Speex resampler quality level
FRAME_SIZE_MILLIS = 3        # target output frame duration in milliseconds (small to reduce latency)
FRAME_SIZE_SAMPLES = 160     # target output frame size in samples (~10ms at 16 kHz)
FILTER_LENGTH = FRAME_SIZE_SAMPLES * 10
AEC_SAMPLE_RATE = 16000

WAV_PATH = Path(__file__).parent / "example_talking.wav"


def load_wav_f32(path: Path):
    """Load a 16-bit PCM WAV and return (samples_interleaved_f32, sample_rate, channels)."""
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        if sampwidth != 2:
            raise ValueError(f"Only 16-bit PCM supported; got {8 * sampwidth}-bit")
        frames = wf.getnframes()
        raw = wf.readframes(frames)
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return pcm, sample_rate, channels


async def main() -> None:
    # Build configs (you can call get_supported_* to enumerate)
    aec = melaec3.AecConfig(
        target_sample_rate=AEC_SAMPLE_RATE,
        frame_size=FRAME_SIZE_SAMPLES,
        filter_length=FILTER_LENGTH,
    )
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
    inp = await melaec3.InputDeviceConfig.from_default(
        host=None,
        device_name="front:CARD=Beyond,DEV=0",
        history_len=HISTORY_LEN,
        calibration_packets=CALIBRATION_PACKETS,
        audio_buffer_seconds=AUDIO_BUFFER_SECONDS,
        resampler_quality=RESAMPLER_QUALITY,
    )
    out = await melaec3.OutputDeviceConfig.from_default(
        host=None,
        device_name="default",
        history_len=HISTORY_LEN,
        calibration_packets=CALIBRATION_PACKETS,
        audio_buffer_seconds=AUDIO_BUFFER_SECONDS,
        resampler_quality=RESAMPLER_QUALITY,
        frame_size_millis=FRAME_SIZE_MILLIS,
    )

    await stream.add_input_device(inp)
    prod = await stream.add_output_device(out)

    # Load audio and play
    audio, file_sample_rate, file_channels = load_wav_f32(WAV_PATH)
    channel_map = {ch: [ch] for ch in range(file_channels)}  # map virtual channels -> hardware channels 1:1
    out_stream = prod.begin_audio_stream(
        file_channels,
        channel_map,
        int((len(audio) * 2) / file_sample_rate),
        file_sample_rate,
        RESAMPLER_QUALITY,
    )

    out_stream.queue_audio(audio.tolist())  # queue WAV audio

    # Optional: background callback receives (bytes, start_us, end_us)
    def on_chunk(buf, start_us, end_us):
        # buf is raw f32 PCM bytes; len(buf)/4 samples interleaved
        pass

    stream.set_callback(on_chunk)

    await asyncio.sleep(4)
    prod.interrupt_all_streams()
    print("Interrupted")
    # stream not usable once interrupted, by design
    await asyncio.sleep(4)
    prod.end_audio_stream(out_stream)
    out_stream = prod.begin_audio_stream(
        file_channels,
        channel_map,
        int((len(audio) * 2) / file_sample_rate),
        file_sample_rate,
        RESAMPLER_QUALITY,
    )
    out_stream.queue_audio(audio.tolist())  # queue WAV audio
    await asyncio.sleep(4)
    prod.interrupt_all_streams()
    await asyncio.sleep(4)
    prod.end_audio_stream(out_stream)
    out_stream = prod.begin_audio_stream(
        file_channels,
        channel_map,
        int((len(audio) * 2) / file_sample_rate),
        file_sample_rate,
        RESAMPLER_QUALITY,
    )
    out_stream.queue_audio(audio.tolist())  # queue WAV audio

    await asyncio.sleep(10)  # hold stream open briefly

    # Cleanup
    prod.end_audio_stream(out_stream)
    stream.clear_callback()
    await stream.remove_input_device(inp)
    await stream.remove_output_device(out)


if __name__ == "__main__":
    asyncio.run(main())
