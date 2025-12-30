use melaec3::EchoCanceller;

fn i16_from_f32(v: f32) -> i16 {
    v.clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

fn energy(buf: &[i16]) -> f64 {
    buf.iter().map(|s| (*s as f64) * (*s as f64)).sum::<f64>() / buf.len() as f64
}

#[test]
fn multichannel_cancels_basic_echo() {
    let frame_size = 160; // 10 ms at 16 kHz
    let filter_length = 800; // 50 ms tail
    let mic_channels = 2;
    let spk_channels = 2;
    let mut aec =
        EchoCanceller::new_multichannel(frame_size, filter_length, mic_channels, spk_channels)
            .expect("alloc");
    aec.set_sampling_rate(16_000);

    // Synthetic far-end: two tones with different delays per channel into the mic.
    let mut far_end_ring = vec![0f32; frame_size * spk_channels * 4];
    let mut ring_pos = 0usize;

    // Pre-fill ring so initial delay works.
    for _ in 0..far_end_ring.len() / spk_channels {
        let t = ring_pos as f32 / 16_000.0;
        let spk0 = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 12_000.0;
        let spk1 = (2.0 * std::f32::consts::PI * 660.0 * t).sin() * 10_000.0;
        let len = far_end_ring.len();
        let idx0 = ring_pos % len;
        let idx1 = (ring_pos + 1) % len;
        far_end_ring[idx0] = spk0;
        far_end_ring[idx1] = spk1;
        ring_pos = (ring_pos + spk_channels) % len;
    }

    let delay_ch0 = 32; // samples
    let delay_ch1 = 48;
    let total_frames = 300;
    let mut input_buf = vec![0i16; frame_size * mic_channels];
    let mut spk_buf = vec![0i16; frame_size * spk_channels];
    let mut out_buf = vec![0i16; frame_size * mic_channels];

    let mut post_energy = 0.0;
    let mut pre_energy = 0.0;
    for frame_idx in 0..total_frames {
        // Synthesize far-end and mic frames.
        for i in 0..frame_size {
            let t = ((frame_idx * frame_size + i) as f32) / 16_000.0;
            let spk0 = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 12_000.0;
            let spk1 = (2.0 * std::f32::consts::PI * 660.0 * t).sin() * 10_000.0;

            // Write far-end interleaved.
            let base = i * spk_channels;
            spk_buf[base] = i16_from_f32(spk0);
            spk_buf[base + 1] = i16_from_f32(spk1);

            // Push into ring buffer for delayed echo synthesis.
            let len = far_end_ring.len();
            let idx0 = ring_pos % len;
            let idx1 = (ring_pos + 1) % len;
            far_end_ring[idx0] = spk0;
            far_end_ring[idx1] = spk1;
            ring_pos = (ring_pos + spk_channels) % len;

            // Read delayed samples to form mic echoes.
            let read0 = (ring_pos + len - delay_ch0 * spk_channels) % len;
            let read1 = (ring_pos + len - delay_ch1 * spk_channels) % len;
            let echo0 = far_end_ring[read0] * 0.6;
            let echo1 = far_end_ring[(read1 + 1) % len] * 0.6;

            // Add a little noise to avoid perfect correlation.
            let noise = ((i * 17 + frame_idx * 23) % 31) as f32 - 15.0;
            let mic_base = i * mic_channels;
            input_buf[mic_base] = i16_from_f32(echo0 + noise);
            input_buf[mic_base + 1] = i16_from_f32(echo1 + noise * 0.5);
        }

        aec.cancel_frame(&input_buf, &spk_buf, &mut out_buf);

        // Skip early adaptation frames when measuring performance.
        if frame_idx > 150 {
            pre_energy += energy(&input_buf) as f64;
            post_energy += energy(&out_buf) as f64;
        }
    }

    // Expect at least ~6 dB reduction on this synthetic scenario.
    assert!(
        post_energy < pre_energy * 0.25,
        "post energy not reduced enough: pre={}, post={}",
        pre_energy,
        post_energy
    );
}
