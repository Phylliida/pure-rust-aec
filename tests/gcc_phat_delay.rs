use melaec3::cpal_aec::{gcc_phat_delay, indexed_chirp};

const SAMPLE_RATE: usize = 16_000;
const TONE_SECONDS: f32 = 0.1;
const SLICE_SECONDS: f32 = 4.0;
const OFFSET_COUNT: usize = 32;

fn build_slice(silence_before_frames: usize, silence_after_frames: usize, slice_samples: usize) -> (Vec<f32>, usize) {
    let mut slice = vec![0.0f32; slice_samples];
    let tone = indexed_chirp(0, SAMPLE_RATE as u32, 0.0, TONE_SECONDS);
    let start = silence_before_frames.min(slice_samples.saturating_sub(1));
    let avail = slice_samples.saturating_sub(start);
    let max_len = tone.len().min(avail);
    slice[start..start + max_len].copy_from_slice(&tone[..max_len]);
    let _ = silence_after_frames; // kept for clarity; layout is controlled by caller
    (slice, start)
}

fn add_deterministic_noise(buf: &mut [f32], amplitude: f32) {
    let mut state: u32 = 0x1234_5678;
    for sample in buf {
        // simple LCG for reproducibility
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let val = ((state >> 9) as f32 / (1u32 << 23) as f32) * 2.0 - 1.0; // [-1, 1]
        *sample += val * amplitude;
    }
}

fn offset_sweep_frames() -> Vec<isize> {
    // 32 evenly spaced offsets from -1.5s to +1.5s inclusive, in frames.
    (0..OFFSET_COUNT)
        .map(|i| {
            let seconds = -1.5 + (3.0 * i as f32 / (OFFSET_COUNT - 1) as f32);
            (seconds * SAMPLE_RATE as f32).round() as isize
        })
        .collect()
}

fn silence_after(start: usize, slice_samples: usize, tone_samples: usize) -> usize {
    slice_samples.saturating_sub(start + tone_samples)
}

#[test]
fn gcc_phat_delay_tracks_offsets_across_range() {
    let slice_samples = (SLICE_SECONDS * SAMPLE_RATE as f32) as usize;
    let tone_samples = (TONE_SECONDS * SAMPLE_RATE as f32) as usize;
    let base_start = slice_samples / 2 - tone_samples / 2;
    let max_probe_start = (slice_samples - tone_samples) as isize;

    let offsets = offset_sweep_frames();
    let base_silence_after = silence_after(base_start, slice_samples, tone_samples);
    let (input, base_start_idx) = build_slice(base_start, base_silence_after, slice_samples);

    for requested_offset_samples in offsets {
        let requested_offset_s = requested_offset_samples as f32 / SAMPLE_RATE as f32;
        let desired_probe_start =
            base_start as isize + requested_offset_samples;
        let probe_start = desired_probe_start.clamp(0, max_probe_start) as usize;
        let probe_silence_after = silence_after(probe_start, slice_samples, tone_samples);
        let (probe, probe_start_idx) = build_slice(probe_start, probe_silence_after, slice_samples);

        let actual_offset_samples = probe_start_idx as isize - base_start_idx as isize;
        let (detected_offset_samples, score) = gcc_phat_delay(&input, &probe);
        let diff = detected_offset_samples - actual_offset_samples;

        println!(
            "diff {:+} | requested {:+.3}s ({:+} samples) actual {} samples detected {} samples (score {:.2})",
            diff,
            requested_offset_s,
            requested_offset_samples,
            actual_offset_samples,
            detected_offset_samples,
            score
        );

        let tolerance_samples = 2; // within two samples
        assert!(
            (detected_offset_samples - actual_offset_samples).abs() <= tolerance_samples,
            "offset mismatch: diff {:+} | requested {:+.3}s ({:+} samples) actual {} samples detected {} samples",
            diff,
            requested_offset_s,
            requested_offset_samples,
            actual_offset_samples,
            detected_offset_samples
        );
    }
}

#[test]
fn gcc_phat_delay_with_noise_still_tracks_offsets() {
    let slice_samples = (SLICE_SECONDS * SAMPLE_RATE as f32) as usize;
    let tone_samples = (TONE_SECONDS * SAMPLE_RATE as f32) as usize;
    let base_start = slice_samples / 2 - tone_samples / 2;
    let max_probe_start = (slice_samples - tone_samples) as isize;

    let offsets = offset_sweep_frames();
    let base_silence_after = silence_after(base_start, slice_samples, tone_samples);
    let (input, base_start_idx) = build_slice(base_start, base_silence_after, slice_samples);

    for requested_offset_samples in offsets {
        let requested_offset_s = requested_offset_samples as f32 / SAMPLE_RATE as f32;
        let desired_probe_start =
            base_start as isize + requested_offset_samples;
        let probe_start = desired_probe_start.clamp(0, max_probe_start) as usize;
        let probe_silence_after = silence_after(probe_start, slice_samples, tone_samples);
        let (mut probe, probe_start_idx) = build_slice(probe_start, probe_silence_after, slice_samples);

        // add noise at roughly the same scale as the tone (tone peak is ~0.3)
        add_deterministic_noise(&mut probe, 0.3);

        let actual_offset_samples = probe_start_idx as isize - base_start_idx as isize;
        let (detected_offset_samples, score) = gcc_phat_delay(&input, &probe);
        let diff = detected_offset_samples - actual_offset_samples;

        println!(
            "[noise] diff {:+} | requested {:+.3}s ({:+} samples) actual {} samples detected {} samples (score {:.2})",
            diff,
            requested_offset_s,
            requested_offset_samples,
            actual_offset_samples,
            detected_offset_samples,
            score
        );

        let tolerance_samples = 8; // looser with noise at tone scale
        assert!(
            (detected_offset_samples - actual_offset_samples).abs() <= tolerance_samples,
            "offset mismatch w/ noise: diff {:+} | requested {:+.3}s ({:+} samples) actual {} samples detected {} samples",
            diff,
            requested_offset_s,
            requested_offset_samples,
            actual_offset_samples,
            detected_offset_samples
        );
    }
}
