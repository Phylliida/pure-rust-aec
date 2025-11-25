//! Example demonstrating how to wire `speex-rust-aec` into a CPAL input/output
//! pipeline.
//!
//! Requires building with `--features cpal-example` so the optional `cpal`
//! dependency is enabled:
//!
//! ```text
//! cargo run --example cpal_aec --features cpal-example
//! ```
//!
//! You can optionally pass the desired input and output device names (substring
//! match) as the first and second command-line arguments:
//!
//! ```text
//! cargo run --example cpal_aec --features cpal-example "USB Microphone" "Loopback"
//! ```
//!
//! ```text
//! cargo run --example cpal_aec --features cpal-example "USB Microphone" "Loopback" 48000
//! ```
//!
//! The optional third argument selects the internal echo-canceller sample rate (Hz). When
//! omitted, the demo defaults to 48 kHz and transparently resamples the devices if needed.

use std::{
    thread,
    collections::{HashMap},
    error::Error,
    mem::{MaybeUninit},
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{self, TryRecvError, RecvError},
        Arc,
    },
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Device, FromSample, Host, InputCallbackInfo, Sample, SampleFormat, SampleRate, SizedSample,
    Stream, SupportedStreamConfig,
};
use ringbuf::{
    traits::{Consumer, Producer, RingBuffer, Split},
    HeapCons, HeapProd, HeapRb, LocalRb,
};
use ringbuf::storage::Heap;
use speex_rust_aec::{
    speex_echo_cancellation, EchoCanceller, Resampler,
};

use std::time::{UNIX_EPOCH, SystemTime};

#[inline]
unsafe fn assume_init_slice_mut<T>(slice: &mut [MaybeUninit<T>]) -> &mut [T] {
    &mut *(slice as *mut [MaybeUninit<T>] as *mut [T])
}

fn input_to_output_frames(input_frames: u128, in_rate: u32, out_rate: u32) -> u128 {
    // u128 to avoid overflow
    (input_frames * (out_rate as u128)) / (in_rate as u128)
}

fn micros_to_frames(microseconds: u128, sample_rate: u128) -> u128 {
    // There are sample_rate samples per second
    // there are sample_rate / 1_000_000 samples per microsecond
    // now that we have samples_per_microsecond, we simply multiply by microseconds to get total samples
    // rearranging:
    microseconds * sample_rate / 1000000
}

fn frames_to_micros(frames: u128, sample_rate: u128) -> u128 {
    // frames = (microseconds * sample_rate / 1 000 000)
    // frames * 1_000_000 = microseconds * sample_rate
    frames * 1000000 / sample_rate // = microseconds
}

/// Producer-side sibling to `BufferedCircularProducer`.
/// Provides chunked, mostly zero-copy write access to a `HeapProd`.
/// Call `chunk_mut()` to obtain a contiguous region and `commit()` afterwards
/// to advance the underlying write index (or copy scratch data in).
struct BufferedCircularProducer<T: Copy> {
    producer: HeapProd<T>,
    scratch: Vec<T>
}

impl<T: Copy> BufferedCircularProducer<T> {
    fn new(producer: HeapProd<T>) -> Self {
        Self {
            producer,
            scratch: Vec::new()
        }
    }

    fn finish_write(&mut self, need_to_write_outputs: bool, num_written: usize) {
        if need_to_write_outputs {
            // wrote to scratch, need to add it to producer
            let _appended = self.producer.push_slice(&self.scratch[..num_written]);
            if _appended < num_written {
                eprintln!("Warning: Producer cannot keep up, increase buffer size or decrease latency")
            }
        } else {
            // wrote directly to producer, simply advance write index
            unsafe { self.producer.advance_write_index(num_written) };
        }
    }
}

impl<T: Copy + Default> BufferedCircularProducer<T> {
    fn get_chunk_to_write(&mut self, size: usize) -> (bool, &mut [T]) {
        let (first, second) = self.producer.vacant_slices_mut();
        // we can simply 
        if first.len() >= size {
            let buf = unsafe { assume_init_slice_mut(first) };
            (false, &mut buf[..size])
        } else if first.is_empty() && second.len() >= size {
            let buf = unsafe { assume_init_slice_mut(second) };
            (false, &mut buf[..size])
        }
        else {
            if self.scratch.len() < size {
                self.scratch.resize_with(size, Default::default);
            }
            (true, &mut self.scratch[..size])
        }
    }
}

/// Helper that makes the consumer half of a ring buffer feel like a stream of contiguous slices.
/// It tries to return zero-copy slices when the occupied region is already contiguous,
/// and otherwise falls back to copying into a scratch buffer.
/// `StreamAligner` can hold one of these alongside its producer half and call `chunk()` /
/// `consume()` whenever it needs to feed the SpeexDSP resampler.
struct BufferedCircularConsumer<T: Copy> {
    consumer: HeapCons<T>,
    scratch: Vec<T>,
}

impl<T: Copy> BufferedCircularConsumer<T> {
    fn new(consumer: HeapCons<T>) -> Self {
        Self {
            consumer,
            scratch: Vec::new(),
        }
    }

    fn finish_read(&mut self, num_read: usize) -> usize {
        self.consumer.skip(num_read)
    }

    fn available(&self) -> usize {
        let (head, tail) = self.consumer.as_slices();
        let head_len = head.len();
        let tail_len = tail.len();
        head_len + tail_len
    }
}

impl<T: Copy> BufferedCircularConsumer<T> {
    fn get_chunk_to_read(&mut self, size: usize) -> &[T] {
        if size == 0 {
            return &[];
        }

        let (head, tail) = self.consumer.as_slices();
        let head_len = head.len();
        let tail_len = tail.len();
        let available = head_len + tail_len;

        if available == 0 {
            return &[];
        }

        let take = size.min(available);
        // all fits in head, just return slice of that
        if head_len >= take {
            &head[..take]
        // head is empty so all fits in tail, return that
        } else if head_len == 0 {
            &tail[..take]
        // we need intermediate buffer to join head and tail, use scratch
        } else {
            if self.scratch.capacity() < take {
                self.scratch.reserve(take - self.scratch.capacity());
            }
            self.scratch.clear(); // this empties it but does not remove allocations

            let from_head = head_len.min(take);
            if from_head > 0 {
                self.scratch.extend_from_slice(&head[..from_head]);
            }
            let remaining = take - from_head;
            if remaining > 0 {
                self.scratch.extend_from_slice(&tail[..remaining]);
            }

            &self.scratch[..take]
        }
    }
}


// a wrapper around BufferedCircularConsumer that resamples the stream before outputting
// you must call .resample(...)? before calling get_chunk_to_read() or there will be nothing available
// a safe choice is .resample(consumer.available_to_resample())
struct ResampledBufferedCircularProducer {
    consumer: BufferedCircularConsumer<f32>,
    resampled_producer: BufferedCircularProducer<f32>,
    input_sample_rate: u32,
    output_sample_rate: u32,
    total_input_samples_remaining: u128,
    resampler: Resampler
}

impl ResampledBufferedCircularProducer {
    fn new(
        input_sample_rate: u32,
        output_sample_rate : u32,
        resampler_quality: i32,
        consumer: BufferedCircularConsumer<f32>,
        resampled_producer: BufferedCircularProducer<f32>) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            consumer: consumer,
            resampled_producer: resampled_producer,
            total_input_samples_remaining: 0,
            input_sample_rate: input_sample_rate,
            output_sample_rate: output_sample_rate,
            resampler: Resampler::new(
                1, // channels, we have one of these StreamAligner each channel
                input_sample_rate,
                output_sample_rate,
                resampler_quality
            )?
        })
    }

    fn set_sample_rate(&mut self, input_sample_rate: u32, output_sample_rate: u32) -> Result<(), Box<dyn std::error::Error>> {
        self.resampler.set_rate(input_sample_rate, output_sample_rate)?;
        self.input_sample_rate = input_sample_rate;
        self.output_sample_rate = output_sample_rate;
        Ok(())
    }

    fn available_to_resample(&self) -> usize {
        self.consumer.available()
    }
}

impl ResampledBufferedCircularProducer {
    // resample all data available
    fn resample_all(&mut self) -> Result<(usize, usize), Box<dyn std::error::Error>>  {
        self.resample(self.available_to_resample() as u32)
    }

    fn resample(&mut self, num_available_frames: u32) -> Result<(usize, usize), Box<dyn std::error::Error>> {
        if num_available_frames == 0 {
            return Ok((0,0));
        }
        // there might be some leftover from last call, so use state
        self.total_input_samples_remaining += num_available_frames as u128;

        let input_buf = self.consumer.get_chunk_to_read(self.total_input_samples_remaining as usize);
        let target_output_samples_count = input_to_output_frames(self.total_input_samples_remaining, self.input_sample_rate, self.output_sample_rate) + 10; // add a few extra for rounding
        let (need_to_write_outputs, output_buf) = self.resampled_producer.get_chunk_to_write(target_output_samples_count as usize);
        let (consumed, produced) = self.resampler.process_interleaved_f32(input_buf, output_buf)?;
        // it may return less consumed and produced than the sizes of stuff we gave it
        // so use actual processed sizes here instead of our lengths from above
        // (worst case this is like 0.6 ms or so, so it's okay to have them slightly delayed like this)
        self.consumer.finish_read(consumed);
        self.resampled_producer.finish_write(need_to_write_outputs, produced);

        self.total_input_samples_remaining -= consumed as u128;
        Ok((consumed, produced))
    }
}

enum AudioBufferMetadata {
    Arrive(u64, u128, u128, bool),
    Teardown(),
}


struct InputStreamAlignerProducer {
    input_sample_rate: u32,
    output_sample_rate: u32,
    input_audio_buffer_producer: HeapProd<f32>,
    input_audio_buffer_metadata_producer: mpsc::Sender<AudioBufferMetadata>,
    chunk_sizes: LocalRb<Heap<usize>>,
    system_time_micros_when_chunk_ended: LocalRb<Heap<u128>>,
    num_calibration_packets: u32,
    num_packets_recieved: u64,
    num_emitted_frames: u128,
    start_time_micros: Option<u128>,
}

impl InputStreamAlignerProducer {
    fn new(input_sample_rate: u32, output_sample_rate: u32, history_len: usize, num_calibration_packets: u32, input_audio_buffer_producer: HeapProd<f32>, input_audio_buffer_metadata_producer: mpsc::Sender<AudioBufferMetadata>) -> Result<Self, Box<dyn Error>>  {
        Ok(Self {
            input_sample_rate: input_sample_rate,
            output_sample_rate: output_sample_rate,
            input_audio_buffer_producer: input_audio_buffer_producer,
            input_audio_buffer_metadata_producer: input_audio_buffer_metadata_producer,
            // alignment data, these are used to adjust resample rate so output stays aligned with true timings (according to sytem clock)
            chunk_sizes: LocalRb::<Heap<usize>>::new(history_len),
            system_time_micros_when_chunk_ended: LocalRb::<Heap<u128>>::new(history_len),
            num_calibration_packets: num_calibration_packets,
            num_packets_recieved: 0,
            num_emitted_frames: 0,
            start_time_micros: None,
        })
    }

    fn estimate_micros_when_most_recent_ended(&self) -> u128 {
        // Take minimum over estimates for all previous recieved
        // Some may be delayed due to cpu being busy, but none can ever arrive too early
        // so this should be a decent estimate
        // it does not account for hardware latency, but we cannot account for that without manual calibration
        // (btw, CPAL timestamps do not work because they may be different for different devices)
        // (wheras this synchronizes us to global system time)
        let mut best_estimate_of_when_most_recent_ended = if let Some(most_recent_time) = self.system_time_micros_when_chunk_ended.last() {
            *most_recent_time
        }
        else {
            u128::MAX
        };
        let mut frames_until_most_recent = 0 as u128;


        // iterate from most recent backwards (that's what .rev() does)
        let mut chunk_iter = self.chunk_sizes.iter().rev();
        let mut time_iter = self.system_time_micros_when_chunk_ended.iter().rev();
        while let (Some(chunk_size), Some(micros_when_chunk_ended)) = (chunk_iter.next(), time_iter.next()) {
            let micros_until_most_recent_ended = frames_to_micros(frames_until_most_recent as u128, self.input_sample_rate as u128);
            let estimate_of_micros_most_recent_ended = *micros_when_chunk_ended + micros_until_most_recent_ended;
            best_estimate_of_when_most_recent_ended = (estimate_of_micros_most_recent_ended).min(best_estimate_of_when_most_recent_ended);
            // timestamps are at end, not at start, so only increment this after
            frames_until_most_recent += *chunk_size as u128;
        }
        best_estimate_of_when_most_recent_ended
    }

    fn process_chunk(&mut self, chunk: &[f32]) -> Result<(), Box<dyn Error>> {
        let micros_when_chunk_received = SystemTime::now().duration_since(UNIX_EPOCH).expect("clock went backwards").as_micros();

       
        let appended_count = self.input_audio_buffer_producer.push_slice(chunk);
        if appended_count < chunk.len() { // todo: auto resize
            eprintln!("Error: cannot keep up with audio, buffer is full, try increasing audio_buffer_seconds")
        }
        if appended_count > 0 {
            // delibrately overwrite once we pass history len, we keep a rolling buffer of last 100 or so
            self.chunk_sizes.push_overwrite(appended_count);
            self.system_time_micros_when_chunk_ended.push_overwrite(micros_when_chunk_received);

            // use our estimate to suggest how many frames we should have emitted
            // this is used to dynamically adjust sample rate until we actually emit that many frames
            // that ensures that we stay synchronized to the system clock and do not drift
            let micros_when_chunk_ended = self.estimate_micros_when_most_recent_ended();

            self.num_emitted_frames += appended_count as u128;

            let (target_emitted_frames, calibrated) = if self.num_packets_recieved < self.num_calibration_packets as u64 {
                // until we've recieved enough calibration packets, we don't have good enough time estimate
                // thus, simply request num_emitted_frames emitted
                // this avoids large amounts of distortion if we get an initial burst of packets on device init
                (self.num_emitted_frames, false)
            } else {
                // calibration finished, setup start_time_micros
                if self.num_packets_recieved == self.num_calibration_packets as u64 {
                    // now we can actually make a good estimate of our current time,
                    // which allows us to make a good estimate of start time (just convert number packets emitted into an offset)
                    // this isn't ideal when calibration involved a dropped packet
                    // but is about as good as we can do
                    self.start_time_micros = Some(micros_when_chunk_ended - frames_to_micros(self.num_emitted_frames, self.input_sample_rate as u128));
                }

                if let Some(start_time_micros_value) = self.start_time_micros {
                    // look at actual elapsed time, and use that to say how many frames we would have preferred to emitted
                    // this can be used later to adjust sample rate slightly to keep us in line with system time
                    let elapsed_micros = micros_when_chunk_ended - start_time_micros_value;
                    (micros_to_frames(elapsed_micros, self.input_sample_rate as u128), true)
                }
                else {
                    (self.num_emitted_frames, false)
                }
            };

            // increment afterwards incase num_calibration_packets = 0
            self.num_packets_recieved += 1;

            let metadata = AudioBufferMetadata::Arrive(
                // num available frames
                appended_count as u64,
                // estimated timestamp after this sample
                micros_when_chunk_ended,
                // target emitted frames
                target_emitted_frames,
                // calibrated
                calibrated
            );
            self.input_audio_buffer_metadata_producer.send(metadata)?;
        }
        Ok(())
    }

}

#[derive(Clone)]
enum ResamplingMetadata {
    Arrive(usize, u128, u128, bool),
}

struct InputStreamAlignerResampler {
    input_sample_rate: u32,
    output_sample_rate: u32,
    dynamic_output_sample_rate: u32,
    input_audio_buffer_consumer: ResampledBufferedCircularProducer,
    input_audio_buffer_metadata_consumer: mpsc::Receiver<AudioBufferMetadata>,
    total_emitted_frames: u128,
    total_received_frames: u128,
    total_processed_input_frames: u128,
    finished_resampling_producer: mpsc::Sender<ResamplingMetadata>
}

impl InputStreamAlignerResampler {
    // Takes input audio and resamples it to the target rate
    // May slightly stretch or squeeze the audio (via resampling)
    // to ensure the outputs stay aligned with system clock
    fn new(
        input_sample_rate: u32,
        output_sample_rate: u32,
        resampler_quality: i32,
        input_audio_buffer_consumer: HeapCons<f32>,
        input_audio_buffer_metadata_consumer: mpsc::Receiver<AudioBufferMetadata>,
        output_audio_buffer_producer: HeapProd<f32>,
        finished_resampling_producer: mpsc::Sender<ResamplingMetadata>
    ) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            input_sample_rate: input_sample_rate,
            output_sample_rate: output_sample_rate,
            dynamic_output_sample_rate: output_sample_rate,
            // we need buffered because this interfaces with speex which expects continuous buffers
            input_audio_buffer_consumer: ResampledBufferedCircularProducer::new(
                input_sample_rate,
                output_sample_rate,
                resampler_quality,
                BufferedCircularConsumer::<f32>::new(input_audio_buffer_consumer),
                BufferedCircularProducer::<f32>::new(output_audio_buffer_producer)
            )?,
            input_audio_buffer_metadata_consumer: input_audio_buffer_metadata_consumer,
            // alignment data, these are used to adjust resample rate so output stays aligned with true timings (according to sytem clock)
            total_emitted_frames: 0,
            total_received_frames: 0,
            total_processed_input_frames: 0,
            finished_resampling_producer: finished_resampling_producer,
        })
    }

    // do it very slowly
    fn decrease_dynamic_sample_rate(&mut self)  -> Result<(), Box<dyn std::error::Error>>  {
        self.dynamic_output_sample_rate = (((self.output_sample_rate as f32) * 0.95) as i128).max((self.dynamic_output_sample_rate-1) as i128) as u32;
        self.input_audio_buffer_consumer.set_sample_rate(self.input_sample_rate, self.dynamic_output_sample_rate)?;
        Ok(())
    }

    fn increase_dynamic_sample_rate(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.dynamic_output_sample_rate = (((self.output_sample_rate as f32) * 1.05) as i128).min((self.dynamic_output_sample_rate+1) as i128) as u32;
        self.input_audio_buffer_consumer.set_sample_rate(self.input_sample_rate, self.dynamic_output_sample_rate)?;
        Ok(())
    }

    fn handle_metadata(&mut self, num_available_frames : u64, target_emitted_input_frames : u128) -> Result<(usize, usize), Box<dyn std::error::Error>> {
        let estimated_emitted_frames = input_to_output_frames(num_available_frames as u128, self.input_sample_rate, self.dynamic_output_sample_rate);
        let updated_total_frames_emitted = self.total_emitted_frames + estimated_emitted_frames;
        let target_emitted_output_frames = input_to_output_frames(target_emitted_input_frames, self.input_sample_rate, self.output_sample_rate);
        // dynamic adjustment to synchronize input devices to global clock:
        // not enough frames, we need to increase dynamic sample rate (to get more samples)
        if updated_total_frames_emitted < target_emitted_output_frames {
            self.increase_dynamic_sample_rate()?;
        }
        // too many frames, we need to decrease dynamic sample rate (to get less samples)
        else if updated_total_frames_emitted > target_emitted_output_frames {
            self.decrease_dynamic_sample_rate()?;
        }

        //// do resampling ////
        let (consumed, produced) = self.input_audio_buffer_consumer.resample(num_available_frames as u32)?;

        // the main downside of this is that it'll be persistently behind by 0.6ms or so (the resample frame size), but we'll quickly adjust for that so this shouldn't be a major issue
        // todo: think about how to fix this better (maybe current solution is as good as we can do, and it should average out to correct since past ones accumulated will result in more for this one, still, it's likely to stay behind by this amount)
        self.total_emitted_frames += produced as u128;

        Ok((consumed, produced))
    }

    fn resample(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        // process all recieved audio chunks
        match self.input_audio_buffer_metadata_consumer.recv() {
            Ok(msg) => match msg {
                AudioBufferMetadata::Arrive(num_available_frames, system_micros_after_packet_finishes, target_emitted_frames, calibrated) => {
                    let num_leftovers_from_prev = self.total_received_frames - self.total_processed_input_frames;
                    self.total_received_frames += num_available_frames as u128;
                    // if it hypothetically consumed all frames every time,
                    // then we would know that current time in system_micros_after_packet_finishes
                    // however, there are a few things that happen:
                    // 1. There are some "leftover" samples from previously
                    // 2. There are some "ignored" samples from cur (that are later leftover)
                    // this logic sets the timestamp to be correct relative to last resampled emitted
                    // which is slightly distinct from system_micros_after_packet_finishes
                    // because resampling may operate at some latency
                    let (consumed, produced) = self.handle_metadata(num_available_frames, target_emitted_frames)?;
                    self.total_processed_input_frames += consumed as u128;
                    let micros_earlier = if consumed as u128 > num_leftovers_from_prev {
                        let num_of_ours_consumed = (consumed as i128) - (num_leftovers_from_prev as i128);
                        let num_of_ours_leftover = (num_available_frames as i128) - (num_of_ours_consumed as i128);
                        frames_to_micros(num_of_ours_leftover as u128, self.input_sample_rate as u128) as i128
                    } else {
                        // none of ours was consumed, skip back even further
                        let additional_frames_back = (num_leftovers_from_prev as u128) - (consumed as u128);
                        frames_to_micros(num_available_frames as u128 + additional_frames_back, self.input_sample_rate as u128) as i128
                    };
                    // will always be positive because it's relative to 1970
                    let system_micros_after_resampled_packet_finishes = (system_micros_after_packet_finishes as i128) - micros_earlier;
                    let system_micros_at_start_of_packet = (system_micros_after_resampled_packet_finishes as u128) - micros_to_frames(consumed as u128, self.input_sample_rate as u128);
                    self.finished_resampling_producer.send(ResamplingMetadata::Arrive(produced, system_micros_at_start_of_packet, system_micros_after_resampled_packet_finishes as u128, calibrated))?;
                    Ok(true)
                },
                AudioBufferMetadata::Teardown() => {
                    Ok(false)
                }
            },
            Err(RecvError) => Err("input metadata channel disconnected".into())   // sender dropped; bail out or log
        }
    }
}

struct InputStreamAlignerConsumer {
    sample_rate: u32,
    final_audio_buffer_consumer: BufferedCircularConsumer<f32>,
    thread_message_sender: mpsc::Sender<AudioBufferMetadata>,
    finished_message_reciever: mpsc::Receiver<ResamplingMetadata>,
    initial_metadata: Vec<ResamplingMetadata>,
    samples_recieved: u128,
    calibrated: bool,
}

impl InputStreamAlignerConsumer {
    fn new(sample_rate: u32, final_audio_buffer_consumer: BufferedCircularConsumer<f32>, thread_message_sender: mpsc::Sender<AudioBufferMetadata>, finished_message_reciever: mpsc::Receiver<ResamplingMetadata>) -> Self {
        Self {
            sample_rate: sample_rate,
            final_audio_buffer_consumer: final_audio_buffer_consumer,
            thread_message_sender: thread_message_sender,
            finished_message_reciever: finished_message_reciever,
            initial_metadata: Vec::new(),
            samples_recieved: 0,
            calibrated: false,
        }
    }

    // used to poll for when an input stream is actually ready to output data
    // we allow some initial calibration time to synchronize the clocks
    // (it needs some extra time because packets can be delayed sometimes 
    // so waiting and min over a history lets us get better estimate)
    fn is_ready_to_read(&mut self, micros_packet_finished: u128, size: usize) -> bool {
        // non blocking cause maybe it's just not ready (initialized) yet
        loop {
            match self.finished_message_reciever.try_recv() {
                Ok(msg) => {
                    match msg {
                        ResamplingMetadata::Arrive(frames_recieved, _system_micros_at_start_of_packet, _system_micros_after_packet_finishes, calibrated) => {
                            self.calibrated = calibrated;
                            self.initial_metadata.push(msg.clone());
                            self.samples_recieved += frames_recieved as u128;
                        }
                    }
                }
                Err(TryRecvError::Empty) => {
                    // no message available right now
                    break;
                }
                Err(TryRecvError::Disconnected) => {
                    // sender dropped; receiver will never get more messages
                    break;
                }
            }
        }

        // we need to skip ahead to be frame aligned
        if self.calibrated {
            let num_samples_that_are_behind_current_packet = self.num_samples_that_are_behind_current_packet(micros_packet_finished, size);
            let available_samples = self.samples_recieved as i128 - (num_samples_that_are_behind_current_packet as i128);
            
            if available_samples < size as i128 {
                // we will be able to get all samples for this packet, block until we get them
                if num_samples_that_are_behind_current_packet > 0 {
                    // skip ahead so we are only getting samples for this packet
                    self.final_audio_buffer_consumer.finish_read(num_samples_that_are_behind_current_packet as usize);
                    let additional_samples_needed = (size as i128) - available_samples;
                    let (_read_success, _samples) = self.get_chunk_to_read(additional_samples_needed as usize);
                    true // we will read them again later, at which point we will do finish_read (this is delibrate reading them twice)
                }
                // we started in the middle of this packet, we can't get enough, wait until next packet
                else {
                    false
                }
            }
            else {
                // enough samples! ignore the ones we need to ignore and then let the sampling happen elsewhere
                self.final_audio_buffer_consumer.finish_read(num_samples_that_are_behind_current_packet as usize);
                true
            }
        } else {
            false
        }
    }

    fn num_samples_that_are_behind_current_packet(&self, micros_packet_finished: u128, size: usize) -> u128 {
        let micros_packet_started = micros_packet_finished - frames_to_micros(size as u128, self.sample_rate as u128);
        let mut samples_to_ignore = 0 as u128;
        for metadata in self.initial_metadata.iter() {
            match metadata {
                ResamplingMetadata::Arrive(frames_recieved, micros_metadata_started, micros_metadata_finished, _calibrated) => {
                    // whole packet is behind, ignore entire thing
                    if *micros_metadata_finished < micros_packet_started {
                        samples_to_ignore += *frames_recieved as u128;
                    }
                    // keep all data
                    else if *micros_metadata_started >= micros_packet_started{
                        
                    } 
                    // it overlaps, only ignore stuff before this packet
                    else {
                        let micros_ignoring = micros_packet_started - *micros_metadata_started;
                        samples_to_ignore += micros_to_frames(micros_ignoring as u128, self.sample_rate as u128);
                    }

                }
            }
        }
        samples_to_ignore
    }

    // waits until we have at least that much data
    // (or something errors)
    // returns (success, audio_buffer)
    fn get_chunk_to_read(&mut self, size: usize) -> (bool, &[f32]) {
        while self.final_audio_buffer_consumer.available() < size {
            // wait for data to arrive
            match self.finished_message_reciever.recv() {
                Ok(_data) => {
                    // this is only called after is_ready_to_read returns true (and is no longer used),
                    // so it's fine to ignore this, we don't use samples_recieved anymore
                }
                Err(err) => {
                    eprintln!("channel closed: {err}");
                    return (false, &[])
                }
            }
        }
        (true, self.final_audio_buffer_consumer.get_chunk_to_read(size))
    }

    fn finish_read(&mut self, size: usize) -> usize {
        self.final_audio_buffer_consumer.finish_read(size)
    }
}

impl Drop for InputStreamAlignerConsumer {
    fn drop(&mut self) {
        if let Err(err) = self.thread_message_sender.send(AudioBufferMetadata::Teardown()) {
            eprintln!("failed to send shutdown signal: {}", err);
        }
    }
}

// make (producer (recieves audio data from device), resampler (resamples input audio to target rate), consumer (contains resampled data)) for input audio alignment
fn create_input_stream_aligner(input_sample_rate: u32, output_sample_rate: u32, history_len: usize, calibration_packets: u32, audio_buffer_seconds: u32, resampler_quality: i32) -> Result<(InputStreamAlignerProducer, InputStreamAlignerResampler, InputStreamAlignerConsumer), Box<dyn Error>> {
    let (input_audio_buffer_producer, input_audio_buffer_consumer) = HeapRb::<f32>::new((audio_buffer_seconds * input_sample_rate) as usize).split();
    let (input_audio_buffer_metadata_producer, input_audio_buffer_metadata_consumer) = mpsc::channel::<AudioBufferMetadata>();
    let additional_input_audio_buffer_metadata_producer = input_audio_buffer_metadata_producer.clone(); // make another one, this is ok because it is multiple producer single consumer 
    // this recieves data from audio buffer
    let producer = InputStreamAlignerProducer::new(
        input_sample_rate,
        output_sample_rate,
        history_len,
        calibration_packets,
        input_audio_buffer_producer,
        input_audio_buffer_metadata_producer
    )?;

    let (finished_resampling_producer, finished_resampling_consumer) = mpsc::channel::<ResamplingMetadata>();

    let (output_audio_buffer_producer, output_audio_buffer_consumer) = HeapRb::<f32>::new((audio_buffer_seconds * output_sample_rate) as usize).split();
    // resampled_consumer: BufferedCircularConsumer::<f32>::new(resampled_consumer))
    // this resamples, designed to run on a seperate thread
    
    let resampler = InputStreamAlignerResampler::new(
        input_sample_rate,
        output_sample_rate,
        resampler_quality,
        input_audio_buffer_consumer,
        input_audio_buffer_metadata_consumer,
        output_audio_buffer_producer,
        finished_resampling_producer,
    )?;

   
    let consumer = InputStreamAlignerConsumer::new(
        output_sample_rate,
        BufferedCircularConsumer::new(output_audio_buffer_consumer),
        additional_input_audio_buffer_metadata_producer, // give it ability to send shutdown signal to thread
        finished_resampling_consumer
    );

    Ok((producer, resampler, consumer))
}

type StreamId = u64;

enum HeapConsSendMsg {
    Add(StreamId, ResampledBufferedCircularProducer, ringbuf::HeapCons<f32>),
    Remove(StreamId),
    InterruptAll(),
}

struct OutputStreamAligner {
    device_sample_rate: u32,
    output_sample_rate: u32,
    frame_size: u32,
    heap_cons_sender: mpsc::Sender<HeapConsSendMsg>,
    heap_cons_reciever: mpsc::Receiver<HeapConsSendMsg>,
    input_audio_buffer_producer: BufferedCircularProducer<f32>,
    resample_audio_buffer_producer: ResampledBufferedCircularProducer,
    resample_audio_buffer_consumer: BufferedCircularConsumer<f32>,
    device_audio_producer: BufferedCircularProducer<f32>,
    stream_consumers: HashMap<StreamId, (ResampledBufferedCircularProducer, BufferedCircularConsumer<f32>)>,
    cur_stream_id: Arc<AtomicU64>,
}

// allows for playing audio on top of each other (mixing) or just appending to buffer
impl OutputStreamAligner {
    fn new(device_sample_rate: u32, output_sample_rate: u32, audio_buffer_seconds: u32, resampler_quality: i32, frame_size: u32, device_audio_producer: HeapProd<f32>) -> Result<Self, Box<dyn Error>>  {

        // used to send across threads
        let (heap_cons_sender, heap_cons_reciever) = mpsc::channel::<HeapConsSendMsg>();
        let (input_audio_buffer_producer, input_audio_buffer_consumer) = HeapRb::<f32>::new((audio_buffer_seconds * device_sample_rate) as usize).split();
        let (resampled_audio_buffer_producer, resampled_audio_buffer_consumer) = HeapRb::<f32>::new((audio_buffer_seconds * output_sample_rate) as usize).split();
        Ok(Self {
            device_sample_rate: device_sample_rate,
            output_sample_rate: output_sample_rate,
            frame_size: frame_size,
            heap_cons_sender: heap_cons_sender,
            heap_cons_reciever: heap_cons_reciever,
            input_audio_buffer_producer: BufferedCircularProducer::new(input_audio_buffer_producer),
            resample_audio_buffer_producer: ResampledBufferedCircularProducer::new(
                device_sample_rate,
                output_sample_rate,
                resampler_quality,
                BufferedCircularConsumer::new(input_audio_buffer_consumer),
                BufferedCircularProducer::new(resampled_audio_buffer_producer)
            )?,
            resample_audio_buffer_consumer: BufferedCircularConsumer::new(resampled_audio_buffer_consumer),
            device_audio_producer: BufferedCircularProducer::new(device_audio_producer),
            stream_consumers: HashMap::new(),
            cur_stream_id: Arc::new(AtomicU64::new(0)),
        })
    }

    fn begin_audio_stream(&self, audio_buffer_seconds: u32, sample_rate: u32, resampler_quality: i32) -> Result<(StreamId, HeapProd<f32>), Box<dyn Error>> {
        // this assigns unique ids in a thread-safe way
        let stream_index = self.cur_stream_id.fetch_add(1, Ordering::Relaxed);
        let (producer, consumer) = HeapRb::<f32>::new((audio_buffer_seconds * sample_rate) as usize).split();
        let (resampled_producer, resampled_consumer) = HeapRb::<f32>::new((audio_buffer_seconds * self.device_sample_rate) as usize).split();

        // send the consumer to the consume thread
        let resampled_producer = ResampledBufferedCircularProducer::new(
            sample_rate,
            self.device_sample_rate,
            resampler_quality,
            BufferedCircularConsumer::<f32>::new(consumer),
            BufferedCircularProducer::<f32>::new(resampled_producer),
        )?;

        self.heap_cons_sender.send(HeapConsSendMsg::Add(stream_index, resampled_producer, resampled_consumer))?;
        Ok((stream_index, producer))
    }

    fn enqueue_audio(audio_data: &[f32], mut audio_producer: HeapProd<f32>) -> HeapProd<f32> {
        let num_pushed = audio_producer.push_slice(audio_data);
        if num_pushed < audio_data.len() {
            eprintln!("Error: output audio buffer got behind, try increasing buffer size");
        }
        audio_producer
    }

    fn end_audio_stream(&self, stream_index: StreamId) -> Result<(), Box<dyn Error>> {
        self.heap_cons_sender.send(HeapConsSendMsg::Remove(stream_index))?;
        Ok(())
    }

    fn interrupt_all_streams(&self) -> Result<(), Box<dyn Error>> { 
        self.heap_cons_sender.send(HeapConsSendMsg::InterruptAll())?;
        Ok(())
    }
    
    fn get_chunk_to_read(&mut self, size: usize) -> Result<&[f32], Box<dyn std::error::Error>> {
        while self.resample_audio_buffer_consumer.available() < size {
            self.mix_audio_streams(self.frame_size as usize)?;
        }
        Ok(self.resample_audio_buffer_consumer.get_chunk_to_read(size))
    }

    fn finish_read(&mut self, size: usize) -> usize {
        self.resample_audio_buffer_consumer.finish_read(size)
    }

    fn mix_audio_streams(&mut self, input_chunk_size: usize) -> Result<(), Box<dyn std::error::Error>> {
        // fetch new audio consumers, non-blocking
        loop {
            match self.heap_cons_reciever.try_recv() {
                Ok(msg) => match msg {
                    HeapConsSendMsg::Add(id, resampled_producer, resampled_consumer) => {
                        self.stream_consumers.insert(id, (resampled_producer, BufferedCircularConsumer::new(resampled_consumer)));
                    }
                    HeapConsSendMsg::Remove(id) => {
                        // remove if present
                        self.stream_consumers.remove(&id);
                    }
                    HeapConsSendMsg::InterruptAll() => {
                        // remove all streams, interrupt requires new streams to be created
                        self.stream_consumers.clear();
                    }
                },
                Err(TryRecvError::Empty) => break,          // nothing waiting; continue processing
                Err(TryRecvError::Disconnected) => break,   // sender dropped; bail out or log
            }
        }

        let (need_to_write_input_values, input_buf_write) = self.input_audio_buffer_producer.get_chunk_to_write(input_chunk_size);
        let actual_input_chunk_size = input_buf_write.len();
        input_buf_write.fill(0.0);
        for (_stream_id, (resample_producer, resample_consumer)) in self.stream_consumers.iter_mut() {
            resample_producer.resample_all()?; // do resampling of any available data
            let buf_from_stream = resample_consumer.get_chunk_to_read(actual_input_chunk_size);
            let samples_to_mix = buf_from_stream.len().min(actual_input_chunk_size);
            if samples_to_mix == 0 {
                continue;
            }
            // just add to mix, do not average or clamp. Average results in too quiet, clamp is non-linear (so confuses eac, which only works with linear transformations), 
            // (fyi, resample is a linear operation in speex so it's safe to do while using eac)
            // see this https://dsp.stackexchange.com/a/3603
            for (dst, &src) in input_buf_write[..samples_to_mix]
                .iter_mut()
                .zip(buf_from_stream.iter())
            {
                *dst += src;
            }
            
            resample_consumer.finish_read(samples_to_mix);
        }
        // send mixed values to the output device
        let (need_to_write_device_values, device_buf_write) = self.device_audio_producer.get_chunk_to_write(actual_input_chunk_size);
        let samples_to_copy = device_buf_write.len().min(actual_input_chunk_size);
        device_buf_write[..samples_to_copy].copy_from_slice(&input_buf_write[..samples_to_copy]);
        self.device_audio_producer.finish_write(need_to_write_device_values, samples_to_copy);
        
        // finish writing to the input buffer
        self.input_audio_buffer_producer.finish_write(need_to_write_input_values, actual_input_chunk_size);
        
        // resample any available data
        self.resample_audio_buffer_producer.resample_all()?;
        Ok(())
    }
}

struct InputDeviceConfig {
    host_id: cpal::HostId,
    device_name: String,
    channels: u16,
    sample_rate: u32,
    sample_format: SampleFormat,
    
    // number of audio chunks to hold in memory, for aligning input devices's values when dropped frames/clock offsets. 100 or so is fine
    history_len: usize,
    // number of packets recieved before we start getting audio data
    // a larger value here will take longer to connect, but result in more accurate timing alignments
    calibration_packets: u32,
    // how long buffer of input audio to store, should only really need a few seconds as things are mostly streamed
    audio_buffer_seconds: u32,
    resampler_quality: i32
}

impl InputDeviceConfig {
    fn new(
        host_id: cpal::HostId,
        device_name: String,
        channels: u16,
        sample_rate: u32,
        sample_format: SampleFormat,
        history_len: usize,
        calibration_packets: u32,
        audio_buffer_seconds: u32,
        resampler_quality: i32,
    ) -> Self {
        Self {
            host_id,
            device_name: device_name.clone(),
            channels,
            sample_rate,
            sample_format,
            history_len,
            calibration_packets,
            audio_buffer_seconds,
            resampler_quality,
        }
    }

    /// Build a config using the device's default input settings plus caller-provided buffer/resampler tuning.
    fn from_default(
        host_id: cpal::HostId,
        device_name: String,
        history_len: usize,
        calibration_packets: u32,
        audio_buffer_seconds: u32,
        resampler_quality: i32,
    ) -> Result<Self, Box<dyn Error>> {
        let host = cpal::host_from_id(host_id)?;
        let input_device = select_device(host.input_devices(), &device_name, "Input")?;
        let default_config = input_device.default_input_config()?;

        Ok(Self::new(
            host_id,
            input_device.name()?,
            default_config.channels(),
            default_config.sample_rate().0,
            default_config.sample_format(),
            history_len,
            calibration_packets,
            audio_buffer_seconds,
            resampler_quality,
        ))
    }
}

struct OutputDeviceConfig {
    host_id: cpal::HostId,
    device_name: String,
    channels: u16,
    sample_rate: u32,
    sample_format: SampleFormat,
    
    // how long buffer of output audio to store, should only really need a few seconds as things are mostly streamed
    audio_buffer_seconds: u32,
    resampler_quality: i32,
    // frame size (in terms of samples) should be small, on the order of 1-2ms or less.
    // otherwise you may get skipping if you do not provide audio via enqueue_audio fast enough
    // larger frame sizes will also prevent immediate interruption, as interruption can only happen between each frame
    frame_size: u32,
}

impl OutputDeviceConfig {
    fn new(
        host_id: cpal::HostId,
        device_name: String,
        channels: u16,
        sample_rate: u32,
        sample_format: SampleFormat,
        audio_buffer_seconds: u32,
        resampler_quality: i32,
        frame_size: u32,
    ) -> Self {
        Self {
            host_id,
            device_name: device_name.clone(),
            channels,
            sample_rate,
            sample_format,
            audio_buffer_seconds,
            resampler_quality,
            frame_size,
        }
    }

    /// Build a config using the device's default output settings plus caller-provided buffer/resampler tuning.
    fn from_default(
        host_id: cpal::HostId,
        device_name: String,
        audio_buffer_seconds: u32,
        resampler_quality: i32,
        frame_size_millis: u32,
    ) -> Result<Self, Box<dyn Error>> {
        let host = cpal::host_from_id(host_id)?;
        let output_device = select_device(host.output_devices(), &device_name, "Output")?;
        let default_config = output_device.default_output_config()?;
        let sample_rate = default_config.sample_rate().0;
        let frame_size = micros_to_frames((frame_size_millis as u128)*1000, sample_rate as u128);

        Ok(Self::new(
            host_id,
            output_device.name()?,
            default_config.channels(),
            default_config.sample_rate().0,
            default_config.sample_format(),
            audio_buffer_seconds,
            resampler_quality,
            frame_size as u32,
        ))
    }
}

struct AecConfig {
    target_sample_rate: u32,
    frame_size: usize,
    filter_length: usize
}

impl AecConfig {
    fn new(target_sample_rate: u32, frame_size: usize, filter_length: usize) -> Self {
        Self { target_sample_rate, frame_size, filter_length }
    }
}

fn get_input_stream_aligners(device_config: &InputDeviceConfig, aec_config: &AecConfig) -> Result<(Stream, Vec<InputStreamAlignerConsumer>), Box<dyn std::error::Error>>  {

    let host = cpal::host_from_id(device_config.host_id)?;
    let device = select_device(
        host.input_devices(),
        &device_config.device_name,
        "Input",
    )?;

    let supported_config = find_matching_device_config(
        &device,
        &device_config.device_name,
        device_config.channels,
        device_config.sample_rate,
        device_config.sample_format,
        "Input",
    )?;
    
    let mut aligner_producers = Vec::new();
    let mut aligner_consumers = Vec::new();

    for _channel in 0..device_config.channels {
        let (producer, mut resampler, consumer) = create_input_stream_aligner(
            device_config.sample_rate,
            aec_config.target_sample_rate,
            device_config.history_len,
            device_config.calibration_packets,
            device_config.audio_buffer_seconds,
            device_config.resampler_quality)?;
        aligner_producers.push(producer);

        // start the resampler thread
        thread::spawn(move || {
            // it returns true when signaled to stop (such as when consumer goes out of scope)
            loop {
                match resampler.resample() {
                    Ok(true) => continue,
                    Ok(false) => break,
                    Err(err) => {
                        eprintln!("resampler error: {err}");
                        break;
                    }
                }
            }
        });

        aligner_consumers.push(consumer);
    }

    let stream = build_input_alignment_stream(
        &device,
        device_config,
        supported_config,
        aligner_producers,
    )?;

    // start input stream
    stream.play()?;

    Ok((stream, aligner_consumers))
}

fn get_output_stream_aligners(device_config: &OutputDeviceConfig, aec_config: &AecConfig) -> Result<(Stream, Vec<OutputStreamAligner>), Box<dyn std::error::Error>> {

    let host = cpal::host_from_id(device_config.host_id)?;

    let device = select_device(
        host.output_devices(),
        &device_config.device_name,
        "Output",
    )?;

    let supported_config = find_matching_device_config(
        &device,
        &device_config.device_name,
        device_config.channels,
        device_config.sample_rate,
        device_config.sample_format,
        "Output",
    )?;
    
    let mut aligners = Vec::<OutputStreamAligner>::new();
    let mut device_audio_channel_consumers = Vec::new();
    for _channel in 0..device_config.channels {
        let (device_audio_producer, device_audio_consumer) = HeapRb::<f32>::new((device_config.audio_buffer_seconds * device_config.sample_rate) as usize).split();
        let aligner = OutputStreamAligner::new(
            device_config.sample_rate,
            aec_config.target_sample_rate,
            device_config.audio_buffer_seconds,
            device_config.resampler_quality,
            device_config.frame_size,
            device_audio_producer,
        )?;
        aligners.push(aligner);
        device_audio_channel_consumers.push(BufferedCircularConsumer::new(device_audio_consumer));
    }

    let stream = build_output_alignment_stream(
        &device,
        device_config,
        supported_config,
        device_audio_channel_consumers
    )?;

    // start output stream
    stream.play()?;

    Ok((stream, aligners))
}


enum DeviceUpdateMessage {
    AddInputDevice(String, Stream, Vec<InputStreamAlignerConsumer>),
    RemoveInputDevice(String),
    AddOutputDevice(String, Stream, Vec<OutputStreamAligner>),
    RemoveOutputDevice(String)
}

struct AecStream {
    aec: Option<EchoCanceller>,
    aec_config: AecConfig,
    device_update_sender: mpsc::Sender<DeviceUpdateMessage>,
    device_update_receiver: mpsc::Receiver<DeviceUpdateMessage>,
    input_streams: HashMap<String, Stream>,
    output_streams: HashMap<String, Stream>,
    input_aligners: HashMap<String, Vec<InputStreamAlignerConsumer>>,
    input_aligners_in_progress: HashMap<String, Vec<InputStreamAlignerConsumer>>,
    output_aligners: HashMap<String, Vec<OutputStreamAligner>>,
    sorted_input_aligners: Vec<String>,
    sorted_output_aligners: Vec<String>,
    input_channels: usize,
    output_channels: usize,
    start_micros: Option<u128>,
    total_frames_emitted: u128,
    input_audio_buffer: Vec<i16>,
    output_audio_buffer: Vec<i16>,
    aec_audio_buffer: Vec<i16>,
    aec_out_audio_buffer: Vec<f32>
}

impl AecStream {
    fn new(
        aec_config: AecConfig
    ) -> Result<Self, Box<dyn Error>> {
        if aec_config.target_sample_rate == 0 {
            return Err(format!("Target sample rate is {}, it must be greater than zero.", aec_config.target_sample_rate).into());
        }
        let (device_update_sender, device_update_receiver) = mpsc::channel::<DeviceUpdateMessage>();
        Ok(Self {
           aec: None,
           aec_config: aec_config,
           device_update_sender: device_update_sender,
           device_update_receiver: device_update_receiver,
           input_streams: HashMap::new(),
           output_streams: HashMap::new(),
           input_aligners: HashMap::new(),
           input_aligners_in_progress: HashMap::new(),
           output_aligners: HashMap::new(),
           sorted_input_aligners: Vec::new(),
           sorted_output_aligners: Vec::new(),
           input_channels: 0,
           output_channels: 0,
           start_micros: None,
           total_frames_emitted: 0,
           input_audio_buffer: Vec::new(),
           output_audio_buffer: Vec::new(),
           aec_audio_buffer: Vec::new(),
           aec_out_audio_buffer: Vec::new(),
        })
    }

    fn num_input_channels(&self) -> usize {
        self.input_aligners.values().flatten().count()
    }

    fn num_output_channels(&self) -> usize {
        self.output_aligners.values().flatten().count()
    }

    fn reinitialize_aec(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.input_channels = self.num_input_channels();
        self.output_channels = self.num_output_channels();

        // store a consistent ordering
        self.sorted_input_aligners = self.input_aligners.keys().cloned().collect();
        self.sorted_input_aligners.sort();

        self.sorted_output_aligners = self.output_aligners.keys().cloned().collect();
        self.sorted_output_aligners.sort();

        self.aec = if self.input_channels > 0 && self.output_channels > 0 {
            EchoCanceller::new_multichannel(
                self.aec_config.frame_size,
                self.aec_config.filter_length,
                self.input_channels,
                self.output_channels,
            )
        } else {
            None
        };

        self.input_audio_buffer.clear();
        self.input_audio_buffer.resize(self.aec_config.frame_size * self.input_channels, 0 as i16);
        self.output_audio_buffer.clear();
        self.output_audio_buffer.resize(self.aec_config.frame_size * self.output_channels, 0 as i16);
        self.aec_audio_buffer.clear();
        self.aec_audio_buffer.resize(self.aec_config.frame_size * self.input_channels, 0 as i16);
        self.aec_out_audio_buffer.clear();
        self.aec_out_audio_buffer.resize(self.aec_config.frame_size * self.input_channels, 0 as f32);
        Ok(())
    }

    fn add_input_device(&mut self, config: &InputDeviceConfig) -> Result<(), Box<dyn std::error::Error>> {
        let (stream, aligners) = get_input_stream_aligners(config, &self.aec_config)?;
        self.device_update_sender.send(DeviceUpdateMessage::AddInputDevice(config.device_name.clone(), stream, aligners))?;
        Ok(())
    }

    fn add_output_device(&mut self, config: &OutputDeviceConfig) -> Result<(), Box<dyn std::error::Error>> {
        let (stream, aligners) = get_output_stream_aligners(config, &self.aec_config)?;
        self.device_update_sender.send(DeviceUpdateMessage::AddOutputDevice(config.device_name.clone(), stream, aligners))?;
        Ok(())
    }

    fn remove_input_device(&mut self, config: &InputDeviceConfig) -> Result<(), Box<dyn std::error::Error>> {
        self.device_update_sender.send(DeviceUpdateMessage::RemoveInputDevice(config.device_name.clone()))?;
        Ok(())
    }

    fn remove_output_device(&mut self, config: &OutputDeviceConfig) -> Result<(), Box<dyn std::error::Error>> {
        self.device_update_sender.send(DeviceUpdateMessage::RemoveOutputDevice(config.device_name.clone()))?;
        Ok(())
    }

    fn update(&mut self) -> Result<&[f32], Box<dyn std::error::Error>> {
        let chunk_size = self.aec_config.frame_size;
        let start_micros = if let Some(start_micros_value) = self.start_micros {
            start_micros_value
        } else {
            let start_micros_value = SystemTime::now().duration_since(UNIX_EPOCH).expect("clock went backwards").as_micros() - frames_to_micros(chunk_size as u128, self.aec_config.target_sample_rate as u128);
            self.start_micros = Some(start_micros_value);
            start_micros_value
        };
        let chunk_start_micros = frames_to_micros(self.total_frames_emitted, self.aec_config.target_sample_rate as u128) + start_micros;
        let chunk_end_micros = chunk_start_micros + frames_to_micros(chunk_size as u128, self.aec_config.target_sample_rate as u128);
        self.total_frames_emitted += chunk_size as u128;
        // todo: move to crossbeam to avoid mpsc locks
        // todo: use actual timestamps on mac osx because they are system-wide (on linux they are device-wide so we have to do something else)
        loop {
            match self.device_update_receiver.try_recv() {
                Ok(msg) => match msg {
                    DeviceUpdateMessage::AddInputDevice(device_name, stream, aligners) => {
                        // old stream is stopped by default when it goes out of scope
                        self.input_streams.insert(device_name.clone(), stream);
                        self.input_aligners_in_progress.insert(device_name.clone(), aligners);
                        self.input_aligners.insert(device_name.clone(), Vec::new());

                        self.reinitialize_aec()?;
                    }
                    DeviceUpdateMessage::RemoveInputDevice(device_name) => {
                        // old stream is stopped by default when it goes out of scope
                        self.input_streams.remove(&device_name);
                        self.input_aligners.remove(&device_name);
                        self.input_aligners_in_progress.remove(&device_name);

                        self.reinitialize_aec()?;
                    }
                    DeviceUpdateMessage::AddOutputDevice(device_name, stream, aligners) => {
                        if let Some(stream) = self.output_streams.get(&device_name) {
                            stream.pause()?;
                        }
                        self.output_streams.insert(device_name.clone(), stream);
                        self.output_aligners.insert(device_name.clone(), aligners);

                        self.reinitialize_aec()?;
                    }
                    DeviceUpdateMessage::RemoveOutputDevice(device_name) => {
                        if let Some(stream) = self.output_streams.get(&device_name) {
                            stream.pause()?;
                        }
                        self.output_streams.remove(&device_name);
                        self.output_aligners.remove(&device_name);

                        self.reinitialize_aec()?;
                    }
                }
                Err(TryRecvError::Empty) => {
                    // no message available right now
                    break;
                }
                Err(TryRecvError::Disconnected) => {
                    // sender dropped; receiver will never get more messages
                    eprintln!("Error: Stream message send disconnected");
                    break;
                }
            } 
        }
        // similarly, if we initialize an output device here
        // we may not get any audio for a little bit
        if chunk_size == 0 {
            return Ok(&[]);
        }

        // initialize any new aligners and align them to our frame step
        let mut modified_aligners = false;
        for key in &self.sorted_input_aligners {
            if let Some(in_progress) = self.input_aligners_in_progress.get_mut(key) {
                let mut ready = Vec::new();
                let mut remaining = Vec::new();

                for mut aligner in in_progress.drain(..) {
                    if aligner.is_ready_to_read(chunk_end_micros, chunk_size) {
                        ready.push(aligner);
                    } else {
                        remaining.push(aligner);
                    }
                }

                *in_progress = remaining;

                if !ready.is_empty() {
                    modified_aligners = true;
                    if let Some(ready_aligners) = self.input_aligners.get_mut(key) {
                        ready_aligners.extend(ready);
                    }
                }
            }
        }

        if modified_aligners {
            self.reinitialize_aec()?;
        }

        let mut input_channel = 0;
        self.input_audio_buffer.fill(0 as i16);
        for key in &self.sorted_input_aligners {
            if let Some(channel_aligners) = self.input_aligners.get_mut(key) {
                for aligner in channel_aligners.iter_mut() {
                    let samples_used = {
                        let (ok, chunk) = aligner.get_chunk_to_read(chunk_size);
                        if ok {
                            Self::write_channel_from_f32(
                                chunk,
                                input_channel,
                                self.input_channels,
                                chunk_size,
                                &mut self.input_audio_buffer,
                            );
                            chunk.len().min(chunk_size) as usize
                        } else {
                            Self::clear_channel(input_channel, self.input_channels, chunk_size, &mut self.input_audio_buffer);
                            0
                        }
                    };
                    
                    aligner.finish_read(samples_used);
                    input_channel += 1;
                }
            }
        }

        let aec_output = if self.output_channels == 0 {
            // simply pass through input_channels, no need for aec
            &self.input_audio_buffer
        }
        else {
            let mut output_channel = 0;
            self.output_audio_buffer.fill(0 as i16);
            for (_output_i, output_key) in self.sorted_output_aligners.iter().enumerate() {
                if let Some(output_aligners) = self.output_aligners.get_mut(output_key) {
                    for output_aligner in output_aligners.iter_mut() {
                        let samples_used = {
                            let output_channel_i_data = output_aligner.get_chunk_to_read(chunk_size)?;
                            Self::write_channel_from_f32(
                                output_channel_i_data,
                                output_channel,
                                self.output_channels,
                                chunk_size,
                                &mut self.output_audio_buffer,
                            );
                            output_channel_i_data.len()
                        };
                        output_aligner.finish_read(samples_used);
                        output_channel += 1;
                    }
                }
            }

            self.aec_audio_buffer.fill(0 as i16);

            if self.input_channels == 0 {
                &self.aec_audio_buffer
            }
            else {
                let Some(aec) = self.aec.as_mut() else { 
                    return Err("no aec".into());
                };

                unsafe {
                    speex_echo_cancellation(
                        aec.as_ptr(),
                        self.input_audio_buffer.as_ptr(),
                        self.output_audio_buffer.as_ptr(),
                        self.aec_audio_buffer.as_mut_ptr(),
                    );
                }
                &self.aec_audio_buffer
            }
        };
        
        for (out, sample) in self.aec_out_audio_buffer.iter_mut().zip(aec_output) {
            *out = f32::from_sample(*sample);
        }

        Ok(self.aec_out_audio_buffer.as_slice())
    }
    fn write_channel_from_f32(
        src: &[f32],
        channel: usize,
        total_channels: usize,
        frames: usize,
        dst: &mut [i16],
    ) {
        for frame in 0..frames {
            let value = src.get(frame).copied().unwrap_or(0.0);
            dst[frame * total_channels + channel] = Self::f32_to_i16(value);
        }
    }
    fn clear_channel(channel: usize, total_channels: usize, frames: usize, dst: &mut [i16]) {
        for frame in 0..frames {
            dst[frame * total_channels + channel] = 0;
        }
    }
    fn f32_to_i16(sample: f32) -> i16 {
        let clamped = sample.clamp(-1.0, 1.0);
        (clamped * i16::MAX as f32).round() as i16
    }
}
    


fn main() -> Result<(), Box<dyn Error>> {
    let frame_size_ms = 10;
    let filter_length_ms = 200;
    let aec_sample_rate = 16000;
    let aec_config = AecConfig::new(
        aec_sample_rate,
        (aec_sample_rate * frame_size_ms / 1000) as usize,
        (aec_sample_rate * filter_length_ms / 1000) as usize,
    );
    let mut stream = AecStream::new(aec_config)?;

    let host_ids = cpal::available_hosts();
    for host_id in host_ids {
        println!("Host: '{}'", host_id.name());

        // If you want to inspect devices:
        let host = cpal::host_from_id(host_id)?;
        for dev in host.input_devices()? {
            println!("  input: '{}'", dev.name()?);
            match supported_device_configs_to_string(&dev, &dev.name()?, "Input") {
                Ok(cfgs) => println!("      {cfgs}"),
                Err(err) => println!("      {err}"),
            }
        }
        for dev in host.output_devices()? {
            println!("  output: '{}'", dev.name()?);
            match supported_device_configs_to_string(&dev, &dev.name()?, "Output") {
                Ok(cfgs) => println!("      {cfgs}"),
                Err(err) => println!("      {err}"),
            }
        }
    }

    let host = get_host_by_name("ALSA").unwrap_or_else(cpal::default_host);
    let input_device_config = InputDeviceConfig::from_default(
        host.id(),
        "front:CARD=Beyond,DEV=0".to_string(),
        // number of audio chunks to hold in memory, for aligning input devices's values when dropped frames/clock offsets. 100 or so is fine
        100, // history_len 
        // number of packets recieved before we start getting audio data
        // a larger value here will take longer to connect, but result in more accurate timing alignments
        20, // calibration_packets
        // how long buffer of input audio to store, should only really need a few seconds as things are mostly streamed
        20, // audio_buffer_seconds
        3 // resampler_quality
    )?;

    let output_device_config = OutputDeviceConfig::from_default(
        host.id(),
        "hdmi:CARD=NVidia,DEV=0".to_string(),
        60*10, // audio_buffer_seconds, 10 minutes (for longer audio you may need longer)
        3, // resampler_quality
        3, // frame_size_millis (3 millis of audio per frame)
    )?;

    stream.add_input_device(&input_device_config)?;
    stream.add_output_device(&output_device_config)?;

    for _i in 0..1000 {
        let _samples = stream.update()?;
    }
    stream.remove_input_device(&input_device_config)?;
    stream.remove_output_device(&output_device_config)?;


    Ok(())
}

fn get_host_by_name(target: &str) -> Option<Host> {
    for host_id in cpal::available_hosts() {
        if host_id.name().eq_ignore_ascii_case(target) {
            return cpal::host_from_id(host_id).ok();
        }
    }
    None
}
fn select_device<I>(
    devices: Result<I, cpal::DevicesError>,
    target: &str,
    kind: &str,
) -> Result<Device, Box<dyn Error>>
where
    I: IntoIterator<Item = Device> {
    match devices {
        Ok(device_iter) => {
            let mut available = Vec::new();

            for device in device_iter {
                let name = device.name().unwrap_or_else(|_| "<unknown device>".to_string());
                available.push(name.clone());

                if &name == target {
                    return Ok(device);
                }
            }

            let quoted = available
                .iter()
                .map(|name| format!("'{name}'"))
                .collect::<Vec<_>>()
                .join(", ");
            Err(format!(
                "{kind} device matching '{target}' not found. Available: {quoted}"
            )
            .into())
        }
        Err(err) => Err(format!("Failed to enumerate {kind} devices: {err}").into()),
    }
}

fn supported_device_configs_to_string(
    device: &Device,
    device_name: &String,
    direction: &'static str
) -> Result<String, Box<dyn Error>> {
    let configs : Vec<_> = match direction {
        "Input" => device.supported_input_configs().map(|configs| configs.collect())
            .map_err(|err| format!("Unable to enumerate input configs for '{device_name}': {err}"))?,
        "Output" => device.supported_output_configs().map(|configs| configs.collect())
            .map_err(|err| format!("Unable to enumerate output configs for '{device_name}': {err}"))?,
        other => {
            return Err(format!("Unknown device direction '{other}' when validating {device_name}., should be input or output").into());
        }
    };

    Ok(configs
        .iter()
        .map(|cfg| {
            let min_rate = cfg.min_sample_rate().0;
            let max_rate = cfg.max_sample_rate().0;
            let rate_desc = if min_rate == max_rate {
                format!("{min_rate} Hz")
            } else {
                format!("{min_rate}-{max_rate} Hz")
            };
            format!(
                "{} channel(s), {:?}, sample rates: {rate_desc}",
                cfg.channels(),
                cfg.sample_format()
            )
        })
        .collect::<Vec<_>>()
        .join("\n      "))
}

fn find_matching_device_config(
    device: &Device,
    device_name: &String,
    channels: u16,
    sample_rate: u32,
    format: SampleFormat,
    direction: &'static str,
) -> Result<SupportedStreamConfig, Box<dyn Error>> {
    let configs : Vec<_> = match direction {
        "Input" => device.supported_input_configs().map(|configs| configs.collect())
            .map_err(|err| format!("Unable to enumerate input configs for '{device_name}': {err}"))?,
        "Output" => device.supported_output_configs().map(|configs| configs.collect())
            .map_err(|err| format!("Unable to enumerate output configs for '{device_name}': {err}"))?,
        other => {
            return Err(format!("Unknown device direction '{other}' when validating {device_name}., should be input or output").into());
        }
    };

    if configs.is_empty() {
        return Err(format!(
            "{} device '{}' reported no supported stream configurations to validate.",
            direction, device_name
        )
        .into());
    }

    let desired_rate = SampleRate(sample_rate);
    let matching_config = configs
        .iter()
        .filter(|cfg| cfg.channels() == channels && cfg.sample_format() == format)
        .find_map(|cfg| cfg.clone().try_with_sample_rate(desired_rate));
    
    if let Some(config) = matching_config {
        Ok(config)
    } else {
        let supported_list = configs
            .iter()
            .map(|cfg| {
                let min_rate = cfg.min_sample_rate().0;
                let max_rate = cfg.max_sample_rate().0;
                let rate_desc = if min_rate == max_rate {
                    format!("{min_rate} Hz")
                } else {
                    format!("{min_rate}-{max_rate} Hz")
                };
                format!(
                    "{} channel(s), {:?}, sample rates: {rate_desc}",
                    cfg.channels(),
                    cfg.sample_format()
                )
            })
            .collect::<Vec<_>>()
            .join("\n      ");

        Err(format!(
            "{} device '{}' does not support {} channel(s), {:?} at {} Hz. Supported configs: {}",
            direction, device_name, channels, format, sample_rate, supported_list
        )
        .into())
    }
}




fn build_input_alignment_stream(
    device: &Device,
    config: &InputDeviceConfig,
    supported_config: SupportedStreamConfig,
    channel_aligners: Vec<InputStreamAlignerProducer>,
) -> Result<Stream, cpal::BuildStreamError> {
    match config.sample_format {
        SampleFormat::I16 => build_input_alignment_stream_typed::<i16>(
            device,
            config,
            supported_config,
            channel_aligners,
        ),
        SampleFormat::F32 => build_input_alignment_stream_typed::<f32>(
            device,
            config,
            supported_config,
            channel_aligners,
        ),
        SampleFormat::U16 => build_input_alignment_stream_typed::<u16>(
            device,
            config,
            supported_config,
            channel_aligners,
        ),
        other => {
            eprintln!(
                "Input device '{0}' uses unsupported sample format {other:?}; cannot build StreamAligner.",
                config.device_name
            );
            Err(cpal::BuildStreamError::StreamConfigNotSupported)
        }
    }
}

fn build_input_alignment_stream_typed<T>(
    device: &Device,
    config: &InputDeviceConfig,
    supported_config: SupportedStreamConfig,
    mut channel_aligners: Vec<InputStreamAlignerProducer>,
) -> Result<Stream, cpal::BuildStreamError>
where
    T: Sample + SizedSample,
    f32: FromSample<T>,
{
    let per_channel_capacity = config.sample_rate
        .saturating_div(20) // ~50 ms of audio per channel
        .max(1024);
    let mut channel_buffers = (0..config.channels)
        .map(|_| Vec::<f32>::with_capacity(per_channel_capacity as usize))
        .collect::<Vec<_>>();
    
    let device_name = config.device_name.clone();
    let device_name_inner = config.device_name.clone();
    let channels = config.channels as usize;
    device.build_input_stream(
        &supported_config.config(),
        move |data: &[T], _info: &InputCallbackInfo| {
            if data.is_empty() {
                return;
            }
            for buffer in channel_buffers.iter_mut() {
                buffer.clear();
            }
            // undo the interleaving and convert to f32
            // todo: do this looping per channel in outermost instead of get_mut each time
            // (if it becomes a performance issue)
            for frame in data.chunks(channels) {
                for (channel_idx, sample) in frame.iter().enumerate() {
                    if let Some(buffer) = channel_buffers.get_mut(channel_idx) {
                        buffer.push(f32::from_sample(*sample));
                    }
                }
            }
            for (buffer, channel_aligner) in channel_buffers.iter_mut().zip(channel_aligners.iter_mut()) {
                if buffer.is_empty() {
                    continue;
                }
                if let Err(err) = channel_aligner.process_chunk(buffer.as_slice()) {
                    eprintln!("Input stream '{device_name_inner}' error when process chunk {err}");
                }
                buffer.clear();
            }
        },
        move |err| eprintln!("Input stream '{device_name}' error: {err}",),
        None,
    )
}

fn build_output_alignment_stream(
    device: &Device,
    config: &OutputDeviceConfig,
    supported_config: SupportedStreamConfig,
    device_audio_channel_consumers: Vec<BufferedCircularConsumer<f32>>
) -> Result<Stream, cpal::BuildStreamError> {
    match config.sample_format {
        SampleFormat::I16 => build_output_alignment_stream_typed::<i16>(
            device,
            config,
            supported_config,
            device_audio_channel_consumers,
        ),
        SampleFormat::F32 => build_output_alignment_stream_typed::<f32>(
            device,
            config,
            supported_config,
            device_audio_channel_consumers,
        ),
        SampleFormat::U16 => build_output_alignment_stream_typed::<u16>(
            device,
            config,
            supported_config,
            device_audio_channel_consumers,
        ),
        other => {
            eprintln!(
                "Output device '{0}' uses unsupported sample format {other:?}; cannot build StreamAligner.",
                config.device_name
            );
            Err(cpal::BuildStreamError::StreamConfigNotSupported)
        }
    }
}

fn build_output_alignment_stream_typed<T>(
    device: &Device,
    config: &OutputDeviceConfig,
    supported_config: SupportedStreamConfig,
    mut device_audio_channel_consumers: Vec<BufferedCircularConsumer<f32>>
) -> Result<Stream, cpal::BuildStreamError>
where
    T: Sample + SizedSample,
    T: FromSample<f32>,
{
    let device_name = config.device_name.clone();
    let channels = config.channels as usize;
    device.build_output_stream(
        &supported_config.config(),
        move |data: &mut [T], _| {
            let frames = data.len() / channels;
            if frames == 0 {
                return;
            }
            // in case one of the device_audio_channel_consumers doesn't have enough yet
            data.fill(T::from_sample(0.0f32));
            
            // interleave the data which is what cpal expects
            for (channel_idx, consumer) in device_audio_channel_consumers.iter_mut().enumerate() {
                let chunk = consumer.get_chunk_to_read(frames);
                if chunk.is_empty() {
                    continue;
                }

                let samples_to_write = chunk.len().min(frames);
                for (frame_idx, &sample) in chunk.iter().take(samples_to_write).enumerate() {
                    let dst = frame_idx * channels + channel_idx;
                    data[dst] = T::from_sample(sample);
                }

                consumer.finish_read(samples_to_write);
            }
        },
        move |err| eprintln!("Output stream '{device_name}' error: {err}"),
        None,
    )
}
