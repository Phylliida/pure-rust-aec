
fn main() -> Result<(), Box<dyn Error>> {
    let frame_size_ms = 10;
    let filter_length_ms = 100;
    let aec_sample_rate = 16000;
    let multi_channel = false;
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
    let frame_size = stream.aec_config.frame_size;
    println!("Frame size {frame_size}");

    let resampler_quality = 5;

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
        resampler_quality // resampler_quality
    )?;

    let output_device_config = OutputDeviceConfig::from_default(
        host.id(),
        "default".to_string(),
        100, // history_len
        20, // calibration_packets
        20, // audio_buffer_seconds, just for resampling (actual audio buffer determined upon begin_audio_stream creation)
        resampler_quality, // resampler_quality
        3, // frame_size_millis (3 millis of audio per frame)
    )?;

    stream.add_input_device(&input_device_config)?;
    let mut stream_output_creator = stream.add_output_device(&output_device_config)?;

    // output wav files for debugging
    let pcm_spec_input = WavSpec {
        channels: input_device_config.channels as u16,
        sample_rate: aec_sample_rate, // 16_000 in your config
        bits_per_sample: 16,
        sample_format: HoundSampleFormat::Int,
    };

    let pcm_spec_output = WavSpec {
        channels: output_device_config.channels as u16,
        ..pcm_spec_input
    };    
    
    let aec_spec = WavSpec {
        sample_format: HoundSampleFormat::Float,
        bits_per_sample: 32,
        ..pcm_spec_input
    };

    let mut in_wav = WavWriter::create("aligned_input.wav", pcm_spec_input)?;
    let mut out_wav = WavWriter::create("aligned_output.wav", pcm_spec_output)?;
    let mut aec_wav = WavWriter::create("aec_applied.wav", aec_spec)?;

    // input wav file to output
    let mut wav = WavReader::open("examples/example_talking.wav")?;
    let spec = wav.spec();
    let wav_channels = spec.channels as usize;
    let wav_rate = spec.sample_rate;
    let wav_samples: Vec<f32> = match spec.sample_format {
        HoundSampleFormat::Int => wav
            .samples::<i16>()
            .map(|s| f32::from_sample(s.unwrap()))
            .collect(),
        HoundSampleFormat::Float => wav.samples::<f32>().map(|s| s.unwrap()).collect(),
    };

    let mut channel_map = HashMap::new();
    for i in 0..wav_channels {
        let mut mapped_to_channels = Vec::new();
        mapped_to_channels.push(0);
        channel_map.insert(i, mapped_to_channels); // map all wav channels to first output channel, for testing
    }

    let mut stream_output = stream_output_creator.begin_audio_stream(
        wav_channels as usize,
        channel_map,
        ((wav_samples.len()/(wav_rate as usize) + 1)*2000) as u32, // audio_buffer_seconds, needs to be long enough to hold all the audio
        wav_rate,
        resampler_quality
    )?;


    // waits for channels to calibrate
    while stream.num_input_channels() == 0 || stream.num_output_channels() == 0 {
        let (aligned_input, aligned_output, aec_applied, _start_time, _end_time) = stream.update_debug()?;
        // don't write to wav files bc if one device is ready before another,
        // that device will have more samples written
        // which makes it annoying to check alignments in audacity
    }

    println!("Computing calibration");
    stream.calibrate(std::slice::from_mut(&mut stream_output_creator), true)?;
    println!("calibrated");

    let silence = vec![0.0f32; wav_samples.len()];
    // enqueues audio samples to be played after each other
    stream_output.queue_audio(wav_samples.as_slice());
    stream_output.queue_audio(&silence);
    stream_output.queue_audio(&silence);
    stream_output.queue_audio(wav_samples.as_slice());
    stream_output.queue_audio(&silence);
    stream_output.queue_audio(&silence);
    stream_output.queue_audio(wav_samples.as_slice());
    stream_output.queue_audio(&silence);
    stream_output.queue_audio(wav_samples.as_slice());


    for _i in 0..6000 {
        let num_input_channels = stream.num_input_channels();
        let (aligned_input, aligned_output, aec_applied, _start_time, _end_time) = stream.update_debug()?;
        let chunk_size = aligned_input.len() / num_input_channels;
        //for &s in aligned_input[..chunk_size].iter() { in_wav.write_sample(s)?; }
        //for &s in aligned_output[..chunk_size].iter() { out_wav.write_sample(s)?; }
        //for &s in aec_applied[..chunk_size].iter() { aec_wav.write_sample(s)?; }
        for &s in aligned_input.iter() { in_wav.write_sample(s)?; }
        for &s in aligned_output.iter() { out_wav.write_sample(s)?; }
        for &s in aec_applied.iter() { aec_wav.write_sample(s)?; }
        //println!("Got {} samples", aec_applied.len());
    }
    
    stream_output_creator.end_audio_stream(&stream_output);
    
    stream_output_creator.interrupt_all_streams();
    
    stream.remove_input_device(&input_device_config)?;
    stream.remove_output_device(&output_device_config)?;

    in_wav.finalize()?;
    out_wav.finalize()?;
    aec_wav.finalize()?;

    Ok(())
}
