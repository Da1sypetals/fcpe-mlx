use fcpe_mlxrs::{
    build_hann_window, build_mel_filterbank, load_weights_safetensors, postprocess_f0, resample_audio,
    wav_to_mel, CFNaiveMelPE,
};
use mlx_rs::Array;

fn load_wav(path: &str) -> (Array, u32) {
    let mut r = hound::WavReader::open(path).unwrap();
    let spec = r.spec();
    let channels = spec.channels as usize;
    let samples: Vec<f32> = r.samples::<f32>().map(|s| s.unwrap()).collect();
    let n_frames = samples.len() / channels;
    let mut mono = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let mut sum = 0.0f32;
        for c in 0..channels {
            sum += samples[i * channels + c];
        }
        mono.push(sum / channels as f32);
    }
    let arr = Array::from_slice(&mono, &[1, mono.len() as i32]);
    (arr, spec.sample_rate)
}

fn main() {
    let weights = load_weights_safetensors("/Users/daisy/develop/fcpe-mlxrs/fcpe.safetensors");
    println!("Loaded {} weights from safetensors", weights.len());

    let (audio, sr) = load_wav("/Users/daisy/develop/fcpe-mlxrs/audio/huaxue.wav");
    println!("audio shape: {:?}, sr: {}", audio.shape(), sr);

    // 1. 测重采样（只做5次取平均）
    let slice = audio.as_slice::<f32>();
    let mut resample_times = Vec::with_capacity(5);
    for _ in 0..5 {
        let t0 = std::time::Instant::now();
        let _ = resample_audio(slice, sr as usize, 16000);
        let t1 = std::time::Instant::now();
        resample_times.push(t1.duration_since(t0).as_secs_f64() * 1000.0);
    }
    let audio_res = if sr != 16000 {
        Array::from_slice(&resample_audio(slice, sr as usize, 16000), &[1, 992000i32])
    } else {
        audio.clone()
    };
    println!("audio after resample shape: {:?}", audio_res.shape());

    // 2. 测 mel 提取（只做5次）
    let mut mel_times = Vec::with_capacity(5);
    for _ in 0..5 {
        let t0 = std::time::Instant::now();
        let mel_basis = build_mel_filterbank(16000.0, 1024, 128, 0.0, 8000.0);
        let hann_window = build_hann_window(1024);
        let _ = wav_to_mel(&audio_res, &mel_basis, &hann_window);
        let t1 = std::time::Instant::now();
        mel_times.push(t1.duration_since(t0).as_secs_f64() * 1000.0);
    }
    let mel_basis = build_mel_filterbank(16000.0, 1024, 128, 0.0, 8000.0);
    let hann_window = build_hann_window(1024);
    let mel = wav_to_mel(&audio_res, &mel_basis, &hann_window);
    println!("mel shape: {:?}", mel.shape());

    // 3. 测模型推理（100次）
    let mut model = CFNaiveMelPE::new(weights);
    for _ in 0..50 {
        let _ = model.infer(&mel, "local_argmax", 0.006);
    }
    let n = 100;
    let mut infer_times = Vec::with_capacity(n);
    for _ in 0..n {
        let t0 = std::time::Instant::now();
        let _ = model.infer(&mel, "local_argmax", 0.006);
        let t1 = std::time::Instant::now();
        infer_times.push(t1.duration_since(t0).as_secs_f64() * 1000.0);
    }

    // 4. 测后处理（100次）
    let f0 = model.infer(&mel, "local_argmax", 0.006);
    let mut post_times = Vec::with_capacity(n);
    for _ in 0..n {
        let t0 = std::time::Instant::now();
        let _ = postprocess_f0(&f0, model.f0_min, Some(model.f0_max), true);
        let t1 = std::time::Instant::now();
        post_times.push(t1.duration_since(t0).as_secs_f64() * 1000.0);
    }

    let f0_processed = postprocess_f0(&f0, model.f0_min, Some(model.f0_max), true);
    println!("f0 min: {:?}", f0_processed.min(None).unwrap().item::<f32>());
    println!("f0 max: {:?}", f0_processed.max(None).unwrap().item::<f32>());

    println!("\n=== Performance Breakdown ===");
    println!("Resample (5 avg):   {:.3} ms", resample_times.iter().sum::<f64>() / resample_times.len() as f64);
    println!("Mel extract (5 avg):{:.3} ms", mel_times.iter().sum::<f64>() / mel_times.len() as f64);
    println!("Model infer (100 avg):{:.3} ms", infer_times.iter().sum::<f64>() / n as f64);
    println!("Postprocess (100 avg):{:.3} ms", post_times.iter().sum::<f64>() / n as f64);
    let total = resample_times[0] + mel_times[0] + infer_times.iter().sum::<f64>() / n as f64 + post_times.iter().sum::<f64>() / n as f64;
    println!("Total pipeline:     {:.3} ms", total);

    // 与历史对比
    println!("\n=== Comparison ===");
    println!("Previous linear resample + model infer: ~0.8 ms (model only)");
    println!("Current sinc resample + model infer: {:.3} ms (model only)", infer_times.iter().sum::<f64>() / n as f64);
}
