use fcpe_mlxrs::{
    build_hann_window, build_mel_filterbank, load_weights_safetensors, postprocess_f0, resample_audio,
    wav_to_mel_profiled, CFNaiveMelPE,
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

    let slice = audio.as_slice::<f32>();
    let audio_res = if sr != 16000 {
        Array::from_slice(&resample_audio(slice, sr as usize, 16000), &[1, 992000i32])
    } else {
        audio.clone()
    };
    println!("audio after resample shape: {:?}", audio_res.shape());

    let mel_basis = build_mel_filterbank(16000.0, 1024, 128, 0.0, 8000.0);
    let hann_window = build_hann_window(1024);
    let mel = wav_to_mel_profiled(&audio_res, &mel_basis, &hann_window);
    println!("mel shape: {:?}", mel.shape());

    let mut model = CFNaiveMelPE::new(weights.clone());
    let f0 = model.infer(&mel, "local_argmax", 0.006);
    println!("raw f0 shape: {:?}", f0.shape());

    // E2E benchmark: GPU postprocess
    let n = 100;
    let mut gpu_times = Vec::with_capacity(n);
    for _ in 0..n {
        let t0 = std::time::Instant::now();
        let _ = postprocess_f0(&f0, model.f0_min, Some(model.f0_max), true);
        let t1 = std::time::Instant::now();
        gpu_times.push(t1.duration_since(t0).as_secs_f64() * 1000.0);
    }

    let f0_gpu = postprocess_f0(&f0, model.f0_min, Some(model.f0_max), true);
    println!("GPU f0 min: {:?}", f0_gpu.min(None).unwrap().item::<f32>());
    println!("GPU f0 max: {:?}", f0_gpu.max(None).unwrap().item::<f32>());

    println!("\n=== Performance Comparison ===");
    println!("GPU postprocess (100 avg): {:.3} ms", gpu_times.iter().sum::<f64>() / n as f64);

    // Full pipeline benchmark
    let mut total_times = Vec::with_capacity(5);
    for _ in 0..5 {
        let t0 = std::time::Instant::now();
        let slice = audio.as_slice::<f32>();
        let audio_r = if sr != 16000 {
            Array::from_slice(&resample_audio(slice, sr as usize, 16000), &[1, 992000i32])
        } else {
            audio.clone()
        };
        let mel_basis = build_mel_filterbank(16000.0, 1024, 128, 0.0, 8000.0);
        let hann_window = build_hann_window(1024);
        let mel = wav_to_mel_profiled(&audio_r, &mel_basis, &hann_window);
        let mut model = CFNaiveMelPE::new(weights.clone());
        let f0 = model.infer(&mel, "local_argmax", 0.006);
        let _ = postprocess_f0(&f0, model.f0_min, Some(model.f0_max), true);
        let t1 = std::time::Instant::now();
        total_times.push(t1.duration_since(t0).as_secs_f64() * 1000.0);
    }
    println!("Full pipeline (5 avg):     {:.3} ms", total_times.iter().sum::<f64>() / 5.0);
}
