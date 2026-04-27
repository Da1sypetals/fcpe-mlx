use fcpe_mlxrs::{
    build_hann_window, build_mel_filterbank, load_weights_safetensors, postprocess_f0, resample_audio,
    resample_audio_metal, wav_to_mel_profiled, CFNaiveMelPE,
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

    // Resample accuracy & performance comparison
    if sr != 16000 {
        let vdsp_res = resample_audio(slice, sr as usize, 16000);
        let metal_res = resample_audio_metal(slice, sr as usize, 16000);

        assert_eq!(vdsp_res.len(), metal_res.len(), "length mismatch");
        let mut max_diff = 0.0f32;
        let mut sum_sq = 0.0f64;
        for i in 0..vdsp_res.len() {
            let diff = (vdsp_res[i] - metal_res[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            sum_sq += (diff as f64).powi(2);
        }
        let rmse = (sum_sq / vdsp_res.len() as f64).sqrt();
        println!("Resample vDSP vs Metal max_diff: {:.8e}, rmse: {:.8e}", max_diff, rmse);

        let n = 50;
        let mut vdsp_times = Vec::with_capacity(n);
        let mut metal_times = Vec::with_capacity(n);
        for _ in 0..n {
            let t0 = std::time::Instant::now();
            let _ = resample_audio(slice, sr as usize, 16000);
            let t1 = std::time::Instant::now();
            vdsp_times.push(t1.duration_since(t0).as_secs_f64() * 1000.0);

            let t0 = std::time::Instant::now();
            let _ = resample_audio_metal(slice, sr as usize, 16000);
            let t1 = std::time::Instant::now();
            metal_times.push(t1.duration_since(t0).as_secs_f64() * 1000.0);
        }
        println!(
            "Resample vDSP ({} avg): {:.3} ms",
            n,
            vdsp_times.iter().sum::<f64>() / n as f64
        );
        println!(
            "Resample Metal ({} avg): {:.3} ms",
            n,
            metal_times.iter().sum::<f64>() / n as f64
        );
    }

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

    // Full pipeline benchmark (vDSP resample)
    let mut total_times_vdsp = Vec::with_capacity(5);
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
        total_times_vdsp.push(t1.duration_since(t0).as_secs_f64() * 1000.0);
    }
    println!("Full pipeline vDSP (5 avg): {:.3} ms", total_times_vdsp.iter().sum::<f64>() / 5.0);

    // Full pipeline benchmark (Metal resample)
    let mut total_times_metal = Vec::with_capacity(5);
    for _ in 0..5 {
        let t0 = std::time::Instant::now();
        let slice = audio.as_slice::<f32>();
        let audio_r = if sr != 16000 {
            Array::from_slice(&resample_audio_metal(slice, sr as usize, 16000), &[1, 992000i32])
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
        total_times_metal.push(t1.duration_since(t0).as_secs_f64() * 1000.0);
    }
    println!("Full pipeline Metal (5 avg): {:.3} ms", total_times_metal.iter().sum::<f64>() / 5.0);
}
