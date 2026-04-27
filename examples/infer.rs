use fcpe_mlxrs::{
    build_hann_window, build_mel_filterbank, load_weights_safetensors, postprocess_f0, resample_audio,
    resample_audio_metal, wav_to_mel, CFNaiveMelPE,
};
use mlx_rs::Array;

fn iqr_mean(mut times: Vec<f64>) -> (f64, f64, f64) {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = times.len();
    let q1 = times[n / 4];
    let q3 = times[n * 3 / 4];
    let slice = &times[n / 4..n * 3 / 4];
    let mean = slice.iter().sum::<f64>() / slice.len() as f64;
    (mean, q1, q3)
}

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
    let mel = wav_to_mel(&audio_res, &mel_basis, &hann_window);
    println!("mel shape: {:?}", mel.shape());

    let mut model = CFNaiveMelPE::new(weights.clone());
    let f0 = model.infer(&mel, "local_argmax", 0.006);
    println!("raw f0 shape: {:?}", f0.shape());

    let f0_gpu = postprocess_f0(&f0, model.f0_min, Some(model.f0_max), true);
    println!("GPU f0 min: {:?}", f0_gpu.min(None).unwrap().item::<f32>());
    println!("GPU f0 max: {:?}", f0_gpu.max(None).unwrap().item::<f32>());

    // Full pipeline benchmark (vDSP vs Metal)
    let pn = 20;
    let mut total_times_vdsp = Vec::with_capacity(pn);
    let mut total_times_metal = Vec::with_capacity(pn);
    for _ in 0..pn {
        let t0 = std::time::Instant::now();
        let slice = audio.as_slice::<f32>();
        let audio_r = if sr != 16000 {
            Array::from_slice(&resample_audio(slice, sr as usize, 16000), &[1, 992000i32])
        } else {
            audio.clone()
        };
        let mel_basis = build_mel_filterbank(16000.0, 1024, 128, 0.0, 8000.0);
        let hann_window = build_hann_window(1024);
        let mel = wav_to_mel(&audio_r, &mel_basis, &hann_window);
        let mut model = CFNaiveMelPE::new(weights.clone());
        let f0 = model.infer(&mel, "local_argmax", 0.006);
        let _ = postprocess_f0(&f0, model.f0_min, Some(model.f0_max), true);
        let t1 = std::time::Instant::now();
        total_times_vdsp.push(t1.duration_since(t0).as_secs_f64() * 1000.0);

        let t0 = std::time::Instant::now();
        let slice = audio.as_slice::<f32>();
        let audio_r = if sr != 16000 {
            Array::from_slice(&resample_audio_metal(slice, sr as usize, 16000), &[1, 992000i32])
        } else {
            audio.clone()
        };
        let mel_basis = build_mel_filterbank(16000.0, 1024, 128, 0.0, 8000.0);
        let hann_window = build_hann_window(1024);
        let mel = wav_to_mel(&audio_r, &mel_basis, &hann_window);
        let mut model = CFNaiveMelPE::new(weights.clone());
        let f0 = model.infer(&mel, "local_argmax", 0.006);
        let _ = postprocess_f0(&f0, model.f0_min, Some(model.f0_max), true);
        let t1 = std::time::Instant::now();
        total_times_metal.push(t1.duration_since(t0).as_secs_f64() * 1000.0);
    }

    let (vdsp_pipe_mean, vdsp_pipe_q1, vdsp_pipe_q3) = iqr_mean(total_times_vdsp);
    let (metal_pipe_mean, metal_pipe_q1, metal_pipe_q3) = iqr_mean(total_times_metal);

    println!("\n=== Full Pipeline E2E Performance ===");
    println!("Full pipeline vDSP ({} IQR mean): {:.3} ms (Q1={:.3} Q3={:.3})", pn, vdsp_pipe_mean, vdsp_pipe_q1, vdsp_pipe_q3);
    println!("Full pipeline Metal ({} IQR mean): {:.3} ms (Q1={:.3} Q3={:.3})", pn, metal_pipe_mean, metal_pipe_q1, metal_pipe_q3);
}
