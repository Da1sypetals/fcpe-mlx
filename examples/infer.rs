use fcpe_mlxrs::{
    CFNaiveMelPE, build_hann_window, build_mel_filterbank, postprocess_f0, resample_audio,
    resample_audio_metal, wav_to_mel,
};
use mlx_rs::Array;
use std::env;
use std::io::Write;
use std::path::Path;

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
    let args: Vec<String> = env::args().collect();
    let wav_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("/Users/daisy/develop/fcpe-mlxrs/audio/huaxue.wav");
    let out_path = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("/tmp/rust_f0.txt");

    let checkpoint_dir = Path::new("/Users/daisy/develop/fcpe-mlxrs/checkpoint");
    let mut model = CFNaiveMelPE::load(checkpoint_dir.join("fcpe_mlx.safetensors"));

    let (audio, sr) = load_wav(wav_path);
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

    let f0 = model.infer(&mel, "local_argmax", 0.006);
    println!("raw f0 shape: {:?}", f0.shape());

    let (f0_gpu, uv_gpu) = postprocess_f0(&f0, model.f0_min, Some(model.f0_max), true);
    println!("GPU f0 min: {:?}", f0_gpu.min(None).unwrap().item::<f32>());
    println!("GPU f0 max: {:?}", f0_gpu.max(None).unwrap().item::<f32>());

    let f0_slice = f0_gpu.as_slice::<f32>();
    let uv_slice = uv_gpu.as_slice::<f32>();
    let mut file = std::fs::File::create(out_path).unwrap();
    for i in 0..f0_slice.len() {
        if uv_slice[i] > 0.5 {
            writeln!(file, "0.0").unwrap();
        } else {
            writeln!(file, "{}", f0_slice[i]).unwrap();
        }
    }
    println!("Saved f0 to {}", out_path);

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
        let mut model = CFNaiveMelPE::load(checkpoint_dir.join("fcpe_mlx.safetensors"));
        let f0 = model.infer(&mel, "local_argmax", 0.006);
        let _ = postprocess_f0(&f0, model.f0_min, Some(model.f0_max), true);
        let t1 = std::time::Instant::now();
        total_times_vdsp.push(t1.duration_since(t0).as_secs_f64() * 1000.0);

        let t0 = std::time::Instant::now();
        let slice = audio.as_slice::<f32>();
        let audio_r = if sr != 16000 {
            Array::from_slice(
                &resample_audio_metal(slice, sr as usize, 16000),
                &[1, 992000i32],
            )
        } else {
            audio.clone()
        };
        let mel_basis = build_mel_filterbank(16000.0, 1024, 128, 0.0, 8000.0);
        let hann_window = build_hann_window(1024);
        let mel = wav_to_mel(&audio_r, &mel_basis, &hann_window);
        let mut model = CFNaiveMelPE::load(checkpoint_dir.join("fcpe_mlx.safetensors"));
        let f0 = model.infer(&mel, "local_argmax", 0.006);
        let _ = postprocess_f0(&f0, model.f0_min, Some(model.f0_max), true);
        let t1 = std::time::Instant::now();
        total_times_metal.push(t1.duration_since(t0).as_secs_f64() * 1000.0);
    }

    let (vdsp_pipe_mean, vdsp_pipe_q1, vdsp_pipe_q3) = iqr_mean(total_times_vdsp);
    let (metal_pipe_mean, metal_pipe_q1, metal_pipe_q3) = iqr_mean(total_times_metal);

    println!("\n=== Full Pipeline E2E Performance ===");
    println!(
        "Full pipeline vDSP ({} IQR mean): {:.3} ms (Q1={:.3} Q3={:.3})",
        pn, vdsp_pipe_mean, vdsp_pipe_q1, vdsp_pipe_q3
    );
    println!(
        "Full pipeline Metal ({} IQR mean): {:.3} ms (Q1={:.3} Q3={:.3})",
        pn, metal_pipe_mean, metal_pipe_q1, metal_pipe_q3
    );
}
