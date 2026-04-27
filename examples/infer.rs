use fcpe_mlxrs::{
    CFNaiveMelPE, load_f32_bin, load_weights_safetensors, postprocess_f0, save_f32_bin, wav_to_mel,
};

fn main() {
    let weights =
        load_weights_safetensors("/Users/daisy/develop/fcpe-mlxrs/checkpoint/fcpe.safetensors");
    println!("Loaded {} weights from safetensors", weights.len());

    let mel_basis = load_f32_bin("/Users/daisy/develop/fcpe-mlxrs/ref_mel_basis.bin");
    let hann_window = load_f32_bin("/Users/daisy/develop/fcpe-mlxrs/ref_hann_window.bin");

    // Load audio
    let audio = load_f32_bin("/Users/daisy/develop/fcpe-mlxrs/audio/huaxue_16000.bin")
        .reshape(&[1, 992000])
        .unwrap();
    println!("audio shape: {:?}", audio.shape());

    // Preprocess
    let mel = wav_to_mel(&audio, &mel_basis, &hann_window);
    println!("mel shape: {:?}", mel.shape());

    // Model
    let mut model = CFNaiveMelPE::new(weights);
    let latent = model.forward(&mel);
    println!("latent shape: {:?}", latent.shape());

    // Inference
    let f0 = model.infer(&mel, "local_argmax", 0.006);
    println!("raw f0 shape: {:?}", f0.shape());

    let f0_processed = postprocess_f0(&f0, model.f0_min, Some(model.f0_max), true);
    println!("processed f0 shape: {:?}", f0_processed.shape());
    println!(
        "f0 min: {:?}",
        f0_processed.min(None).unwrap().item::<f32>()
    );
    println!(
        "f0 max: {:?}",
        f0_processed.max(None).unwrap().item::<f32>()
    );

    // Benchmark
    let n = 100;
    let mut times = Vec::with_capacity(n);
    for _ in 0..10 {
        let _ = model.infer(&mel, "local_argmax", 0.006);
    }
    for _ in 0..n {
        let t0 = std::time::Instant::now();
        let _ = model.infer(&mel, "local_argmax", 0.006);
        let t1 = std::time::Instant::now();
        times.push(t1.duration_since(t0).as_secs_f64() * 1000.0);
    }
    let avg = times.iter().sum::<f64>() / times.len() as f64;
    println!("Rust mlx-rs infer: {:.3} ms per run (avg of {})", avg, n);

    // Save f0 output
    save_f32_bin(&f0, "/Users/daisy/develop/fcpe-mlxrs/rust_f0_raw.bin");
    save_f32_bin(
        &f0_processed,
        "/Users/daisy/develop/fcpe-mlxrs/rust_f0_processed.bin",
    );
    println!("Saved f0 outputs");
}
