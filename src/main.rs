use mlx_rs::fft::rfft;
use mlx_rs::module::Module;
use mlx_rs::nn::{Conv1d, GroupNorm, LayerNorm, Linear};
use mlx_rs::ops::indexing::{IndexOp, IntoStrideBy, take_along_axis, argmax_axis};
use mlx_rs::ops::{
    as_strided, concatenate_axis, le, maximum, minimum, pad, sqrt, square, transpose_axes, which,
};
use mlx_rs::{array, Array};
use std::collections::HashMap;

// ===================== 模型结构 =====================

pub struct ConformerConvModule {
    norm: LayerNorm,
    conv1: Conv1d,
    dw_conv: Conv1d,
    conv2: Conv1d,
}

impl ConformerConvModule {
    pub fn new(dim: i32, inner_dim: i32) -> Self {
        let mut conv1 = Conv1d::new(dim, inner_dim * 2, 1).unwrap();
        conv1.padding = 0;

        let mut dw_conv = Conv1d::new(inner_dim, inner_dim, 31).unwrap();
        dw_conv.padding = 15;
        dw_conv.groups = inner_dim;

        let mut conv2 = Conv1d::new(inner_dim, dim, 1).unwrap();
        conv2.padding = 0;

        Self {
            norm: LayerNorm::new(dim).unwrap(),
            conv1,
            dw_conv,
            conv2,
        }
    }

    pub fn forward(&mut self, x: &Array) -> Array {
        let y = self.norm.forward(x).unwrap();
        let y = self.conv1.forward(&y).unwrap();
        let c = y.shape()[2];
        let split = c / 2;
        let a = y.index((.., .., ..split));
        let b = y.index((.., .., split..));
        let sig_b = mlx_rs::nn::sigmoid(&b).unwrap();
        let y = a.multiply(&sig_b).unwrap();
        let y = self.dw_conv.forward(&y).unwrap();
        let y = mlx_rs::nn::silu(&y).unwrap();
        let y = self.conv2.forward(&y).unwrap();
        y
    }
}

pub struct CFNaiveMelPE {
    pub weights: HashMap<String, Array>,
    pub hidden_dims: i32,
    pub out_dims: i32,
    pub n_layers: i32,
    pub f0_max: f32,
    pub f0_min: f32,

    input_conv1: Conv1d,
    input_gn: GroupNorm,
    input_conv2: Conv1d,

    conformer_layers: Vec<ConformerConvModule>,
    #[allow(dead_code)]
    layer_norms: Vec<LayerNorm>,

    output_norm: LayerNorm,
    output_proj: Linear,

    pub cent_table: Array,
    pub gaussian_blurred_cent_mask: Array,
}

fn apply_weight_norm(weight_v: &Array, weight_g: &Array) -> Array {
    let norm = mlx_rs::linalg::norm_l2(weight_v, &[-1], true).unwrap();
    let normed = weight_v / (norm + array!(1e-8f32));
    weight_g * normed
}

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Array {
    weights.get(key).unwrap_or_else(|| panic!("missing weight: {}", key)).clone()
}

impl CFNaiveMelPE {
    pub fn new(weights: HashMap<String, Array>) -> Self {
        let hidden_dims = 512;
        let out_dims = 360;
        let n_layers = 6;
        let input_channels = 128;
        let f0_max = 1975.5f32;
        let f0_min = 32.7f32;

        let mut input_conv1 = Conv1d::new(input_channels, hidden_dims, 3).unwrap();
        input_conv1.padding = 1;
        input_conv1.weight.value = transpose_axes(
            &get_weight(&weights, "input_stack_0_weight"),
            &[0, 2, 1],
        ).unwrap();
        input_conv1.bias.value = Some(get_weight(&weights, "input_stack_0_bias"));

        let mut input_gn = GroupNorm::new(4, hidden_dims).unwrap();
        input_gn.pytorch_compatible = true;
        input_gn.weight.value = Some(get_weight(&weights, "input_stack_1_weight"));
        input_gn.bias.value = Some(get_weight(&weights, "input_stack_1_bias"));

        let mut input_conv2 = Conv1d::new(hidden_dims, hidden_dims, 3).unwrap();
        input_conv2.padding = 1;
        input_conv2.weight.value = transpose_axes(
            &get_weight(&weights, "input_stack_3_weight"),
            &[0, 2, 1],
        ).unwrap();
        input_conv2.bias.value = Some(get_weight(&weights, "input_stack_3_bias"));

        let mut conformer_layers = Vec::new();
        let mut layer_norms = Vec::new();
        let inner_dim = hidden_dims * 2;

        for i in 0..n_layers {
            let prefix = format!("net_encoder_layers_{}", i);

            let mut cm = ConformerConvModule::new(hidden_dims, inner_dim);
            cm.norm.weight.value = Some(get_weight(&weights, &format!("{}_conformer_net_0_weight", prefix)));
            cm.norm.bias.value = Some(get_weight(&weights, &format!("{}_conformer_net_0_bias", prefix)));
            cm.conv1.weight.value = transpose_axes(
                &get_weight(&weights, &format!("{}_conformer_net_2_weight", prefix)),
                &[0, 2, 1],
            ).unwrap();
            cm.conv1.bias.value = Some(get_weight(&weights, &format!("{}_conformer_net_2_bias", prefix)));
            cm.dw_conv.weight.value = transpose_axes(
                &get_weight(&weights, &format!("{}_conformer_net_4_conv_weight", prefix)),
                &[0, 2, 1],
            ).unwrap();
            cm.dw_conv.bias.value = Some(get_weight(&weights, &format!("{}_conformer_net_4_conv_bias", prefix)));
            cm.conv2.weight.value = transpose_axes(
                &get_weight(&weights, &format!("{}_conformer_net_6_weight", prefix)),
                &[0, 2, 1],
            ).unwrap();
            cm.conv2.bias.value = Some(get_weight(&weights, &format!("{}_conformer_net_6_bias", prefix)));

            conformer_layers.push(cm);

            let mut ln = LayerNorm::new(hidden_dims).unwrap();
            ln.weight.value = Some(get_weight(&weights, &format!("{}_norm_weight", prefix)));
            ln.bias.value = Some(get_weight(&weights, &format!("{}_norm_bias", prefix)));
            layer_norms.push(ln);
        }

        let mut output_norm = LayerNorm::new(hidden_dims).unwrap();
        output_norm.weight.value = Some(get_weight(&weights, "norm_weight"));
        output_norm.bias.value = Some(get_weight(&weights, "norm_bias"));

        let mut output_proj = Linear::new(hidden_dims, out_dims).unwrap();
        let w_v = get_weight(&weights, "output_proj_weight_v");
        let w_g = get_weight(&weights, "output_proj_weight_g");
        output_proj.weight.value = apply_weight_norm(&w_v, &w_g);
        output_proj.bias.value = Some(get_weight(&weights, "output_proj_bias"));

        let cent_min_val = 1200.0f32 * ((f0_min / 10.0f32).log2());
        let cent_max_val = 1200.0f32 * ((f0_max / 10.0f32).log2());
        let cent_table = mlx_rs::ops::linspace::<f32, f32>(cent_min_val, cent_max_val, out_dims).unwrap();
        let mask = 1200.0f32 * ((f0_max / 10.0f32).log2());
        let gaussian_blurred_cent_mask = Array::from(mask);

        Self {
            weights,
            hidden_dims,
            out_dims,
            n_layers,
            f0_max,
            f0_min,
            input_conv1,
            input_gn,
            input_conv2,
            conformer_layers,
            layer_norms,
            output_norm,
            output_proj,
            cent_table,
            gaussian_blurred_cent_mask,
        }
    }

    pub fn forward(&mut self, x: &Array) -> Array {
        let x = self.input_conv1.forward(x).unwrap();
        let x = self.input_gn.forward(&x).unwrap();
        let x = mlx_rs::nn::leaky_relu(&x, 0.01).unwrap();
        let x = self.input_conv2.forward(&x).unwrap();

        let mut y = x;
        for i in 0..self.n_layers as usize {
            let residual = y.clone();
            y = self.conformer_layers[i].forward(&y);
            y = y.add(&residual).unwrap();
        }

        y = self.output_norm.forward(&y).unwrap();
        y = self.output_proj.forward(&y).unwrap();
        y = mlx_rs::nn::sigmoid(&y).unwrap();
        y
    }

    pub fn latent2cents_local_decoder(&self, y: &Array, threshold: f32) -> Array {
        let shape = y.shape();
        let b = shape[0];
        let n = shape[1];

        let ci = self.cent_table.reshape(&[1, 1, self.out_dims]).unwrap();

        let confident = mlx_rs::ops::max_axis(&y, -1, true).unwrap();
        let max_index = argmax_axis(&y, -1, true).unwrap();

        let arange_9 = Array::from_iter(0..9, &[9]);
        let local_offset = max_index.subtract(&Array::from(4i32)).unwrap();
        let local_argmax_index = arange_9.add(&local_offset).unwrap();

        let zero = Array::from(0i32);
        let max_idx = Array::from(self.out_dims - 1);
        let local_argmax_index = maximum(&local_argmax_index, &zero).unwrap();
        let local_argmax_index = minimum(&local_argmax_index, &max_idx).unwrap();

        let ci_l = take_along_axis(&ci, &local_argmax_index, -1).unwrap();
        let y_l = take_along_axis(&y, &local_argmax_index, -1).unwrap();

        let num = ci_l.multiply(&y_l).unwrap().sum_axis(-1, false).unwrap();
        let den = y_l.sum_axis(-1, false).unwrap();
        let mut rtn = num.divide(&den).unwrap();
        rtn = rtn.reshape(&[b, n, 1]).unwrap();

        let inf_neg = Array::from(f32::NEG_INFINITY);
        let one = Array::from(1.0f32);
        let mask = which(
            &le(&confident, &Array::from(threshold)).unwrap(),
            &inf_neg,
            &one,
        ).unwrap();
        rtn = rtn.multiply(&mask).unwrap();
        rtn
    }

    pub fn cent_to_f0(&self, cent: &Array) -> Array {
        let exp_term = cent.divide(&Array::from(1200.0f32)).unwrap()
            .multiply(&Array::from(2.0f32.ln())).unwrap()
            .exp().unwrap();
        Array::from(10.0f32).multiply(&exp_term).unwrap()
    }

    pub fn infer(&mut self, mel: &Array, decoder_mode: &str, threshold: f32) -> Array {
        let latent = self.forward(mel);
        let cents = if decoder_mode == "local_argmax" {
            self.latent2cents_local_decoder(&latent, threshold)
        } else {
            panic!("unsupported decoder: {}", decoder_mode)
        };
        self.cent_to_f0(&cents)
    }
}

fn load_weights(dir: &str) -> HashMap<String, Array> {
    let mut weights = HashMap::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if let Some(ext) = path.extension() {
            if ext == "npy" {
                let name = path.file_stem().unwrap().to_str().unwrap().to_string();
                if name.starts_with("weight_") {
                    let key = name.strip_prefix("weight_").unwrap().to_string();
                    let arr = Array::load_numpy(path.to_str().unwrap()).unwrap();
                    weights.insert(key, arr);
                }
            }
        }
    }
    weights
}

fn reflect_pad_1d_last_dim(x: &Array, pad_left: i32, pad_right: i32) -> Array {
    let shape = x.shape();
    let t = shape[shape.len() - 1];
    let _batch = shape[0];

    let mut parts = Vec::new();

    if pad_left > 0 {
        let left = x.index((.., ..pad_left));
        let left_rev = left.index((.., (..).stride_by(-1)));
        parts.push(left_rev);
    }
    parts.push(x.clone());
    if pad_right > 0 {
        let right = x.index((.., (t - pad_right) as i32..));
        let right_rev = right.index((.., (..).stride_by(-1)));
        parts.push(right_rev);
    }

    if parts.len() == 1 {
        parts[0].clone()
    } else {
        concatenate_axis(&parts.iter().map(|a| a).collect::<Vec<_>>()[..], shape.len() as i32 - 1).unwrap()
    }
}

fn wav_to_mel(wav: &Array, mel_basis: &Array, hann_window: &Array) -> Array {
    let batch = wav.shape()[0];
    let t = wav.shape()[1];
    let win_size = 1024;
    let hop_length = 160;
    let n_fft = 1024;
    let n_mels = 128;
    let clip_val = 1e-5f32;

    let pad_left = (win_size - hop_length) / 2;
    let pad_right = std::cmp::max(
        (win_size - hop_length + 1) / 2,
        win_size as i32 - t - pad_left,
    );
    let mode_reflect = pad_right < t;

    let y_pad = if mode_reflect {
        reflect_pad_1d_last_dim(wav, pad_left, pad_right)
    } else {
        pad(&wav.reshape(&[batch, t, 1]).unwrap(), &[(0, 0), (pad_left, pad_right), (0, 0)], Array::from(0.0f32), Some(mlx_rs::ops::PadMode::Constant)).unwrap()
            .reshape(&[batch, t + pad_left + pad_right]).unwrap()
    };

    let t_padded = y_pad.shape()[1];
    let n_frames = (t_padded - win_size) / hop_length + 1;

    let frames = as_strided(
        &y_pad,
        &[batch, n_frames, win_size],
        &[t_padded as i64, hop_length as i64, 1],
        0,
    ).unwrap();

    let hann_reshaped = hann_window.reshape(&[1, 1, win_size]).unwrap();
    let windowed = frames.multiply(&hann_reshaped).unwrap();

    let spec_complex = rfft(&windowed, n_fft, None).unwrap();
    let real = spec_complex.real().unwrap();
    let imag = spec_complex.imag().unwrap();
    let mag = sqrt(&square(&real).unwrap().add(&square(&imag).unwrap()).unwrap().add(&Array::from(1e-9f32)).unwrap()).unwrap();

    let mag_t = transpose_axes(&mag, &[0, 2, 1]).unwrap();

    let mel_basis_3d = mel_basis.reshape(&[1, n_mels, n_fft / 2 + 1]).unwrap();
    let spec_mel = mel_basis_3d.matmul(&mag_t).unwrap();

    let clamped = maximum(&spec_mel, &Array::from(clip_val)).unwrap();
    let compressed = clamped.multiply(&Array::from(1.0f32)).unwrap().log().unwrap();

    let spec_out = transpose_axes(&compressed, &[0, 2, 1]).unwrap();

    let target_n_frames = t / hop_length + 1;
    let mel_final = if target_n_frames > spec_out.shape()[1] {
        let last_frame = spec_out.index((.., -1i32, ..));
        concatenate_axis(&[&spec_out, &last_frame.reshape(&[batch, 1, n_mels]).unwrap()], 1).unwrap()
    } else if target_n_frames < spec_out.shape()[1] {
        spec_out.index((.., ..target_n_frames as i32, ..))
    } else {
        spec_out
    };

    mel_final
}

fn batch_interp_with_replacement_detach(uv: &Array, f0: &Array) -> Array {
    let shape = uv.shape();
    let b = shape[0];
    let t = shape[1];

    let uv_slice = uv.as_slice::<bool>();
    let f0_slice = f0.as_slice::<f32>();

    let mut result = vec![0.0f32; (b * t) as usize];

    for i in 0..b as usize {
        let mut voiced_idx = Vec::new();
        let mut voiced_val = Vec::new();
        for j in 0..t as usize {
            let idx = i * t as usize + j;
            if !uv_slice[idx] {
                voiced_idx.push(j as f32);
                voiced_val.push(f0_slice[idx]);
            }
        }

        if voiced_idx.is_empty() {
            continue;
        }

        for j in 0..t as usize {
            let idx = i * t as usize + j;
            if uv_slice[idx] {
                let x = j as f32;
                let pos = voiced_idx.binary_search_by(|v| v.partial_cmp(&x).unwrap());
                let right_i = match pos {
                    Ok(p) => p,
                    Err(p) => p,
                };
                let right_i = right_i.min(voiced_idx.len() - 1);
                let left_i = if right_i > 0 { right_i - 1 } else { 0 };

                let x_left = voiced_idx[left_i];
                let x_right = voiced_idx[right_i];
                let y_left = voiced_val[left_i];
                let y_right = voiced_val[right_i];

                let interp_val = if (x_right - x_left).abs() < 1e-8 {
                    y_left
                } else if x < x_left {
                    y_left
                } else if x > x_right {
                    y_right
                } else {
                    y_left + (x - x_left) * (y_right - y_left) / (x_right - x_left)
                };
                result[idx] = interp_val;
            } else {
                result[idx] = f0_slice[idx];
            }
        }
    }

    Array::from_slice(&result, &[b, t])
}

fn postprocess_f0(
    f0: &Array,
    f0_min: f32,
    f0_max: Option<f32>,
    interp_uv: bool,
) -> Array {
    let mut f0 = f0.clone();

    // uv = (f0 < f0_min).astype(f0.dtype)
    let uv = mlx_rs::ops::lt(&f0, &Array::from(f0_min)).unwrap();
    let one = Array::from(1.0f32);
    let zero = Array::from(0.0f32);
    let uv_float = mlx_rs::ops::r#where(&uv, &one, &zero).unwrap();

    // f0 = f0 * (1 - uv)
    f0 = f0.multiply(&(&one - &uv_float)).unwrap();

    // interp_uv
    if interp_uv {
        let uv_bool = uv;
        let f0_squeezed = f0.reshape(&[f0.shape()[0], f0.shape()[1]]).unwrap();
        f0 = batch_interp_with_replacement_detach(&uv_bool, &f0_squeezed)
            .reshape(&[f0.shape()[0], f0.shape()[1], 1]).unwrap();
    }

    // f0_max clipping
    if let Some(fmax) = f0_max {
        f0 = mlx_rs::ops::r#where(
            &mlx_rs::ops::gt(&f0, &Array::from(fmax)).unwrap(),
            &Array::from(fmax),
            &f0,
        ).unwrap();
    }

    f0
}

fn load_audio(path: &str) -> (Array, u32) {
    if path.ends_with(".npy") {
        let arr = Array::load_numpy(path).unwrap();
        let len = arr.shape()[0];
        (arr.reshape(&[1, len]).unwrap(), 16000)
    } else {
        let r = hound::WavReader::open(path).unwrap();
        let spec = r.spec();
        let sr = spec.sample_rate;
        let samples: Vec<f32> = r.into_samples::<f32>().map(|s| s.unwrap()).collect();
        let arr = Array::from_slice(&samples, &[1, samples.len() as i32]);
        (arr, sr)
    }
}

fn max_diff_ignore_nan_inf(a: &Array, b: &Array) -> f32 {
    let diff = (a - b).abs().unwrap();
    let finite_diff = mlx_rs::ops::r#where(
        &diff.is_nan().unwrap().add(&diff.is_inf().unwrap()).unwrap(),
        &Array::from(0.0f32),
        &diff,
    ).unwrap();
    finite_diff.max(None).unwrap().item::<f32>()
}

fn main() {
    let weights = load_weights("/Users/daisy/develop/fcpe-mlxrs");
    println!("Loaded {} weights", weights.len());

    let mel_basis = Array::load_numpy("/Users/daisy/develop/fcpe-mlxrs/ref_mel_basis.npy").unwrap();
    let hann_window = Array::load_numpy("/Users/daisy/develop/fcpe-mlxrs/ref_hann_window.npy").unwrap();

    // Load audio
    let audio_path = "/Users/daisy/develop/fcpe-mlxrs/audio/huaxue_16000.npy";
    let (audio, sr) = load_audio(audio_path);
    println!("audio shape: {:?}, sr: {}", audio.shape(), sr);

    // Preprocess
    let mel = wav_to_mel(&audio, &mel_basis, &hann_window);
    println!("mel shape: {:?}", mel.shape());

    // Model
    let mut model = CFNaiveMelPE::new(weights);
    let latent = model.forward(&mel);
    println!("latent shape: {:?}", latent.shape());

    // Postprocessing
    let f0 = model.infer(&mel, "local_argmax", 0.006);
    println!("raw f0 shape: {:?}", f0.shape());

    let f0_processed = postprocess_f0(&f0, model.f0_min, Some(model.f0_max), true);
    println!("processed f0 shape: {:?}", f0_processed.shape());
    println!("f0 min: {:?}", f0_processed.min(None).unwrap().item::<f32>());
    println!("f0 max: {:?}", f0_processed.max(None).unwrap().item::<f32>());

    // Compare with reference if available
    if let Ok(ref_mel) = Array::load_numpy("/Users/daisy/develop/fcpe-mlxrs/ref_mel.npy") {
        let diff_mel = (&mel - &ref_mel).abs().unwrap();
        println!("mel max_diff vs ref: {}", diff_mel.max(None).unwrap().item::<f32>());
    }
    if let Ok(ref_latent) = Array::load_numpy("/Users/daisy/develop/fcpe-mlxrs/ref_latent.npy") {
        let diff_latent = (&latent - &ref_latent).abs().unwrap();
        println!("latent max_diff vs ref: {}", diff_latent.max(None).unwrap().item::<f32>());
    }
    if let Ok(ref_f0) = Array::load_numpy("/Users/daisy/develop/fcpe-mlxrs/ref_f0.npy") {
        let diff_f0 = max_diff_ignore_nan_inf(&f0, &ref_f0);
        println!("raw f0 max_diff vs ref (ignore inf): {}", diff_f0);
    }

    // Save Rust raw f0 and processed f0 for plotting
    let raw_slice = f0.as_slice::<f32>();
    let mut buf = Vec::with_capacity(raw_slice.len() * 4);
    for &v in raw_slice {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write("/Users/daisy/develop/fcpe-mlxrs/rust_f0_raw.bin", buf).unwrap();
    println!("Saved rust_f0_raw.bin");

    let proc_slice = f0_processed.as_slice::<f32>();
    let mut buf2 = Vec::with_capacity(proc_slice.len() * 4);
    for &v in proc_slice {
        buf2.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write("/Users/daisy/develop/fcpe-mlxrs/rust_f0_processed.bin", buf2).unwrap();
    println!("Saved rust_f0_processed.bin");
}
