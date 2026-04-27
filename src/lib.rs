use mlx_rs::fft::rfft;
use mlx_rs::module::Module;
use mlx_rs::nn::{Conv1d, GroupNorm, LayerNorm, Linear};
use mlx_rs::ops::indexing::{argmax_axis, take_along_axis, IndexOp, IntoStrideBy};
use mlx_rs::ops::{
    as_strided, concatenate_axis, le, maximum, minimum, pad, sqrt, square, transpose_axes, which,
};
use mlx_rs::{array, Array};
use safetensors::tensor::TensorView;
use safetensors::SafeTensors;
use std::collections::HashMap;

pub fn tensor_to_array(tensor: &TensorView) -> Array {
    let shape: Vec<i32> = tensor.shape().iter().map(|&s| s as i32).collect();
    let data = tensor.data();
    let f32_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4)
    };
    Array::from_slice(f32_slice, &shape)
}

pub fn load_weights_safetensors(path: &str) -> HashMap<String, Array> {
    let data = std::fs::read(path).unwrap();
    let tensors = SafeTensors::deserialize(&data).unwrap();
    let mut weights = HashMap::new();
    for name in tensors.names() {
        let tensor = tensors.tensor(name).unwrap();
        let arr = tensor_to_array(&tensor);
        weights.insert(name.to_string(), arr);
    }
    weights
}

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Array {
    weights.get(key).unwrap_or_else(|| panic!("missing weight: {}", key)).clone()
}

fn apply_weight_norm(weight_v: &Array, weight_g: &Array) -> Array {
    let norm = mlx_rs::linalg::norm_l2(weight_v, &[-1], true).unwrap();
    let normed = weight_v / (norm + array!(1e-8f32));
    weight_g * normed
}

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

        let cent_table = get_weight(&weights, "cent_table");
        let gaussian_blurred_cent_mask = get_weight(&weights, "gaussian_blurred_cent_mask");

        Self {
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

fn hz_to_mel(freq: f32) -> f32 {
    let f_min = 0.0f32;
    let f_sp = 200.0f32 / 3.0f32;
    let min_log_hz = 1000.0f32;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (6.4f32).ln() / 27.0f32;

    if freq >= min_log_hz {
        min_log_mel + (freq / min_log_hz).ln() / logstep
    } else {
        (freq - f_min) / f_sp
    }
}

fn mel_to_hz(mel: f32) -> f32 {
    let f_min = 0.0f32;
    let f_sp = 200.0f32 / 3.0f32;
    let min_log_hz = 1000.0f32;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (6.4f32).ln() / 27.0f32;

    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        f_min + f_sp * mel
    }
}

fn fft_frequencies(sr: f32, n_fft: i32) -> Vec<f32> {
    let n = (n_fft / 2 + 1) as usize;
    (0..n).map(|i| i as f32 * sr / n_fft as f32).collect()
}

fn mel_frequencies(n_mels: i32, fmin: f32, fmax: f32) -> Vec<f32> {
    let min_mel = hz_to_mel(fmin);
    let max_mel = hz_to_mel(fmax);
    let step = (max_mel - min_mel) / (n_mels - 1) as f32;
    (0..n_mels).map(|i| mel_to_hz(min_mel + step * i as f32)).collect()
}

pub fn build_mel_filterbank(sr: f32, n_fft: i32, n_mels: i32, fmin: f32, fmax: f32) -> Array {
    let n_freqs = (n_fft / 2 + 1) as usize;
    let fftfreqs = fft_frequencies(sr, n_fft);
    let mel_f = mel_frequencies(n_mels + 2, fmin, fmax);

    let mut weights = vec![0.0f32; (n_mels as usize) * n_freqs];

    for i in 0..n_mels as usize {
        let fdiff_lower = mel_f[i + 1] - mel_f[i];
        let fdiff_upper = mel_f[i + 2] - mel_f[i + 1];

        for j in 0..n_freqs {
            let lower = - (mel_f[i] - fftfreqs[j]) / fdiff_lower;
            let upper = (mel_f[i + 2] - fftfreqs[j]) / fdiff_upper;
            let w = lower.min(upper).max(0.0);
            weights[i * n_freqs + j] = w;
        }

        // Slaney normalization
        let enorm = 2.0f32 / (mel_f[i + 2] - mel_f[i]);
        for j in 0..n_freqs {
            weights[i * n_freqs + j] *= enorm;
        }
    }

    Array::from_slice(&weights, &[n_mels, n_freqs as i32])
}

pub fn build_hann_window(size: i32) -> Array {
    // Match PyTorch torch.hann_window(size, periodic=True)
    let data: Vec<f32> = (0..size)
        .map(|i| 0.5f32 * (1.0f32 - (2.0f32 * std::f32::consts::PI * i as f32 / size as f32).cos()))
        .collect();
    Array::from_slice(&data, &[size])
}

fn compute_resample_kernel(
    input: &[f32],
    input_sr: usize,
    output_sr: usize,
) -> Option<(
    Vec<f32>,
    Vec<Vec<f32>>,
    usize,
    usize,
    usize,
    usize,
    usize,
)> {
    if input_sr == output_sr {
        return None;
    }

    let rolloff = 0.99f64;
    let lowpass_filter_width = 128i32;

    let gcd = {
        let mut a = input_sr;
        let mut b = output_sr;
        while b != 0 {
            let tmp = a % b;
            a = b;
            b = tmp;
        }
        a
    };

    let orig_freq = (input_sr / gcd) as f64;
    let new_freq = (output_sr / gcd) as f64;
    let base_freq = orig_freq.min(new_freq) * rolloff;
    let width = ((lowpass_filter_width as f64 * orig_freq / base_freq).ceil()) as i32;

    let kernel_len = (2 * width + orig_freq as i32) as usize;
    let new_freq_i = new_freq as usize;
    let mut kernel = vec![vec![0.0f64; kernel_len]; new_freq_i];

    for j in 0..new_freq_i {
        let t_offset = -(j as f64) / new_freq;
        for k in 0..kernel_len {
            let idx_val = (-width + k as i32) as f64 / orig_freq;
            let mut t = (t_offset + idx_val) * base_freq;
            t = t.clamp(-lowpass_filter_width as f64, lowpass_filter_width as f64);

            let window = (t * std::f64::consts::PI / lowpass_filter_width as f64 / 2.0).cos().powi(2);
            let t_pi = t * std::f64::consts::PI;
            let sinc = if t_pi.abs() < 1e-10 { 1.0 } else { t_pi.sin() / t_pi };
            let scale = base_freq / orig_freq;
            kernel[j][k] = sinc * window * scale;
        }
    }

    let input_len = input.len();
    let pad_left = width as usize;
    let pad_right = (width + orig_freq as i32) as usize;
    let padded_len = input_len + pad_left + pad_right;
    let output_len = (padded_len - kernel_len) / orig_freq as usize + 1;
    let target_len = ((new_freq * input_len as f64) / orig_freq).ceil() as usize;

    let mut padded = vec![0.0f32; padded_len];
    padded[pad_left..pad_left + input_len].copy_from_slice(input);

    let kernel_f32: Vec<Vec<f32>> = kernel.iter().map(|kj| kj.iter().map(|&v| v as f32).collect()).collect();

    Some((padded, kernel_f32, orig_freq as usize, new_freq_i, kernel_len, target_len, output_len))
}

pub fn resample_audio(input: &[f32], input_sr: usize, output_sr: usize) -> Vec<f32> {
    if input_sr == output_sr {
        return input.to_vec();
    }

    let (padded, kernel_f32, orig_freq, new_freq_i, kernel_len, target_len, output_len) =
        compute_resample_kernel(input, input_sr, output_sr).unwrap();

    #[cfg(target_os = "macos")]
    {
        #[link(name = "Accelerate", kind = "framework")]
        unsafe extern "C" {
            fn vDSP_desamp(
                source: *const f32,
                decimation_factor: isize,
                filter: *const f32,
                result: *mut f32,
                n: usize,
                p: usize,
            );
        }

        let mut output = vec![0.0f32; target_len];
        let mut temp = vec![vec![0.0f32; output_len]; new_freq_i];

        for j in 0..new_freq_i {
            unsafe {
                vDSP_desamp(
                    padded.as_ptr(),
                    orig_freq as isize,
                    kernel_f32[j].as_ptr(),
                    temp[j].as_mut_ptr(),
                    output_len,
                    kernel_len,
                );
            }
        }

        for i in 0..output_len {
            for j in 0..new_freq_i {
                let idx = i * new_freq_i + j;
                if idx < target_len {
                    output[idx] = temp[j][i];
                }
            }
        }

        output
    }

    #[cfg(not(target_os = "macos"))]
    {
        let mut output = vec![0.0f32; target_len];

        for i in 0..output_len {
            let start = i * orig_freq;
            for j in 0..new_freq_i {
                let out_idx = i * new_freq_i + j;
                if out_idx >= target_len {
                    break;
                }
                let mut sum = 0.0f64;
                for k in 0..kernel_len {
                    sum += padded[start + k] as f64 * kernel_f32[j][k] as f64;
                }
                output[out_idx] = sum as f32;
            }
        }

        output
    }
}

#[cfg(target_os = "macos")]
pub fn resample_audio_metal(input: &[f32], input_sr: usize, output_sr: usize) -> Vec<f32> {
    use metal::{Device, MTLSize};

    if input_sr == output_sr {
        return input.to_vec();
    }

    let (padded, kernel_f32, orig_freq, new_freq_i, kernel_len, target_len, _output_len) =
        compute_resample_kernel(input, input_sr, output_sr).unwrap();

    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue();

    let shader_src = include_str!("resample.metal");

    let library = device
        .new_library_with_source(shader_src, &metal::CompileOptions::new())
        .expect("failed to compile metal shader");
    let kernel = library
        .get_function("resample_sinc_hann", None)
        .expect("failed to get kernel function");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .expect("failed to create pipeline state");

    let input_buffer = device.new_buffer_with_data(
        padded.as_ptr() as *const _,
        (padded.len() * std::mem::size_of::<f32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let output_buffer = device.new_buffer(
        (target_len * std::mem::size_of::<f32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let kernel_flat: Vec<f32> = kernel_f32
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    let kernel_buffer = device.new_buffer_with_data(
        kernel_flat.as_ptr() as *const _,
        (kernel_flat.len() * std::mem::size_of::<f32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let cmd_buffer = queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input_buffer), 0);
    encoder.set_buffer(1, Some(&output_buffer), 0);
    encoder.set_buffer(2, Some(&kernel_buffer), 0);

    let kl = kernel_len as i32;
    let of = orig_freq as i32;
    let nf = new_freq_i as i32;
    let tl = target_len as i32;

    encoder.set_bytes(3, std::mem::size_of::<i32>() as u64, &kl as *const _ as *const _);
    encoder.set_bytes(4, std::mem::size_of::<i32>() as u64, &of as *const _ as *const _);
    encoder.set_bytes(5, std::mem::size_of::<i32>() as u64, &nf as *const _ as *const _);
    encoder.set_bytes(6, std::mem::size_of::<i32>() as u64, &tl as *const _ as *const _);

    let grid_size = MTLSize::new(target_len as u64, 1, 1);
    let threadgroup_size = MTLSize::new(256, 1, 1);
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();

    let output_ptr = output_buffer.contents() as *const f32;
    let output_slice = unsafe { std::slice::from_raw_parts(output_ptr, target_len) };
    output_slice.to_vec()
}

fn reflect_pad_1d_last_dim(x: &Array, pad_left: i32, pad_right: i32) -> Array {
    let shape = x.shape();
    let t = shape[shape.len() - 1];

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

pub fn wav_to_mel_profiled(wav: &Array, mel_basis: &Array, hann_window: &Array) -> Array {
    let batch = wav.shape()[0];
    let t = wav.shape()[1];
    let win_size = 1024;
    let hop_length = 160;
    let n_fft = 1024;
    let n_mels = 128;
    let clip_val = 1e-5f32;

    let t0 = std::time::Instant::now();
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
    let t1 = std::time::Instant::now();

    let t_padded = y_pad.shape()[1];
    let n_frames = (t_padded - win_size) / hop_length + 1;

    let frames = as_strided(
        &y_pad,
        &[batch, n_frames, win_size],
        &[t_padded as i64, hop_length as i64, 1],
        0,
    ).unwrap();
    let t2 = std::time::Instant::now();

    let hann_reshaped = hann_window.reshape(&[1, 1, win_size]).unwrap();
    let windowed = frames.multiply(&hann_reshaped).unwrap();
    let t3 = std::time::Instant::now();

    let spec_complex = rfft(&windowed, n_fft, None).unwrap();
    let t4 = std::time::Instant::now();
    let real = spec_complex.real().unwrap();
    let imag = spec_complex.imag().unwrap();
    let mag = sqrt(&square(&real).unwrap().add(&square(&imag).unwrap()).unwrap().add(&Array::from(1e-9f32)).unwrap()).unwrap();
    let t5 = std::time::Instant::now();

    let mag_t = transpose_axes(&mag, &[0, 2, 1]).unwrap();
    let t6 = std::time::Instant::now();

    let mel_basis_3d = mel_basis.reshape(&[1, n_mels, n_fft / 2 + 1]).unwrap();
    let spec_mel = mel_basis_3d.matmul(&mag_t).unwrap();
    let t7 = std::time::Instant::now();

    let clamped = maximum(&spec_mel, &Array::from(clip_val)).unwrap();
    let compressed = clamped.multiply(&Array::from(1.0f32)).unwrap().log().unwrap();
    let t8 = std::time::Instant::now();

    let spec_out = transpose_axes(&compressed, &[0, 2, 1]).unwrap();
    let t9 = std::time::Instant::now();

    let target_n_frames = t / hop_length + 1;
    let mel_final = if target_n_frames > spec_out.shape()[1] {
        let last_frame = spec_out.index((.., -1i32, ..));
        concatenate_axis(&[&spec_out, &last_frame.reshape(&[batch, 1, n_mels]).unwrap()], 1).unwrap()
    } else if target_n_frames < spec_out.shape()[1] {
        spec_out.index((.., ..target_n_frames as i32, ..))
    } else {
        spec_out
    };
    let t10 = std::time::Instant::now();

    eprintln!("[PROFILE mel] pad={:.3}us as_strided={:.3}us window={:.3}us rfft={:.3}us mag={:.3}us transpose1={:.3}us matmul={:.3}us clamp_log={:.3}us transpose2={:.3}us trim={:.3}us",
        t1.duration_since(t0).as_secs_f64() * 1e6,
        t2.duration_since(t1).as_secs_f64() * 1e6,
        t3.duration_since(t2).as_secs_f64() * 1e6,
        t4.duration_since(t3).as_secs_f64() * 1e6,
        t5.duration_since(t4).as_secs_f64() * 1e6,
        t6.duration_since(t5).as_secs_f64() * 1e6,
        t7.duration_since(t6).as_secs_f64() * 1e6,
        t8.duration_since(t7).as_secs_f64() * 1e6,
        t9.duration_since(t8).as_secs_f64() * 1e6,
        t10.duration_since(t9).as_secs_f64() * 1e6,
    );

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

fn batch_interp_with_replacement_detach_gpu(uv: &Array, f0: &Array) -> Array {
    let shape = uv.shape();
    let b = shape[0];
    let t = shape[1];

    // uv: (B, T) bool, true = unvoiced
    // f0: (B, T) f32, already zeroed for uv frames

    // Create position indices: (1, T)
    let x = mlx_rs::ops::arange::<_, f32>(0.0, t as f32, None).unwrap()
        .reshape(&[1, t]).unwrap();

    // voiced_mask: (B, T), 1.0 for voiced, 0.0 for unvoiced
    let voiced_mask = mlx_rs::ops::r#where(
        uv,
        &Array::from(0.0f32),
        &Array::from(1.0f32),
    ).unwrap();

    // voiced_pos: position where voiced, -inf where unvoiced (so cummax ignores them)
    let neg_inf = Array::from(f32::NEG_INFINITY);
    let voiced_pos = mlx_rs::ops::r#where(
        &voiced_mask,
        &x,
        &neg_inf,
    ).unwrap();

    // left_pos: forward cummax, gives position of last voiced point seen so far
    let left_pos = voiced_pos.cummax(1, None, Some(true)).unwrap();

    // right_pos: reverse cummax, gives first voiced point from the right
    let right_pos = voiced_pos.cummax(1, Some(true), Some(true)).unwrap();

    // For positions with no voiced neighbor, cummax returns -inf
    // Find the first and last voiced positions globally
    let first_voiced = argmax_axis(&voiced_mask, 1, false).unwrap()
        .as_type::<f32>().unwrap();
    let rev_voiced_mask = mlx_rs::ops::indexing::take_along_axis_device(
        &voiced_mask,
        &mlx_rs::ops::arange::<_, i32>((t - 1) as i32, -1i32, Some(-1i32)).unwrap().reshape(&[1, t]).unwrap(),
        1,
        mlx_rs::StreamOrDevice::default(),
    ).unwrap();
    let last_voiced_rev = argmax_axis(&rev_voiced_mask, 1, false).unwrap()
        .as_type::<f32>().unwrap();
    let last_voiced = Array::from((t - 1) as f32).subtract(&last_voiced_rev).unwrap();

    let left_pos = mlx_rs::ops::r#where(
        &mlx_rs::ops::eq(&left_pos, &neg_inf).unwrap(),
        &first_voiced,
        &left_pos,
    ).unwrap();
    let right_pos = mlx_rs::ops::r#where(
        &mlx_rs::ops::eq(&right_pos, &neg_inf).unwrap(),
        &last_voiced,
        &right_pos,
    ).unwrap();

    // Convert to integer indices
    let left_idx_i = left_pos.as_type::<i32>().unwrap();
    let right_idx_i = right_pos.as_type::<i32>().unwrap();

    // Clamp
    let max_idx = Array::from(t - 1);
    let zero = Array::from(0i32);
    let left_idx_i = mlx_rs::ops::maximum(&left_idx_i, &zero).unwrap();
    let left_idx_i = mlx_rs::ops::minimum(&left_idx_i, &max_idx).unwrap();
    let right_idx_i = mlx_rs::ops::maximum(&right_idx_i, &zero).unwrap();
    let right_idx_i = mlx_rs::ops::minimum(&right_idx_i, &max_idx).unwrap();

    // Use index op for gathering: f0[b, left_idx] 
    // mlx index: f0.index((.., left_idx_i)) should work for per-batch indexing
    // But mlx-rs indexing may not support Array indices directly
    // Alternative: use take_along_axis with proper shapes

    // f0 is (B, T), we need to gather along axis=1
    // take_along_axis expects indices of shape (B, T) and input of shape (B, T)
    // But we want to gather f0 at left_idx for each position
    // Actually, we can use advanced indexing via mlx::ops::take

    // Use take_along_axis for gathering: input (B, T), indices (B, T), axis=1
    // But take_along_axis expects indices to have the same shape as input
    // and gathers along the specified axis
    // We need to expand indices to (B, T, 1) and input to (B, T, 1) for axis=2? No.
    
    // Actually, for take_along_axis with axis=1:
    // input: (B, T), indices: (B, T), output: (B, T)
    // Each element in output is input[b, indices[b, t]]
    // This is exactly what we need!
    
    let f0_left = mlx_rs::ops::indexing::take_along_axis_device(&f0, &left_idx_i, 1, mlx_rs::StreamOrDevice::default()).unwrap();
    let f0_right = mlx_rs::ops::indexing::take_along_axis_device(&f0, &right_idx_i, 1, mlx_rs::StreamOrDevice::default()).unwrap();

    // Position values as float
    let left_pos_f = left_pos;
    let right_pos_f = right_pos;

    // x positions broadcasted to (B, T)
    let x_b = mlx_rs::ops::broadcast_to_device(&x, &[b, t], mlx_rs::StreamOrDevice::default()).unwrap();

    // Linear interpolation
    let dx = right_pos_f.subtract(&left_pos_f).unwrap();
    let dx_safe = mlx_rs::ops::maximum(&dx, &Array::from(1e-8f32)).unwrap();

    let ratio = x_b.subtract(&left_pos_f).unwrap()
        .divide(&dx_safe).unwrap();
    let ratio = mlx_rs::ops::minimum(&ratio, &Array::from(1.0f32)).unwrap();
    let ratio = mlx_rs::ops::maximum(&ratio, &Array::from(0.0f32)).unwrap();

    let interp = f0_left.add(
        &f0_right.subtract(&f0_left).unwrap()
            .multiply(&ratio).unwrap()
    ).unwrap();

    // Where voiced (not uv), use original f0; where unvoiced, use interp
    mlx_rs::ops::r#where(uv, &interp, f0).unwrap()
}

pub fn postprocess_f0(
    f0: &Array,
    f0_min: f32,
    f0_max: Option<f32>,
    interp_uv: bool,
) -> Array {
    let mut f0 = f0.clone();

    let uv = mlx_rs::ops::lt(&f0, &Array::from(f0_min)).unwrap();
    let one = Array::from(1.0f32);
    let zero = Array::from(0.0f32);
    let uv_float = mlx_rs::ops::r#where(&uv, &one, &zero).unwrap();

    f0 = f0.multiply(&(&one - &uv_float)).unwrap();

    if interp_uv {
        let uv_bool = uv.reshape(&[uv.shape()[0], uv.shape()[1]]).unwrap();
        let f0_squeezed = f0.reshape(&[f0.shape()[0], f0.shape()[1]]).unwrap();
        f0 = batch_interp_with_replacement_detach_gpu(&uv_bool, &f0_squeezed)
            .reshape(&[f0.shape()[0], f0.shape()[1], 1]).unwrap();
    }

    if let Some(fmax) = f0_max {
        f0 = mlx_rs::ops::r#where(
            &mlx_rs::ops::gt(&f0, &Array::from(fmax)).unwrap(),
            &Array::from(fmax),
            &f0,
        ).unwrap();
    }

    f0
}

pub fn postprocess_f0_profiled(
    f0: &Array,
    f0_min: f32,
    f0_max: Option<f32>,
    interp_uv: bool,
) -> Array {
    let t0 = std::time::Instant::now();
    let mut f0 = f0.clone();

    let t1 = std::time::Instant::now();
    let uv = mlx_rs::ops::lt(&f0, &Array::from(f0_min)).unwrap();
    let t2 = std::time::Instant::now();
    let one = Array::from(1.0f32);
    let zero = Array::from(0.0f32);
    let t3 = std::time::Instant::now();
    let uv_float = mlx_rs::ops::r#where(&uv, &one, &zero).unwrap();
    let t4 = std::time::Instant::now();

    f0 = f0.multiply(&(&one - &uv_float)).unwrap();
    let t5 = std::time::Instant::now();

    if interp_uv {
        let uv_bool = uv;
        let f0_squeezed = f0.reshape(&[f0.shape()[0], f0.shape()[1]]).unwrap();
        let t6 = std::time::Instant::now();
        f0 = batch_interp_with_replacement_detach(&uv_bool, &f0_squeezed)
            .reshape(&[f0.shape()[0], f0.shape()[1], 1]).unwrap();
        let t7 = std::time::Instant::now();
        eprintln!("[PROFILE postprocess] lt={:.3}us where1={:.3}us multiply={:.3}us interp={:.3}us",
            t2.duration_since(t1).as_secs_f64() * 1e6,
            t4.duration_since(t3).as_secs_f64() * 1e6,
            t5.duration_since(t4).as_secs_f64() * 1e6,
            t7.duration_since(t6).as_secs_f64() * 1e6,
        );
    }

    if let Some(fmax) = f0_max {
        f0 = mlx_rs::ops::r#where(
            &mlx_rs::ops::gt(&f0, &Array::from(fmax)).unwrap(),
            &Array::from(fmax),
            &f0,
        ).unwrap();
    }

    f0
}
