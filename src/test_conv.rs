use mlx_rs::module::Module;
use mlx_rs::nn::Conv1d;
use mlx_rs::ops::transpose_axes;
use mlx_rs::Array;

fn main() {
    let w = Array::load_numpy("/Users/daisy/develop/fcpe-mlxrs/weight_input_stack_0_weight.npy").unwrap();
    let b = Array::load_numpy("/Users/daisy/develop/fcpe-mlxrs/weight_input_stack_0_bias.npy").unwrap();
    let x = Array::load_numpy("/Users/daisy/develop/fcpe-mlxrs/test_conv1d_input.npy").unwrap();
    let ref_out = Array::load_numpy("/Users/daisy/develop/fcpe-mlxrs/test_conv1d_output.npy").unwrap();

    println!("w shape: {:?}", w.shape());
    println!("x shape: {:?}", x.shape());

    let mut conv = Conv1d::new(128, 512, 3).unwrap();
    conv.weight.value = transpose_axes(&w, &[0, 2, 1]).unwrap();
    conv.bias.value = Some(b);
    conv.padding = 1;

    let out = conv.forward(&x).unwrap();
    println!("out shape: {:?}", out.shape());
    println!("out first 5: {:?}", &out.as_slice::<f32>()[..5]);

    let diff = (&out - &ref_out).abs().unwrap();
    let max_diff = diff.max(None).unwrap().item::<f32>();
    let mean_diff = diff.mean(None).unwrap().item::<f32>();
    println!("max_diff: {}, mean_diff: {}", max_diff, mean_diff);
}
