import sys
sys.path.insert(0, 'FCPE')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchfcpe.models_infer import spawn_infer_model_from_pt
import subprocess
import os
import soundfile as sf


def load_wav(path):
    wav, sr = sf.read(path, dtype='float32')
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, sr


def run_python(wav_path):
    model = spawn_infer_model_from_pt('checkpoint/fcpe.pt', device='cpu')
    wav, sr = load_wav(wav_path)
    wav_t = torch.from_numpy(wav).unsqueeze(0).unsqueeze(-1)
    f0 = model.infer(wav_t, sr, decoder_mode='local_argmax', threshold=0.006,
                     f0_min=32.7, f0_max=1975.5, interp_uv=True)
    f0 = f0.squeeze(-1).squeeze(0).numpy()
    # 获取uv mask
    f0_raw, uv = model.infer(wav_t, sr, decoder_mode='local_argmax', threshold=0.006,
                             f0_min=32.7, f0_max=1975.5, interp_uv=False, return_uv=True,
                             output_interp_target_length=f0.shape[0])
    uv = uv.squeeze(-1).squeeze(0).numpy()
    # 将uv位置设为0
    f0[uv > 0.5] = 0.0
    return f0


def run_rust(wav_path):
    abs_wav = os.path.abspath(wav_path)
    out_path = '/tmp/rust_f0.txt'
    result = subprocess.run(
        ['cargo', 'run', '--example', 'infer', '--release', '--', abs_wav, out_path],
        capture_output=True, text=True, cwd='/Users/daisy/develop/fcpe-mlxrs'
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError('Rust example failed')
    with open(out_path, 'r') as f:
        values = [float(line.strip()) for line in f if line.strip()]
    return np.array(values, dtype=np.float32)


def save_f0_svg(f0_py, f0_rust, output_path):
    fig, ax = plt.subplots(figsize=(14, 4))

    f0_py_nan = f0_py.copy()
    f0_py_nan[f0_py_nan == 0] = np.nan
    ax.plot(f0_py_nan, color='blue', linewidth=1.0, label='Python (torchfcpe)')

    f0_rust_nan = f0_rust.copy()
    f0_rust_nan[f0_rust_nan == 0] = np.nan
    ax.plot(f0_rust_nan, color='red', linewidth=1.0, label='Rust (mlx-rs)')

    ax.set_xlabel('Frame')
    ax.set_ylabel('f0 (Hz)')
    ax.set_title('f0 Comparison: Python vs Rust')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, format='svg')
    print(f'Saved SVG to {output_path}')


if __name__ == '__main__':
    wav_path = 'audio/huaxue.wav'

    print('Running Python inference...')
    f0_py = run_python(wav_path)
    print(f'Python f0 shape: {f0_py.shape}')

    print('Running Rust inference...')
    f0_rust = run_rust(wav_path)
    print(f'Rust f0 shape: {f0_rust.shape}')

    print('Saving SVG...')
    save_f0_svg(f0_py, f0_rust, 'f0_comparison.svg')
