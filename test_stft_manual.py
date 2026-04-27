import numpy as np
import torch

wav = np.load('/Users/daisy/develop/fcpe-mlxrs/ref_wav.npy')
wav_tensor = torch.tensor(wav).unsqueeze(0)

n_fft = 1024
win_size = 1024
hop_length = 160

# PyTorch STFT
pad_left = (win_size - hop_length) // 2
pad_right = max((win_size - hop_length + 1) // 2, win_size - wav_tensor.size(-1) - pad_left)
mode = 'reflect' if pad_right < wav_tensor.size(-1) else 'constant'
y_pad = torch.nn.functional.pad(wav_tensor.unsqueeze(1), (pad_left, pad_right), mode=mode).squeeze(1)
hann_window = torch.hann_window(win_size)

spec_complex = torch.stft(y_pad, n_fft, hop_length=hop_length, win_length=win_size,
                          window=hann_window,
                          center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
spec_pytorch = torch.sqrt(spec_complex.real.pow(2) + spec_complex.imag.pow(2) + 1e-9)

# Manual STFT using unfold
frames = y_pad.unfold(dimension=-1, size=win_size, step=hop_length)
frames = frames * hann_window
frames_complex = torch.fft.rfft(frames, n=n_fft, dim=-1)
spec_manual = torch.sqrt(frames_complex.real.pow(2) + frames_complex.imag.pow(2) + 1e-9)
spec_manual = spec_manual.transpose(1, 2)  # [B, n_frames, n_fft//2+1] -> [B, n_fft//2+1, n_frames]

print("PyTorch spec shape:", spec_pytorch.shape)
print("Manual spec shape:", spec_manual.shape)
print("Max diff:", (spec_pytorch - spec_manual).abs().max().item())
print("Mean diff:", (spec_pytorch - spec_manual).abs().mean().item())
