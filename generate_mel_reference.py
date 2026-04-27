import sys
import numpy as np
import torch
import torch.nn.functional as F
import librosa

sys.path.insert(0, '/Users/daisy/develop/fcpe-mlxrs/FCPE')
import importlib.util
mel_fn = importlib.util.spec_from_file_location('torchfcpe.mel_fn_librosa', '/Users/daisy/develop/fcpe-mlxrs/FCPE/torchfcpe/mel_fn_librosa.py').loader.load_module()

wav, sr = librosa.load('/Users/daisy/develop/fcpe-mlxrs/audio/huaxue.wav', sr=16000)
wav_tensor = torch.tensor(wav).unsqueeze(0).unsqueeze(-1)

n_fft = 1024
win_size = 1024
hop_length = 160
clip_val = 1e-5

y = wav_tensor.squeeze(-1)
np.save('/Users/daisy/develop/fcpe-mlxrs/ref_y_squeeze.npy', y.detach().numpy())

pad_left = (win_size - hop_length) // 2
pad_right = max((win_size - hop_length + 1) // 2, win_size - y.size(-1) - pad_left)
mode = 'reflect' if pad_right < y.size(-1) else 'constant'
y_pad = F.pad(y.unsqueeze(1), (pad_left, pad_right), mode=mode).squeeze(1)

np.save('/Users/daisy/develop/fcpe-mlxrs/ref_y_pad.npy', y_pad.detach().numpy())

hann_window = torch.hann_window(win_size)

spec_complex = torch.stft(y_pad, n_fft, hop_length=hop_length, win_length=win_size,
                          window=hann_window,
                          center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)

np.save('/Users/daisy/develop/fcpe-mlxrs/ref_spec_complex_real.npy', spec_complex.real.detach().numpy())
np.save('/Users/daisy/develop/fcpe-mlxrs/ref_spec_complex_imag.npy', spec_complex.imag.detach().numpy())

spec = torch.sqrt(spec_complex.real.pow(2) + spec_complex.imag.pow(2) + 1e-9)
np.save('/Users/daisy/develop/fcpe-mlxrs/ref_spec_mag.npy', spec.detach().numpy())

mel_basis_np = mel_fn.mel(sr=16000, n_fft=1024, n_mels=128, fmin=0, fmax=8000)
mel_basis = torch.tensor(mel_basis_np).float()
np.save('/Users/daisy/develop/fcpe-mlxrs/ref_mel_basis.npy', mel_basis_np)

spec_mel = torch.matmul(mel_basis, spec)
np.save('/Users/daisy/develop/fcpe-mlxrs/ref_spec_mel.npy', spec_mel.detach().numpy())

spec_compressed = torch.log(torch.clamp(spec_mel, min=clip_val) * 1)
np.save('/Users/daisy/develop/fcpe-mlxrs/ref_spec_compressed.npy', spec_compressed.detach().numpy())

spec_out = spec_compressed.transpose(-1, -2)
np.save('/Users/daisy/develop/fcpe-mlxrs/ref_spec_out.npy', spec_out.detach().numpy())

n_frames = int(wav_tensor.shape[1] // hop_length) + 1
if n_frames > int(spec_out.shape[1]):
    spec_out = torch.cat((spec_out, spec_out[:, -1:, :]), 1)
if n_frames < int(spec_out.shape[1]):
    spec_out = spec_out[:, :n_frames, :]

np.save('/Users/daisy/develop/fcpe-mlxrs/ref_mel_final.npy', spec_out.detach().numpy())

print('y shape:', y.shape)
print('y_pad shape:', y_pad.shape)
print('spec_complex shape:', spec_complex.shape)
print('spec shape:', spec.shape)
print('spec_mel shape:', spec_mel.shape)
print('spec_compressed shape:', spec_compressed.shape)
print('spec_out shape:', spec_out.shape)
print('mel_final shape:', spec_out.shape)
