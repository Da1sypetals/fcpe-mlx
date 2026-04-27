import sys
sys.path.insert(0, '/Users/daisy/develop/fcpe-mlxrs/FCPE')
import importlib.util
mel_fn = importlib.util.spec_from_file_location('torchfcpe.mel_fn_librosa', '/Users/daisy/develop/fcpe-mlxrs/FCPE/torchfcpe/mel_fn_librosa.py').loader.load_module()
sys.modules['torchfcpe.mel_fn_librosa'] = mel_fn
mel_ext = importlib.util.spec_from_file_location('torchfcpe.mel_extractor', '/Users/daisy/develop/fcpe-mlxrs/FCPE/torchfcpe/mel_extractor.py').loader.load_module()

import torch
import numpy as np
import librosa

wav, sr = librosa.load('/Users/daisy/develop/fcpe-mlxrs/audio/huaxue.wav', sr=16000)
wav_tensor = torch.tensor(wav).unsqueeze(0).unsqueeze(-1)

wav2mel = mel_ext.Wav2MelModule(
    sr=16000, n_mels=128, n_fft=1024, win_size=1024, hop_length=160,
    fmin=0, fmax=8000, clip_val=1e-5, mel_type='default',
)

mel = wav2mel(wav_tensor, sr)

np.save('/Users/daisy/develop/fcpe-mlxrs/ref_mel_from_wav.npy', mel.detach().numpy())
print('mel shape:', mel.shape)
print('mel first 5:', mel[0, 0, :5])

# Also save mel_basis and hann_window
mel_basis = wav2mel.mel_extractor.mel_basis.detach().numpy()
np.save('/Users/daisy/develop/fcpe-mlxrs/ref_mel_basis.npy', mel_basis)

hann = torch.hann_window(1024).numpy()
np.save('/Users/daisy/develop/fcpe-mlxrs/ref_hann_window.npy', hann)
