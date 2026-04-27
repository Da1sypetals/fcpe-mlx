import sys
import os
import json
import importlib.util
import numpy as np
import torch

sys.path.insert(0, '/Users/daisy/develop/fcpe-mlxrs/FCPE')

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mel_fn = load_module('torchfcpe.mel_fn_librosa', '/Users/daisy/develop/fcpe-mlxrs/FCPE/torchfcpe/mel_fn_librosa.py')
sys.modules['torchfcpe.mel_fn_librosa'] = mel_fn
mel_ext = load_module('torchfcpe.mel_extractor', '/Users/daisy/develop/fcpe-mlxrs/FCPE/torchfcpe/mel_extractor.py')
conf = load_module('torchfcpe.model_conformer_naive', '/Users/daisy/develop/fcpe-mlxrs/FCPE/torchfcpe/model_conformer_naive.py')
sys.modules['torchfcpe.model_conformer_naive'] = conf
models = load_module('torchfcpe.models', '/Users/daisy/develop/fcpe-mlxrs/FCPE/torchfcpe/models.py')

checkpoint = torch.load('/Users/daisy/develop/fcpe-mlxrs/checkpoint/fcpe.pt', map_location='cpu')
model_cfg = checkpoint['config_dict']['model']
mel_cfg = checkpoint['config_dict']['mel']

pe_model = models.CFNaiveMelPE(
    input_channels=mel_cfg['num_mels'],
    out_dims=model_cfg['out_dims'],
    hidden_dims=model_cfg['hidden_dims'],
    n_layers=model_cfg['n_layers'],
    n_heads=model_cfg['n_heads'],
    f0_max=model_cfg['f0_max'],
    f0_min=model_cfg['f0_min'],
    use_fa_norm=model_cfg['use_fa_norm'],
    conv_only=model_cfg['conv_only'],
    conv_dropout=model_cfg['conv_dropout'],
    atten_dropout=model_cfg['atten_dropout'],
    use_harmonic_emb=model_cfg['use_harmonic_emb'],
)
pe_model.load_state_dict(checkpoint['model'])
pe_model.eval()

wav2mel = mel_ext.Wav2MelModule(
    sr=mel_cfg['sr'],
    n_mels=mel_cfg['num_mels'],
    n_fft=mel_cfg['n_fft'],
    win_size=mel_cfg['win_size'],
    hop_length=mel_cfg['hop_size'],
    fmin=mel_cfg['fmin'],
    fmax=mel_cfg['fmax'],
    clip_val=1e-5,
    mel_type='default',
)

import librosa
wav, sr = librosa.load('/Users/daisy/develop/fcpe-mlxrs/audio/huaxue.wav', sr=16000)
wav_tensor = torch.tensor(wav).unsqueeze(0).unsqueeze(-1)

mel = wav2mel(wav_tensor, sr)

np.save('/Users/daisy/develop/fcpe-mlxrs/ref_wav.npy', wav)
np.save('/Users/daisy/develop/fcpe-mlxrs/ref_mel.npy', mel.detach().numpy())

# Forward with hooks
intermediates = {}

def hook_input_stack(mod, inp, out):
    intermediates['input_stack'] = out.detach().numpy()

def hook_layer(i):
    def hook(mod, inp, out):
        intermediates[f'layer_{i}'] = out.detach().numpy()
    return hook

def hook_norm(mod, inp, out):
    intermediates['norm'] = out.detach().numpy()

def hook_output_proj(mod, inp, out):
    intermediates['output_proj'] = out.detach().numpy()

def hook_sigmoid(mod, inp, out):
    intermediates['sigmoid'] = out.detach().numpy()

pe_model.input_stack.register_forward_hook(hook_input_stack)
for i, layer in enumerate(pe_model.net.encoder_layers):
    layer.register_forward_hook(hook_layer(i))
pe_model.norm.register_forward_hook(hook_norm)
pe_model.output_proj.register_forward_hook(hook_output_proj)

latent = pe_model.forward(mel)
intermediates['latent'] = latent.detach().numpy()

cents = pe_model.latent2cents_local_decoder(latent, threshold=0.006)
f0 = pe_model.cent_to_f0(cents)

np.save('/Users/daisy/develop/fcpe-mlxrs/ref_latent.npy', latent.detach().numpy())
np.save('/Users/daisy/develop/fcpe-mlxrs/ref_cents.npy', cents.detach().numpy())
np.save('/Users/daisy/develop/fcpe-mlxrs/ref_f0.npy', f0.detach().numpy())

for k, v in intermediates.items():
    np.save(f'/Users/daisy/develop/fcpe-mlxrs/ref_{k}.npy', v)

# Save model weights
for k, v in checkpoint['model'].items():
    np.save(f'/Users/daisy/develop/fcpe-mlxrs/weight_{k.replace(".", "_")}.npy', v.detach().numpy())

# Save mel basis
mel_basis = wav2mel.mel_extractor.mel_basis.detach().numpy()
np.save('/Users/daisy/develop/fcpe-mlxrs/ref_mel_basis.npy', mel_basis)

print('Reference values saved')
print('Mel shape:', mel.shape)
print('Latent shape:', latent.shape)
print('f0 shape:', f0.shape)
