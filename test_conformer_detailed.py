import sys
sys.path.insert(0, '/Users/daisy/develop/fcpe-mlxrs/FCPE')
import importlib.util
conf = importlib.util.spec_from_file_location('torchfcpe.model_conformer_naive', '/Users/daisy/develop/fcpe-mlxrs/FCPE/torchfcpe/model_conformer_naive.py').loader.load_module()

import torch
import numpy as np

layer = conf.CFNEncoderLayer(512, 8, use_norm=False, conv_only=True, conv_dropout=0.0, atten_dropout=0.0)
layer.eval()

checkpoint = torch.load('/Users/daisy/develop/fcpe-mlxrs/checkpoint/fcpe.pt', map_location='cpu')
state_dict = checkpoint['model']
layer_state = {k.replace('net.encoder_layers.0.', ''): v for k, v in state_dict.items() if k.startswith('net.encoder_layers.0.')}
layer.load_state_dict(layer_state)

x = torch.randn(1, 100, 512)

# Hook conformer internals
intermediates = {}

def hook_fn(name):
    def hook(mod, inp, out):
        intermediates[name] = out.detach().numpy()
    return hook

# Register hooks on each sub-module
layer.conformer.net[0].register_forward_hook(hook_fn('norm'))
layer.conformer.net[2].register_forward_hook(hook_fn('conv1'))
layer.conformer.net[3].register_forward_hook(hook_fn('glu'))
layer.conformer.net[4].register_forward_hook(hook_fn('dwconv'))
layer.conformer.net[5].register_forward_hook(hook_fn('silu'))
layer.conformer.net[6].register_forward_hook(hook_fn('conv2'))

with torch.no_grad():
    out = layer(x)

np.save('/Users/daisy/develop/fcpe-mlxrs/test_conformer_detailed_input.npy', x.detach().numpy())
np.save('/Users/daisy/develop/fcpe-mlxrs/test_conformer_detailed_output.npy', out.detach().numpy())

for k, v in intermediates.items():
    np.save(f'/Users/daisy/develop/fcpe-mlxrs/test_conformer_detailed_{k}.npy', v)
    print(f"{k}: shape={v.shape}, mean={v.mean():.6f}, std={v.std():.6f}")
