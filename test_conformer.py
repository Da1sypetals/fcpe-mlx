import sys
sys.path.insert(0, '/Users/daisy/develop/fcpe-mlxrs/FCPE')
import importlib.util
conf = importlib.util.spec_from_file_location('torchfcpe.model_conformer_naive', '/Users/daisy/develop/fcpe-mlxrs/FCPE/torchfcpe/model_conformer_naive.py').loader.load_module()

import torch
import numpy as np

# Create a single layer
layer = conf.CFNEncoderLayer(512, 8, use_norm=False, conv_only=True, conv_dropout=0.0, atten_dropout=0.0)
layer.eval()

# Load checkpoint weights
checkpoint = torch.load('/Users/daisy/develop/fcpe-mlxrs/checkpoint/fcpe.pt', map_location='cpu')
state_dict = checkpoint['model']

# Filter weights for layer 0
layer_state = {k.replace('net.encoder_layers.0.', ''): v for k, v in state_dict.items() if k.startswith('net.encoder_layers.0.')}
layer.load_state_dict(layer_state)

x = torch.randn(1, 100, 512)
with torch.no_grad():
    out = layer(x)

np.save('/Users/daisy/develop/fcpe-mlxrs/test_conformer_input.npy', x.detach().numpy())
np.save('/Users/daisy/develop/fcpe-mlxrs/test_conformer_output.npy', out.detach().numpy())

print("Input shape:", x.shape)
print("Output shape:", out.shape)
print("Output first 5:", out[0, 0, :5].detach().numpy())
