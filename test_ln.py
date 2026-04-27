import numpy as np
import torch

w = np.load('/Users/daisy/develop/fcpe-mlxrs/weight_net_encoder_layers_0_conformer_net_0_weight.npy')
b = np.load('/Users/daisy/develop/fcpe-mlxrs/weight_net_encoder_layers_0_conformer_net_0_bias.npy')

ln = torch.nn.LayerNorm(512)
ln.weight.data = torch.from_numpy(w).float()
ln.bias.data = torch.from_numpy(b).float()
ln.eval()

x = torch.randn(1, 100, 512)
out = ln(x)

np.save('/Users/daisy/develop/fcpe-mlxrs/test_ln_input.npy', x.detach().numpy())
np.save('/Users/daisy/develop/fcpe-mlxrs/test_ln_output.npy', out.detach().numpy())

print("Input shape:", x.shape)
print("Output shape:", out.shape)
