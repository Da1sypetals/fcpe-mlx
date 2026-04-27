import numpy as np
import torch

# Load weights
w = np.load('/Users/daisy/develop/fcpe-mlxrs/weight_net_encoder_layers_0_conformer_net_4_conv_weight.npy')
b = np.load('/Users/daisy/develop/fcpe-mlxrs/weight_net_encoder_layers_0_conformer_net_4_conv_bias.npy')

dwconv = torch.nn.Conv1d(1024, 1024, 31, padding=15, groups=1024)
dwconv.weight.data = torch.from_numpy(w).float()
dwconv.bias.data = torch.from_numpy(b).float()
dwconv.eval()

# Create test input: NCL format for PyTorch
x = torch.randn(1, 1024, 100)
out_pytorch = dwconv(x)

np.save('/Users/daisy/develop/fcpe-mlxrs/test_dwconv_input_ncl.npy', x.detach().numpy())
np.save('/Users/daisy/develop/fcpe-mlxrs/test_dwconv_input_nlc.npy', x.transpose(1, 2).detach().numpy())
np.save('/Users/daisy/develop/fcpe-mlxrs/test_dwconv_output.npy', out_pytorch.detach().numpy())

print("Input shape:", x.shape)
print("Output shape:", out_pytorch.shape)
print("Output first 5:", out_pytorch[0, :5, 0].detach().numpy())
