import numpy as np
import torch

# Load weights
w = np.load('/Users/daisy/develop/fcpe-mlxrs/weight_input_stack_0_weight.npy')
b = np.load('/Users/daisy/develop/fcpe-mlxrs/weight_input_stack_0_bias.npy')

conv = torch.nn.Conv1d(128, 512, 3, stride=1, padding=1)
conv.weight.data = torch.from_numpy(w).float()
conv.bias.data = torch.from_numpy(b).float()
conv.eval()

# Create test input
x = torch.randn(1, 6201, 128)

# PyTorch way (NLC -> NCL -> conv -> NLC)
out_pytorch = conv(x.transpose(1, 2)).transpose(1, 2)

np.save('/Users/daisy/develop/fcpe-mlxrs/test_conv1d_input.npy', x.detach().numpy())
np.save('/Users/daisy/develop/fcpe-mlxrs/test_conv1d_output.npy', out_pytorch.detach().numpy())

print("Input shape:", x.shape)
print("Output shape:", out_pytorch.shape)
print("Output first 5:", out_pytorch[0, 0, :5].detach().numpy())
