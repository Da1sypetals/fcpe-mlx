import numpy as np
import torch

# Load weights
w = np.load('/Users/daisy/develop/fcpe-mlxrs/weight_input_stack_1_weight.npy')
b = np.load('/Users/daisy/develop/fcpe-mlxrs/weight_input_stack_1_bias.npy')

gn = torch.nn.GroupNorm(4, 512)
gn.weight.data = torch.from_numpy(w).float()
gn.bias.data = torch.from_numpy(b).float()
gn.eval()

# Create test input: NCL format for PyTorch
x = torch.randn(1, 512, 6201)
out_pytorch = gn(x)

# Save both NCL and NLC for mlx testing
np.save('/Users/daisy/develop/fcpe-mlxrs/test_gn_input_ncl.npy', x.detach().numpy())
np.save('/Users/daisy/develop/fcpe-mlxrs/test_gn_input_nlc.npy', x.transpose(1, 2).detach().numpy())
np.save('/Users/daisy/develop/fcpe-mlxrs/test_gn_output.npy', out_pytorch.detach().numpy())

print("Input shape:", x.shape)
print("Output shape:", out_pytorch.shape)
print("Output first 5:", out_pytorch[0, :5, 0].detach().numpy())
