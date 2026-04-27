import numpy as np
import torch

x = torch.randn(1, 2048, 100)
glu = torch.nn.GLU(dim=1)
out = glu(x)

np.save('/Users/daisy/develop/fcpe-mlxrs/test_glu_input_ncl.npy', x.detach().numpy())
np.save('/Users/daisy/develop/fcpe-mlxrs/test_glu_input_nlc.npy', x.transpose(1, 2).detach().numpy())
np.save('/Users/daisy/develop/fcpe-mlxrs/test_glu_output.npy', out.detach().numpy())

print("Input shape:", x.shape)
print("Output shape:", out.shape)
