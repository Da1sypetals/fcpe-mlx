import numpy as np
import torch

# Simple rfft test
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
r = torch.fft.rfft(x, n=4)
print("PyTorch rfft:", r)

# Save for Rust test
np.save('/Users/daisy/develop/fcpe-mlxrs/test_rfft_input.npy', x.numpy())
np.save('/Users/daisy/develop/fcpe-mlxrs/test_rfft_output.npy', r.numpy())
