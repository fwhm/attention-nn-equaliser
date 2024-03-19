import torch
from positional_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, \
    PositionalEncodingPermute1D, FixEncoding
from summer import Summer
# Returns the position encoding only
p_enc_1d_model = PositionalEncoding1D(10)
p_enc_1d_permute_model = PositionalEncodingPermute1D(10)  # permute time step and channels (dimensions)

# Return the inputs with the position encoding added
p_enc_1d_model_sum = Summer(PositionalEncoding1D(10))

# Returns the same as p_enc_1d_model but saves it for later
p_enc_1d_model_fixed = FixEncoding(PositionalEncoding1D(10), (6, ))

x = torch.rand(1, 6, 10)
penc_no_sum = p_enc_1d_model(x)  # penc_no_sum.shape == (1, 6, 10)
penc_sum = p_enc_1d_model_sum(x)
penc_fixed = p_enc_1d_model_fixed(x)  # The encoding is saved for later, making subsequent forward passes faster.
print(penc_no_sum + x == penc_sum)  # True
