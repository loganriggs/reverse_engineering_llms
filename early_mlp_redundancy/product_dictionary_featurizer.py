"""M95 input-only coefficients over frozen native bilinear products.

Output excludes native Down_bias, which the intervention caller adds once.
The 128 output coordinates do not imply only 128 input computations.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ProductDictionaryFeaturizer(nn.Module):
    def __init__(self, state):
        super().__init__()
        for name in ('left', 'right', 'rms', 'dictionary', 'offset'):
            self.register_buffer(name, state[name].detach().clone())
        self.coefficients = nn.Parameter(state['coefficients'].detach().clone())
        assert self.left.shape == self.right.shape == (4608, 1152)
        assert self.rms.shape == (4608,) and (self.rms > 0).all()
        assert self.dictionary.shape == (1152, 128)
        assert self.offset.shape == (1152,)
        assert self.coefficients.shape == (128, 4608)

    def forward(self, z):
        products = F.linear(z, self.left) * F.linear(z, self.right)
        coordinates = F.linear(products / self.rms, self.coefficients)
        prediction = F.linear(coordinates, self.dictionary) + self.offset
        return prediction, torch.ones_like(coordinates), coordinates
