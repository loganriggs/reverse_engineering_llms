"""Frozen KL-to-joint write direction; endpoints retain exact float32 replay.

The tangent uses the realized frozen float32 products and float64 coefficient
subtraction/multiplication. Endpoint anchoring avoids changing old outputs merely
because of a different floating-point addition order.
"""
import json
import torch
from torch.nn import functional as F
from fit_pca_encoder import V, sha
from product_dictionary_featurizer import ProductDictionaryFeaturizer


class FrozenDictionaryDirection:
    def __init__(self, seed):
        entries = json.loads((V / 'product_dictionary_training_results.json').read_text())['models']
        self.models = []
        for arm in ('intact_KL', 'joint_three_corner'):
            entry = next(e for e in entries if e['seed'] == seed and e['arm'] == arm)
            assert sha(entry['path']) == entry['sha256']
            model = ProductDictionaryFeaturizer(torch.load(entry['path'], weights_only=True)['state']).cuda().eval()
            model.requires_grad_(False)
            self.models.append(model)
        assert all(torch.equal(a, b) for a, b in zip(self.models[0].buffers(), self.models[1].buffers()))

    @torch.no_grad()
    def prepare(self, z):
        a, b = self.models
        products = F.linear(z, a.left) * F.linear(z, a.right) / a.rms
        difference = b.coefficients.double() - a.coefficients.double()
        tangent = ((products.double() @ difference.T) @ a.dictionary.double().T).float()
        endpoints = [a(z)[0], b(z)[0]]
        return endpoints, tangent

    @staticmethod
    def at(endpoints, tangent, t, anchor):
        assert anchor in (0, 1)
        return endpoints[anchor] + (t - anchor) * tangent


def scalar_derivative(value, t, retain_graph=True):
    if not value.requires_grad:
        return 0.0
    gradient = torch.autograd.grad(value, t, retain_graph=retain_graph, allow_unused=True)[0]
    return 0.0 if gradient is None else float(gradient.detach())
