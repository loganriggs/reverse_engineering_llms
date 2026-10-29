"""Independent float64 NumPy evaluation of frozen product/coordinate matrices."""
import json
import numpy as np
import torch
from fit_pca_encoder import V, sha


class IndependentDictionaryPrediction:
    def __init__(self, entry):
        self.entry = entry
        assert sha(entry['path']) == entry['sha256']
        state = torch.load(entry['path'], weights_only=True, map_location='cpu')['state']
        ref = json.loads((V / 'native_input_reference.json').read_text())
        assert sha(ref['path']) == ref['sha256']
        native = torch.load(ref['path'], weights_only=True, map_location='cpu')
        assert torch.equal(state['left'], native['L']) and torch.equal(state['right'], native['R'])
        self.left, self.right, self.rms, self.B, self.C, self.offset = (
            state[k].double().numpy() for k in ('left', 'right', 'rms', 'dictionary', 'coefficients', 'offset'))

    def prepare(self, z):
        x = z.detach().cpu().double().numpy()
        return (x @ self.left.T) * (x @ self.right.T)

    def verify(self, products, prediction, support, code):
        expected_code = (products / self.rms) @ self.C.T
        expected = expected_code @ self.B.T + self.offset
        actual = prediction.detach().cpu().double().numpy()
        coordinates = code.detach().cpu().double().numpy()
        active = support.detach().cpu().numpy()
        assert active.shape == expected_code.shape and np.all(active == 1)
        relative = float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30))
        code_relative = float(np.linalg.norm(coordinates - expected_code) / max(np.linalg.norm(expected_code), 1e-30))
        assert relative <= 1e-5 and code_relative <= 1e-5, (relative, code_relative)
        return relative, code_relative
