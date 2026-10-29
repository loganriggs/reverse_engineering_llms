"""M95 evaluation adapter: normalized input only, native bias added by caller."""
import torch
from product_dictionary_featurizer import ProductDictionaryFeaturizer
from fit_pca_encoder import sha


class ProductDictionaryPredictor:
    def __init__(self, entry):
        assert sha(entry['path']) == entry['sha256']
        self.model = ProductDictionaryFeaturizer(torch.load(entry['path'], weights_only=True)['state']).cuda().eval()
        self.kind = entry['kind']
        self.variant = 'input'

    @torch.no_grad()
    def reconstruct(self, z, write=None):
        prediction, support, coordinates = self.model(z)
        return prediction, support, coordinates, {
            'rank_mean': None,
            'numerical_nonzero_mean': float((coordinates != 0).sum(-1).float().mean()),
            'retained_native_products': 4608,
            'output_dimension': 128,
        }
