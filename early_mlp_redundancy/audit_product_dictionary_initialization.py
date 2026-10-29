"""Independent Torch/QR audit of NumPy/SVD M95 initialization; train data only."""
import json
import torch
from fit_pca_encoder import A, V, sha
from product_dictionary_featurizer import ProductDictionaryFeaturizer


def main():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    manifest_path = V / 'product_dictionary_initial_states.json'
    manifest = json.loads(manifest_path.read_text())
    assert sha(V / 'product_dictionary_featurizer_design.json') == manifest['design_sha256']
    assert sha(V.parent / 'product_dictionary_featurizer.py') == manifest['runtime_sha256']
    assert sha(V.parent / 'initialize_product_dictionary.py') == manifest['source_sha256']
    ref = json.loads((V / 'native_input_reference.json').read_text())
    assert sha(ref['path']) == ref['sha256'] == manifest['reference_sha256']
    r = torch.load(ref['path'], weights_only=True, map_location='cpu')
    z = r['z'][r['indices']].double()
    h = (z @ r['L'].double().T) * (z @ r['R'].double().T)
    rms = h.square().mean(0).sqrt()
    zero = rms == 0
    rms[zero] = 1
    assert sha(manifest['statistics_path']) == manifest['statistics_sha256']
    statistics = torch.load(manifest['statistics_path'], weights_only=True)
    assert torch.equal(statistics['indices'], r['indices'])
    assert torch.equal(statistics['zero_rms'], zero)
    torch.testing.assert_close(statistics['rms'], rms, rtol=1e-12, atol=1e-12)
    rows = []
    for entry in manifest['models']:
        assert sha(entry['path']) == entry['sha256']
        assert sha(entry['source']['path']) == entry['source']['sha256']
        source = torch.load(entry['source']['path'], weights_only=True)['state']
        bundle = torch.load(entry['path'], weights_only=True)
        B = source['features.dictionary'].double()
        B = B / B.norm(dim=0)
        Q = torch.linalg.qr(B, mode='reduced').Q
        center = source['output_mean'].double() + source['output_scale'].double() * source['features.bias'].double()
        native = h @ r['D'].double().T
        target = ((native - center) @ Q) @ Q.T + center
        errors = {}
        for dtype, key in [(torch.float64, 'float64_state'), (torch.float32, 'state')]:
            f = ProductDictionaryFeaturizer(bundle[key])
            assert [(n, p.numel()) for n, p in f.named_parameters()] == [('coefficients', 589824)]
            assert all(not b.requires_grad for b in f.buffers())
            torch.testing.assert_close(f.dictionary.double(), B, rtol=1e-6, atol=1e-8)
            with torch.no_grad():
                actual, support, coordinates = f(z.to(dtype))
            relative = float((actual.double() - target).norm() / target.norm())
            assert relative <= (1e-10 if dtype == torch.float64 else 1e-5)
            assert torch.equal(support, torch.ones_like(coordinates))
            assert torch.isfinite(actual).all()
            errors[key] = relative
        rows.append({'seed': entry['seed'], 'relative_projection_errors': errors,
                     'training_positions': len(z), 'trainable_parameters': 589824})
    result = {'status': 'passed', 'scope': 'Initialization, normalization, trainable-state audit only; native bypass, loss and gradient preflight still required before fitting.',
              'source_sha256': sha(__file__), 'manifest_sha256': sha(manifest_path),
              'zero_rms_count': int(zero.sum()), 'rows': rows}
    out = V / 'product_dictionary_initialization_audit.json'
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
