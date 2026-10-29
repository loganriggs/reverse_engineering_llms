"""M95 guarded native bypass and independent causal-loss arithmetic, no fitting."""
import json
import numpy as np
import torch
from product_dictionary_featurizer import ProductDictionaryFeaturizer
from bilinear_input_causal_loss import InputCausalExperiment
from inspect_model import trace
from fit_pca_encoder import A, V, sha


def lp(t):
    x = t.detach().cpu().double().numpy()
    x -= x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def main():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    manifest_path = V / 'product_dictionary_initial_states.json'
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads((V / 'product_dictionary_initialization_audit.json').read_text())
    assert audit['status'] == 'passed' and audit['manifest_sha256'] == sha(manifest_path)
    assert manifest['runtime_sha256'] == sha(V.parent / 'product_dictionary_featurizer.py')
    protocol_path = V / 'product_dictionary_native_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    for name, digest in protocol['references'].items():
        assert sha(V / name) == digest
    e = InputCausalExperiment()
    pairs = json.loads((V / 'training_causal_coverage_protocol.json').read_text())['pairs']
    pair_ids = [pairs[0]['id'], next(p['id'] for p in pairs if not p['equal_length'])]
    rows, samples, guards = [], [], []
    maximum = 0

    def forbidden(*args):
        raise AssertionError('Native MLP1 product or down module was called')

    for entry in manifest['models']:
        assert sha(entry['path']) == entry['sha256']
        f = ProductDictionaryFeaturizer(torch.load(entry['path'], weights_only=True)['state']).cuda()
        for pid in pair_ids:
            for endpoint in (0, 1):
                key = f'{pid}|{endpoint}'
                donor = e.donor_key(key)
                f.zero_grad(set_to_none=True)
                qr = f(e.input_cache[key]['input'].cuda())[0]
                qd = f(e.input_cache[donor]['input'].cuda())[0]
                K, B, intact, branches = e.joint_losses(qr, qd, key)
                c = e.cache[key]
                native_lp = lp(c['native'])
                independent_K = float(np.sum(np.exp(native_lp) * (native_lp - lp(intact))))
                assert np.isclose(independent_K, float(K.detach()), rtol=1e-7, atol=1e-9)
                normalized = []
                for branch in branches:
                    target = next(t for t in c['targets'] if t['region'] == branch['region'])
                    teacher = lp(target['native_changed']) - native_lp
                    difference = lp(branch['changed']) - lp(branch['base']) - teacher
                    value = float(np.sum(np.exp(native_lp) * difference ** 2))
                    maximum = max(maximum, abs(value - float(branch['R'].detach())))
                    assert np.isclose(value, float(branch['R'].detach()), rtol=1e-7, atol=1e-9)
                    if branch['region'] == 'identity':
                        assert value == 0 and not branch['delta'].count_nonzero()
                    else:
                        normalized.append(value / e.den[c['family'], branch['region']])
                assert np.isclose(np.mean(normalized), float(B.detach()), rtol=1e-7, atol=1e-9)
                gradients = torch.autograd.grad(K + B, (qr, qd, f.coefficients))
                assert all(torch.isfinite(g).all() and g.norm() > 0 for g in gradients)
                assert all(p.grad is None for p in e.model.parameters())
                rows.append({'seed': entry['seed'], 'key': key, 'branches': len(branches),
                             'K': float(K.detach()), 'B': float(B.detach()),
                             'gradient_norms': [float(g.norm()) for g in gradients]})
                samples.append({'row': len(rows) - 1, 'intact': intact.detach().cpu(),
                                'branches': [{'region': b['region'], 'corner': b['corner'],
                                              'base': b['base'].detach().cpu(),
                                              'changed': b['changed'].detach().cpu()} for b in branches]})
                # Compare callback replacement to an actual forward override while
                # native product modules are guarded. The predictor sees only z.
                if pid == pair_ids[0]:
                    original = e.mlp.forward
                    handles = [m.register_forward_pre_hook(forbidden) for m in (e.mlp.Left, e.mlp.Right, e.mlp.Down)]
                    try:
                        e.mlp.forward = lambda z: f(z)[0] + e.mlp.Down_bias
                        with torch.no_grad():
                            out, _ = trace(e.model, c['ids'].cuda(), layers=(0, 2))
                        e.calls += 1
                    finally:
                        e.mlp.forward = original
                        for handle in handles:
                            handle.remove()
                    error = float((out[0, -1] - intact.detach()).abs().max())
                    assert error == 0
                    guards.append({'seed': entry['seed'], 'key': key, 'max_logit_difference': error})
                del qr, qd, K, B, intact, branches, gradients
        print('CHECKED', entry['seed'], flush=True)
    teacher_max = 0
    with torch.no_grad():
        for pid in pair_ids:
            for endpoint in (0, 1):
                key = f'{pid}|{endpoint}'
                K, B, _, branches = e.joint_losses(e.cache[key]['write'].cuda(), e.cache[e.donor_key(key)]['write'].cuda(), key)
                error = max(float(K), max(float(b['R']) for b in branches))
                assert error < 1e-9
                teacher_max = max(teacher_max, error)
    path = A / 'emlp_m95_native_preflight_samples.pt'
    assert not path.exists()
    torch.save(samples, path)
    result = {'status': 'passed', 'source_sha256': sha(__file__), 'protocol_sha256': sha(protocol_path),
              'manifest_sha256': sha(manifest_path), 'loss_source_sha256': sha(V.parent / 'bilinear_input_causal_loss.py'),
              'runtime_sha256': manifest['runtime_sha256'], 'rows': rows, 'native_bypass_checks': guards,
              'max_response_arithmetic_error': maximum, 'max_teacher_recovery_loss': teacher_max,
              'samples_path': str(path), 'samples_sha256': sha(path), 'forward_count': e.calls,
              'scope': '8 fixed training cases, all branches and 4 guarded forward overrides. Finite-difference checks and training-only calibration still required; no optimizer updates.'}
    out = V / 'product_dictionary_native_checks.json'
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2) + '\n')
    print('PASSED', len(rows), e.calls, maximum, teacher_max, flush=True)


if __name__ == '__main__':
    main()
