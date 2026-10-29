"""M97 training-only endpoint replay, independent tangent and native derivative checks."""
import json
import numpy as np
import torch
from fit_pca_encoder import V, A, sha
from product_dictionary_direction import FrozenDictionaryDirection, scalar_derivative
from bilinear_input_causal_loss import InputCausalExperiment


def logp(t):
    x = t.detach().cpu().double().numpy()
    x -= x.max()
    return x - np.log(np.exp(x).sum())


def main():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    pp = V / 'product_dictionary_direction_preflight_protocol.json'
    protocol = json.loads(pp.read_text())
    for name, digest in protocol['references'].items():
        assert sha(V / name) == digest
    e = InputCausalExperiment()
    key = json.loads((V / 'training_causal_coverage_protocol.json').read_text())['pairs'][0]['id'] + '|0'
    donor = e.donor_key(key)
    previous = json.loads((V / 'product_dictionary_final_loss_audit.json').read_text())
    assert sha(previous['samples_path']) == previous['samples_sha256']
    old_samples = torch.load(previous['samples_path'], weights_only=True)
    rows, algebra, saved = [], [], []
    maximum = 0.
    for seed in (71, 72):
        direction = FrozenDictionaryDirection(seed)
        prepared = [direction.prepare(e.input_cache[k]['input'].cuda()) for k in (key, donor)]
        # Independently compute the real-arithmetic product/coordinate tangent
        # using NumPy. The production tangent uses realized float32 products.
        a, b = direction.models
        cpu = lambda x: x.detach().cpu().double().numpy()
        for endpoint, (k, (ends, tangent)) in enumerate(zip((key, donor), prepared)):
            z = cpu(e.input_cache[k]['input'])
            h = ((z @ cpu(a.left).T) * (z @ cpu(a.right).T)) / cpu(a.rms)
            expected = (h @ (cpu(b.coefficients) - cpu(a.coefficients)).T) @ cpu(a.dictionary).T
            relative = float(np.linalg.norm(cpu(tangent) - expected) / np.linalg.norm(expected))
            assert relative <= 1e-5
            gap = float(np.linalg.norm(cpu(ends[1]) - cpu(ends[0]) - expected) / np.linalg.norm(expected))
            algebra.append({'seed': seed, 'endpoint': endpoint, 'tangent_relative_error': relative,
                            'float32_endpoint_secant_relative_error': gap})
        for anchor, arm in enumerate(('intact_KL', 'joint_three_corner')):
            t = torch.tensor(float(anchor), device='cuda', requires_grad=True)
            qr, qd = [direction.at(ends, tangent, t, anchor) for ends, tangent in prepared]
            K, B, intact, branches = e.joint_losses(qr, qd, key)
            old_row = next(x for x in previous['rows'] if x['seed'] == seed and x['arm'] == arm)
            old_sample = old_samples[previous['rows'].index(old_row)]
            assert float(K.detach()) == old_row['task_K'] and float(B.detach()) == old_row['task_B']
            assert torch.equal(intact.detach().cpu(), old_sample['intact'])
            n = logp(e.cache[key]['native'])
            independent_K = float((np.exp(n) * (n - logp(intact))).sum())
            assert np.isclose(independent_K, float(K.detach()), rtol=1e-7, atol=1e-9)
            values = []
            slopes = []
            for branch in branches:
                old = next(x for x in old_sample['branches'] if (x['region'], x['corner']) == (branch['region'], branch['corner']))
                assert torch.equal(branch['changed'].detach().cpu(), old['changed'])
                assert torch.equal(branch['base'].detach().cpu(), old['base'])
                teacher = next(x for x in e.cache[key]['targets'] if x['region'] == branch['region'])
                target = logp(teacher['native_changed']) - n
                error = logp(branch['changed']) - logp(branch['base']) - target
                independent = float((np.exp(n) * error ** 2).sum())
                maximum = max(maximum, abs(independent - float(branch['R'].detach())))
                assert np.isclose(independent, float(branch['R'].detach()), rtol=1e-7, atol=1e-9)
                slope = scalar_derivative(branch['R'], t)
                slopes.append({'region': branch['region'], 'corner': branch['corner'], 'R': float(branch['R'].detach()), 'slope': slope})
                if branch['region'] == 'identity':
                    assert independent == slope == 0
                else:
                    values.append(independent / e.den[e.cache[key]['family'], branch['region']])
            assert np.isclose(np.mean(values), float(B.detach()), rtol=1e-7, atol=1e-9)
            ad = scalar_derivative(K + B, t)
            k_slope = scalar_derivative(K, t)
            expected_slope = k_slope + sum(x['slope'] / e.den[e.cache[key]['family'], x['region']] for x in slopes if x['region'] != 'identity') / len(values)
            assert abs(ad - expected_slope) <= 1e-7 + 1e-5 * abs(ad)
            steps = []
            with torch.no_grad():
                for step in protocol['finite_difference_steps']:
                    losses = []
                    for sign in (-1, 1):
                        probe = torch.tensor(anchor + sign * step, device='cuda')
                        pair = [direction.at(ends, tangent, probe, anchor) for ends, tangent in prepared]
                        k, b, _, _ = e.joint_losses(*pair, key)
                        losses.append(float(k + b))
                    fd = (losses[1] - losses[0]) / (2 * step)
                    steps.append({'step': step, 'finite_difference': fd, 'absolute_error': abs(fd - ad),
                                  'passed': abs(fd - ad) <= .001 + .05 * abs(ad)})
            passed = sum(x['passed'] for x in steps) >= 2
            rows.append({'seed': seed, 'anchor': anchor, 'arm': arm, 'key': key, 'K': float(K.detach()),
                         'B': float(B.detach()), 'K_slope': k_slope, 'K_plus_B_slope': ad,
                         'branches': slopes, 'steps': steps, 'passed': passed})
            saved.append({'row': len(rows) - 1, 'intact': intact.detach().cpu(),
                          'branches': [{'region': x['region'], 'corner': x['corner'],
                                        'base': x['base'].detach().cpu(), 'changed': x['changed'].detach().cpu()} for x in branches]})
            assert all(p.grad is None for p in e.model.parameters())
            print('DIRECTION_CHECK', seed, anchor, passed, flush=True)
            del qr, qd, K, B, branches, intact
    sample_path = A / 'emlp_m97_direction_preflight_samples.pt'
    assert not sample_path.exists()
    torch.save(saved, sample_path)
    result = {'status': 'passed' if all(r['passed'] for r in rows) else 'unresolved',
              'source_sha256': sha(__file__), 'runtime_sha256': sha(V.parent / 'product_dictionary_direction.py'),
              'loss_source_sha256': sha(V.parent / 'bilinear_input_causal_loss.py'), 'protocol_sha256': sha(pp),
              'rows': rows, 'algebra': algebra, 'max_metric_absolute_error': maximum, 'forward_count': e.calls,
              'samples_path': str(sample_path), 'samples_sha256': sha(sample_path),
              'scope': 'Four fixed training endpoint directions, twelve finite-difference steps, exact endpoint replay and independent tangent/loss arithmetic. Existing derivative tolerance retained; small slopes below absolute tolerance are not certified to high relative accuracy. No optimization or heldout scoring.'}
    path = V / 'product_dictionary_direction_preflight.json'
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'], e.calls, flush=True)


if __name__ == '__main__':
    main()
