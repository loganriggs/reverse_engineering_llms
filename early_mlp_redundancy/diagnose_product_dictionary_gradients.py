"""One bounded M95 diagnostic: analytic linear pullback and finite-step resolution."""
import json
import torch
from fit_pca_encoder import A, V, sha
from product_dictionary_featurizer import ProductDictionaryFeaturizer
from bilinear_input_causal_loss import InputCausalExperiment


def main():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    protocol_path = V / 'product_dictionary_gradient_diagnostic_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    for name, digest in protocol['references'].items():
        assert sha(V / name) == digest
    e = InputCausalExperiment()
    manifest = json.loads((V / 'product_dictionary_initial_states.json').read_text())
    key = json.loads((V / 'training_causal_coverage_protocol.json').read_text())['pairs'][0]['id'] + '|0'
    donor = e.donor_key(key)
    zs = [e.input_cache[k]['input'].cuda() for k in (key, donor)]
    records = []
    for entry in manifest['models']:
        assert sha(entry['path']) == entry['sha256']
        f = ProductDictionaryFeaturizer(torch.load(entry['path'], weights_only=True)['state']).cuda()
        initial = f.coefficients.detach().clone()
        q = [f(z)[0] for z in zs]
        K, B, _, _ = e.joint_losses(*q, key)
        gr, gd, gc = torch.autograd.grad(K + B, (*q, f.coefficients))
        # Closed-form pullback for q = B C H + offset; use float64 to
        # separate algebraic derivative agreement from float32 perturbations.
        features = [((z @ f.left.T) * (z @ f.right.T) / f.rms).detach() for z in zs]
        analytic = sum((g.double().reshape(-1, 1152) @ f.dictionary.double()).T @ h.double().reshape(-1, 4608)
                       for g, h in zip((gr, gd), features))
        pullback_error = float((gc.double() - analytic).norm() / analytic.norm())
        assert pullback_error <= 1e-5
        direction = gc.detach() / gc.norm()
        ad = float((gc * direction).sum())
        tangents = [(h.double() @ direction.double().T @ f.dictionary.double().T).float() for h in features]
        base = [x.detach() for x in q]
        rows = []
        with torch.no_grad():
            for step in protocol['steps']:
                for mode in ('coefficient_perturbation', 'exact_write_tangent'):
                    values, write_errors, realized_steps = [], [], []
                    for sign in (-1, 1):
                        if mode == 'coefficient_perturbation':
                            f.coefficients.copy_(initial + sign * step * direction)
                            args = [f(z)[0] for z in zs]
                            realized = (f.coefficients - initial).double()
                            realized_steps.append(float((realized - sign * step * direction.double()).norm() / step))
                            write_errors.append(float(torch.sqrt(sum(((a - b).double() - sign * step * t.double()).square().sum() for a, b, t in zip(args, base, tangents))) /
                                                      (step * torch.sqrt(sum(t.double().square().sum() for t in tangents)))))
                        else:
                            args = [b + sign * step * t for b, t in zip(base, tangents)]
                        k, b, _, _ = e.joint_losses(*args, key)
                        values.append(float(k + b))
                    f.coefficients.copy_(initial)
                    fd = (values[1] - values[0]) / (2 * step)
                    rows.append({'step': step, 'mode': mode, 'finite_difference': fd,
                                 'absolute_error': abs(fd - ad), 'agreement': abs(fd - ad) <= .001 + .05 * abs(ad),
                                 'relative_write_perturbation_errors': write_errors,
                                 'relative_coefficient_step_errors': realized_steps})
        modes_pass = {mode: sum(row['agreement'] for row in rows if row['mode'] == mode and row['step'] in protocol['acceptance_steps']) >= 2
                      for mode in ('coefficient_perturbation', 'exact_write_tangent')}
        records.append({'seed': entry['seed'], 'autograd': ad, 'analytic_pullback_relative_error': pullback_error,
                        'rows': rows, 'modes_pass': modes_pass})
        print('DIAGNOSED', entry['seed'], modes_pass, flush=True)
    result = {'status': 'passed' if all(all(r['modes_pass'].values()) for r in records) else 'unresolved',
              'source_sha256': sha(__file__), 'protocol_sha256': sha(protocol_path),
              'runtime_sha256': sha(V.parent / 'product_dictionary_featurizer.py'),
              'loss_source_sha256': sha(V.parent / 'bilinear_input_causal_loss.py'),
              'records': records, 'forward_count': e.calls,
              'scope': 'Same two initial states, same first training pair and gradient directions. Original small-step failures remain recorded. No optimizer updates or tolerance changes; this tests numerical derivative resolution, not training stability.'}
    out = V / 'product_dictionary_gradient_diagnostic.json'
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'], e.calls, flush=True)


if __name__ == '__main__':
    main()
