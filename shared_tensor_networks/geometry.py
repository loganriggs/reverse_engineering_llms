"""E2 bounded metric control: plain SGD, same initial function, one fixed teacher."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiment import Network, coefficients, gram, write_csv


def spectrum(model, x):
    jac = []
    for y in model(x):
        grads = torch.autograd.grad(y, tuple(model.parameters()), retain_graph=True)
        jac.append(torch.cat([g.flatten() for g in grads]))
    sv = torch.linalg.svdvals(torch.stack(jac)).detach().numpy()
    return dict(jacobian_singular_values=sv.tolist(), ntk_eigenvalues=(sv**2/len(x)).tolist())


def main():
    torch.set_num_threads(1)
    out = Path(__file__).parent/'results/e2_metric'
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(steps=300, learning_rate=.001, samples=4096, seeds=list(range(5)),
                    d=4, width=6, top_width=4, teacher_seed=10000, optimizer='plain full-batch SGD; no clipping',
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    dependency_sha256=hashlib.sha256((Path(__file__).parent/'experiment.py').read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    teacher = Network(4, 6, 4, True, 10000)
    coeff = coefficients(teacher); keys = sorted(coeff); H = gram(keys)
    target = np.array([coeff[k] for k in keys]); scale = np.sqrt(target @ H @ target)
    with torch.no_grad(): teacher.w.div_(scale)
    target /= scale
    x = torch.randn(4096, 4, generator=torch.Generator().manual_seed(50000))
    y = teacher(x).detach()
    rows, results = [], []
    for seed in range(5):
        tied = Network(4, 6, 4, True, 60000+seed)
        untied = Network(4, 6, 4, False, 60000+seed)
        with torch.no_grad():
            untied.AR.copy_(untied.A); untied.BR.copy_(untied.B)
        models = {'tied_euclidean': tied, 'tied_metric': copy.deepcopy(tied),
                  'occurrence_projected': copy.deepcopy(untied), 'occurrence_free': untied}
        initial_spectra = {name:spectrum(model, x[:16]) for name,model in models.items()}
        initial = tied(x).detach()
        for model in models.values(): torch.testing.assert_close(model(x), initial, rtol=0, atol=0)
        max_difference = 0.
        for step in range(300):
            for name, model in models.items():
                model.zero_grad(set_to_none=True)
                loss = (model(x)-y).square().mean()
                if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss: {seed} {name} {step}')
                loss.backward()
                with torch.no_grad():
                    for key, p in model.named_parameters():
                        factor = .5 if name == 'tied_metric' and key in ['A', 'B'] else 1.
                        p.add_(p.grad, alpha=-.001*factor)
                    if name == 'occurrence_projected':
                        for a, b in [(model.A, model.AR), (model.B, model.BR)]:
                            mean = (a+b)/2; a.copy_(mean); b.copy_(mean)
                if step % 25 == 0 or step == 299:
                    with torch.no_grad():
                        rows.append(dict(seed=seed, step=step+1, model=name, train_mse=float((model(x)-y).square().mean())))
            for name, param in models['tied_metric'].named_parameters():
                other = dict(models['occurrence_projected'].named_parameters())[name]
                difference = float((param-other).detach().abs().max())
                max_difference = max(max_difference, difference)
                assert difference < 1e-10
        for name, model in models.items():
            pc = coefficients(model); diff = np.array([pc[k] for k in keys])-target
            results.append(dict(seed=seed, model=name, exact_gaussian_mse=float(diff@H@diff),
                                max_projected_equivalence_parameter_error=max_difference,
                                initial_spectrum=initial_spectra[name], final_spectrum=spectrum(model, x[:16])))
        print('Completed seed', seed, 'max equivalence error', max_difference, flush=True)
    write_csv(out/'curves.csv', rows)
    (out/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    (out/'completed.json').write_text(json.dumps({'status':'complete'})+'\n')


if __name__ == '__main__': main()
