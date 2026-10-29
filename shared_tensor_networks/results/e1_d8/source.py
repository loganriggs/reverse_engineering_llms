"""CPU-first shared bilinear polynomial study. No external datasets/checkpoints."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

torch.set_default_dtype(torch.float64)


class Network(nn.Module):
    def __init__(self, d, h, q, tied, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.tied, self.d, self.h, self.q = tied, d, h, q
        for name, shape in [('A', (h, d)), ('B', (h, d)),
                            ('C', (q, h)), ('D', (q, h)), ('w', (q,))]:
            fan = shape[-1]
            self.register_parameter(name, nn.Parameter(torch.randn(shape, generator=g) / math.sqrt(fan)))
        if not tied:
            self.AR = nn.Parameter(torch.randn(h, d, generator=g) / math.sqrt(d))
            self.BR = nn.Parameter(torch.randn(h, d, generator=g) / math.sqrt(d))

    def forward(self, x, recompute=False):
        left = (x @ self.A.T) * (x @ self.B.T)
        if self.tied:
            right = (x @ self.A.T) * (x @ self.B.T) if recompute else left
        else:
            right = (x @ self.AR.T) * (x @ self.BR.T)
        return ((left @ self.C.T) * (right @ self.D.T)) @ self.w


def count(d, h, q, tied):
    return (2 if tied else 4) * d * h + 2 * q * h + q


def multiply(a, b):
    out = {}
    for i, x in a.items():
        for j, y in b.items():
            k = tuple(u + v for u, v in zip(i, j))
            out[k] = out.get(k, 0.) + x * y
    return out


def coefficients(model):
    """Exact monomial expansion, independent of the forward evaluation path."""
    p = {k: v.detach().cpu().numpy() for k, v in model.named_parameters()}
    basis = [tuple(int(i == j) for i in range(model.d)) for j in range(model.d)]
    def hidden(A, B):
        return [multiply(dict(zip(basis, a)), dict(zip(basis, b))) for a, b in zip(A, B)]
    left = hidden(p['A'], p['B'])
    right = left if model.tied else hidden(p['AR'], p['BR'])
    def combine(weights, polys):
        return {k: sum(w * poly.get(k, 0.) for w, poly in zip(weights, polys))
                for k in set().union(*polys)}
    out = {}
    for c, d, w in zip(p['C'], p['D'], p['w']):
        for k, value in multiply(combine(c, left), combine(d, right)).items():
            out[k] = out.get(k, 0.) + w * value
    return out


def gaussian_moment(power):
    if power % 2:
        return 0.
    return float(math.prod(range(1, power, 2)))


def gram(keys):
    return np.array([[math.prod(gaussian_moment(a + b) for a, b in zip(i, j))
                      for j in keys] for i in keys])


def check():
    torch.manual_seed(3001)
    errors = {}
    def verify(name, a, b):
        denom = float(torch.linalg.vector_norm(b.detach()))
        err = float(torch.linalg.vector_norm((a-b).detach())) / (denom if denom > 0 else 1.)
        assert err < 1e-8, (name, err)
        errors[name] = err
    x, A, B, w = torch.randn(3), torch.randn(4, 3, requires_grad=True), torch.randn(4, 3), torch.randn(4)
    y = w @ ((A @ x) * (B @ x))
    verify('single_layer_derivative', torch.autograd.grad(y, A)[0], (w * (B @ x))[:, None] * x)
    S = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [1., 0.], [0., 1.]])
    theta = torch.tensor([.6, -.7], requires_grad=True)
    def loss(z):
        return .5 * (z[0]*z[1]*z[2] + z[3]*z[4] - .8)**2
    z = (S @ theta).detach().requires_grad_()
    gz = torch.autograd.grad(loss(z), z)[0]
    gt = torch.autograd.grad(loss(S @ theta), theta)[0]
    hz = torch.autograd.functional.hessian(loss, z)
    ht = torch.autograd.functional.hessian(lambda t: loss(S @ t), theta)
    verify('gradient_pullback', gt, S.T @ gz)
    verify('hessian_pullback', ht, S.T @ hz @ S)
    G = S.T @ S
    P = S @ torch.linalg.solve(G, S.T)
    verify('projected_metric', S @ torch.linalg.solve(G, gt), P @ gz)
    assert not torch.allclose(S @ gt, P @ gz)
    verify('projector_idempotence', P @ P, P)
    model = Network(3, 3, 2, True, 74)
    inputs = torch.randn(8, 3)
    a, b = model(inputs), model(inputs, recompute=True)
    verify('DAG_formula_values', a, b)
    ga = torch.autograd.grad(a.square().mean(), tuple(model.parameters()))
    gb = torch.autograd.grad(b.square().mean(), tuple(model.parameters()))
    verify('DAG_formula_gradients', torch.cat([v.flatten() for v in ga]), torch.cat([v.flatten() for v in gb]))
    for tied in [True, False]:
        m = Network(3, 3, 2, tied, 84)
        poly = coefficients(m)
        reconstructed = sum(v * inputs.pow(torch.tensor(k)).prod(-1) for k, v in poly.items())
        verify(f'monomial_expansion_{tied}', reconstructed, m(inputs))
        verify(f'degree_four_{tied}', m(1.7 * inputs), 1.7**4 * m(inputs))
        assert sum(p.numel() for p in m.parameters()) == count(3, 3, 2, tied)
    for c in [0., 1.]:
        for t in [0., -.7, 1.2]:
            t = torch.tensor(t, requires_grad=True)
            f = .5*(t*t-c)**2
            first = torch.autograd.grad(f, t, create_graph=True)[0]
            second = torch.autograd.grad(first, t)[0]
            verify(f'scalar_gradient_{c}_{float(t.detach())}', first, 2*t*(t*t-c))
            verify(f'scalar_hessian_{c}_{float(t.detach())}', second, 6*t*t-2*c)
    # Early A occurs twice: y(alpha*A)=alpha^2*y(A), not alpha*y(A).
    base = model(inputs).detach()
    with torch.no_grad():
        model.A.mul_(1.4)
    verify('early_matrix_degree_two', model(inputs), 1.4**2 * base)
    return {'status': 'passed', 'tolerance': 1e-8, 'errors': errors}


def train(args, out):
    sizes = [int(n) for n in args.ntrain.split(',')]
    if min(sizes) < 1 or args.seeds < 1 or args.steps < 1:
        raise ValueError('Positive sample counts, seeds, and steps required')
    d, h, q = args.d, args.width, args.top_width
    budget = count(d, h, q, True)
    small = min(range(1, h+1), key=lambda width: (abs(count(d, width, q, False)-budget), width))
    rows, curves = [], []
    dev = torch.device(args.device)
    for teacher_kind in args.teacher.split(','):
        assert teacher_kind in ['tied', 'untied']
        for seed in range(args.seeds):
            teacher = Network(d, h, q, teacher_kind == 'tied', 10000+seed)
            # Exact Gaussian RMS, not an estimated test-set normalization.
            tc = coefficients(teacher)
            keys = sorted(tc)
            H = gram(keys)
            target = np.array([tc[k] for k in keys])
            rms = float(np.sqrt(target @ H @ target))
            with torch.no_grad():
                teacher.w.div_(rms)
            target /= rms
            teacher.to(dev)
            gen = torch.Generator().manual_seed(20000+seed)
            xt = torch.randn(max(sizes), d, generator=gen).to(dev)
            xv = torch.randn(1024, d, generator=gen).to(dev)
            xe = torch.randn(4096, d, generator=gen).to(dev)
            xo = torch.randn(4096, d, generator=gen).to(dev) * torch.linspace(.5, 1.5, d, device=dev)
            xs = torch.randn(4096, d, generator=gen)
            xs *= (torch.rand(4096, d, generator=gen) < .25) * 2.
            xs = xs.to(dev)
            with torch.no_grad():
                yt, yv, ye, yo, ys = [teacher(x) for x in [xt, xv, xe, xo, xs]]
            for n, (kind, width, tied) in itertools.product(sizes, [
                ('tied', h, True), ('untied_width', h, False), ('untied_budget', small, False)]):
                model = Network(d, width, q, tied, 30000+seed).to(dev)
                # Width-matched models have identical A/B/C/D/w initialization;
                # right branch is independently drawn unless matched requested.
                if args.init == 'matched' and kind == 'untied_width':
                    with torch.no_grad():
                        model.AR.copy_(model.A); model.BR.copy_(model.B)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                x, y = xt[:n], yt[:n]
                with torch.no_grad():
                    initial = model(xv)
                    init_rms = float(initial.square().mean().sqrt())
                jacnorm = 0.
                for pred in model(xv[:16]):
                    grads = torch.autograd.grad(pred, tuple(model.parameters()), retain_graph=True)
                    jacnorm += sum(float(g.square().sum()) for g in grads) / 16
                start = time.monotonic()
                status, clipped, gradnorm = 'ok', 0, float('nan')
                for step in range(args.steps):
                    optimizer.zero_grad(set_to_none=True)
                    loss = (model(x)-y).square().mean()
                    if not torch.isfinite(loss):
                        status = 'nonfinite_loss'; break
                    loss.backward()
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    gradnorm = float(gn)
                    if not math.isfinite(gradnorm):
                        status = 'nonfinite_gradient'; break
                    clipped += gradnorm > args.clip
                    optimizer.step()
                    if step % 50 == 0 or step == args.steps-1:
                        with torch.no_grad():
                            curves.append(dict(teacher=teacher_kind, seed=seed, n=n, student=kind,
                                               step=step+1, train_mse=float((model(x)-y).square().mean()),
                                               validation_mse=float((model(xv)-yv).square().mean()), gradient_norm=gradnorm))
                with torch.no_grad():
                    metrics = {name: float((model(a)-b).square().mean()) for name, a, b in [
                        ('train_mse', x, y), ('validation_mse', xv, yv), ('test_mse', xe, ye),
                        ('covariance_shift_mse', xo, yo), ('sparse_input_mse', xs, ys)]}
                mc = coefficients(model)
                delta = np.array([mc.get(k, 0.) for k in keys]) - target
                evidence = f'{teacher_kind}_{seed}_{n}_{kind}.npz'
                np.savez_compressed(out/evidence,
                    **{'student_'+k: p.detach().cpu().numpy() for k, p in model.named_parameters()},
                    **{'teacher_'+k: p.detach().cpu().numpy() for k, p in teacher.named_parameters()})
                row = dict(teacher=teacher_kind, seed=seed, n=n, student=kind, width=width,
                           top_width=q, parameters=sum(p.numel() for p in model.parameters()),
                           init=args.init, status=status, completed_steps=step+1, clipped_steps=clipped,
                           init_output_rms=init_rms, init_jacobian_squared_frobenius_per_probe=jacnorm,
                           final_gradient_norm=gradnorm, seconds=time.monotonic()-start,
                           exact_gaussian_mse=float(delta @ H @ delta),
                           coefficient_relative_error=float(np.linalg.norm(delta)/np.linalg.norm(target)),
                           evidence=evidence, **metrics)
                rows.append(row)
                write_csv(out/'results.csv', rows)
                write_csv(out/'curves.csv', curves)
                print(json.dumps(row), flush=True)
    return rows


def write_csv(path, rows):
    temp = path.with_suffix('.tmp')
    with temp.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['check', 'train'], required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--ntrain', default='256')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--teacher', default='tied,untied')
    parser.add_argument('--d', type=int, default=4)
    parser.add_argument('--width', type=int, default=6)
    parser.add_argument('--top-width', type=int, default=4)
    parser.add_argument('--lr', type=float, default=.01)
    parser.add_argument('--clip', type=float, default=10.)
    parser.add_argument('--init', choices=['independent', 'matched'], default='independent')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    meta = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                torch=torch.__version__, numpy=np.__version__, python=platform.python_version(),
                dtype='float64', threads=1)
    (args.out/'protocol.json').write_text(json.dumps(meta, indent=2)+'\n')
    checks = check()
    (args.out/'checks.json').write_text(json.dumps(checks, indent=2)+'\n')
    if args.mode == 'train':
        train(args, args.out)
    (args.out/'completed.json').write_text(json.dumps({'status': 'complete'})+'\n')
    print(json.dumps(checks))


if __name__ == '__main__':
    main()
