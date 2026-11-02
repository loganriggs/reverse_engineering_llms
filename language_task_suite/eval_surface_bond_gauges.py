"""Compare exact local bond gauges on the readable mixed-rule language."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from surface_mixture_language import SurfaceMixtureLanguage
from surface_mixture_v2 import SurfaceMixtureV2
from train_surface_mixture import make_batch, make_eval


def bases_for(activations, weight):
    d = activations.shape[-1]
    z = activations.reshape(-1, d).double()
    _, u = torch.linalg.eigh(z.T @ z)
    _, _, vh = torch.linalg.svd(weight.double())
    random, _ = np.linalg.qr(np.random.default_rng(6511).normal(size=(d, d)))
    return dict(physical=torch.eye(d),
                random=torch.tensor(random, dtype=torch.float32),
                activation_pca=torch.flip(u, (1,)).float(),
                weight_svd=vh.T.float())


def scores(logits, data, reference_logp):
    logp = F.log_softmax(logits.float(), -1)
    q = data['q']
    entropy = -(q*q.clamp_min(1e-30).log()).sum(-1)
    kl = -(q*logp).sum(-1)-entropy
    model_kl = (reference_logp.exp()*(reference_logp-logp)).sum(-1)
    masks = {'all': torch.ones_like(data['verb']),
             'verb': data['verb'],
             'verb_with_induction': data['verb'] & data['induction'],
             'verb_without_induction': data['verb'] & ~data['induction']}
    return {name: dict(n=int(mask.sum()), oracle_kl=float(kl[mask].mean()),
                       model_to_truncated_kl=float(model_kl[mask].mean()))
            for name, mask in masks.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', default='language_task_suite/surface_mixture_models_v1')
    p.add_argument('--generator', choices=['v1', 'v2'], default='v1')
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--layer', type=int, default=0)
    p.add_argument('--nfit', type=int, default=256)
    p.add_argument('--ntest', type=int, default=256)
    p.add_argument('--length', type=int, default=70)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    torch.set_num_threads(4)
    generator_class = SurfaceMixtureLanguage if a.generator == 'v1' else SurfaceMixtureV2
    saved = json.loads((Path(a.model_dir)/'results.json').read_text())
    record = next(row for row in saved['records'] if row['depth'] == a.depth)
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(Path(a.model_dir)/f'model_depth{a.depth}.pt',
                                     weights_only=True, map_location='cpu'))
    model.eval()
    bond = model.blocks[a.layer].attn.out
    xfit, _ = make_batch(generator_class(seed=8501), a.nfit,
                         a.length, 'cpu')
    test = make_eval(generator_class(seed=8502), a.ntest,
                     a.length, 'cpu')
    captured = []
    def collect(_, inputs):
        captured.append(inputs[0].detach())
    hook = bond.register_forward_pre_hook(collect)
    with torch.no_grad():
        for batch in xfit.split(32):
            model(batch)
    hook.remove()
    bases = bases_for(torch.cat(captured), bond.weight.detach())
    with torch.no_grad():
        reference_logits = torch.cat([model(batch)[0].float()
                                      for batch in test['x'].split(32)])
    reference_logp = F.log_softmax(reference_logits, -1)
    widths = sorted(set([4, 8, 16, 32, 48, model.config.n_embd]))
    rows = []
    for name, basis in bases.items():
        assert torch.linalg.matrix_norm(basis.T@basis-torch.eye(basis.shape[1])) < 1e-4
        for width in widths:
            uk = basis[:, :width]
            def truncate(_, inputs):
                x = inputs[0]
                return ((x@uk)@uk.T,)
            hook = bond.register_forward_pre_hook(truncate)
            with torch.no_grad():
                logits = torch.cat([model(batch)[0].float()
                                    for batch in test['x'].split(32)])
            hook.remove()
            if width == model.config.n_embd:
                assert torch.max(torch.abs(logits-reference_logits)) < 3e-4
            rows.append(dict(basis=name, width=width,
                             groups=scores(logits, test, reference_logp)))
    output = dict(config=vars(a), model_config=record['model_config'],
                  widths=widths, rows=rows,
                  caveat='This is an orthogonal basis/truncation at one attention '
                         'output bond. Full width is an exact gauge. Sparse '
                         'truncation probes model compression, not uniqueness or '
                         'recovery of planted rule mechanisms.')
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')
    for row in rows:
        if row['width'] in (16, 32) and row['basis'] != 'random':
            print(row['basis'], row['width'],
                  round(row['groups']['verb']['oracle_kl'], 3),
                  round(row['groups']['verb']['model_to_truncated_kl'], 3))


if __name__ == '__main__':
    main()
