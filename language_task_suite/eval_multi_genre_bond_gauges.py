"""Compare local weight-only and data-aware attention-bond gauges by task."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from eval_surface_bond_gauges import bases_for
from multi_genre_language import GENRES, MultiGenreLanguage
from train_multi_genre import make_batch, make_eval


def group_scores(logits, data, reference_logp):
    logp = F.log_softmax(logits.float(), -1)
    q = data['q']
    entropy = -(q*q.clamp_min(1e-30).log()).sum(-1)
    kl = -(q*logp).sum(-1)-entropy
    model_kl = (reference_logp.exp()*(reference_logp-logp)).sum(-1)
    return {name: dict(n=int(mask.sum()),
                       oracle_kl=float(kl[mask].mean()) if mask.any() else None,
                       model_to_truncated_kl=float(model_kl[mask].mean())
                       if mask.any() else None)
            for name, mask in data['masks'].items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', required=True)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--layer', type=int, default=1)
    p.add_argument('--nfit', type=int, default=256)
    p.add_argument('--ntest', type=int, default=256)
    p.add_argument('--length', type=int, default=71)
    p.add_argument('--fit-genres', default=','.join(GENRES),
                   help='Comma-separated genres for activation-PCA fitting; all genres are tested')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    torch.set_num_threads(4)
    saved = json.loads((Path(a.model_dir)/'results.json').read_text())
    record = next(row for row in saved['records'] if row['depth'] == a.depth)
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(Path(a.model_dir)/f'model_depth{a.depth}.pt',
                                     weights_only=True, map_location='cpu'))
    model.eval()
    bond = model.blocks[a.layer].attn.out
    fit_genres = set(a.fit_genres.split(','))
    assert fit_genres and fit_genres.issubset(GENRES)
    fit_weights = tuple(float(g in fit_genres) for g in GENRES)
    xfit, _ = make_batch(MultiGenreLanguage(seed=9501,
                                            genre_weights=fit_weights), a.nfit,
                         a.length, 'cpu')
    test = make_eval(MultiGenreLanguage(seed=9502), a.ntest,
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
                             groups=group_scores(logits, test, reference_logp)))
    output = dict(config=vars(a), model_config=record['model_config'],
                  widths=widths, rows=rows,
                  caveat='Full-width orthogonal rotations are exact local '
                         'gauges. Truncation probes bond compression, not '
                         'uniqueness or recovery of the planted mechanism.')
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')
    for row in rows:
        if row['width'] in (16, 32) and row['basis'] != 'random':
            print(row['basis'], row['width'],
                  {k: round(row['groups'][k]['model_to_truncated_kl'], 3)
                   if row['groups'][k]['model_to_truncated_kl'] is not None else None
                   for k in ('report_mixed_verb', 'policy_mixed_action',
                             'faq_answer_copy', 'faq_place_recall',
                             'list_number')})


if __name__ == '__main__':
    main()
