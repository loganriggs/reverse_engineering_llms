"""Compare weight-only and data-fitted local bond bases on joint v4 tasks.

This is loss-preserving bond compression, not mechanism identification. The
activation basis can be fitted with all tasks or with questions held out.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from eval_surface_bond_gauges import bases_for
from surface_mixture_v4 import SurfaceMixtureV4
from surface_mixture_v5 import SurfaceMixtureV5
from train_surface_mixture import make_batch, make_eval


def group_scores(logits, data, reference_logp):
    logp = F.log_softmax(logits.float(), -1)
    q = data['q']
    entropy = -(q*q.clamp_min(1e-30).log()).sum(-1)
    oracle_kl = -(q*logp).sum(-1)-entropy
    model_kl = (reference_logp.exp()*(reference_logp-logp)).sum(-1)
    masks = {'all': torch.ones_like(data['verb']),
             'verb': data['verb'], 'object': data['obj'],
             'question_answer': data['answer']}
    if data['count_answer'].any():
        masks['count_answer'] = data['count_answer']
    return {name: dict(n=int(mask.sum()),
                       oracle_kl=float(oracle_kl[mask].mean()) if mask.any() else None,
                       model_to_truncated_kl=float(model_kl[mask].mean())
                       if mask.any() else None)
            for name, mask in masks.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', required=True)
    p.add_argument('--layer', type=int, default=0)
    p.add_argument('--nfit', type=int, default=256)
    p.add_argument('--ntest', type=int, default=512)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--fit-without-questions', action='store_true')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    folder = Path(a.model_dir)
    saved = json.loads((folder/'results.json').read_text())
    generator_class = (SurfaceMixtureV5 if saved['config']['generator'] == 'v5'
                       else SurfaceMixtureV4)
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config
    results = {}
    for record in saved['records']:
        depth = record['depth']
        assert 0 <= a.layer < depth
        model = BilinearGPT(Config(**record['model_config']))
        model.load_state_dict(torch.load(folder/f'model_depth{depth}.pt',
                                         map_location='cpu', weights_only=True))
        model.eval()
        query_kwargs = dict(hard_query_rate=saved['config'].get('hard_query_rate'),
                            counterbalanced_queries=saved['config'].get(
                                'counterbalanced_queries', False))
        question_rate = saved['config'].get('question_rate', .05)
        if generator_class is SurfaceMixtureV5:
            query_kwargs['count_rate'] = saved['config'].get('count_rate', .20)
        fit_kwargs = dict(query_kwargs)
        if a.fit_without_questions and generator_class is SurfaceMixtureV5:
            fit_kwargs['count_rate'] = 0
        fit = generator_class(seed=9501,
                              question_rate=0 if a.fit_without_questions
                              else question_rate, **fit_kwargs)
        test = generator_class(seed=9502, question_rate=question_rate,
                               **query_kwargs)
        xfit, _ = make_batch(fit, a.nfit, a.length, 'cpu')
        data = make_eval(test, a.ntest, a.length, 'cpu')
        bond = model.blocks[a.layer].attn.out
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
            reference = torch.cat([model(batch)[0].float()
                                   for batch in data['x'].split(32)])
        reference_logp = F.log_softmax(reference, -1)
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
                                        for batch in data['x'].split(32)])
                hook.remove()
                if width == model.config.n_embd:
                    assert torch.max(torch.abs(logits-reference)) < 3e-4
                rows.append(dict(basis=name, width=width,
                                 groups=group_scores(logits, data, reference_logp)))
        results[str(depth)] = dict(widths=widths, rows=rows,
                                   reference=group_scores(reference, data,
                                                          reference_logp))
    output = dict(config=vars(a), model_config=saved['records'][0]['model_config'],
                  results=results,
                  caveat='Full-width orthogonal rotations are exact local gauges. '
                         'Truncation compares preservation of model and oracle '
                         'predictions; it does not identify the planted retrieval '
                         'mechanism, especially where the model fails causal edits.')
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')
    for depth, result in results.items():
        for row in result['rows']:
            if row['width'] in (16, 32) and row['basis'] in ('weight_svd', 'activation_pca'):
                print('depth', depth, 'basis', row['basis'], 'width', row['width'],
                      'model KL', {k: round(v['model_to_truncated_kl'], 3)
                                  if v['model_to_truncated_kl'] is not None else None
                                  for k, v in row['groups'].items()}, flush=True)


if __name__ == '__main__':
    main()
