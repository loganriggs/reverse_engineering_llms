"""Test individual weight-SVD bond directions for causal rule selectivity."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from eval_multi_genre_bond_gauges import group_scores
from eval_multi_genre_v3_agreement_edits import build_pairs
from eval_surface_mixture_edits import predict, score
from multi_genre_v3_agreement import GENRES_V3, MultiGenreV3Agreement
from train_multi_genre import TASK_LABELS, make_eval


AGREEMENT_LABELS = ('agreement_main_match', 'agreement_main_conflict',
                    'agreement_embedded')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', required=True)
    p.add_argument('--depth', type=int, default=3)
    p.add_argument('--layer', type=int, default=2)
    p.add_argument('--axes', type=int, default=12)
    p.add_argument('--ntest', type=int, default=256)
    p.add_argument('--nedit', type=int, default=64)
    p.add_argument('--seed', type=int, default=13551)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    torch.set_num_threads(4)
    saved = json.loads((Path(a.model_dir)/'results.json').read_text())
    assert saved['config']['language'] == 'v3'
    record = next(r for r in saved['records'] if r['depth'] == a.depth)
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(
        Path(a.model_dir)/f'model_depth{a.depth}.pt',
        weights_only=True, map_location='cpu'))
    model.eval()
    bond = model.blocks[a.layer].attn.out
    _, _, vh = torch.linalg.svd(bond.weight.detach().double())
    basis = vh.T.float()
    assert a.axes <= basis.shape[1]
    weights = tuple(float(w) for w in
                    saved['config']['genre_weights'].split(','))
    length = saved['config']['length']
    test = make_eval(MultiGenreV3Agreement(
        seed=a.seed, genre_weights=weights), a.ntest, length, 'cpu',
        TASK_LABELS + AGREEMENT_LABELS, GENRES_V3)
    edit_gen = MultiGenreV3Agreement(
        seed=a.seed+1, genre_weights=(0, 0, 0, 0, 0, 1))
    pairs = build_pairs(edit_gen, a.nedit, length)
    originals = [pair['original'] for pair in pairs]
    edits = [pair['edited'] for pair in pairs]
    vocab = len(saved['vocabulary'])
    with torch.no_grad():
        ref_logits = torch.cat([model(batch)[0].float()
                                for batch in test['x'].split(32)])
    ref_logp = F.log_softmax(ref_logits, -1)
    def eval_current():
        with torch.no_grad():
            logits = torch.cat([model(batch)[0].float()
                                for batch in test['x'].split(32)])
        before = predict(model, originals, vocab)
        after = predict(model, edits, vocab)
        return dict(groups=group_scores(logits, test, ref_logp),
                    edits=score(pairs, before, after, edit_gen))
    native = eval_current()
    results = []
    for axis in list(range(a.axes)) + ['all']:
        if axis == 'all':
            def ablate(_, inputs):
                return (torch.zeros_like(inputs[0]),)
        else:
            u = basis[:, axis]
            def ablate(_, inputs):
                x = inputs[0]
                return (x-(x@u).unsqueeze(-1)*u,)
        hook = bond.register_forward_pre_hook(ablate)
        result = eval_current()
        hook.remove()
        result['removed'] = axis
        results.append(result)
        print(axis, 'agreement KL',
              round(result['groups']['agreement_main_conflict']['oracle_kl'], 4),
              'main edit L1',
              round(result['edits']['main_subject_flip']['effect_l1_error'], 4),
              'policy KL',
              round(result['groups']['policy_mixed_action']['oracle_kl'], 4),
              flush=True)
    output = dict(config=vars(a), n_pairs=len(pairs), native=native,
                  ablations=results,
                  caveat='Individual axes are ordered by weight singular '
                         'value. This tests selectivity under removal, not '
                         'uniqueness or semantic interpretability.')
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
