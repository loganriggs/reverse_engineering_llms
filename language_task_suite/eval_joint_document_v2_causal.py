"""Compare source-attribute and bracket edits for V2 bilinear models."""

import argparse
import json
from pathlib import Path

import torch

from audit_joint_document_v2 import attribute_edit, supported
from eval_joint_document_v1_causal import candidate_prob
from joint_document_v1 import NAMES, ROLES, TOKEN2ID, _split
from joint_document_v2 import next_distribution, sample_sequence
from train_joint_document_v1 import DeepModel, SPECS


def nuisance_role_edit(prefix, regime):
    start = 0
    for end, word in enumerate(prefix):
        if word != '.':
            continue
        sentence = prefix[start:end]
        if (len(sentence) == 4 and sentence[0] in NAMES and
                sentence[1] == 'is'):
            old = sentence[3]
            for role in ROLES:
                if role == old or (role[0] in 'aeiou') != (old[0] in 'aeiou'):
                    continue
                edited = list(prefix)
                edited[start+3] = role
                if supported(edited, regime):
                    return edited
        start = end+1
    return None


def opener_edit(prefix, regime):
    _, raw = _split(prefix)
    opener = next(word for word in raw if word in ('(', '['))
    edited = list(prefix)
    edited[len(prefix)-len(raw)+raw.index(opener)] = (
        '[' if opener == '(' else '(')
    assert supported(edited, regime)
    return edited


def mean(rows, key):
    return sum(row[key] for row in rows)/len(rows) if rows else None


@torch.no_grad()
def run(checkpoint, regime, n_streams, seed, device):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    depth, cfg = saved['depth'], saved['model_kwargs']
    model = DeepModel(len(TOKEN2ID), cfg['d_model'], cfg['n_head'],
                      SPECS[f'block{depth}'], saved['n_ctx'],
                      norm=cfg['norm'], residual=cfg['residual'],
                      attention=cfg['attention']).to(device).eval()
    model.load_state_dict(saved['state_dict'])
    source, bracket = [], []
    for row in range(n_streams):
        words, labels, _ = sample_sequence(seed+row, saved['n_ctx']+1, regime)
        for i, label in enumerate(labels):
            if i == 0:
                continue
            prefix = words[:i]
            if label == 'source_answer':
                changed = attribute_edit(prefix, regime)
                nuisance = nuisance_role_edit(prefix, regime)
                if changed is None or nuisance is None:
                    continue
                gold = next(iter(next_distribution(prefix, regime)))
                new_gold = next(iter(next_distribution(changed, regime)))
                assert gold != new_gold
                assert next_distribution(nuisance, regime) == {gold: 1.}
                candidates = (gold, new_gold)
                base = candidate_prob(model, prefix, candidates, device)
                edit = candidate_prob(model, changed, candidates, device)
                no_op = candidate_prob(model, nuisance, candidates, device)
                source.append(dict(base_gold=base[0], edited_gold=edit[1],
                                   base_hit=int(base[0] > base[1]),
                                   edited_hit=int(edit[1] > edit[0]),
                                   attribute_tv=abs(base[0]-edit[0]),
                                   role_nuisance_tv=abs(base[0]-no_op[0])))
            elif label == 'bracket_close':
                changed = opener_edit(prefix, regime)
                gold = next(iter(next_distribution(prefix, regime)))
                new_gold = next(iter(next_distribution(changed, regime)))
                assert gold != new_gold
                candidates = (gold, new_gold)
                base = candidate_prob(model, prefix, candidates, device)
                edit = candidate_prob(model, changed, candidates, device)
                bracket.append(dict(base_gold=base[0], edited_gold=edit[1],
                                    base_hit=int(base[0] > base[1]),
                                    edited_hit=int(edit[1] > edit[0]),
                                    opener_tv=abs(base[0]-edit[0])))
    return dict(checkpoint=str(checkpoint), depth=depth, regime=regime,
                n_streams=n_streams, seed=seed,
                source=dict(n=len(source), means={key: mean(source, key)
                                                  for key in source[0]})
                if source else dict(n=0),
                bracket=dict(n=len(bracket), means={key: mean(bracket, key)
                                                    for key in bracket[0]})
                if bracket else dict(n=0),
                oracle=dict(source_attribute_tv=1., source_role_nuisance_tv=0.,
                            bracket_opener_tv=1.),
                caveat='Candidate-normalized two-answer probabilities on '
                       'fresh synthetic streams; one trained seed per depth, '
                       'no whole-task training holdout.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--regime', choices=('reference', 'task_rich'), required=True)
    p.add_argument('--n-streams', type=int, default=100)
    p.add_argument('--seed', type=int, default=37001)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = run(args.checkpoint, args.regime, args.n_streams, args.seed,
                 args.device)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result, flush=True)
