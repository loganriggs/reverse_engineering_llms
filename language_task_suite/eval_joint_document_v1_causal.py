"""Paired supported-edit checks on the joint-document bilinear baseline."""

import argparse
import json
from pathlib import Path

import torch

from audit_joint_document_v1 import _query_key_edit, _supported
from joint_document_v1 import (NAMES, ROLES, TOKEN2ID, _split,
                               next_distribution, sample_sequence)
from train_joint_document_v1 import DeepModel, SPECS


def role_edit(prefix):
    edited = list(prefix)
    start = 0
    for i, token in enumerate(prefix):
        if token != '.':
            continue
        sentence = prefix[start:i]
        if (len(sentence) == 4 and sentence[0] in NAMES and
                sentence[1] == 'is'):
            old = sentence[3]
            choices = [role for role in ROLES if role != old and
                       (role[0] in 'aeiou') == (old[0] in 'aeiou')]
            if choices:
                edited[start+3] = choices[0]
                if _supported(edited):
                    return edited
        start = i+1
    return None


def bracket_edit(prefix):
    _, raw = _split(prefix)
    opener = next(word for word in raw if word in ('(', '['))
    edited = list(prefix)
    edited[len(prefix)-len(raw)+raw.index(opener)] = (
        '[' if opener == '(' else '(')
    assert _supported(edited)
    return edited


@torch.no_grad()
def candidate_prob(model, prefix, candidates, device):
    ids = torch.tensor([[TOKEN2ID[word] for word in prefix]],
                       device=device, dtype=torch.long)
    logits = model(ids)[0, -1]
    values = logits[torch.tensor([TOKEN2ID[word] for word in candidates],
                                 device=device)]
    return torch.softmax(values, dim=-1).cpu().tolist()


def _mean(rows, key):
    return sum(row[key] for row in rows)/len(rows) if rows else None


def run(checkpoint, n_streams, seed, device):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    depth = saved['depth']
    cfg = saved['model_kwargs']
    model = DeepModel(len(TOKEN2ID), cfg['d_model'], cfg['n_head'],
                      SPECS[f'block{depth}'], saved['n_ctx'],
                      norm=cfg['norm'], residual=cfg['residual'],
                      attention=cfg['attention']).to(device).eval()
    model.load_state_dict(saved['state_dict'])
    source, bracket = [], []
    for row in range(n_streams):
        words, labels, _ = sample_sequence(seed+row, saved['n_ctx']+1)
        for i, label in enumerate(labels):
            if i == 0:
                continue
            prefix = words[:i]
            if label == 'source_answer':
                changed = _query_key_edit(prefix)
                nuisance = role_edit(prefix)
                if changed is None or nuisance is None:
                    continue
                gold = next(iter(next_distribution(prefix)))
                new_gold = next(iter(next_distribution(changed)))
                assert gold != new_gold
                assert next_distribution(nuisance) == {gold: 1.}
                candidates = (gold, new_gold)
                base = candidate_prob(model, prefix, candidates, device)
                edit = candidate_prob(model, changed, candidates, device)
                no_op = candidate_prob(model, nuisance, candidates, device)
                source.append(dict(base_gold=base[0], edited_gold=edit[1],
                                   base_hit=int(base[0] > base[1]),
                                   edited_hit=int(edit[1] > edit[0]),
                                   key_tv=abs(base[0]-edit[0]),
                                   nuisance_tv=abs(base[0]-no_op[0])))
            elif label == 'bracket_close':
                changed = bracket_edit(prefix)
                gold = next(iter(next_distribution(prefix)))
                new_gold = next(iter(next_distribution(changed)))
                assert gold != new_gold
                candidates = (gold, new_gold)
                base = candidate_prob(model, prefix, candidates, device)
                edit = candidate_prob(model, changed, candidates, device)
                bracket.append(dict(base_gold=base[0], edited_gold=edit[1],
                                    base_hit=int(base[0] > base[1]),
                                    edited_hit=int(edit[1] > edit[0]),
                                    opener_tv=abs(base[0]-edit[0])))
    return dict(checkpoint=str(checkpoint), depth=depth, seed=seed,
                n_streams=n_streams,
                source=dict(n=len(source), means={key: _mean(source, key)
                                                  for key in source[0]})
                if source else dict(n=0),
                bracket=dict(n=len(bracket), means={key: _mean(bracket, key)
                                                    for key in bracket[0]})
                if bracket else dict(n=0),
                oracle=dict(source_key_tv=1., source_role_nuisance_tv=0.,
                            bracket_opener_tv=1.),
                caveat='Candidate-normalized probabilities over the two '
                       'original/edited answers; one trained seed, fresh '
                       'synthetic streams, no whole-task holdout.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--n-streams', type=int, default=100)
    p.add_argument('--seed', type=int, default=36001)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = run(args.checkpoint, args.n_streams, args.seed, args.device)
    args.output.write_text(json.dumps(out, indent=2)+'\n')
    print(out, flush=True)
