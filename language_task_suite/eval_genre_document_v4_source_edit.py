"""Supported contrastive source-attribute edits for a V4 bilinear model."""

import argparse
import json
from pathlib import Path

import torch

from genre_document_v3 import _base_prefix
from genre_document_v4 import (TOKEN2ID, next_distribution, sample_sequence,
                               to_v3)
from joint_document_v1 import _split, visible_state
from train_joint_document_v1 import DeepModel, SPECS


def pairs(n, seed):
    out = []
    inspected = 0
    while len(out) < n:
        assert inspected < n*20, 'insufficient supported contrastive sites'
        words, labels, _, _ = sample_sequence(seed+inspected, 256,
                                               'task_rich')
        inspected += 1
        for i, label in enumerate(labels):
            if label != 'source_answer' or i == 0:
                continue
            prefix = words[:i]
            history, raw = _split(_base_prefix(to_v3(prefix)))
            claims = visible_state(history)['claims']
            item, old_attr = raw[3], raw[5]
            old_answer = claims[(item, old_attr)]
            alternates = [(attr, speaker) for (key, attr), speaker
                          in claims.items() if key == item and
                          speaker != old_answer]
            if not alternates:
                continue
            new_attr, new_answer = alternates[0]
            edited = list(prefix)
            assert edited[-4] == old_attr
            edited[-4] = new_attr
            assert next_distribution(prefix) == {old_answer: 1.}
            assert next_distribution(edited) == {new_answer: 1.}
            out.append((prefix, edited, old_answer, new_answer))
            if len(out) == n:
                break
    return out, inspected


def bracket_pairs(n, seed):
    out = []
    inspected = 0
    while len(out) < n:
        assert inspected < n*20, 'insufficient supported bracket sites'
        words, labels, _, _ = sample_sequence(seed+inspected, 256,
                                               'task_rich')
        inspected += 1
        for i, label in enumerate(labels):
            if label != 'bracket_close' or i == 0:
                continue
            prefix = words[:i]
            _, raw = _split(prefix)
            old = '(' if '(' in raw else '['
            new = '[' if old == '(' else '('
            edited = list(prefix)
            edited[len(prefix)-len(raw)+raw.index(old)] = new
            old_answer = ')' if old == '(' else ']'
            new_answer = ']' if old == '(' else ')'
            assert next_distribution(prefix) == {old_answer: 1.}
            assert next_distribution(edited) == {new_answer: 1.}
            out.append((prefix, edited, old_answer, new_answer))
            if len(out) == n:
                break
    return out, inspected


@torch.no_grad()
def run(checkpoint, n, seed, device):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['vocabulary'] == TOKEN2ID
    depth, cfg = saved['depth'], saved['model_kwargs']
    model = DeepModel(len(TOKEN2ID), cfg['d_model'], cfg['n_head'],
                      SPECS[f'block{depth}'], saved['n_ctx'],
                      norm=cfg['norm'], residual=cfg['residual'],
                      attention=cfg['attention']).to(device).eval()
    model.load_state_dict(saved['state_dict'])
    def evaluate_examples(examples):
        totals = dict(original_answer_prob=0., edited_answer_prob=0.,
                      original_full_vocab_hit=0., edited_full_vocab_hit=0.,
                      two_answer_effect=0., full_vocab_tv=0.)
        for original, edited, old, new in examples:
            p = []
            for prefix in (original, edited):
                x = torch.tensor([[TOKEN2ID[w] for w in prefix]], device=device)
                p.append(model(x)[0, -1].softmax(dim=-1))
            old_id, new_id = TOKEN2ID[old], TOKEN2ID[new]
            totals['original_answer_prob'] += p[0][old_id].item()
            totals['edited_answer_prob'] += p[1][new_id].item()
            totals['original_full_vocab_hit'] += int(
                p[0].argmax().item() == old_id)
            totals['edited_full_vocab_hit'] += int(
                p[1].argmax().item() == new_id)
            totals['full_vocab_tv'] += .5*(p[0]-p[1]).abs().sum().item()
            base_pair = p[0][old_id]/(p[0][old_id]+p[0][new_id])
            edit_pair = p[1][old_id]/(p[1][old_id]+p[1][new_id])
            totals['two_answer_effect'] += (base_pair-edit_pair).item()
        return {key: value/len(examples) for key, value in totals.items()}

    examples, inspected = pairs(n, seed)
    bracket_examples, bracket_inspected = bracket_pairs(n, seed+100000)
    return dict(checkpoint=str(checkpoint), n=n, seed=seed,
                documents_inspected=inspected,
                oracle=dict(full_vocab_tv=1., two_answer_effect=1.),
                means=evaluate_examples(examples),
                bracket_control=dict(n=n,
                                     documents_inspected=bracket_inspected,
                                     means=evaluate_examples(bracket_examples)),
                caveat='Fresh supported query-attribute edits switching '
                       'source answers. Two-answer effect normalizes over '
                       'the old/new answers; bracket opener swaps are a '
                       'competent control. Full-vocabulary accuracy and TV '
                       'are separate. One model seed, no task holdout.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--n', type=int, default=200)
    p.add_argument('--seed', type=int, default=79001)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = run(args.checkpoint, args.n, args.seed, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(dict(source=result['means'],
               bracket=result['bracket_control']['means']), flush=True)
