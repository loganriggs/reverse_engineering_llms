"""Check whether a V4 model treats equivalent source questions alike."""

import argparse
import json
from pathlib import Path

import torch

from genre_document_v4 import TOKEN2ID, next_distribution, sample_sequence
from joint_document_v1 import _split
from train_joint_document_v1 import DeepModel, SPECS


def paired_questions(n, seed):
    pairs = []
    inspected = 0
    while len(pairs) < n:
        assert inspected < n*10, 'too few source questions'
        words, labels, _, _ = sample_sequence(seed+inspected, 256, 'task_rich')
        inspected += 1
        for i, label in enumerate(labels):
            if label != 'source_answer' or i >= 254:
                continue
            prefix = words[:i]
            _, raw = _split(prefix)
            if raw[0] == 'Who':
                revised_raw = ['Which', 'source'] + list(raw[1:])
            else:
                assert raw[:2] == ('Which', 'source')
                revised_raw = ['Who'] + list(raw[2:])
            edited = prefix[:-len(raw)] + revised_raw
            q0 = next_distribution(prefix, 'task_rich')
            q1 = next_distribution(edited, 'task_rich')
            assert len(q0) == len(q1) == 1 and q0 == q1
            assert max(q0.values()) == 1.
            pairs.append((prefix, edited, next(iter(q0)), raw[0]))
            if len(pairs) == n:
                break
    return pairs, inspected


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
    pairs, inspected = paired_questions(n, seed)
    totals = dict(original_answer_probability=0.,
                  edited_answer_probability=0.,
                  original_accuracy=0., edited_accuracy=0.,
                  model_tv=0., answer_probability_abs_change=0.)
    for original, edited, answer, _ in pairs:
        p = []
        for prefix in (original, edited):
            x = torch.tensor([[TOKEN2ID[w] for w in prefix]], device=device)
            p.append(model(x)[0, -1].softmax(dim=-1))
        target = TOKEN2ID[answer]
        totals['original_answer_probability'] += p[0][target].item()
        totals['edited_answer_probability'] += p[1][target].item()
        totals['original_accuracy'] += int(p[0].argmax().item() == target)
        totals['edited_accuracy'] += int(p[1].argmax().item() == target)
        totals['model_tv'] += .5*(p[0]-p[1]).abs().sum().item()
        totals['answer_probability_abs_change'] += abs(
            p[0][target].item()-p[1][target].item())
    return dict(checkpoint=str(checkpoint), n=n, seed=seed,
                documents_inspected=inspected,
                original_forms=dict(Who=sum(form == 'Who'
                                            for *_, form in pairs),
                                    Which=sum(form == 'Which'
                                              for *_, form in pairs)),
                exact_oracle_answer_edit_tv=0.,
                means={key: value/n for key, value in totals.items()},
                caveat='Surface-only query paraphrase at supported source '
                       'answer sites. Oracle answer remains the same; model '
                       'accuracy is prerequisite to interpret an invariant '
                       'source-binding mechanism.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--n', type=int, default=200)
    p.add_argument('--seed', type=int, default=69001)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = run(args.checkpoint, args.n, args.seed, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['means'], flush=True)
