"""Test held-out live-ledger models under coherent score edits."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
import random
import sys

import torch


OLD_REPO = Path(__file__).resolve().parents[2]/'reverse_engineer/tensor_language'
sys.modules.pop('model', None)
sys.path.insert(0, str(OLD_REPO))
from deep_model import DeepModel, SPECS  # noqa: E402
sys.modules.pop('model', None)
sys.path.remove(str(OLD_REPO))
from sports_live_ledger_dgp import (answer_from_visible_prefix,
                                    sample_document)  # noqa: E402


def score_lines(tokens):
    return [i for i in range(len(tokens)-8)
            if tuple(tokens[i:i+2]) == ('Score', ':') and
            tokens[i+4] == ',' and tuple(tokens[i+7:i+9]) == ('.', '\n')]


def edited(doc, edit, side=0):
    tokens = list(doc.tokens)
    lines = score_lines(tokens)
    assert len(lines) >= 5
    codes = (tokens[lines[0]+2], tokens[lines[0]+5])
    index = 3 if side == 0 else 6
    first_changed_line = -1
    if edit == 'add_play':
        assert not doc.latent['descriptions']
        play = next(i for i in range(len(tokens)-6)
                    if tuple(tokens[i:i+2]) == ('Play', ':') and
                    tokens[i+2] == codes[side] and tokens[i+3] == 'scored')
        tokens[play+4] = str(int(tokens[play+4])+1)
        first_changed_line = play
    elif edit != 'offset':
        raise ValueError(edit)
    for line in lines:
        if line > first_changed_line:
            tokens[line+index] = str(int(tokens[line+index])+1)
    sites = []
    for site in doc.sites:
        queried_side = codes.index(tokens[site.index-1])
        changed = (queried_side == side and
                   (site.family == 'live_score_copy' or
                    edit == 'add_play'))
        target = str(int(site.target)+int(changed))
        tokens[site.index] = target
        sites.append(replace(site, target=target, distribution={target: 1.}))
    output = replace(doc, tokens=tuple(tokens), sites=tuple(sites),
                     latent=doc.latent | {'edit': edit, 'edit_side': side})
    for site in output.sites:
        kind = 'copy' if site.family == 'live_score_copy' else 'run'
        assert answer_from_visible_prefix(output.tokens[:site.index], kind) \
            == site.target
    return output


def model_from_checkpoint(path, device):
    data = torch.load(path, map_location='cpu', weights_only=False)
    kw = data['model_kwargs']
    model = DeepModel(len(data['vocabulary']), kw['d_model'], kw['n_head'],
                      SPECS[f"block{data['depth']}"], data['n_ctx'],
                      norm=kw['norm'], residual=kw['residual'],
                      attention=kw['attention']).to(device)
    model.load_state_dict(data['state_dict'])
    model.eval()
    return model, data['vocabulary'], data


@torch.no_grad()
def predictions(model, vocab, doc, device):
    ids = [vocab.get(token, vocab['<UNK>']) for token in doc.tokens[:-1]]
    x = torch.tensor(ids, device=device)[None]
    logits = model(x)[0]
    out = {}
    for site in doc.sites:
        limit = 201 if site.family == 'live_score_copy' else 41
        candidates = torch.tensor([vocab[str(i)] for i in range(limit)],
                                  device=device)
        out[site.family] = int(logits[site.index-1, candidates].argmax())
    return out


def panel(model, vocab, seed, n, arm, device):
    counts = defaultdict(Counter)
    for i in range(n):
        doc = sample_document(random.Random(seed+i), 'rotowire',
                              'heldout', arm == 'described')
        baseline = predictions(model, vocab, doc, device)
        first_site = next(site for site in doc.sites
                          if site.family == 'live_run_first')
        changed_tokens = list(doc.tokens)
        changed_tokens[first_site.index] = str(int(first_site.target)+1)
        changed_first = replace(doc, tokens=tuple(changed_tokens))
        changed_first_prediction = predictions(model, vocab, changed_first,
                                               device)['live_run_second']
        row = counts['first_answer_plus_one/live_run_second/target_same']
        row['n'] += 1
        correct = baseline['live_run_second'] == int(doc.sites[-1].target)
        row['baseline_correct'] += correct
        row['edited_correct'] += changed_first_prediction == \
            int(doc.sites[-1].target)
        row['response_exact'] += changed_first_prediction == \
            baseline['live_run_second']
        row['response_exact_given_baseline_correct'] += correct and \
            changed_first_prediction == baseline['live_run_second']
        for edit, side in (('offset', 0),
                           *((('add_play', s) for s in (0, 1))
                             if arm == 'bare' else ())):
            changed = edited(doc, edit, side)
            after = predictions(model, vocab, changed, device)
            run_values = [int(site.target) for site in changed.sites
                          if site.family.startswith('live_run_')]
            in_sampling_support = (min(run_values) >= 1 and
                                   max(run_values) >= 6 and
                                   abs(run_values[0]-run_values[1]) >= 3)
            for before_site, after_site in zip(doc.sites, changed.sites):
                family = before_site.family
                expected = int(after_site.target)-int(before_site.target)
                row = counts[f'{edit}_{side}/{family}/'
                             f'{"target_up" if expected else "target_same"}/'
                             f'{"in_support" if in_sampling_support else "out_of_support"}']
                prior_correct = baseline[family] == int(before_site.target)
                row['n'] += 1
                row['baseline_correct'] += prior_correct
                row['edited_correct'] += after[family] == \
                    int(after_site.target)
                row['response_exact'] += after[family]-baseline[family] \
                    == expected
                row['response_exact_given_baseline_correct'] += \
                    prior_correct and after[family]-baseline[family] == \
                    expected
    return {key: dict(value) for key, value in counts.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--seed', type=int, default=10308)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--device', default='cuda' if
                        torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    out = {'config': {'seed': args.seed, 'n': args.n,
                      'checkpoints': [path.name for path in
                                      args.checkpoints]},
           'edits': {'offset': 'Add one point to every printed home score '
                              'and latest-score copy target; run totals '
                              'must stay invariant.',
                     'first_answer_plus_one': 'Deliberately corrupt the '
                              'printed first run answer by +1; the second '
                              'run answer is still determined by score lines.',
                     'add_play': 'Bare arm only: add one point to one '
                                 'scoring play and all later printed '
                                 'scores for its team; corresponding run '
                                 'total and possibly latest home score '
                                 'increase by one.'},
           'results': {}}
    for path in args.checkpoints:
        model, vocab, data = model_from_checkpoint(path, device)
        arm = data['arm']
        result = panel(model, vocab, args.seed, args.n, arm, device)
        out['results'][path.stem] = {'arm': arm, 'depth': data['depth'],
                                    'counts': result}
        print(path.name, result, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    main()
