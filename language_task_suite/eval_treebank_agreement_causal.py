"""Paired number-changing and number-preserving agreement interventions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
import random
import sys

import torch
from tokenizers import Tokenizer


OLD_REPO = Path(__file__).resolve().parents[2]/'reverse_engineer/tensor_language'
sys.modules.pop('model', None)
sys.path.insert(0, str(OLD_REPO))
from deep_model import DeepModel, SPECS  # noqa: E402
sys.modules.pop('model', None)
sys.path.remove(str(OLD_REPO))
from treebank_agreement_dgp import (  # noqa: E402
    PAIRS, PLURAL_DETERMINERS, SINGULAR_DETERMINERS, load_world,
)


def with_subject(base, new_subject):
    tokens = new_subject.subject+(PAIRS[base.lemma][new_subject.number],)+ \
        base.continuation+('.',)
    return replace(base, tokens=tokens, target_index=len(new_subject.subject),
                   number=new_subject.number, subject=new_subject.subject,
                   attractor=new_subject.attractor)


def load(path, device):
    data = torch.load(path, map_location='cpu', weights_only=False)
    cfg = data['config']
    tokenizer = Tokenizer.from_str(data['vocab'])
    model = DeepModel(tokenizer.get_vocab_size(), cfg['d_model'],
                      cfg['heads'], SPECS[f"block{data['depth']}"],
                      data['n_ctx'], norm='rms', residual='lerp',
                      attention='bilinear').to(device)
    model.load_state_dict(data['state_dict'])
    model.eval()
    return model, tokenizer, data['depth']


@torch.no_grad()
def predict(model, tokenizer, clauses, device, batch_size=64):
    outputs = []
    pad = tokenizer.token_to_id('<PAD>')
    for start in range(0, len(clauses), batch_size):
        batch = clauses[start:start+batch_size]
        prefixes = [tokenizer.encode(' '.join(
            clause.tokens[:clause.target_index])).ids
            for clause in batch]
        n_ctx = max(map(len, prefixes))
        x = torch.full((len(batch), n_ctx), pad, dtype=torch.long,
                       device=device)
        for i, ids in enumerate(prefixes):
            x[i, :len(ids)] = torch.tensor(ids, device=device)
        logits = model(x)
        for i, clause in enumerate(batch):
            choices = [tokenizer.encode(word).ids for word in
                       PAIRS[clause.lemma]]
            assert len(choices[0]) == len(choices[1]) == 1
            a, b = choices[0][0], choices[1][0]
            row = logits[i, len(prefixes[i])-1]
            outputs.append((int(row[b] > row[a]),
                            float((row[b]-row[a]).item())))
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--treebank', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--fit-trees', type=int, default=20000)
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--seed', type=int, default=41401)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    world = load_world(str(args.treebank), args.fit_trees)
    rng = random.Random(args.seed)
    baseline, same_number, opposite_number = [], [], []
    for i in range(args.n):
        base = world.sample(rng, number=i % 2,
                            attractor=(i//2) % 2 == 1,
                            subject_split='heldout')
        same = world.sample(rng, number=base.number, lemma=base.lemma,
                            attractor=base.attractor,
                            subject_split='heldout')
        opposite = world.sample(rng, number=1-base.number,
                                lemma=base.lemma,
                                attractor=base.attractor,
                                subject_split='heldout')
        same = with_subject(base, same)
        opposite = with_subject(base, opposite)
        for clause in (base, same, opposite):
            assert world.number_from_visible_prefix(
                clause.tokens[:clause.target_index]) == clause.number
        assert base.continuation == same.continuation == \
            opposite.continuation
        baseline.append(base)
        same_number.append(same)
        opposite_number.append(opposite)
    out = {'config': {'seed': args.seed, 'n': args.n,
                      'subject_split': 'heldout',
                      'checkpoints': [p.name for p in args.checkpoints]},
           'edits': 'Same and opposite head-number noun-phrase swaps, '
                    'holding the source VP continuation and verb lemma '
                    'fixed; attractor status matched.',
           'results': {}}
    for path in args.checkpoints:
        model, tokenizer, depth = load(path, device)
        before = predict(model, tokenizer, baseline, device)
        same = predict(model, tokenizer, same_number, device)
        opposite = predict(model, tokenizer, opposite_number, device)
        counts = defaultdict(Counter)
        for i, clause in enumerate(baseline):
            group = 'attractor' if clause.attractor else 'plain'
            revealing = clause.subject[0].lower() in \
                SINGULAR_DETERMINERS | PLURAL_DETERMINERS
            keys = ['all', group, 'lemma/'+clause.lemma]
            if clause.attractor:
                keys.append('attractor_revealing_determiner' if revealing
                            else 'attractor_neutral_determiner')
            for key in keys:
                row = counts[key]
                row['n'] += 1
                row['baseline_correct'] += before[i][0] == clause.number
                row['same_correct'] += same[i][0] == clause.number
                row['opposite_correct'] += opposite[i][0] != clause.number
                row['same_invariant'] += same[i][0] == before[i][0]
                row['opposite_flip'] += opposite[i][0] != before[i][0]
                row['both_causal'] += same[i][0] == before[i][0] and \
                    opposite[i][0] != before[i][0]
        out['results'][path.stem] = {
            'depth': depth,
            'counts': {key: dict(value)
                       for key, value in counts.items()}}
        print(path.name, out['results'][path.stem]['counts']['all'],
              flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    main()
