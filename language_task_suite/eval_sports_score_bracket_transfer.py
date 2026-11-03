"""Evaluate score-copy and bracket behavior on the same fresh joint panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from eval_sports_ov_canonical_truncation import model_from_checkpoint
from train_sports_score_bracket import draw, evaluate, stack


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--worlds', type=int, default=28)
    p.add_argument('--seed', type=int, default=55901)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    model, cfg = model_from_checkpoint(a.checkpoint, a.device)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    records, attempts = draw(random.Random(a.seed), a.worlds,
                              'heldout', tokenizer, cfg['n_ctx'],
                              cfg['carrier_words'], 'winning_score',
                              unique=True)
    report = {
        'config': {k: str(v) if isinstance(v, Path) else v
                   for k, v in vars(a).items()},
        'attempts': attempts,
        'game_sha256': [hashlib.sha256(repr(key).encode()).hexdigest()
                        for key, _ in records],
        'score': evaluate(model, stack(records), a.device),
        'caveat': 'Game-disjoint from training partition; may overlap earlier '
                  'heldout panels. All four joint rule arms are evaluated.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print('saved', a.output)


if __name__ == '__main__':
    main()
