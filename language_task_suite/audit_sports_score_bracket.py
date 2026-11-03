"""Check score arithmetic and bracket edits, token alignment, and form."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from tokenizers import Tokenizer

from audit_document_structure_vs_fineweb import structure
from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from compare_v12_fineweb_fresh import first_words
from sports_document_dgp import _detokenize
from sports_entity_alias_dgp import _rows
from sports_score_bracket_dgp import realize, sample_plan


ARMS = ((a, b) for a in ('base', 'increment')
        for b in ('round', 'square'))
ARMS = tuple(ARMS)


def aligned(doc, tokenizer, n_ctx):
    enc = tokenizer.encode(doc.text, add_special_tokens=False)
    found = []
    for site in doc.sites:
        rows = [i for i, (start, end) in enumerate(enc.offsets)
                if start <= site.start < end]
        if len(rows) != 1 or rows[0] < 1 or rows[0] >= n_ctx or \
                enc.offsets[rows[0]][1] != site.end:
            return None
        found.append(rows[0])
    return found


def audit(group, n, seed, words, n_ctx, tokenizer, query_kind):
    rng = random.Random(seed)
    seen, hashes, counts = set(), [], Counter()
    source_text, base_text, edited_text = [], [], []
    bracket_windows, numeric_windows = [], []
    attempts = 0
    while len(seen) < n:
        attempts += 1
        if attempts > n*500:
            raise RuntimeError('Insufficient unique joint-rule games')
        plan = sample_plan(rng, group, carrier_words=words,
                           query_kind=query_kind)
        key = plan.score.source_game_key
        if key in seen:
            continue
        seen.add(key)
        hashes.append(hashlib.sha256(repr(key).encode()).hexdigest())
        docs = {arm: realize(plan, *arm) for arm in ARMS}
        for doc in docs.values():
            for site in doc.sites:
                assert doc.text[site.start:site.end] == site.answer
                counts['visible_site_checks'] += 1
        for left in ARMS:
            for right in ARMS:
                if left >= right:
                    continue
                for k in (0, 1):
                    assert (docs[left].sites[k].answer !=
                            docs[right].sites[k].answer) == \
                        (left[1-k] != right[1-k])
        counts['independent_factorials'] += 1
        all_aligned = True
        for doc in docs.values():
            ok = aligned(doc, tokenizer, n_ctx) is not None
            counts['single_bpe_in_context_arms'] += ok
            all_aligned &= ok
        counts['complete_aligned_factorials'] += all_aligned
        original = docs[('base', 'round')]
        edited = docs[('increment', 'square')]
        source_text.append(first_words(_detokenize(
            _rows()[key]['summary']), 100))
        base_text.append(first_words(original.text, 100))
        edited_text.append(first_words(edited.text, 100))
        counts[query_kind+'/'+original.sites[1].answer] += 1
        if len(bracket_windows) < 64:
            for site, output in ((original.sites[0], bracket_windows),
                                 (original.sites[1], numeric_windows)):
                output.append(' '.join(original.text[:site.start].split()[-25:]
                                        +original.text[site.start:].split()[:16]))
    return {'counts': dict(counts), 'draw_attempts': attempts,
            'game_sha256': hashes,
            'first100_form': {'source': structure(source_text[:64]),
                              'base': structure(base_text[:64]),
                              'edited': structure(edited_text[:64])},
            'site41_form': {'bracket': structure(bracket_windows),
                            'numeric': structure(numeric_windows)}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-n', type=int, default=100)
    p.add_argument('--heldout-n', type=int, default=30)
    p.add_argument('--words', type=int, default=100)
    p.add_argument('--n-ctx', type=int, default=256)
    p.add_argument('--query-kind', choices=('margin', 'winning_score'),
                   default='margin')
    p.add_argument('--seed', type=int, default=55601)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    result = {'config': {k: str(v) if isinstance(v, Path) else v
                         for k, v in vars(a).items()},
              'groups': {},
              'caveat': 'Human carrier prose has no full next-token oracle; '
                        'winning margin is an inserted natural-style query. '
                        'The audit checks visible rules and surface form, '
                        'not human judgement of every edited report.'}
    for group, n, seed in (('train', a.train_n, a.seed),
                           ('heldout', a.heldout_n, a.seed+100)):
        result['groups'][group] = audit(group, n, seed, a.words,
                                        a.n_ctx, tokenizer, a.query_kind)
        print(group, result['groups'][group]['counts'], flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
