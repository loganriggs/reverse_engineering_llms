"""Locate cross-document phrase repetition in the sampled sports DGP.

Reports aggregate counts only; no natural source prose is written to disk.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re

from sports_document_dgp import realize, sample_plan


WORD = re.compile(r"[A-Za-z]+|\d+")
SENTENCE = re.compile(r"[^.]+\.", re.S)
SCORE = re.compile(r"\b\d{2,3}\s*-\s*\d{2,3}\b")


def words(text):
    return [m.group().lower() for m in WORD.finditer(text)]


def trigrams(text):
    ws = words(text)
    return list(zip(ws, ws[1:], ws[2:]))


def sentences(doc):
    for match in SENTENCE.finditer(doc.text):
        sentence = match.group().strip()
        if any(match.start() <= site.start < match.end()
               for site in doc.sites):
            kind = 'answer'
        elif SCORE.search(sentence):
            kind = 'score'
        elif any(name in sentence for name in doc.plan.players):
            kind = 'player_stat'
        else:
            kind = 'other'
        yield kind, sentence


def summarize(rows):
    # The repeated-phrase metric is computed within a category. Each phrase
    # occurrence counts only if that phrase appears in a different document.
    document_frequency = Counter()
    occurrences = Counter()
    token_count = 0
    document_count = len(rows)
    for row in rows:
        grams = [gram for sentence in row for gram in trigrams(sentence)]
        document_frequency.update(set(grams))
        occurrences.update(grams)
        token_count += sum(len(words(sentence)) for sentence in row)
    total = sum(occurrences.values())
    repeated = sum(n for gram, n in occurrences.items()
                   if document_frequency[gram] > 1)
    return {'documents': document_count, 'word_tokens': token_count,
            'trigram_occurrences': total,
            'cross_document_repeated_occurrences': repeated,
            'cross_document_repeated_share': round(repeated / total, 4)
            if total else None,
            'unique_trigrams': len(occurrences)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=128)
    parser.add_argument('--seed', type=int, default=37731)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = {'config': {'n': args.n, 'seed': args.seed}, 'arms': {}}
    for source in ('rotowire', 'sbnation'):
        rng = random.Random(args.seed + int(source == 'sbnation') * 1000)
        plans = [sample_plan(rng, sbnation_rate=int(source == 'sbnation'),
                             source_group='train') for _ in range(args.n)]
        for empirical in (False, True):
            rows = defaultdict(list)
            sentence_counts = Counter()
            for plan in plans:
                by_kind = defaultdict(list)
                doc = realize(plan, pair_rate=0., stat_richness=1.,
                              discourse_entropy=1., lead_stat_rate=.25,
                              empirical_stat_rate=float(empirical))
                for kind, sentence in sentences(doc):
                    by_kind[kind].append(sentence)
                    sentence_counts[kind] += 1
                assert {kind: len(by_kind[kind]) for kind in
                        ('score', 'player_stat', 'answer', 'other')} == {
                            'score': 1, 'player_stat': 5,
                            'answer': 3, 'other': 0}
                for kind in ('score', 'player_stat', 'answer', 'other'):
                    rows[kind].append(by_kind[kind])
            key = source + ('_empirical' if empirical else '_fixed')
            out['arms'][key] = {
                'sentence_counts': dict(sentence_counts),
                'by_kind': {kind: summarize(group) for kind, group in rows.items()},
            }
            print(key, out['arms'][key], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
