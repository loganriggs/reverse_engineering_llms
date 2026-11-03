"""Frozen word-rank panels for two human sports-report corpora.

Only aggregate frequencies and source hashes are saved. No report text is
written to the repository.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from audit_rotowire_document_source import DEFAULT_ARCHIVE, game_key, load
from audit_surface_repetition import WORD


HERE = Path(__file__).resolve().parent


def summarize(rows):
    counts = Counter(word for row in rows for word in row)
    ranks = sorted(counts.values(), reverse=True)
    total = sum(ranks)
    assert all(len(row) == 100 for row in rows)
    slope = float(np.polyfit(np.log(np.arange(10, 101)),
                             np.log(np.asarray(ranks[9:100])/total), 1)[0])
    return {'summary': {'n_windows': len(rows), 'words_per_window': 100,
                        'total_words': total,
                        'observed_vocabulary': len(counts),
                        'top1_mass': ranks[0]/total,
                        'top10_mass': sum(ranks[:10])/total,
                        'tail_slope_rank10_to100': slope,
                        'hapax_types': sum(v == 1 for v in counts.values()),
                        'top25': counts.most_common(25)},
            'ranks': ranks}


def audit(n, seed, source_dir):
    sources = {}
    for dataset in ('rotowire', 'sbnation'):
        archive = source_dir/f'{dataset}.tar.bz2'
        train = load(archive, dataset)['train']
        unique = {}
        for row in train:
            unique.setdefault(game_key(row), row)
        sources[dataset] = {'rows': unique,
                            'archive_sha256': hashlib.sha256(
                                archive.read_bytes()).hexdigest()}
    common = set.intersection(*(set(s['rows']) for s in sources.values()))
    eligible = sorted(key for key in common if all(
        len(WORD.findall(' '.join(s['rows'][key]['summary']))) >= 100
        for s in sources.values()))
    if len(eligible) < n:
        raise RuntimeError(f'Only {len(eligible)} paired eligible games')
    selected = random.Random(seed).sample(eligible, n)
    panels = {}
    for dataset, source in sources.items():
        rows = [source['rows'][key] for key in selected]
        words = [[word.lower() for word in WORD.findall(
            ' '.join(row['summary']))[:100]] for row in rows]
        result = summarize(words)
        result['eligible_paired_games'] = len(eligible)
        result['archive_sha256'] = source['archive_sha256']
        result['panel_game_hashes'] = [hashlib.sha256(
            repr(key).encode()).hexdigest()[:16] for key in selected]
        panels[dataset] = result
    return {'config': {'n_windows_per_corpus': n, 'seed': seed,
                       'word_rule': 'first 100 lowercase WORD-regex words'},
            'panels': panels,
            'caveat': ('Human sports reports are a narrow genre, and these '
                       'are paired unique-game training-source samples. They '
                       'cannot supply a complete next-token oracle or '
                       'prove that a synthetic generator matches broader '
                       'web language. The FineWeb reference is independent.')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path,
                        default=DEFAULT_ARCHIVE.parent)
    parser.add_argument('--n', type=int, default=480)
    parser.add_argument('--seed', type=int, default=92031)
    parser.add_argument('--output', type=Path,
                        default=HERE/'results/natural_sports_zipf_n480.json')
    args = parser.parse_args()
    result = audit(args.n, args.seed, args.source_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for name, panel in result['panels'].items():
        s = panel['summary']
        print(name, s['observed_vocabulary'], s['top1_mass'],
              s['top10_mass'], s['hapax_types'], flush=True)
    print('Saved', args.output)


if __name__ == '__main__':
    main()
