"""Find same-game human reports with distinct styles and consistent scores."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from audit_rotowire_document_source import game_key, load, score_pair_present


DEFAULT_DIR = Path('/home/loganriggs/Coding/neuron_circuits/'
                   'artifacts/rotowire')
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+")


def _score(row):
    return (row['home_line']['TEAM-PTS'],
            row['vis_line']['TEAM-PTS'])


def _first_words(row, n=100):
    return set(word.lower() for word in WORD.findall(
        ' '.join(row['summary']))[:n])


def _jaccard(a, b):
    return len(a & b)/len(a | b)


def audit(directory, seed=71403):
    datasets = {}
    for name in ('rotowire', 'sbnation'):
        archive = directory/f'{name}.tar.bz2'
        rows = load(archive, name)
        by_game = defaultdict(list)
        for split, items in rows.items():
            for row in items:
                by_game[game_key(row)].append((split, row))
        datasets[name] = by_game
    a, b = datasets['rotowire'], datasets['sbnation']
    common = sorted(set(a) & set(b))
    counts = Counter()
    pairs = []
    for key in common:
        left = [row for _, row in a[key]]
        right = [row for _, row in b[key]]
        compatible = [(x, y) for x in left for y in right
                      if _score(x) == _score(y)]
        if not compatible:
            counts['same_date_teams_score_conflict'] += 1
            continue
        counts['same_game_score_consistent'] += 1
        x, y = compatible[0]
        has_x, has_y = score_pair_present(x), score_pair_present(y)
        counts[f'exact_score_pair_rotowire{int(has_x)}_sbnation{int(has_y)}'] += 1
        pairs.append((x, y))
    rng = random.Random(seed)
    same = [_jaccard(_first_words(x), _first_words(y)) for x, y in pairs]
    shuffled = [y for _, y in pairs]
    rng.shuffle(shuffled)
    random_game = [_jaccard(_first_words(x), _first_words(y))
                   for (x, _), y in zip(pairs, shuffled)]
    return dict(source='https://github.com/harvardnlp/boxscore-data',
                seed=seed, common_date_team_keys=len(common),
                counts=dict(counts),
                first100_word_type_jaccard=dict(
                    same_game_mean=statistics.mean(same),
                    same_game_median=statistics.median(same),
                    random_game_mean=statistics.mean(random_game),
                    random_game_median=statistics.median(random_game)),
                caveat='A shared date/team key can have score conflicts; '
                       'only score-consistent pairs are considered matched. '
                       'Human summaries are not exact paraphrases and may '
                       'contain facts absent from the box scores. Group '
                       'all records from a game across both corpora before '
                       'any model split.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, default=DEFAULT_DIR)
    p.add_argument('--seed', type=int, default=71403)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.directory, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['common_date_team_keys'], result['counts'])
    print(result['first100_word_type_jaccard'])


if __name__ == '__main__':
    main()
