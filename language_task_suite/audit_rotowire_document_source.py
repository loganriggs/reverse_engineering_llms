"""Screen full human-written game reports as a grounded DGP surface source.

Reads the official RotoWire archive locally. Saves aggregates and source
hashes only, not game summaries or box scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import tarfile
from collections import Counter
from pathlib import Path

from audit_document_structure_vs_fineweb import structure
from compare_v12_fineweb_fresh import first_words


DEFAULT_ARCHIVE = Path('/home/loganriggs/Coding/neuron_circuits/'
                       'artifacts/rotowire/rotowire.tar.bz2')


def load(archive, dataset):
    with tarfile.open(archive, 'r:bz2') as handle:
        return {split: json.load(handle.extractfile(f'{dataset}/{split}.json'))
                for split in ('train', 'valid', 'test')}


def game_key(row):
    teams = sorted(((row['home_city'], row['home_name']),
                    (row['vis_city'], row['vis_name'])))
    return (row['day'], *teams)


def score_pair_present(row):
    tokens = row['summary']
    a, b = (row['home_line']['TEAM-PTS'],
            row['vis_line']['TEAM-PTS'])
    for left, sep, right in zip(tokens, tokens[1:], tokens[2:]):
        if sep in ('-', 'to', '–') and (left, right) in ((a, b), (b, a)):
            return True
    return False


def player_point_mentions(row):
    """Conservative surface alignment, not semantic assertion."""
    tokens = row['summary']
    table = row['box_score']
    matches = 0
    for key, name in table['PLAYER_NAME'].items():
        points = table['PTS'].get(key)
        name_tokens = name.split()
        if not points or not name_tokens:
            continue
        width = len(name_tokens)
        for i in range(len(tokens)-width+1):
            if tokens[i:i+width] != name_tokens:
                continue
            window = tokens[i+width:i+width+16]
            if points in window and any(word.lower() in ('points', 'point',
                                                        'pts') for word in window):
                matches += 1
                break
    return matches


def audit(archive, dataset='rotowire', seed=20261103,
          form_n=64, form_words=100):
    splits = load(archive, dataset)
    keys = {split: {game_key(row) for row in rows}
            for split, rows in splits.items()}
    hashes = {split: {hashlib.sha256(' '.join(row['summary']).encode())
                      .hexdigest() for row in rows}
              for split, rows in splits.items()}
    train = splits['train']
    counts = Counter()
    lengths = []
    for row in train:
        lengths.append(len(row['summary']))
        counts['score_pair_exact_surface'] += score_pair_present(row)
        mentions = player_point_mentions(row)
        counts['player_points_surface_mentions'] += mentions
        counts['documents_with_player_points_surface'] += mentions > 0
        counts['documents_100_words'] += len(row['summary']) >= form_words
    eligible = [row for row in train if len(row['summary']) >= form_words]
    panel = random.Random(seed).sample(eligible, form_n)
    texts = [first_words(' '.join(row['summary']), form_words)
             for row in panel]
    assert all(len(text.split()) >= form_words for text in texts)
    return dict(source='https://github.com/harvardnlp/boxscore-data',
                dataset=dataset,
                archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                split_sizes={name: len(rows) for name, rows in splits.items()},
                unique_game_keys={name: len(values)
                                  for name, values in keys.items()},
                game_key_overlap={
                    f'{a}_{b}': len(keys[a] & keys[b])
                    for a, b in (('train', 'valid'), ('train', 'test'),
                                 ('valid', 'test'))},
                exact_summary_overlap={
                    f'{a}_{b}': len(hashes[a] & hashes[b])
                    for a, b in (('train', 'valid'), ('train', 'test'),
                                 ('valid', 'test'))},
                train_counts=dict(counts),
                train_summary_tokens=dict(min=min(lengths),
                                          median=statistics.median(lengths),
                                          mean=statistics.mean(lengths),
                                          max=max(lengths)),
                form_config=dict(seed=seed, n=form_n, words=form_words),
                form=structure(texts),
                caveat='Exact score pairs and player point mentions are '
                       'surface matches, not validated factual entailments. '
                       'The corpus is empirical; its full next-token law '
                       'and the writers\' unobserved evidence are unknown. '
                       'Group by game before any training split.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, default=DEFAULT_ARCHIVE)
    p.add_argument('--dataset', choices=('rotowire', 'sbnation'),
                   default='rotowire')
    p.add_argument('--seed', type=int, default=20261103)
    p.add_argument('--form-n', type=int, default=64)
    p.add_argument('--form-words', type=int, default=100)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.archive, args.dataset, args.seed, args.form_n,
                   args.form_words)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['split_sizes'])
    print(result['game_key_overlap'])
    print(result['train_counts'])
    print('form', result['form']['word_types'],
          result['form']['repeated_across_docs_trigram_occurrence_share'])


if __name__ == '__main__':
    main()
