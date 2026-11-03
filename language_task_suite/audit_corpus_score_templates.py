"""Audit reversible corpus-derived score-sentence realization on novel games."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from audit_document_structure_vs_fineweb import structure
from audit_rotowire_document_source import game_key, load
from corpus_score_sentence_renderer import extract, render
from score_template_guard import safe_for


DEFAULT_DIR = Path('/home/loganriggs/Coding/neuron_circuits/'
                   'artifacts/rotowire')


def _roles(row):
    home_score = int(row['home_line']['TEAM-PTS'])
    away_score = int(row['vis_line']['TEAM-PTS'])
    home_win = home_score > away_score
    home = row['home_city']+' '+row['home_name']
    away = row['vis_city']+' '+row['vis_name']
    winner, loser = (home, away) if home_win else (away, home)
    return home_win, winner, loser, max(home_score, away_score), min(
        home_score, away_score)


def audit(directory, seed=90213, n=128):
    rng = random.Random(seed)
    sources = {}
    all_train_keys = set()
    for name in ('rotowire', 'sbnation'):
        splits = load(directory/f'{name}.tar.bz2', name)
        templates, extraction_counts = extract(splits['train'])
        sources[name] = (splits, templates, extraction_counts)
        all_train_keys.update(map(game_key, splits['train']))
    heldout = []
    seen = set()
    for splits, _, _ in sources.values():
        for row in splits['valid']+splits['test']:
            key = game_key(row)
            if key in all_train_keys or key in seen:
                continue
            seen.add(key)
            heldout.append(row)
    panel = {}
    for source_name, (splits, templates, counts) in sources.items():
        by_role = {role: [template for template in templates
                          if template.home_win == role]
                   for role in (False, True)}
        assert all(by_role.values())
        texts = []
        guarded_texts = []
        for _ in range(n):
            game = rng.choice(heldout)
            home_win, winner, loser, high, low = _roles(game)
            template = rng.choice(by_role[home_win])
            text = render(template, winner, loser, high, low)
            changed = render(template, winner, loser, high+1, low)
            assert text != changed
            assert [i for i, (a, b) in enumerate(zip(text, changed))
                    if a != b] == [text.index(str(high))]
            assert all(token not in text for token in (
                '{WINNER}', '{LOSER}', '{WIN_SCORE}', '{LOSE_SCORE}'))
            texts.append(' '.join(text))
            guarded = [candidate for candidate in by_role[home_win]
                       if safe_for(candidate, high, low)]
            assert guarded, (source_name, home_win, high-low)
            safe_template = rng.choice(guarded)
            guarded_texts.append(' '.join(render(
                safe_template, winner, loser, high, low)))
        panel[source_name] = dict(
            extraction_counts=dict(counts),
            unique_templates=len(templates),
            by_home_win={str(role): len(rows)
                         for role, rows in by_role.items()},
            generated_unique_sentences=len(set(texts)),
            generated_form=structure(texts),
            guarded_template_counts_by_margin={str(margin): sum(
                safe_for(template, 100+margin, 100)
                for template in templates)
                for margin in (3, 8, 25)},
            guarded_generated_unique_sentences=len(set(guarded_texts)),
            guarded_generated_form=structure(guarded_texts))
    return dict(config=dict(seed=seed, n_per_source=n,
                            heldout_distinct_games=len(heldout),
                            training_game_keys=len(all_train_keys)),
                sources=panel,
                caveat='Only full-team names and the two score tokens are '
                       'certified slots. Other claims in extracted sentences '
                       'may depend on the source game. The guarded subset '
                       'is conservative but still requires manual semantic '
                       'validation before DGP use. '
                       'This is not yet a sampled multi-rule document DGP.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, default=DEFAULT_DIR)
    p.add_argument('--seed', type=int, default=90213)
    p.add_argument('--n', type=int, default=128)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.directory, args.seed, args.n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['config'])
    for name, row in result['sources'].items():
        print(name, row['unique_templates'], row['by_home_win'],
              row['generated_form']['word_types'],
              row['generated_form'][
                  'repeated_across_docs_trigram_occurrence_share'])


if __name__ == '__main__':
    main()
