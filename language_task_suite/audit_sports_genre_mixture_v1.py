"""Audit shared-game genre mixture without saving empirical text."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from audit_document_structure_vs_fineweb import structure
from audit_rotowire_document_source import game_key
from audit_sports_event_world_renderer import semantic_check
from compare_v12_fineweb_fresh import first_words
from sports_document_dgp import game_partition, resources
from sports_genre_mixture_v1 import sample_document
from sports_live_ledger_dgp import answer_from_visible_prefix, ledger_pool


def panel(seed, n, source, split, live_rate):
    rng = random.Random(seed)
    counts = Counter()
    by_genre = defaultdict(list)
    windows = defaultdict(list)
    selected_games = set()
    for _ in range(n):
        genre, chosen, pair = sample_document(
            rng, source, split, live_rate=live_rate)
        assert pair.retrospective.world.game_id == pair.game_id
        assert pair.live.latent['game_id'] == pair.game_id
        assert pair.retrospective.world.partition == split
        assert pair.live.latent['split'] == split
        selected_games.add(pair.game_id)
        counts['paired_worlds_checked'] += 1
        kinds, _ = semantic_check(pair.retrospective)
        counts['retrospective_run_sites_checked'] += len(kinds)
        for site in pair.live.sites:
            assert pair.live.tokens[site.index] == site.target
            task = 'copy' if site.family == 'live_score_copy' else 'run'
            assert answer_from_visible_prefix(
                pair.live.tokens[:site.index], task) == site.target
            counts['live_sites_checked'] += 1
        counts['drawn_genre/'+genre] += 1
        if genre == 'live':
            text = chosen.text()
            for site in chosen.sites:
                tokens = chosen.tokens
                windows['live/'+site.family].append(' '.join(
                    tokens[max(0, site.index-25):site.index+16]))
                counts['drawn_site/'+site.family] += 1
        else:
            text = chosen.text
            for site in chosen.sites:
                windows['retrospective/'+site.kind].append(' '.join(
                    text[:site.start].split()[-25:] +
                    text[site.start:].split()[:16]))
                counts['drawn_site/'+site.kind] += 1
        by_genre[genre].append(first_words(text, 30))
    return {'counts': dict(counts), 'distinct_games': len(selected_games),
            'first30': {kind: structure(rows)
                        for kind, rows in by_genre.items()},
            'task_windows': {kind: structure(rows)
                             for kind, rows in windows.items()}}


def human_reference(seed, n, source, split):
    records = {}
    for row in resources()[0][source][0]:
        if game_partition(game_key(row)) != split:
            continue
        crop = first_words(' '.join(row['summary']), 30)
        if len(crop.split()) >= 30:
            records.setdefault(game_key(row), crop)
    if len(records) < n:
        raise RuntimeError('Insufficient human reference reports')
    sample = random.Random(seed).sample(list(records.values()), n)
    return structure(sample)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=200)
    p.add_argument('--seed', type=int, default=52101)
    p.add_argument('--live-rate', type=float, default=.25)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    pools = ledger_pool()
    for source in ('rotowire', 'sbnation'):
        train = {row[0][0] for row in pools[source, 'train']}
        heldout = {row[0][0] for row in pools[source, 'heldout']}
        assert not train & heldout
    out = {'config': {'n_per_source_split': a.n, 'seed': a.seed,
                      'live_rate': a.live_rate},
           'eligible_games': {f'{source}/{split}': len(pools[source, split])
                              for source in ('rotowire', 'sbnation')
                              for split in ('train', 'heldout')},
           'arms': {},
           'caveat': 'The two genre views share a real game ID but not '
                     'necessarily the same scoring interval. Retrospective '
                     'run sites are rare at reference rates. Live play '
                     'descriptions are empirical text and no matched '
                     'human live-feed form reference is provided. '
                     'Only aggregate counts and forms are saved.'}
    for source in ('rotowire', 'sbnation'):
        for split in ('train', 'heldout'):
            key = f'{source}/{split}'
            result = panel(a.seed+len(out['arms'])*1000, a.n,
                           source, split, a.live_rate)
            live_n = result['counts'].get('drawn_genre/live', 0)
            retro_n = result['counts'].get('drawn_genre/retrospective', 0)
            result['human_retrospective_first30'] = human_reference(
                a.seed+len(out['arms'])*1000+500, max(1, retro_n),
                source, split)
            out['arms'][key] = result
            print(key, 'live', live_n, 'retro', retro_n,
                  'run sites', result['counts'].get('drawn_site/run_for', 0),
                  flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2) + '\n')
    print('saved', a.output, flush=True)


if __name__ == '__main__':
    main()
