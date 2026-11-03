"""Check the sampled event-world state before adding a text renderer."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

from sports_event_world_dgp import sample_world, world_pool


def check(world):
    assert world.final_score == world.period_end_scores[-1]
    assert world.period_end_scores[0][0] >= 0
    assert world.period_end_scores[0][1] >= 0
    assert all(a[0] <= b[0] and a[1] <= b[1]
               for a, b in zip(world.period_end_scores,
                               world.period_end_scores[1:]))
    run = world.interval
    assert run.start_action < run.end_action
    assert 1 <= run.period <= len(world.period_end_scores)
    assert run.points_for > run.points_against > 0
    assert run.points_for-run.points_against >= 6
    delta = (run.end_score[0]-run.start_score[0],
             run.end_score[1]-run.start_score[1])
    if run.side == 'home':
        assert delta == (run.points_for, run.points_against)
    else:
        assert run.side == 'away'
        assert delta == (run.points_against, run.points_for)
    assert all(a <= b for a, b in zip(run.end_score, world.final_score))
    assert world.top_player_points >= 0 and world.top_player


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=10303)
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pool = world_pool()
    for source in ('rotowire', 'sbnation'):
        train = {row[0] for row in pool[source, 'train']}
        heldout = {row[0] for row in pool[source, 'heldout']}
        assert not train & heldout
    for group in ('train', 'heldout'):
        all_train = {row[0] for source in ('rotowire', 'sbnation')
                     for row in pool[source, group]}
        other = 'heldout' if group == 'train' else 'train'
        all_other = {row[0] for source in ('rotowire', 'sbnation')
                     for row in pool[source, other]}
        assert not all_train & all_other
    out = {'config': {'seed': args.seed, 'n_per_arm': args.n},
           'arms': {},
           'caveat': 'Empirical-game latent worlds with event and box-score '
                     'facts; no text renderer, visible-prefix oracle, '
                     'language-form audit, or model training yet.'}
    for source in ('rotowire', 'sbnation'):
        for group in ('train', 'heldout'):
            rng = random.Random(args.seed+100*len(out['arms']))
            counts = Counter()
            examples = []
            for _ in range(args.n):
                world = sample_world(rng, source, group)
                check(world)
                counts['worlds_checked'] += 1
                counts['run_period/'+str(world.interval.period)] += 1
                counts['run_points_for_sum'] += world.interval.points_for
                counts['run_points_against_sum'] += \
                    world.interval.points_against
                if len(examples) < 2:
                    run = world.interval
                    examples.append({'home': world.home, 'away': world.away,
                                     'final': world.final_score,
                                     'top_scorer': world.top_player,
                                     'top_scorer_points':
                                     world.top_player_points,
                                     'run': {'team': world.home if
                                             run.side == 'home' else world.away,
                                             'period': run.period,
                                             'points_for': run.points_for,
                                             'points_against':
                                             run.points_against}})
            key = source+'_'+group
            out['arms'][key] = {'distinct_eligible_games':
                                len(pool[source, group]),
                                'counts': dict(counts),
                                'examples': examples}
            print(key, out['arms'][key]['distinct_eligible_games'],
                  dict(counts), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    main()
