"""Semantic and held-out form screens for the event-world text pilot."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re

from audit_document_structure_vs_fineweb import structure
from audit_rotowire_document_source import game_key
from compare_v12_fineweb_fresh import first_words
from sports_document_dgp import game_partition, resources
from sports_event_world_dgp import sample_world
from sports_event_world_renderer import (render_world,
                                         parse_result_sentence,
                                         run_values_from_visible_prefix)


SENTENCE = re.compile(r'[^.!?]*[.!?]')
PAIR = re.compile(r'\b(\d+)-(\d+)\b')
PLAYER = re.compile(r'(.+?) (?:scored|finished with|posted) (\d+) points\.$')


def semantic_check(doc):
    world = doc.world
    results = []
    player_facts = []
    period_facts = []
    for raw in doc.sentences:
        sentence = raw.strip()
        try:
            results.append(parse_result_sentence(sentence))
        except ValueError:
            pass
        player = PLAYER.fullmatch(sentence)
        if player:
            player_facts.append((player.group(1), int(player.group(2))))
        pair = PAIR.search(sentence)
        if pair and not ('run' in sentence or
                         'Later that quarter' in sentence):
            for label, p in (('halftime', 2), ('first quarter', 1),
                             ('third quarter', 3)):
                if label in sentence.lower() and 'had' not in sentence:
                    period_facts.append((p, tuple(map(int, pair.groups())),
                                         sentence))
    home_score, away_score = world.final_score
    winner = world.home if home_score > away_score else world.away
    assert results == [(max(world.final_score), min(world.final_score),
                        winner)]
    assert player_facts == [(world.top_player, world.top_player_points)]
    for period, pair, sentence in period_facts:
        home, away = world.period_end_scores[period-1]
        assert pair == tuple(sorted((home, away), reverse=True))
        if home == away:
            assert 'tied' in sentence.lower()
        else:
            leader = world.home if home > away else world.away
            if ' led ' in sentence:
                assert sentence.startswith(leader+' led ')
            else:
                assert f'{leader} held a ' in sentence
    kinds = {site.kind for site in doc.sites}
    assert kinds in (set(), {'run_for', 'run_against'})
    for site in doc.sites:
        assert doc.text[site.start:site.end] == site.answer
        run_values = run_values_from_visible_prefix(doc.text[:site.start])
        expected = (world.interval.points_for,
                    world.interval.points_against)
        assert run_values == expected
        assert site.answer == str(expected[0] if site.kind == 'run_for'
                                  else expected[1])
    return kinds, len(period_facts)


def semantic_panel(seed, n, source, partition):
    rng = random.Random(seed)
    counts = Counter()
    examples = []
    for _ in range(n):
        world = sample_world(rng, source, partition)
        doc = render_world(world, rng, run_rate=1)
        kinds, period_facts = semantic_check(doc)
        assert kinds == {'run_for', 'run_against'}
        counts['documents_checked'] += 1
        counts['run_answer_sites_checked'] += len(doc.sites)
        counts['period_claims_checked'] += period_facts
        if len(examples) < 2:
            examples.append(doc.text)
    return {'counts': dict(counts), 'generated_examples': examples}


def form_panel(seed, n, words, source, partition):
    rng = random.Random(seed)
    generated = []
    attempts = 0
    selected_run_docs = 0
    while len(generated) < n:
        attempts += 1
        if attempts > 300*n:
            raise RuntimeError('Too few long generated documents')
        world = sample_world(rng, source, partition)
        doc = render_world(world, rng)
        crop = first_words(doc.text, words)
        if len(crop.split()) >= words:
            generated.append(crop)
            selected_run_docs += bool(doc.sites)
    human = {}
    for row in resources()[0][source][0]:
        if game_partition(game_key(row)) != partition:
            continue
        crop = first_words(' '.join(row['summary']), words)
        if len(crop.split()) >= words:
            human.setdefault(game_key(row), crop)
    if len(human) < n:
        raise RuntimeError('Too few human reference games')
    reference = random.Random(seed+50000).sample(list(human.values()), n)
    return {'matched_crops': n, 'draw_attempts': attempts,
            'selected_run_docs': selected_run_docs,
            'generated': structure(generated),
            'human_reference': structure(reference)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=10304)
    parser.add_argument('--semantic-n', type=int, default=100)
    parser.add_argument('--form-n', type=int, default=64)
    parser.add_argument('--words', type=int, default=40)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = {'config': vars(args) | {'output': str(args.output)},
           'arms': {},
           'caveat': 'Run semantic panel forces run insertion; form panel '
                     'uses genre-conditioned default run rate. Full-token '
                     'next-token oracle and model training are absent.'}
    for source in ('rotowire', 'sbnation'):
        for partition in ('train', 'heldout'):
            seed = args.seed+100*len(out['arms'])
            key = source+'_'+partition
            out['arms'][key] = {
                'semantic': semantic_panel(seed, args.semantic_n,
                                           source, partition),
                'form': form_panel(seed+10000, args.form_n, args.words,
                                   source, partition)}
            result = out['arms'][key]['form']
            metric = 'repeated_across_docs_trigram_occurrence_share'
            print(key, out['arms'][key]['semantic']['counts'],
                  'repeated_trigram',
                  result['generated'][metric],
                  result['human_reference'][metric], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    main()
