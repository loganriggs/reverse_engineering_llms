"""Visible-rule, edit-selectivity, and natural-form audit for multirule carrier."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from audit_document_structure_vs_fineweb import structure
from audit_rotowire_document_source import game_key
from compare_v12_fineweb_fresh import first_words
from sports_document_dgp import game_partition, resources
from sports_report_carrier_dgp import eligible
from sports_report_multirule_dgp import (
    increment_player, increment_score, realize, sample_plan,
    template_digest, visible_site_answer,
)


def window(doc, site):
    return ' '.join(doc.text[:site.start].split()[-25:]+
                    doc.text[site.start:].split()[:16])


def audit(group, seed, n, form_n, carrier_words):
    rng = random.Random(seed)
    counts = Counter()
    seen = set()
    form_keys = []
    output_texts = []
    original_texts = []
    full_texts = []
    length_matched_sources = []
    windows = {kind: [] for kind in ('winning_score', 'margin', 'player_points')}
    row_by_key = {game_key(row): row for row in resources()[0]['rotowire'][0]}
    examples = []
    for _ in range(n):
        plan = sample_plan(rng, group=group, carrier_words=carrier_words)
        assert game_partition(plan.carrier_plan.source_game_key) == group
        original = realize(plan)
        score_edit = realize(increment_score(plan))
        player_edit = realize(increment_player(plan))
        documents = (original, score_edit, player_edit)
        for doc in documents:
            assert [s.kind for s in doc.sites] == \
                [s.kind for s in original.sites]
            for site in doc.sites:
                assert doc.text[site.start:site.end] == site.answer
                assert visible_site_answer(doc, site) == site.answer
                counts['visible_site_checks'] += 1
        answers = [{site.kind: site.answer for site in doc.sites}
                   for doc in documents]
        counts['query_order/'+'>'.join(site.kind for site in original.sites)] += 1
        assert int(answers[1]['winning_score']) == \
            int(answers[0]['winning_score'])+1
        assert int(answers[1]['margin']) == int(answers[0]['margin'])+1
        assert answers[1]['player_points'] == answers[0]['player_points']
        assert answers[2]['winning_score'] == answers[0]['winning_score']
        assert answers[2]['margin'] == answers[0]['margin']
        assert int(answers[2]['player_points']) == \
            int(answers[0]['player_points'])+1
        assert score_edit.plan.carrier_plan.carrier == \
            player_edit.plan.carrier_plan.carrier == \
            original.plan.carrier_plan.carrier
        counts['selective_edit_triplets'] += 1
        key = plan.carrier_plan.source_game_key
        if key not in seen and len(output_texts) < form_n:
            seen.add(key)
            form_keys.append(key)
            output_texts.append(first_words(original.text, 100))
            source_text = ' '.join(row_by_key[key]['summary'])
            original_texts.append(first_words(source_text, 100))
            full_texts.append(original.text)
            length_matched_sources.append(first_words(
                source_text, len(original.text.split())))
            for site in original.sites:
                windows[site.kind].append(window(original, site))
        if len(examples) < 2:
            examples.append(original.text)
    if len(output_texts) < form_n:
        raise RuntimeError('Too few distinct games in form panel')
    return {'counts': dict(counts), 'eligible_games': len(eligible(
                'rotowire', group, carrier_words)),
            'distinct_form_games': len(form_keys),
            'form_game_sha256': [hashlib.sha256(repr(k).encode()).hexdigest()
                                 for k in form_keys],
            'generated_form': structure(output_texts),
            'paired_source_form': structure(original_texts),
            'full_generated_form': structure(full_texts),
            'length_matched_source_form': structure(length_matched_sources),
            'site_windows': {kind: structure(texts)
                             for kind, texts in windows.items()},
            'manual_examples': examples}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=96601)
    p.add_argument('--n', type=int, default=300)
    p.add_argument('--form-n', type=int, default=64)
    p.add_argument('--carrier-words', type=int, default=100)
    a = p.parse_args()
    result = {'config': {**vars(a), 'output': str(a.output),
                         'stat_template_sha256': template_digest()},
              'arms': {},
              'caveat': 'Source prose is empirical and lacks a full symbolic '
                        'oracle. Only inserted fact and answer sites are '
                        'scored. Player edits avoid name mentions in the '
                        'carrier and rank changes but cannot prove every '
                        'surrounding human claim unaffected.'}
    for group in ('train', 'heldout'):
        record = audit(group, a.seed+(group == 'heldout')*100,
                       a.n, a.form_n, a.carrier_words)
        examples = record.pop('manual_examples')
        result['arms'][group] = record
        print(group, record['counts'], 'eligible', record['eligible_games'],
              'examples', examples, flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
