"""Independent exact-oracle, frame-equivalence and form audit for V16 pilot."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from audit_document_structure_vs_fineweb import structure
from compare_v12_fineweb_fresh import first_words
from event_frame_language_v1 import (
    Event, EventFrameLanguage, ID2TOKEN, TOKEN2ID, VOCAB,
    parse_event, render, sentence, split_prefix)


def probability(gen, history, event, frame):
    prefix = [token for old in history for token in
              sentence(old, 'topical')]
    value = 1.
    for token in sentence(event, frame):
        q = gen.next_distribution(prefix)
        value *= q[token]
        prefix.append(token)
    return value


def prefix_checks(gen):
    old = Event('blue', 'parcel', 'sent', 'gallery', 'Mara')
    changed_verb = Event('blue', 'parcel', 'carried', 'gallery', 'Mara')
    changed_object = Event('blue', 'letter', 'sent', 'gallery', 'Mara')
    new = Event('old', 'device', 'moved', 'clinic', 'Nina')
    history = [old]
    for frame in ('topical', 'passive'):
        assert parse_event(sentence(new, frame))[0] == new
        ratio = probability(gen, history, new, frame) / probability(
            gen, history, new, 'topical')
        expected = (1. if frame == 'topical' else
                    (1-gen.topical_rate)/gen.topical_rate)
        assert abs(ratio-expected) < 1e-10
    # Both grammatical orders expose the same object and verb rule inputs.
    base = list(sentence(old, 'topical'))
    object_topical = base+['As', 'for', 'the', 'blue']
    object_passive = base+['The', 'blue']
    assert gen.next_distribution(object_topical) == gen.next_distribution(
        object_passive)
    verb_topical = base+['As', 'for', 'the', 'blue', 'parcel', ',', 'Leo']
    verb_passive = base+['The', 'blue', 'parcel', 'was']
    assert gen.next_distribution(verb_topical) == gen.next_distribution(
        verb_passive)
    # A supported adjective edit changes the local proposal, with all
    # candidate verbs still in support through the lexicon component.
    changed_adjective = base+['The', 'old', 'parcel', 'was']
    before = gen.next_distribution(verb_passive)
    after = gen.next_distribution(changed_adjective)
    assert before != after and all(before[v] > 0 and after[v] > 0
                                       for v in ('sent', 'carried'))
    # Editing only the earlier verb moves precisely the induction mass.
    other = list(sentence(changed_verb, 'topical'))+verb_passive[len(base):]
    q = gen.next_distribution(other)
    assert abs(before['sent']-q['sent']-.15) < 1e-10
    assert abs(q['carried']-before['carried']-.15) < 1e-10
    # Editing the earlier object changes the object-induction component.
    other = list(sentence(changed_object, 'topical'))+object_passive[len(base):]
    q = gen.next_distribution(other)
    old_q = gen.next_distribution(object_passive)
    assert abs(old_q['parcel']-q['parcel']-.20) < 1e-10
    assert abs(q['letter']-old_q['letter']-.20) < 1e-10
    return {'same_event_probability_ratio': True,
            'object_distribution_frame_equivalence': True,
            'verb_distribution_frame_equivalence': True,
            'local_adjective_edit_supported': True,
            'earlier_verb_induction_shift': .15,
            'earlier_object_induction_shift': .20}


def panel(seed, n, length):
    gen = EventFrameLanguage(seed)
    checks = prefix_checks(gen)
    counts = Counter()
    texts = []
    for _ in range(n):
        ids, labels, returned = gen.sample_sequence(length)
        tokens = [ID2TOKEN[index] for index in ids]
        prefix = []
        for token, label, expected in zip(tokens, labels, returned):
            q = gen.next_distribution(prefix)
            assert q == expected and abs(sum(q.values())-1) < 1e-10
            assert all(word in VOCAB and mass > 0 for word, mass in q.items())
            assert q[token] > 0 and gen.label(prefix) == label
            if label in ('mixed_object', 'mixed_verb'):
                history, current = split_prefix(prefix)
                adjective = current[3] if current[0] == 'As' else current[1]
                if label == 'mixed_object':
                    parts = gen.object_components(history, adjective)
                else:
                    object_name = (current[4] if current[0] == 'As'
                                   else current[2])
                    parts = gen.verb_components(history, adjective,
                                                object_name)
                supports = Counter(word for name, (_, dist) in parts.items()
                                   if name != 'lexicon' for word in dist)
                counts['nonbaseline_collision_sites'] += any(
                    n > 1 for n in supports.values())
                counts['active_proposals/'+label] += len(parts)
            counts['label/'+label] += 1
            prefix.append(token)
        history, _ = split_prefix(tokens)
        counts['complete_events'] += len(history)
        segment = []
        for token in tokens:
            segment.append(token)
            if token == '.':
                counts['topical_events'] += segment[0] == 'As'
                segment = []
        crop = first_words(render(ids), 100)
        assert len(crop.split()) >= 100
        texts.append(crop)
    return {'checks': checks, 'counts': dict(counts),
            'first100_form': structure(texts)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=128)
    p.add_argument('--length', type=int, default=256)
    p.add_argument('--seed', type=int, default=52401)
    p.add_argument('--reference', type=Path, default=Path(
        'language_task_suite/results/v14_frame_vs_fineweb_skip94500_n128.json'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    reference = json.loads(a.reference.read_text())
    assert reference['config']['n'] == a.n
    first, second = panel(a.seed, a.n, a.length), panel(
        a.seed+1, a.n, a.length)
    result = {'config': {'n': a.n, 'length': a.length,
                         'seeds': [a.seed, a.seed+1],
                         'reference': str(a.reference)},
              'panels': {'seed_a': first, 'seed_b': second},
              'form_reference': {
                  name: (value if name == 'fineweb' else value['form'])
                  for name, value in reference['panels'].items()},
              'caveat': 'Two-frame event grammar is a correctness fixture, '
                        'not a realistic language distribution or complete '
                        'successor to V14. It lacks V14 source, bracket, '
                        'attribute and price families.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    for name, row in result['panels'].items():
        form = row['first100_form']
        print(name, row['counts'], 'starts',
              form['sentence_start_types'], 'trigram',
              form['repeated_across_docs_trigram_occurrence_share'],
              flush=True)


if __name__ == '__main__':
    main()
