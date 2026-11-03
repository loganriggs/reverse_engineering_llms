"""Exact visible-prefix, semantic-order, causal-edit and form audit for V2."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from audit_document_structure_vs_fineweb import structure
from compare_v12_fineweb_fresh import first_words
from event_frame_language_v1 import Event, mixture
from event_frame_language_v2 import (
    ADJECTIVES, OBJECTS, VERBS, FRAME_WEIGHTS, TEMPLATES, ID2TOKEN,
    EventFrameLanguageV2, parse_event, render, sentence, split_prefix)


def sentence_probability(gen, history, event, frame):
    prefix = [word for old in history for word in sentence(old, 'active')]
    p = 1.
    for word in sentence(event, frame):
        q = gen.next_distribution(prefix)
        p *= q[word]
        prefix.append(word)
    return p


def independent_semantic_distribution(gen, history, observed, target):
    """Enumerate the joint law directly, independently of the oracle table."""
    counts = Counter()
    for adj in ADJECTIVES:
        pa = gen._prior(ADJECTIVES)[adj]
        for obj, po in mixture(gen.object_components(history, adj)).items():
            for verb, pv in mixture(gen.verb_components(
                    history, adj, obj)).items():
                row = {'adjective': adj, 'object': obj, 'verb': verb}
                if all(row[k] == v for k, v in observed.items()
                       if k in row):
                    counts[row[target]] += pa*po*pv
    total = sum(counts.values())
    return {word: mass/total for word, mass in counts.items()}


def compare_distributions(actual, expected):
    assert set(actual) == set(expected)
    assert max(abs(actual[k]-expected[k]) for k in actual) < 1e-10


def semantic_checks(gen):
    old = Event('blue', 'parcel', 'sent', 'gallery', 'Mara')
    target = Event('old', 'device', 'moved', 'clinic', 'Nina')
    history = (old,)
    for frame in TEMPLATES:
        assert parse_event(sentence(target, frame)) == (target, frame)
        p = sentence_probability(gen, history, target, frame)
        base = sentence_probability(gen, history, target, 'active')
        assert abs(p/base-FRAME_WEIGHTS[frame]/FRAME_WEIGHTS['active']) < 1e-10
    irregular = Event('blue', 'parcel', 'took', 'gallery', 'Mara')
    assert sentence(irregular, 'active')[1] == 'took'
    assert sentence(irregular, 'passive')[4] == 'taken'
    for frame in TEMPLATES:
        assert parse_event(sentence(irregular, frame)) == (irregular, frame)
        ratio = (sentence_probability(gen, history, irregular, frame) /
                 sentence_probability(gen, history, irregular, 'active'))
        assert abs(ratio-FRAME_WEIGHTS[frame]/FRAME_WEIGHTS['active']) < 1e-10
    prefix = list(sentence(old, 'active'))
    cases = [
        (prefix+['Leo'], {}, 'verb'),
        (prefix+['It', 'was', 'Leo', 'who'], {}, 'verb'),
        (prefix+['The', 'blue'], {'adjective': 'blue'}, 'object'),
        (prefix+['What', 'Leo', 'sent', 'to', 'the', 'gallery',
                 'was', 'the', 'blue'],
         {'verb': 'sent', 'adjective': 'blue'}, 'object'),
    ]
    for text, observed, target_field in cases:
        compare_distributions(gen.next_distribution(text),
                              independent_semantic_distribution(
                                  gen, history, observed, target_field))
    passive_verb = gen.next_distribution(prefix+['The', 'blue',
                                                  'parcel', 'was'])
    expected_passive = independent_semantic_distribution(
        gen, history, {'adjective': 'blue', 'object': 'parcel'}, 'verb')
    assert passive_verb['taken'] == expected_passive['took']
    assert 'took' not in passive_verb
    # A prior-event edit must change the posterior, including the branch
    # where the current verb precedes its adjective/object.
    prior_verb_edit = Event('blue', 'parcel', 'carried', 'gallery', 'Mara')
    active_before = gen.next_distribution(prefix+['Leo'])
    active_after = gen.next_distribution(
        list(sentence(prior_verb_edit, 'active'))+['Leo'])
    assert active_before != active_after
    assert active_before['sent'] > active_after['sent']
    # The late object posterior must also respond to the earlier adjective
    # binding after conditioning on the already visible verb.
    prior_obj_edit = Event('blue', 'letter', 'sent', 'gallery', 'Mara')
    suffix = ['What', 'Leo', 'sent', 'to', 'the', 'gallery',
              'was', 'the', 'blue']
    object_before = gen.next_distribution(prefix+suffix)
    object_after = gen.next_distribution(
        list(sentence(prior_obj_edit, 'active'))+suffix)
    assert object_before != object_after
    assert object_before['parcel'] > object_after['parcel']
    return {'frame_equivalence': True, 'independent_semantic_marginals':
            len(cases), 'prior_verb_edit': True, 'prior_object_edit': True}


def panel(seed, n, length):
    gen = EventFrameLanguageV2(seed)
    checks = semantic_checks(gen)
    counts = Counter()
    texts = []
    for _ in range(n):
        ids, labels, returned = gen.sample_sequence(length)
        words = [ID2TOKEN[index] for index in ids]
        prefix = []
        for word, label, expected in zip(words, labels, returned):
            q = gen.next_distribution(prefix)
            compare_distributions(q, expected)
            assert abs(sum(q.values())-1) < 1e-10
            assert word in q and q[word] > 0
            assert label == gen.label(prefix)
            counts['label/'+label] += 1
            prefix.append(word)
        history, _ = split_prefix(words)
        counts['complete_events'] += len(history)
        starts = []
        for word in words:
            starts.append(word)
            if word == '.':
                counts['frame/'+parse_event(starts)[1]] += 1
                starts = []
        cropped = first_words(render(ids), 100)
        assert len(cropped.split()) >= 100
        texts.append(cropped)
    return {'checks': checks, 'counts': dict(counts),
            'first100_form': structure(texts),
            'examples': [render(gen.sample_sequence(80)[0]) for _ in range(3)]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=128)
    p.add_argument('--length', type=int, default=256)
    p.add_argument('--seed', type=int, default=52901)
    p.add_argument('--reference', type=Path, default=Path(
        'language_task_suite/results/v14_frame_vs_fineweb_skip94500_n128.json'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    ref = json.loads(args.reference.read_text())
    assert ref['config']['n'] == args.n
    first, second = (panel(args.seed+i, args.n, args.length)
                     for i in range(2))
    result = {'config': {'n': args.n, 'length': args.length,
                         'seeds': [args.seed, args.seed+1],
                         'reference': str(args.reference)},
              'panels': {'seed_a': first, 'seed_b': second},
              'form_reference': {name: (value if name == 'fineweb'
                                       else value['form'])
                                 for name, value in ref['panels'].items()},
              'caveat': 'Five-frame event grammar is a correctness and '
                        'form experiment, not an accepted full multi-task DGP.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for name, row in result['panels'].items():
        f = row['first100_form']
        print(name, row['counts'], 'starts', f['sentence_start_types'],
              'trigram', f['repeated_across_docs_trigram_occurrence_share'],
              flush=True)


if __name__ == '__main__':
    main()
