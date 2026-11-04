"""Verify variable event frames, hidden-genre marginalization, and edits."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from audit_surface_repetition import WORD, form_cues, score
from event_frame_oracle import ACTION, PASSIVE, EventFrameOracle, GENRES


def action_edit_fixture(generator):
    obj = generator.objects[0]
    person = generator.people[0]
    checked = 0
    for fact_frame in range(3):
        for action_frame in range(3):
            fact = [row for row in generator.candidates((), GENRES[0])
                    if row.kind == 'fact' and row.obj == obj and
                    row.attribute == 'clean'][fact_frame]
            # The candidate order is object, attribute, then frame.
            assert fact.tokens in [row.tokens for row in generator.candidates(
                (), GENRES[0]) if row.kind == 'fact' and row.obj == obj and
                row.attribute == 'clean']
            state = ((obj, 'clean'),)
            action = [row for row in generator.candidates(state, GENRES[0])
                      if row.kind == 'action' and row.obj == obj and
                      row.attribute == 'clean' and person in row.tokens][
                          action_frame]
            prefix = fact.tokens + action.tokens[:action.action_position]
            q = generator.next_distribution(prefix)
            expected = PASSIVE['clean'] if action_frame == 1 else ACTION['clean']
            edited_expected = (PASSIVE['dusty'] if action_frame == 1
                               else ACTION['dusty'])
            assert q.get(expected, 0.) > 0 and q.get(edited_expected, 0.) == 0
            edited_fact = tuple('dusty' if token == 'clean' else token
                                for token in fact.tokens)
            edited = edited_fact + action.tokens[:action.action_position]
            edited_q = generator.next_distribution(edited)
            assert edited_q.get(edited_expected, 0.) > 0
            assert edited_q.get(expected, 0.) == 0
            assert abs(q[expected]-edited_q[edited_expected]) < 1e-10
            assert {key: value for key, value in q.items()
                    if key != expected} == {
                    key: value for key, value in edited_q.items()
                    if key != edited_expected}
            checked += 1
    return checked


def summarize(generator, n):
    counts = Counter()
    texts, examples = [], []
    worst_relative_error = 0.
    for _ in range(n):
        sample = generator.sample_document()
        direct = generator.direct_document_probability(sample.tokens)
        chain = 1.
        for t, token in enumerate(sample.tokens):
            q = generator.next_distribution(sample.tokens[:t])
            assert abs(sum(q.values()) - 1.) < 1e-10
            assert q.get(token, 0.) > 0
            chain *= q[token]
            counts['sampled_tokens'] += 1
        relative = abs(direct-chain)/direct
        worst_relative_error = max(worst_relative_error, relative)
        assert relative < 1e-8
        counts[f'genre_{sample.genre}'] += 1
        counts['actions'] += len(sample.actions)
        counts['facts'] += sample.kinds.count('fact')
        counts['action_documents'] += bool(sample.actions)
        for site in sample.actions:
            q = generator.next_distribution(sample.tokens[:site.position])
            assert q.get(site.answer, 0.) > 0
            counts['action_site_probability_sum'] += q[site.answer]
            counts['visible_deterministic_actions'] += q == {site.answer: 1.}
            counts['visible_ambiguous_actions'] += len(q) > 1
        text = generator.render(sample.tokens)
        texts.append(text)
        if len(examples) < 3:
            examples.append({'text': text, 'genre': sample.genre,
                             'actions': [site.__dict__ for site in
                                         sample.actions]})
    starts = Counter()
    for text in texts:
        for sentence in text.split('.'):
            words = WORD.findall(sentence)
            if words:
                starts[words[0].lower()] += 1
    all_words = [word.lower() for text in texts for word in WORD.findall(text)]
    top = Counter(all_words)
    return {'n': n, 'counts': dict(counts),
            'worst_relative_direct_vs_chain': worst_relative_error,
            'forms': form_cues(texts),
            'first_40_repetition': score(texts, 40),
            'distinct_words': len(top),
            'top_ten_word_fraction': sum(v for _, v in
                                         top.most_common(10))/len(all_words),
            'sentence_start_types': len(starts),
            'top_five_sentence_start_fraction': sum(v for _, v in
                                                    starts.most_common(5)) / sum(starts.values()),
            'examples': examples}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--seed', type=int, default=42001)
    p.add_argument('--objects', type=int, default=16)
    p.add_argument('--people', type=int, default=8)
    p.add_argument('--frequency-exponent', type=float, default=.5)
    p.add_argument('--sentences', type=int, default=8)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    kwargs = {'seed': a.seed, 'n_objects': a.objects,
              'n_people': a.people, 'frequency_exponent': a.frequency_exponent,
              'n_sentences': a.sentences}
    variable = EventFrameOracle(**kwargs, variable_frames=True)
    fixed = EventFrameOracle(**kwargs, variable_frames=False)
    fixture_checks = action_edit_fixture(variable)
    paired_variable = EventFrameOracle(**kwargs, variable_frames=True)
    paired_fixed = EventFrameOracle(**kwargs, variable_frames=False)
    for _ in range(a.n):
        left, right = (paired_variable.sample_document(),
                       paired_fixed.sample_document())
        assert left.genre == right.genre and left.events == right.events
    result = {'config': {**vars(a), 'output': str(a.output)},
              'action_frame_edit_checks': fixture_checks,
              'paired_event_checks': a.n,
              'variable': summarize(variable, a.n),
              'fixed': summarize(fixed, a.n),
              'scope': 'Both arms sample exactly paired latent event sequences. '
                       'The exact visible-prefix '
                       'oracle marginalizes two latent genre choices. '
                       'Three fact and three action frames are only a '
                       'grammar feasibility fixture, not a realistic DGP.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print('action frame edit checks', fixture_checks)
    for name in ('variable', 'fixed'):
        row = result[name]
        print(name, row['counts']['actions'], row['sentence_start_types'],
              row['top_ten_word_fraction'], row['worst_relative_direct_vs_chain'])


if __name__ == '__main__':
    main()
