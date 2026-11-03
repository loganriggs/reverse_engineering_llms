"""Audit corpus background mixtures and their retained task oracle."""

import argparse
import collections
import json
import math
from pathlib import Path

import corpus_backed_document_v5 as bank
import corpus_markov_document_v6 as markov
from genre_document_v3 import _base_prefix
from genre_document_v4 import to_v3
from joint_document_v1 import _split, visible_state


def audit(regime='task_rich', n=100, length=256, seed=81001,
          generator='v5'):
    assert generator in ('v5', 'v6')
    source = bank if generator == 'v5' else markov
    labels = collections.Counter()
    checks = collections.Counter()
    background_prefix = collections.defaultdict(lambda: collections.Counter())
    for row in range(n):
        words, sites, distributions, genre = source.sample_sequence(
            seed+row, length, regime)
        trace = source.PosteriorTrace(regime)
        for i, (word, label, recorded) in enumerate(zip(
                words, sites, distributions, strict=True)):
            q, qs = trace.predict()
            assert abs(sum(q.values())-1) < 1e-10
            assert q.get(word, 0.) > 0
            assert all(trace.posterior[g] > 0 for g in trace.posterior)
            assert all(abs(q.get(w, 0.)-p) < 1e-10
                       for w, p in recorded.items())
            labels[label] += 1
            if label == 'background_word':
                _, raw_bg = _split(words[:i])
                k = len(raw_bg)
                bucket = ('1' if k == 1 else '2' if k == 2 else
                          '3-4' if k <= 4 else '5-9' if k <= 9 else '10+')
                record = background_prefix[bucket]
                record['n'] += 1
                record['deterministic'] += int(len(q) == 1)
                record['entropy'] -= sum(p*math.log(p)
                                         for p in q.values() if p)
            if label == 'document_sentence_start' and word in bank.BG_STARTS:
                checks['background_sentence_starts'] += 1
            if row < 20 and i in (0, 1, 2, 3, 4, 32, 64, 128, 255):
                replay = source.next_distribution(words[:i], regime)
                assert all(abs(replay.get(w, 0.)-p) < 1e-10
                           for w, p in q.items())
                checks['independent_prefix_replay'] += 1
            if label in ('source_answer', 'arithmetic_answer',
                         'step_number', 'bracket_close', 'section_number'):
                v3_prefix = to_v3(bank._task_prefix(words[:i]))
                history, raw = _split(_base_prefix(v3_prefix))
                if label == 'source_answer':
                    speaker = visible_state(history)['claims'][(raw[3], raw[5])]
                    assert q == {speaker: 1.}
                elif label == 'arithmetic_answer':
                    _, problem = _split(v3_prefix)
                    assert q == {str(int(problem[2])+int(problem[4])): 1.}
                elif label == 'step_number':
                    step_history, _ = _split(v3_prefix)
                    assert q == {str(1+sum(s[0] == 'Step'
                                          for s in step_history)): 1.}
                elif label == 'bracket_close':
                    assert q == {')' if '(' in raw else ']': 1.}
                elif label == 'section_number':
                    assert q == {str(visible_state(history)['section_number']+1): 1.}
                checks[label] += 1
            trace.observe(word, qs)
        task_history, _ = _split(_base_prefix(to_v3(
            bank._task_prefix(words))))
        visible_state(task_history)
        checks['semantic_document_replay'] += 1
    assert checks['background_sentence_starts'] > 0
    if regime == 'task_rich':
        assert checks['source_answer'] > 0 and checks['bracket_close'] > 0
    return dict(version=('corpus_backed_document_v5' if generator == 'v5'
                         else 'corpus_markov_document_v6'), regime=regime,
                n=n, length=length, seed=seed,
                vocab_size=len(bank.TOKEN2ID),
                background_bank_sentences=bank.BANK_SIZE,
                background_source_row_sha256=bank.SOURCE_HASH,
                sampled_token_support=n*length, labels=dict(labels),
                checks=dict(checks),
                background_prefix={k: dict(n=int(v['n']),
                                           deterministic_fraction=(
                                               v['deterministic']/v['n']),
                                           mean_next_token_entropy_nats=(
                                               v['entropy']/v['n']))
                                   for k, v in background_prefix.items()},
                caveat=('Exact for the finite empirical-bank/Markov '
                        'background distribution plus planted task rules. '
                        'Neither sentence mosaics nor local Markov '
                        'continuations establish natural document coherence.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--regime', choices=('task_rich', 'reference'),
                   default='task_rich')
    p.add_argument('--generator', choices=('v5', 'v6'), default='v5')
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--seed', type=int, default=81001)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.regime, args.n, seed=args.seed,
                   generator=args.generator)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print({k: result[k] for k in ('labels', 'checks')}, flush=True)
