"""Replay v6 and independently check source binding and legacy mixed rules."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_ROWS, DEFAULT_TOKENIZER
from audit_surface_mixture_v2 import check_clause
from audit_surface_repetition import WORD, form_cues, score
from compare_realism import summary
from surface_mixture_v2 import _frame, _indices, _split
from surface_mixture_v3 import SurfaceMixtureV3
from surface_mixture_v4 import _content
from surface_mixture_v6 import (SurfaceMixtureV6, render,
                                source_pair_heldout)


def reference_facts(history):
    result = {}
    for raw in history:
        clause = _content(raw)
        if not clause or clause[0] != 'As':
            continue
        if clause[4] == '"':
            assert len(clause) == 11
            assert clause[2:6] == ['said', ',', '"', 'the']
            assert clause[7:] == ['was', 'missing', '"', '.']
            obj = clause[6]
        else:
            assert len(clause) == 9
            assert clause[2:5] == ['said', ',', 'the']
            assert clause[6:] == ['was', 'missing', '.']
            obj = clause[5]
        assert obj not in result
        result[obj] = clause[1]
    return result


def narrative(history):
    return [_content(raw) for raw in history
            if _content(raw) and _content(raw)[0]
            not in ('Where', 'How', 'As', 'Who')]


def reference_place(history, obj):
    from surface_mixture_v3 import OBJECT_INDEX
    for clause in reversed(narrative(history)):
        frame = _frame(clause)
        if clause[OBJECT_INDEX[frame]] == obj:
            return clause[_indices(frame)[2]]
    raise AssertionError(obj)


def audit(n, length, seed, quote_rate, source_rate, question_rate,
          rows_path=DEFAULT_ROWS, exclude_source_pairs=False,
          unquoted_source_rate=0.):
    gen = SurfaceMixtureV6(seed=seed, quote_rate=quote_rate,
                           source_rate=source_rate, question_rate=question_rate,
                           counterbalanced_queries=True,
                           exclude_source_pairs=exclude_source_pairs,
                           unquoted_source_rate=unquoted_source_rate)
    v3 = SurfaceMixtureV3()
    ids, labels, qs = gen.sample_batch(n, length)
    counts = Counter()
    statement_forms = Counter()
    examples = [render(row) for row in ids[:6]]
    for row, labs, distributions in zip(ids, labels, qs):
        words = [gen.id2token[i] for i in row]
        history, _ = _split(words)
        for clause_index, raw in enumerate(history):
            clause = _content(raw)
            if not clause:
                continue
            if clause[0] == 'As':
                if clause[4] == '"':
                    statement_forms['quoted'] += 1
                    assert len(clause) == 11
                    assert clause[2:6] == ['said', ',', '"', 'the']
                    assert clause[7:] == ['was', 'missing', '"', '.']
                    obj = clause[6]
                else:
                    statement_forms['unquoted'] += 1
                    assert len(clause) == 9
                    assert clause[2:5] == ['said', ',', 'the']
                    assert clause[6:] == ['was', 'missing', '.']
                    obj = clause[5]
                if exclude_source_pairs:
                    assert not source_pair_heldout(clause[1], obj)
            elif clause[0] == 'Who':
                assert len(clause) == 11
                assert clause[:3] == ['Who', 'said', 'the']
                assert clause[4:9] == ['was', 'missing', '?', 'Answer', ':']
                assert clause[9] == reference_facts(
                    history[:clause_index])[clause[3]]
                assert clause[10] == '.'
            elif clause[0] == 'Where':
                assert clause[:3] == ['Where', 'did', 'the']
                assert clause[4:] == ['appear', '?', 'Answer', ':',
                                      reference_place(history[:clause_index],
                                                      clause[3]), '.']
            elif clause[0] == 'How':
                raise AssertionError('Count questions disabled in this audit')
            else:
                check_clause(clause)
        for t, (word, q, label) in enumerate(zip(words, distributions, labs)):
            assert q == gen.next_distribution(words[:t])
            assert abs(sum(q.values())-1) < 1e-12
            assert q.get(word, 0) > 0
            counts[label] += 1
            if label in ('mixed_verb', 'mixed_object'):
                earlier, raw = _split(words[:t])
                projected = [w for clause in narrative(earlier) for w in clause]
                projected += _content(raw)
                assert q == v3.next_distribution(projected)
            elif label == 'source_answer':
                earlier, raw = _split(words[:t])
                clause = _content(raw)
                facts = reference_facts(earlier)
                assert clause[:3] == ['Who', 'said', 'the']
                assert len(clause) == 9
                assert q == {facts[clause[3]]: 1.}
                key_q = gen.next_distribution(words[:t-6])
                assert key_q == {obj: 1/len(facts) for obj in facts}
                assert any(speaker != facts[clause[3]]
                           for speaker in facts.values())
            elif label == 'question_answer':
                earlier, raw = _split(words[:t])
                clause = _content(raw)
                assert q == {reference_place(earlier, clause[3]): 1.}
    natural_rows = torch.load(rows_path, map_location='cpu', weights_only=True)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    m = min(n, len(natural_rows))
    synthetic = [render(row) for row in ids[:m]]
    natural = [tokenizer.decode(row[:length].tolist(),
                                skip_special_tokens=False)
               for row in natural_rows[:m]]
    eligible = [i for i in range(m)
                if len(WORD.findall(synthetic[i])) >= 40
                and len(WORD.findall(natural[i])) >= 40]
    return dict(config=dict(n=n, length=length, seed=seed,
                            quote_rate=quote_rate, source_rate=source_rate,
                            question_rate=question_rate,
                            exclude_source_pairs=exclude_source_pairs,
                            unquoted_source_rate=unquoted_source_rate,
                            natural_rows=str(rows_path)),
                n_vocab=len(gen.vocab), slots=dict(counts),
                statement_forms=dict(statement_forms),
                form_cues=dict(synthetic=form_cues(synthetic),
                               fineweb=form_cues(natural)),
                matched_surface=dict(n=m, synthetic=summary(synthetic),
                                     fineweb=summary(natural)),
                word_matched=dict(n=len(eligible), words=40,
                                  synthetic=score([synthetic[i] for i in eligible], 40),
                                  fineweb=score([natural[i] for i in eligible], 40)),
                examples=examples,
                caveat='Source statements and answers independently replayed on '
                       'every complete clause; mixed sites checked against v3. '
                       'Surface forms and tokenizers differ from FineWeb; this '
                       'is not a natural dependency prevalence estimate.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=3000)
    parser.add_argument('--length', type=int, default=128)
    parser.add_argument('--seed', type=int, default=12001)
    parser.add_argument('--quote-rate', type=float, default=.30)
    parser.add_argument('--source-rate', type=float, default=.30)
    parser.add_argument('--question-rate', type=float, default=.05)
    parser.add_argument('--exclude-source-pairs', action='store_true')
    parser.add_argument('--unquoted-source-rate', type=float, default=0.)
    parser.add_argument('--rows', type=Path, default=DEFAULT_ROWS)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('surface_mixture_v6_audit.json'))
    args = parser.parse_args()
    result = audit(args.n, args.length, args.seed, args.quote_rate,
                   args.source_rate, args.question_rate, args.rows,
                   args.exclude_source_pairs, args.unquoted_source_rate)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('slots', 'form_cues', 'word_matched')},
                     indent=2))
