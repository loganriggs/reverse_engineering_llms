"""Audit exact hidden-genre filtering, planted tasks, and surface statistics."""

import argparse
import collections
import json
import math
from pathlib import Path

from genre_document_v3 import (GENRES, PosteriorTrace, _base_prefix,
                               next_distribution, sample_sequence)
from joint_document_v1 import _split, visible_state


def _entropy(q):
    return -sum(p*math.log(p) for p in q.values() if p)


def audit(regime, n, length, seed):
    sites = collections.Counter()
    genre_sites = {g: collections.Counter() for g in GENRES}
    lexical = collections.Counter()
    identification = collections.defaultdict(list)
    posterior_entropies = collections.defaultdict(list)
    independent_prefix_checks = 0
    source_checks = 0
    source_last_speaker_correct = 0
    source_multi_speaker = 0
    source_attribute_edits = 0
    arithmetic_checks = 0
    step_checks = 0
    title_edit_tvs = []
    for row in range(n):
        words, labels, distributions, genre = sample_sequence(
            seed+row, length, regime)
        title = words[:4]
        replacement = 'arithmetic' if title[2] == 'evidence' else 'evidence'
        edited_title = [title[0], title[1], replacement, title[3]]
        base_title = next_distribution(title, regime)
        changed_title = next_distribution(edited_title, regime)
        title_edit_tvs.append(.5*sum(abs(base_title.get(w, 0.)-
                                          changed_title.get(w, 0.))
                                      for w in base_title.keys() |
                                      changed_title.keys()))
        trace = PosteriorTrace(regime)
        for i, (word, label, recorded) in enumerate(zip(
                words, labels, distributions, strict=True)):
            marginal, qs = trace.predict()
            assert abs(sum(marginal.values())-1) < 1e-10
            assert all(trace.posterior[g] > 0 for g in GENRES)
            assert all(abs(marginal.get(w, 0.)-p) < 1e-10
                       for w, p in recorded.items())
            assert marginal.get(word, 0.) > 0
            sites[label] += 1
            genre_sites[genre][label] += 1
            history, raw = _split(words[:i])
            if label == 'arithmetic_answer':
                assert marginal == {str(int(raw[2])+int(raw[4])): 1.}
                arithmetic_checks += 1
            elif label == 'step_number':
                number = 1+sum(s[0] == 'Step' for s in history)
                assert marginal == {str(number): 1.}
                step_checks += 1
            elif label == 'source_answer':
                base_history, base_raw = _split(_base_prefix(words[:i]))
                claims = visible_state(base_history)['claims']
                expected = claims[(base_raw[3], base_raw[5])]
                assert marginal == {expected: 1.}
                source_checks += 1
                source_last_speaker_correct += (expected ==
                                                next(reversed(claims.values())))
                source_multi_speaker += len({speaker for (item, _), speaker
                                             in claims.items()
                                             if item == base_raw[3]}) >= 2
                if source_attribute_edits < 100:
                    alternatives = [attr for (item, attr), speaker in claims.items()
                                    if item == base_raw[3] and speaker != expected]
                    for attr in alternatives:
                        edited = list(words[:i])
                        edited[-4] = attr
                        q_edit = next_distribution(edited, regime)
                        if q_edit == {claims[(base_raw[3], attr)]: 1.}:
                            source_attribute_edits += 1
                            break
            if row < 20 and i in (0, 1, 2, 3, 4, 32, 64, 128, 255):
                direct = next_distribution(words[:i], regime)
                assert all(abs(direct.get(w, 0.)-p) < 1e-10
                           for w, p in marginal.items())
                independent_prefix_checks += 1
            if i in (4, 32, 64, 128, 255):
                identification[i].append(int(max(trace.posterior,
                                                 key=trace.posterior.get) == genre))
                posterior_entropies[i].append(_entropy(trace.posterior))
            trace.observe(word, qs)
        if row < 49:
            alpha = [w.lower() for w in words if w.isalpha()]
            assert len(alpha) >= 100
            lexical.update(alpha[:100])
    return dict(version='genre_document_v3', regime=regime,
                n_streams=n, length=length, seed=seed,
                oracle_checks=n*length,
                independent_prefix_checks=independent_prefix_checks,
                semantic_checks=dict(source=source_checks,
                                     source_last_speaker_correct=source_last_speaker_correct,
                                     source_multi_speaker=source_multi_speaker,
                                     source_attribute_edits=source_attribute_edits,
                                     arithmetic=arithmetic_checks,
                                     step_number=step_checks),
                sites=dict(sites),
                by_hidden_genre={g: dict(row) for g, row in genre_sites.items()},
                genre_identification={str(i): dict(
                    accuracy=sum(identification[i])/len(identification[i]),
                    mean_posterior_entropy=sum(posterior_entropies[i])/
                    len(posterior_entropies[i])) for i in identification},
                title_topic_edit=dict(n=len(title_edit_tvs),
                                      mean_next_token_tv=sum(title_edit_tvs)/
                                      len(title_edit_tvs)),
                first49_x100_alpha=dict(types=len(lexical),
                                        singletons=sum(c == 1 for c in lexical.values()),
                                        top10_share=sum(c for _, c in lexical.most_common(10))/4900),
                caveat='Structural hidden-genre and semantic audit only; '
                       'genre rates and words are not natural-text calibrated.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--regime', choices=('reference', 'task_rich'), required=True)
    p.add_argument('--n', type=int, default=200)
    p.add_argument('--length', type=int, default=256)
    p.add_argument('--seed', type=int, default=38001)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.regime, args.n, args.length, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print({k: result[k] for k in ('semantic_checks', 'genre_identification',
                                  'first49_x100_alpha')}, flush=True)
