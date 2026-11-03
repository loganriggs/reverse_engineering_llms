"""Falsify a lexicon-only repair of V3's natural-text frequency gap.

The counterfactual changes sampled open-slot spellings but preserves all
sentence frames and slot placements. Its top-ten lower bound is therefore
the observed concentration of the ten most frequent non-open words.
It is *not* a bound on a redesigned grammar or on resampling after edits.
"""

import argparse
import collections
import json
from pathlib import Path

from genre_document_v3 import sample_sequence, STEP_ACTIONS
from joint_document_v1 import (ASIDES, ATTRIBUTES, EVENTS, NAMES, OBJECTS,
                               PLACES, ROLES, VERBS)


OPEN_WORDS = set().union(NAMES, OBJECTS, VERBS, PLACES, EVENTS, ROLES,
                         ATTRIBUTES, ASIDES, STEP_ACTIONS)
OPEN_WORDS_LOWER = {word.lower() for word in OPEN_WORDS}


def audit(n=49, seed=38001, regime='task_rich', first_alpha=100):
    counts = collections.Counter()
    open_counts = collections.Counter()
    frame_counts = collections.Counter()
    by_doc = []
    skipped_short = 0
    candidate = 0
    while len(by_doc) < n:
        assert candidate < 8*n, 'insufficient documents with 100 alpha words'
        doc_seed = seed+candidate
        candidate += 1
        words, _, _, _ = sample_sequence(doc_seed, 256, regime)
        alpha = [(i, word.lower()) for i, word in enumerate(words)
                 if word.isalpha()][:first_alpha]
        if len(alpha) < first_alpha:
            skipped_short += 1
            continue
        counts.update(word for _, word in alpha)
        # `Document` can also be an object word, but token zero is the title
        # frame. Classify its occurrence by position rather than spelling.
        op = [word for i, word in alpha if i != 0 and word in OPEN_WORDS_LOWER]
        frame = [word for i, word in alpha if i == 0 or word not in OPEN_WORDS_LOWER]
        open_counts.update(op)
        frame_counts.update(frame)
        by_doc.append(dict(seed=doc_seed, open_slots=len(op),
                           frame_slots=len(frame)))
    total = n*first_alpha
    natural = json.loads((Path(__file__).parent/'results'/
                          'fineweb_predictive_inventory_skip23000_n64.json').read_text())
    ref = natural['first1200_diagnostics']
    assert ref['at_least_100_alphabetic_words'] == n
    frame_floor = sum(c for _, c in frame_counts.most_common(10))/total
    actual_top = sum(c for _, c in counts.most_common(10))/total
    return dict(version='genre_document_v3', regime=regime, n_documents=n,
                seed=seed, candidates_scanned=candidate,
                skipped_short=skipped_short,
                first_alpha=first_alpha, n_alpha=total,
                sampled=dict(types=len(counts),
                             singletons=sum(c == 1 for c in counts.values()),
                             top10_share=actual_top,
                             top10=counts.most_common(10)),
                open_slot=dict(n=sum(open_counts.values()),
                               types=len(open_counts),
                               singletons=sum(c == 1 for c in open_counts.values())),
                frame_slot=dict(n=sum(frame_counts.values()),
                                types=len(frame_counts),
                                top10=frame_counts.most_common(10)),
                lexical_replacement_counterfactual=dict(
                    maximum_types_if_all_open_occurrences_unique=(
                        len(frame_counts)+sum(open_counts.values())),
                    top10_share_floor_given_fixed_frame_tokens=frame_floor,
                    natural_top10_share_below_frame_floor=(
                        ref['top10_share_in_first100_words_of_eligible'] < frame_floor)),
                fineweb_development_screen=dict(
                    types=ref['types_in_first100_words_of_eligible'],
                    singletons=ref['singletons_in_first100_words_of_eligible'],
                    top10_share=ref['top10_share_in_first100_words_of_eligible']),
                by_document=by_doc,
                caveat='A fixed-sample post-hoc lexical replacement bound only. '
                       'It does not bound richer syntax, resampling with a '
                       'different article rule, or a different document mix. '
                       'FineWeb comparison is a small development screen.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--regime', choices=('reference', 'task_rich'),
                   default='task_rich')
    p.add_argument('--n', type=int, default=49)
    p.add_argument('--seed', type=int, default=38001)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(n=args.n, seed=args.seed, regime=args.regime)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print({key: result[key] for key in ('sampled', 'open_slot', 'frame_slot',
                                        'lexical_replacement_counterfactual')})
