"""Small manual panel of FineWeb opening forms; saves no source prose.

Labels describe only roughly the first 500 whitespace-normalized characters
of each independent sample-10BT document. They are one reader's exploratory
coding, not population mixture estimates or verified prediction sites.
"""

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path

from datasets import load_dataset


# Dominant page form and ordered, visible opening moves. Each row was read
# at its fixed offset; the audit records hashes to make the labeling panel
# independently retrievable without committing copyrighted passages.
LABELS = {
    42000: ('procedure', ('item_list', 'ordered_step', 'measure', 'review')),
    42001: ('report', ('headline', 'background', 'change', 'crosslink')),
    42002: ('reference', ('question', 'definition', 'enumeration', 'qualification')),
    42003: ('report', ('lead_claim', 'change', 'consequence', 'attribution')),
    42004: ('quoted_feature', ('byline', 'question', 'profile', 'attribution')),
    42005: ('personal', ('headline', 'analogy', 'background', 'example')),
    42006: ('collection', ('scene', 'crosslink', 'scene', 'crosslink')),
    42007: ('personal', ('personal_scene', 'question', 'crosslink', 'ordered_step')),
    42008: ('report', ('lead_claim', 'background', 'question', 'quotation')),
    42009: ('commercial', ('form_notice', 'review', 'example')),
    42010: ('quoted_feature', ('attribution', 'quotation', 'background')),
    42011: ('reference', ('headline', 'absence_notice', 'instruction', 'crosslink')),
    42012: ('report', ('dateline', 'scene', 'quotation', 'measure')),
    42013: ('report', ('headline', 'dateline', 'change', 'evidence')),
    42014: ('personal', ('lead_claim', 'measure', 'aside', 'example')),
    42015: ('reference', ('enumeration', 'definition', 'example')),
    42016: ('quoted_feature', ('series_context', 'attribution', 'quotation')),
    42017: ('personal', ('question', 'profile', 'background', 'change')),
    42018: ('reference', ('measure', 'enumeration', 'definition')),
    42019: ('report', ('change', 'measure', 'comparison', 'attribution')),
    42020: ('report', ('bullet_summary', 'profile', 'position')),
    42021: ('personal', ('problem', 'attempt', 'constraint')),
    42022: ('commercial', ('headline', 'specification', 'enumeration', 'seller')),
    42023: ('procedure', ('headline', 'ordered_step', 'measure', 'alternative')),
    42024: ('report', ('profile', 'dateline', 'event_summary')),
    42025: ('personal', ('personal_scene', 'background', 'ordered_step')),
    42026: ('collection', ('lead_claim', 'item', 'social_quote', 'item')),
    42027: ('personal', ('question', 'position', 'example')),
    42028: ('personal', ('reflection', 'temporal_sequence', 'reflection')),
    42029: ('report', ('dateline', 'quotation', 'comparison', 'quotation')),
    42030: ('report', ('headline', 'subheadline', 'change', 'attribution')),
    42031: ('report', ('headline', 'specification', 'measure', 'enumeration')),
}
assert tuple(LABELS) == tuple(range(42000, 42032))


def audit():
    dataset = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                           split='train', streaming=True)
    rows = list(islice(iter(dataset), 42000, 42032))
    assert len(rows) == 32
    forms, moves = Counter(), Counter()
    transitions = defaultdict(Counter)
    records = []
    for offset, row in zip(LABELS, rows, strict=True):
        form, sequence = LABELS[offset]
        opening = ' '.join(row['text'].split())[:500]
        digest = hashlib.sha256(opening.encode('utf-8')).hexdigest()
        forms[form] += 1
        moves.update(sequence)
        transitions[form].update(zip(sequence, sequence[1:]))
        records.append(dict(offset=offset, opening_sha256=digest,
                            form=form, moves=list(sequence)))
    assert len(set(record['opening_sha256'] for record in records)) == 32
    return dict(source='HuggingFaceFW/fineweb sample-10BT train',
                unit='first 500 whitespace-normalized characters',
                records=records,
                form_counts=dict(forms), move_counts=dict(moves),
                within_form_transitions={
                    form: [{'from': a, 'to': b, 'count': n}
                           for (a, b), n in counts.items()]
                    for form, counts in transitions.items()},
                limitations=('One reader, 32 contiguous corpus rows, '
                             'opening-only labels. This supplies a design '
                             'inventory, not empirical mixture probabilities '
                             'or verified predictive-rule prevalence.'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('forms', result['form_counts'], flush=True)
    print('moves', result['move_counts'], flush=True)
    # Streaming datasets can keep background threads alive after the work.
    os._exit(0)


if __name__ == '__main__':
    main()
