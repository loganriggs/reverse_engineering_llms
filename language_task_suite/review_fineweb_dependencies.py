"""Reproduce the small manually classified FineWeb cue sample.

The labels below were assigned after reading local decoded context around each
match. We store source coordinates and categories, not copied article text.
This is qualitative calibration, not a prevalence estimate.
"""

import json
import hashlib
import random
import re
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_ROWS, DEFAULT_TOKENIZER, PATTERNS


LABELS = {
    'pronoun': ('specific_antecedent', 'generic_reference', 'generic_reference',
                'generic_reference', 'specific_antecedent', 'group_antecedent'),
    'state_change_verb': ('stored_information', 'idiom', 'idiom',
                          'phrasal_verb', 'physical_location', 'list_membership'),
    'if_then': ('instruction_condition', 'inference_condition',
                'consequence_condition', 'instruction_condition',
                'counterfactual_condition', 'technical_rule'),
    'numbered_list_line': ('instruction_list', 'bibliography',
                           'instruction_list', 'inline_enumeration',
                           'ranking', 'bibliography'),
    'comparative': ('quantity_threshold', 'duration_threshold', 'idiom',
                    'quantity_threshold', 'duration_threshold',
                    'quantity_threshold'),
    'parenthesis_pair': ('clarification', 'statistic', 'date',
                         'citation', 'gloss', 'geographic_note'),
    'weekday_name': ('schedule_label', 'event_date', 'narrative_time',
                     'date_stamp', 'opening_hours', 'event_date'),
}


def main():
    rows = torch.load(DEFAULT_ROWS, map_location='cpu', weights_only=True)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    texts = [tokenizer.decode(row.tolist(), skip_special_tokens=False)
             for row in rows]
    records = []
    for name, labels in LABELS.items():
        rx = re.compile(PATTERNS[name], re.IGNORECASE | re.MULTILINE)
        matches = [(i, m) for i, text in enumerate(texts)
                   for m in rx.finditer(text)]
        random.Random(239+len(name)).shuffle(matches)
        assert len(matches) >= len(labels)
        for sample_index, ((row, match), label) in enumerate(zip(matches, labels)):
            records.append(dict(cue=name, sample_index=sample_index,
                                row=row, char_start=match.start(),
                                char_end=match.end(), label=label))
    digest = hashlib.sha256(rows.numpy().tobytes()).hexdigest()
    prior = json.loads(Path(__file__).with_name('fineweb_surface_audit.json').read_text())
    assert prior['source_sha256'] == digest
    output = dict(source=str(DEFAULT_ROWS), source_sha256=digest,
                  n_windows=len(texts),
                  sample_method='For each regex, shuffle all matches with '
                                'random.Random(239+len(cue_name)); review first six.',
                  n_per_cue=6, records=records,
                  counts={name: dict(Counter(r['label'] for r in records
                                             if r['cue'] == name))
                          for name in LABELS},
                  caveat='A small manually classified sample of regex matches. '
                         'Some labels require judgment and the sample estimates '
                         'neither corpus prevalence nor prediction difficulty.')
    path = Path(__file__).with_name('fineweb_dependency_review.json')
    path.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output['counts'], indent=2))


if __name__ == '__main__':
    main()
