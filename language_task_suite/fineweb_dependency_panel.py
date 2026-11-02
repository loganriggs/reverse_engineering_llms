"""Reproduce a small, manually labeled FineWeb dependency-cue panel.

The frozen 480-window panel is a development sample, not a prevalence sample.
Only row identifiers, cue matches, and qualitative labels are exported; no
source prose is copied into the repository.
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_ROWS, DEFAULT_TOKENIZER


PATTERNS = {
    'attribution': r'\b(?:according to|said|told|wrote|reported)\b',
    'correction': r'\b(?:corrected|correction|instead|actually|rather than|updated)\b',
    'alias': r'\b(?:also known as|aka|formerly called|renamed)\b',
    'number': r'\b(?:remaining|left|total|another)\b[^.!?\n]{0,60}\b\d+\b',
    'scope': r'(?m)^\s{0,5}\d{1,3}[.)]\s+\w',
}

# A positive means the *specific relationship* is reasonably clear within
# this 513-token window. It does not mean the window supports an exact next-
# token target, that the dependency requires a language model, or that it is
# common in FineWeb. Unclear labels remain explicit instead of being counted
# as positive or negative.
LABELS = {
    'attribution': {
        305: ('yes', 'Named author is linked to a book.'),
        25: ('unclear', 'The phrase said as much has an unclear local source.'),
        186: ('yes', 'Named scientist is linked to quoted claims.'),
        342: ('yes', 'Named official is linked to a quotation.'),
        353: ('yes', 'Named speaker is linked to a stated claim.'),
    },
    'correction': {
        98: ('no', 'Instead introduces a contrast in advice.'),
        78: ('no', 'Instead selects a different summary statistic.'),
        248: ('no', 'Actually expresses evaluation, not a revised fact.'),
        33: ('no', 'Actually is an emphasis in a comment exchange.'),
        410: ('no', 'Updated is a software changelog item.'),
    },
    'alias': {
        386: ('no', 'Aka marks a figurative description of a place.'),
        444: ('yes', 'Aka links two names for a country.'),
        464: ('yes', 'Also known as links two labels for a procedure.'),
        39: ('yes', 'Renamed links historical names of a region.'),
        405: ('unclear', 'The nickname referent is unclear in the local context.'),
    },
    'number': {
        57: ('no', 'Numbers identify a series, anniversary, and date.'),
        324: ('no', 'Numbers describe sports statistics, no update calculation.'),
        267: ('no', 'Numbers are scores and ranks, no update calculation.'),
        205: ('unclear', 'Recipe uses an amount remaining, but earlier use is not locally clear.'),
        303: ('yes', 'A postponed total is divided over remaining payments.'),
    },
    'scope': {
        385: ('yes', 'Numbered heading introduces its own explanatory item.'),
        341: ('yes', 'Numbered heading scopes following shopping advice.'),
        212: ('yes', 'Numbered thought has following supporting prose.'),
        35: ('yes', 'Numbered travel tip has its own supporting explanation.'),
        115: ('yes', 'Numbered list items have distinct descriptions.'),
    },
}


def build(rows_path=DEFAULT_ROWS, tokenizer_path=DEFAULT_TOKENIZER):
    rows = torch.load(rows_path, map_location='cpu', weights_only=True)
    assert rows.ndim == 2 and rows.shape == (480, 513)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    texts = [tokenizer.decode(row.tolist(), skip_special_tokens=False)
             for row in rows]
    rng = random.Random(20261102)
    seen = set()
    groups = {}
    for name, expression in PATTERNS.items():
        regex = re.compile(expression, re.IGNORECASE)
        eligible = [i for i, text in enumerate(texts)
                    if i not in seen and regex.search(text)]
        chosen = rng.sample(eligible, min(5, len(eligible)))
        assert set(chosen) == set(LABELS[name]), (name, chosen)
        seen.update(chosen)
        groups[name] = dict(eligible_after_exclusions=len(eligible),
                            rows=[dict(index=i, label=LABELS[name][i][0],
                                       reason=LABELS[name][i][1])
                                  for i in chosen])
    return dict(rows=str(rows_path), tokenizer=str(tokenizer_path),
                source_sha256=hashlib.sha256(rows.numpy().tobytes()).hexdigest(),
                selection_seed=20261102, regexes=PATTERNS, groups=groups,
                note='Five pattern-positive windows per cue, sampled without '
                     'replacement from the frozen development panel. Manual '
                     'labels describe the local semantic relation, not the '
                     'prevalence of a learnable task or exact target.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('fineweb_dependency_panel.json'))
    args = parser.parse_args()
    result = build()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for name, group in result['groups'].items():
        print(name, group['eligible_after_exclusions'],
              {label: sum(row['label'] == label for row in group['rows'])
               for label in ('yes', 'no', 'unclear')})
