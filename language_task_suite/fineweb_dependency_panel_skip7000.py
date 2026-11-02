"""Independent cue-selected review on a disjoint FineWeb development panel.

Store only row IDs and qualitative judgments, never source prose. Labels were
assigned after inspecting the matched local context in each token window.
"""

import hashlib
import json
import random
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from fineweb_dependency_panel import PATTERNS


HERE = Path(__file__).resolve().parent
ROWS = (HERE.parents[1] / 'reverse_engineer/tensor_language/'
        'basis_aligned/bilinear_quotient/.rowcache/fineweb_n192_skip7000.pt')
SEED = 20261103

# The question is whether the *local relation* targeted by the proposed toy
# rule is identifiable, not merely whether a keyword appears. "Unclear" is
# retained rather than forced into a positive or negative count.
LABELS = {
    'attribution': {
        25: ('unclear', 'Quoted criticism is attributed only to an unspecified analysis.'),
        38: ('no', 'According to men is not an attribution of a local claim to a named source.'),
        41: ('unclear', 'As I said refers to the current author, with no separately recoverable source.'),
        47: ('yes', 'A named speaker is explicitly linked to a quotation.'),
        36: ('unclear', 'Speech is quoted, but the local speaker is a plural referent.'),
    },
    'correction': {
        16: ('no', 'Instead compares physical particle sizes; no prior fact is replaced.'),
        158: ('no', 'Updated refers to an administrative review.'),
        70: ('unclear', 'Actually may correct a character-height claim outside the visible context.'),
        152: ('no', 'Actually emphasizes a personal reaction.'),
        40: ('no', 'Instead connects two sides of an argument.'),
    },
    'alias': {
        85: ('yes', 'A device is described with an acronym and its expansion.'),
        93: ('yes', 'A person is linked to an alternate name.'),
        175: ('yes', 'Circular file is explicitly glossed as trash can.'),
    },
    'number': {
        143: ('no', 'A total repair cost is stated, with no local update operation.'),
        169: ('no', 'Numbers describe dates, arrests, and a monetary charge.'),
        55: ('no', 'Totals are descriptive election statistics.'),
        145: ('no', 'Total sales volume is a projection, not a computed update.'),
        191: ('no', 'Remaining days are stated without a local arithmetic update.'),
    },
    'scope': {
        183: ('yes', 'A numbered section heading scopes the following agreement text.'),
        129: ('yes', 'A numbered advice item scopes its explanation.'),
        124: ('yes', 'A numbered recruiting step scopes the following instructions.'),
        60: ('yes', 'A numbered claim is followed by a scoped facts paragraph.'),
        182: ('yes', 'A numbered service item scopes its details.'),
    },
}


def build(rows_path=ROWS):
    rows = torch.load(rows_path, map_location='cpu', weights_only=True)
    assert rows.shape == (192, 513)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    texts = [tokenizer.decode(row.tolist(), skip_special_tokens=False)
             for row in rows]
    rng = random.Random(SEED)
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
    return dict(rows=str(rows_path),
                source_sha256=hashlib.sha256(rows.numpy().tobytes()).hexdigest(),
                selection_seed=SEED, regexes=PATTERNS, groups=groups,
                caveat='Cue-selected development windows, at most five per '
                       'cue. A positive label does not imply prevalence, a '
                       'next-token target, or model competence.')


if __name__ == '__main__':
    result = build()
    output = HERE / 'fineweb_dependency_panel_skip7000.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    for name, group in result['groups'].items():
        print(name, group['eligible_after_exclusions'],
              {label: sum(row['label'] == label for row in group['rows'])
               for label in ('yes', 'no', 'unclear')})
