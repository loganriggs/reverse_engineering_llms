"""Count corpus rule opportunities that coexist with surname reuse.

This is source-document eligibility, not token-aligned task density or proof
that two rules can be observed in the same context window.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from corpus_numbered_list_dgp import candidates as list_candidates
from corpus_transformed_delimiters import candidates as bracket_candidates
from corpus_uppercase_copy_dgp import candidates as code_candidates
from corpus_transformed_spelling_style import candidate_indices as style_candidates
from fineweb_name_copy_dgp import NameCopyDGP


FAMILIES = ('bracket', 'list', 'code', 'style')


def audit(rows):
    surname = NameCopyDGP(rows)
    totals = Counter()
    for index, row in enumerate(rows):
        text = row['preview']
        flags = {'surname': bool(surname.candidates[index]),
                 'bracket': bool(bracket_candidates(text)),
                 'list': bool(list_candidates(text)),
                 'code': any(code_candidates(text)),
                 'style': bool(style_candidates(text)[1])}
        totals['documents'] += 1
        for name, present in flags.items():
            totals[f'eligible_{name}'] += present
        if flags['surname']:
            other_count = sum(flags[name] for name in FAMILIES)
            totals['surname_with_any_other'] += other_count >= 1
            totals['surname_with_two_or_more_other'] += other_count >= 2
            for name in FAMILIES:
                totals[f'surname_and_{name}'] += flags[name]
    return {'source':'HuggingFaceFW/fineweb sample-10BT train',
            'offsets':[rows[0]['index'],rows[-1]['index']],
            'counts':dict(totals),
            'scope':'Full-preview, heuristic source eligibility only. '
                    'No BPE alignment or local same-window check; surname '
                    'PERSON tagging is fallible. All text stays local.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preview', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows = json.loads(a.preview.read_text())
    result = audit(rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(result['counts'], flush=True)


if __name__ == '__main__':
    main()
