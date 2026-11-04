"""Audit paired surname sites alongside existing corpus rule sites."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from corpus_rule_reference_dgp import chunk_document
from corpus_surname_joint_dgp import paired_document
from fineweb_name_copy_dgp import NameCopyDGP


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preview', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--n-ctx', type=int, default=256)
    p.add_argument('--seed', type=int, default=46301)
    a = p.parse_args()
    rows = json.loads(a.preview.read_text())
    dgp = NameCopyDGP(rows)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    counts = Counter()
    other_at_name = Counter()
    for index in range(len(rows)):
        rng = random.Random(a.seed+rows[index]['index'])
        pair = paired_document(dgp,index,rng,[],
                               list_rate=1.,bracket_rate=1.,
                               literal_copy_rate=1.,natural_style_rate=1.,
                               nonce_rate=0.,square_rate=.5,
                               list_shift_rate=.5)
        if pair is None:
            continue
        counts['pairable_source_documents'] += 1
        chunks = [chunk_document(doc,tokenizer,a.n_ctx,max_tokens=2048)
                  for doc in pair]
        surname_windows = [[chunk for chunk in arm if any(
            site['kind']=='surname_copy' for site in chunk['sites'])]
                           for arm in chunks]
        if not all(surname_windows):
            continue
        counts['pairable_token_aligned_surname_documents'] += 1
        counts['surname_windows_left'] += len(surname_windows[0])
        counts['surname_windows_right'] += len(surname_windows[1])
        left_families = {site['kind'] for chunk in surname_windows[0]
                         for site in chunk['sites']}
        right_families = {site['kind'] for chunk in surname_windows[1]
                          for site in chunk['sites']}
        joint = (left_families & right_families)-{'surname_copy'}
        counts['same_context_with_other_rule'] += bool(joint)
        for name in joint:
            other_at_name[name] += 1
    result = {'source':'HuggingFaceFW/fineweb sample-10BT train',
              'offsets':[rows[0]['index'],rows[-1]['index']],
              'n_ctx':a.n_ctx,'seed':a.seed,
              'counts':dict(counts),
              'same_context_other_rule_families':dict(other_at_name),
              'scope':'Heuristic paired same-length surname edits; both '
                      'arms need a source-visible one-BPE surname target '
                      'inside a nonoverlapping context chunk. Other-rule '
                      'co-occurrence means a local site in both arms; it '
                      'does not establish model competence. Raw prose is '
                      'local only.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(result,flush=True)


if __name__ == '__main__':
    main()
