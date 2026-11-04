"""Screen a repeated-noun copy edit for yield, alignment, and joint sites."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from corpus_repeated_noun_copy_dgp import load_nlp, paired_document
from corpus_rule_mixture_dgp import Config, draw
from corpus_rule_reference_dgp import chunk_document


def audit(preview: Path, n: int, seed: int, ctx: int):
    rows = json.loads(preview.read_text())[:n]
    nlp = load_nlp()
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    counts = Counter()
    for row, parsed in zip(rows, nlp.pipe(
            (item['preview'] for item in rows), batch_size=64), strict=True):
        base = draw(row, random.Random(seed*1000003+row['index']),
                    Config(.5,.4)).document
        pair = paired_document(base, parsed, tokenizer,
                               random.Random(seed+row['index']))
        counts['sources'] += 1
        if pair is None:
            continue
        left, right, plan = pair
        counts['paired_sources'] += 1
        counts[f'group/{plan.group}'] += 1
        assert left.sites[-1].answer != right.sites[-1].answer
        chunks = chunk_document(left, tokenizer, ctx, max_tokens=2048)
        aligned = any(any(site['kind']=='noun_copy' and
                          len(site['positions'])==1
                          for site in chunk['sites']) for chunk in chunks)
        joint = any({'noun_copy','bracket'} <= {
            site['kind'] for site in chunk['sites']
            if len(site['positions'])==1} for chunk in chunks)
        counts['aligned_noun_sources'] += aligned
        counts['joint_bracket_noun_chunks'] += joint
    return {'config': {'source_first_index': rows[0]['index'],
                       'source_last_index': rows[-1]['index'],
                       'preview_sha256': hashlib.sha256(
                           preview.read_bytes()).hexdigest(),
                       'documents': n, 'context_tokens': ctx,
                       'seed': seed},
            'counts': dict(counts),
            'scope': 'Strict lowercase singular spaCy-noun matches in '
                     'five hand-coded coarse groups. Same-length, '
                     'same-vowel-class, one-BPE replacement pair. '
                     'Every edit passes local visible-copy checks, but '
                     'this does not establish grammatical or semantic '
                     'naturalness. No source text or edits are saved.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preview', type=Path, required=True)
    p.add_argument('--documents', type=int, default=2000)
    p.add_argument('--seed', type=int, default=91337)
    p.add_argument('--context-tokens', type=int, default=256)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.preview, a.documents, a.seed, a.context_tokens)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['counts'], flush=True)


if __name__ == '__main__':
    main()
