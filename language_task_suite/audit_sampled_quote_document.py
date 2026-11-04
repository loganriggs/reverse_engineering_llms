"""Aggregate semantic, context, and form screen for sampled quote documents."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import spacy
from tokenizers import Tokenizer

from audit_document_structure_vs_fineweb import structure
from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from sampled_quote_document_dgp import (
    WORD, check_visible, sample, swap_source_names,
)


def _first_words(text, count):
    matches = list(WORD.finditer(text))
    if len(matches) < count:
        return text
    return text[:matches[count-1].end()]


def audit(preview: Path, limit: int, seed: int, tokenizer: Path):
    rows = json.loads(preview.read_text())[:limit]
    if not rows:
        raise ValueError('No preview rows')
    nlp = spacy.load('en_core_web_sm')
    tok = Tokenizer.from_file(str(tokenizer))
    rng = random.Random(seed)
    counts = Counter()
    generated, natural = [], []
    phrase_lengths, context_lengths = Counter(), Counter()
    for row, doc in zip(rows, nlp.pipe((r['preview'] for r in rows),
                                       batch_size=32), strict=True):
        counts['source_documents'] += 1
        drawn = sample(doc, row['index'], rng)
        if drawn is None:
            continue
        counts['eligible_documents'] += 1
        assert drawn.text[slice(*drawn.source_span)] == drawn.answer
        assert drawn.text[slice(*drawn.target_span)] == drawn.answer
        assert drawn.source_span[1] < drawn.target_span[0]
        assert check_visible(drawn.text) == (drawn.speaker, drawn.answer)
        counts['visible_checker_pass'] += 1
        swapped = swap_source_names(drawn.text)
        try:
            check_visible(swapped)
        except ValueError:
            counts['source_swap_rejected'] += 1
        else:
            raise AssertionError('Source swap kept the original phrase valid')
        bpe = len(tok.encode(drawn.text).ids)
        context_lengths[bpe//64*64] += 1
        if bpe <= 256:
            counts['fits_256_bpe'] += 1
        phrase_lengths[len(tok.encode(drawn.answer).ids)] += 1
        generated.append(drawn.text)
        natural.append(_first_words(
            row['preview'], len(WORD.findall(drawn.text))))
    if not generated:
        raise ValueError('No eligible documents')
    return {
        'scope': {'preview_sha256': hashlib.sha256(preview.read_bytes()).hexdigest(),
                  'first_index': rows[0]['index'],
                  'last_index': rows[-1]['index'],
                  'documents': len(rows), 'seed': seed,
                  'tokenizer': str(tokenizer)},
        'counts': dict(counts),
        'bpe_context_length_64_bins': dict(context_lengths),
        'answer_bpe_length': dict(phrase_lengths),
        'form': {'generated': structure(generated),
                 'same_source_natural_crop': structure(natural)},
        'caveat': 'The claim spans are natural source sentences, but speaker '
                  'wrappers and recap are synthetic. A valid phrase is not '
                  'the unique next-token target: the generator samples among '
                  'eligible noun chunks. The swap check tests source '
                  'specificity, not model competence or naturalness. No '
                  'source passages are exported.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=73141)
    parser.add_argument('--tokenizer', type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.preview, args.limit, args.seed, args.tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['scope']['first_index'], result['counts'], flush=True)


if __name__ == '__main__':
    main()
