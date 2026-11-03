"""Audit two independent sampled rules inside heterogeneous FineWeb prose."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import islice
import json
import os
from pathlib import Path
import random
import sys

from datasets import load_dataset
from tokenizers import Tokenizer

from audit_document_structure_vs_fineweb import structure
from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from compare_v12_fineweb_fresh import first_words
from corpus_joint_style_delimiter_dgp import plans, validate_factorial
from corpus_transformed_delimiters import candidates as delimiter_candidates
from corpus_transformed_spelling_style import candidate_indices, LOOKUP


def aligned(arms, tokenizer, n_ctx):
    result = {}
    for key, doc in arms.items():
        enc = tokenizer.encode(doc.text, add_special_tokens=False)
        indices = []
        for site in doc.sites:
            hits = [i for i, (start, end) in enumerate(enc.offsets)
                    if start <= site.start < end and end == site.end]
            if len(hits) != 1 or hits[0] < 1:
                return None
            indices.append(hits[0])
        sources = []
        for site in doc.sites:
            hits = [i for i, (start, end) in enumerate(enc.offsets)
                    if start <= site.source_start < end]
            if len(hits) != 1:
                return None
            sources.append(hits[0])
        left = max(0, min(sources)-32)
        if max(indices)-left >= n_ctx:
            return None
        result[key] = {'site_token': indices,
                       'source_token': sources,
                       'target_ids': [enc.ids[i] for i in indices],
                       'window_start': left}
    if not (result['us', 'round']['target_ids'][0] !=
            result['uk', 'round']['target_ids'][0] and
            result['us', 'round']['target_ids'][1] !=
            result['us', 'square']['target_ids'][1]):
        return None
    return result


def audit(skip, n, max_chars, max_joint_span, n_ctx, seed):
    source = load_dataset('HuggingFaceFW/fineweb',
                          name='sample-10BT', split='train', streaming=True)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    rng = random.Random(seed)
    counts = Counter()
    originals, edited = [], []
    gap_words, joint_spans = [], []
    bracket_gaps = Counter()
    stream = iter(source)
    try:
        for row in islice(stream, skip, skip+n):
            text = row['text'][:max_chars]
            counts['documents'] += 1
            counts['has_style'] += bool(candidate_indices(text)[1])
            counts['has_delimiter'] += bool(delimiter_candidates(text))
            options = plans(text, max_joint_span)
            if not options:
                continue
            counts['has_joint_plan'] += 1
            plan = rng.choice(options)
            arms = validate_factorial(plan)
            counts['validated_factorial'] += 1
            alignment = aligned(arms, tokenizer, n_ctx)
            if alignment is None:
                continue
            counts['aligned_factorial'] += 1
            source_rows, _ = candidate_indices(text)
            cue_row = source_rows[plan.style_occurrence_index-1]
            target_row = source_rows[plan.style_occurrence_index]
            for key, doc in arms.items():
                cue = doc.text[doc.sites[0].source_start:
                               doc.sites[0].source_end]
                assert cue == (cue_row.us if key[0] == 'us'
                               else cue_row.uk)
                assert doc.sites[0].answer == (
                    target_row.us if key[0] == 'us' else target_row.uk)
                assert LOOKUP[cue][0] != LOOKUP[doc.sites[0].answer][0]
            gap_words.append(next(gap for index, gap in
                candidate_indices(text)[1]
                if index == plan.style_occurrence_index))
            bracket_gaps[delimiter_candidates(text)[
                plan.delimiter_pair_index][2]] += 1
            joint_spans.append(plan.joint_span_characters)
            if len(originals) < 64:
                originals.append(first_words(text, 100))
                edited.append(first_words(arms['uk', 'square'].text, 100))
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    if counts['documents'] != n:
        raise RuntimeError(f'Expected {n} documents, got {counts["documents"]}')
    return {'config': {'skip': skip, 'n': n, 'max_chars': max_chars,
                       'max_joint_span': max_joint_span, 'n_ctx': n_ctx,
                       'seed': seed},
            'counts': dict(counts),
            'aligned_joint_span_characters': {
                'min': min(joint_spans) if joint_spans else None,
                'max': max(joint_spans) if joint_spans else None},
            'aligned_style_cue_to_target_words': {
                'min': min(gap_words) if gap_words else None,
                'max': max(gap_words) if gap_words else None},
            'bracket_gap_words': dict(bracket_gaps),
            'form_first100': {'original': structure(originals),
                              'edited_uk_square': structure(edited)},
            'caveat': 'FineWeb prose is the carrier; only selected spelling '
                      'and delimiter sites have known controlled rules. '
                      'No raw text or full LM oracle is saved.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip', type=int, default=83500)
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--max-chars', type=int, default=8000)
    p.add_argument('--max-joint-span', type=int, default=1200)
    p.add_argument('--n-ctx', type=int, default=320)
    p.add_argument('--seed', type=int, default=98501)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.skip, args.n, args.max_chars,
                   args.max_joint_span, args.n_ctx, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['counts'], flush=True)
    sys.stdout.flush()
    os._exit(0)  # datasets/pyarrow streaming may crash during interpreter exit


if __name__ == '__main__':
    main()
