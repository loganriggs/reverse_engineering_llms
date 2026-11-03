"""Text-free aggregate audit of controlled name reuse in a natural panel."""

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_document_structure_vs_fineweb import structure
from audit_fineweb_surfaces import DEFAULT_ROWS, DEFAULT_TOKENIZER
from compare_v12_fineweb_fresh import first_words
from corpus_transformed_name_reuse import (
    NAME, NAME_GROUPS, NAME_TO_GROUP, WORD, candidates,
    reverse_transform, transform,
)


def around_site(text, site_start, width=100):
    words = list(WORD.finditer(text))
    if len(words) < width:
        return None
    target = next(i for i, match in enumerate(words)
                  if match.start() == site_start)
    begin = max(0, min(target-width//2, len(words)-width))
    return text[words[begin].start():words[begin+width-1].end()]


def run(rows_path=DEFAULT_ROWS, tokenizer_path=DEFAULT_TOKENIZER,
        seed=58103, min_gap_words=20, max_gap_words=250,
        identity_policy='strict_full_name'):
    rows = torch.load(rows_path, map_location='cpu', weights_only=True)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    raw = rows.numpy()
    assert rows.ndim == 2 and rows.shape[1] == 513
    original, transformed, original_site, transformed_site = [], [], [], []
    counts = Counter()
    gaps = []
    pool_sizes = []
    examples = []
    for index, row in enumerate(rows):
        text = tokenizer.decode(row.tolist(), skip_special_tokens=False)
        rng = random.Random(seed+index)
        possible, _ = candidates(text, min_gap_words, max_gap_words,
                                 identity_policy)
        counts['windows'] += 1
        counts['eligible_windows'] += bool(possible)
        sample = transform(text, rng, min_gap_words, max_gap_words,
                           identity_policy)
        if sample is None:
            continue
        assert reverse_transform(sample) == text
        assert sample.text[slice(*sample.answer_span)] == sample.sampled_name
        assert sample.sampled_name in sample.prefix
        assert not re.search(r'\b'+re.escape(sample.source_name)+r'\b',
                             sample.text)
        assert sample.name_group == NAME_TO_GROUP[sample.source_name]
        counts['transformed_windows'] += 1
        counts['source_occurrences'] += len(sample.source_spans)
        gaps.append(sample.gap_words)
        pool_sizes.append(sample.target_pool_size)

        present = {word.lower() for word in WORD.findall(text)}
        alternatives = [name for name in NAME_GROUPS[sample.name_group]
                        if name.lower() not in present and
                        name != sample.sampled_name]
        alternate = transform(
            text, random.Random(seed+index), min_gap_words,
            max_gap_words, identity_policy,
            forced_source=sample.source_name,
            forced_target=alternatives[0],
            forced_occurrence=sample.scored_occurrence)
        assert alternate is not None
        assert reverse_transform(alternate) == text
        assert alternate.sampled_name != sample.sampled_name
        assert alternate.gap_words == sample.gap_words
        counts['supported_name_flips'] += 1

        other_names = [match.group() for match in NAME.finditer(text)
                       if match.group() in NAME_TO_GROUP and
                       match.group() != sample.source_name]
        if other_names:
            other = other_names[0]
            other_group = NAME_TO_GROUP[other]
            replacements = [name for name in NAME_GROUPS[other_group]
                            if name.lower() not in present and
                            name != sample.sampled_name]
            if replacements:
                altered = re.sub(r'\b'+re.escape(other)+r'\b',
                                 replacements[0], text)
                control = transform(
                    altered, random.Random(seed+index), min_gap_words,
                    max_gap_words, identity_policy,
                    forced_source=sample.source_name,
                    forced_target=sample.sampled_name,
                    forced_occurrence=sample.scored_occurrence)
                assert control is not None
                assert control.sampled_name == sample.sampled_name
                assert control.text[slice(*control.answer_span)] == \
                    sample.sampled_name
                counts['irrelevant_name_no_effect_checks'] += 1

        original.append(first_words(text, 100))
        transformed.append(first_words(sample.text, 100))
        before = around_site(text, sample.source_spans[
            sample.scored_occurrence][0])
        after = around_site(sample.text, sample.answer_span[0])
        if before and after:
            original_site.append(before)
            transformed_site.append(after)
        if len(examples) < 20:
            examples.append(dict(row=index,
                                 source_name=sample.source_name,
                                 sampled_name=sample.sampled_name,
                                 occurrences=len(sample.source_spans),
                                 gap_words=sample.gap_words,
                                 target_character_offset=sample.answer_span[0]))
    assert counts['transformed_windows'] == len(original)
    assert len(original_site) == len(transformed_site)
    return dict(config=dict(seed=seed, min_gap_words=min_gap_words,
                            max_gap_words=max_gap_words,
                            identity_policy=identity_policy,
                            source='frozen local FineWeb GPT-2 window panel',
                            source_sha256=hashlib.sha256(
                                raw.tobytes()).hexdigest(),
                            tokenizer=str(tokenizer_path)),
                counts=dict(counts),
                gap_words=dict(min=min(gaps), max=max(gaps),
                               mean=sum(gaps)/len(gaps)),
                target_pool=dict(min=min(pool_sizes), max=max(pool_sizes),
                                 mean=sum(pool_sizes)/len(pool_sizes),
                                 blind_accuracy_mean=sum(
                                     1/n for n in pool_sizes)/len(pool_sizes)),
                first100_form=dict(original=structure(original),
                                   transformed=structure(transformed)),
                target100_form=dict(original=structure(original_site),
                                    transformed=structure(transformed_site)),
                examples=examples,
                caveat='Corpus transformation with exact edit sites, not a '
                       'symbolic oracle for natural source text. A token '
                       'window is not a document; candidate names are '
                       'heuristic. No source prose is saved.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--identity-policy',
                   choices=('strict_full_name', 'first_name'),
                   default='strict_full_name')
    args = p.parse_args()
    result = run(identity_policy=args.identity_policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['counts'])
    for key in ('first100_form', 'target100_form'):
        for arm, row in result[key].items():
            print(key, arm, row['word_types'],
                  row['repeated_across_docs_trigram_occurrence_share'])


if __name__ == '__main__':
    main()
