"""Causal and form audit for delimiter-style resampling in natural prose."""

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_corpus_transformed_name_reuse import around_site
from audit_document_structure_vs_fineweb import structure
from audit_fineweb_surfaces import DEFAULT_ROWS, DEFAULT_TOKENIZER
from compare_v12_fineweb_fresh import first_words
from corpus_transformed_delimiters import WORD, candidates, transform, validate


def around_close(text, close_index, width=100):
    prior = [match for match in WORD.finditer(text)
             if match.end() <= close_index]
    return around_site(text, prior[-1].start(), width) if prior else None


def audit(rows_path=DEFAULT_ROWS, tokenizer_path=DEFAULT_TOKENIZER,
          seed=23117):
    rows = torch.load(rows_path, map_location='cpu', weights_only=True)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    raw = rows.numpy()
    counts, gaps, rows_seen = Counter(), [], []
    original_first, square_first = [], []
    original_site, square_site = [], []
    style_sweep = {rate: Counter() for rate in (0., .01, .04, .1, .5, 1.)}
    square_pair = re.compile(r'\[[^\]\n]{1,220}\]')
    for index, row in enumerate(rows):
        source = tokenizer.decode(row.tolist(), skip_special_tokens=False)
        options = candidates(source)
        counts['windows'] += 1
        counts['eligible_windows'] += bool(options)
        counts['eligible_pairs'] += len(options)
        for rate, setting in style_sweep.items():
            drawn = transform(source, random.Random(seed+index),
                              square_rate=rate) if options else None
            surface = drawn.text if drawn else source
            setting['square_selected'] += bool(drawn and
                                                  drawn.style == 'square')
            setting['windows_with_square_pair'] += bool(
                square_pair.search(surface))
        if not options:
            continue
        rng = random.Random(seed+index)
        pair_index = rng.randrange(len(options))
        round_ex = transform(source, rng, style='round',
                             pair_index=pair_index)
        square_ex = transform(source, rng, style='square',
                              pair_index=pair_index)
        assert round_ex and square_ex
        assert validate(round_ex) and validate(square_ex)
        assert round_ex.text == source
        assert square_ex.text != source
        assert round_ex.prefix != square_ex.prefix
        assert round_ex.answer == ')' and square_ex.answer == ']'
        assert (round_ex.prefix[:round_ex.open_index] ==
                square_ex.prefix[:square_ex.open_index])
        assert (round_ex.prefix[round_ex.open_index+1:] ==
                square_ex.prefix[square_ex.open_index+1:])
        counts['supported_delimiter_flips'] += 1
        gaps.append(square_ex.gap_words)
        rows_seen.append(index)

        next_token_ok = []
        pure_delimiter_token = []
        for ex in (round_ex, square_ex):
            prefix = tokenizer.encode(ex.prefix,
                                      add_special_tokens=False).ids
            full = tokenizer.encode(ex.text, add_special_tokens=False)
            next_index = len(prefix)
            okay = (next_index < len(full.ids) and
                    full.ids[:next_index] == prefix and
                    full.offsets[next_index][0] == ex.close_index and
                    full.offsets[next_index][1] > ex.close_index)
            next_token_ok.append(okay)
            if okay:
                counts['single_next_token_sites'] += 1
            pure_delimiter_token.append(
                okay and full.offsets[next_index][1] == ex.close_index+1)
        counts['paired_single_next_token_sites'] += all(next_token_ok)
        counts['paired_pure_delimiter_token_sites'] += all(
            pure_delimiter_token)
        if len(options) >= 2:
            earlier = [row for j, row in enumerate(options)
                       if j != pair_index and row[1] < square_ex.open_index]
            if earlier:
                other_start, other_end, _ = earlier[-1]
                both = (square_ex.text[:other_start]+'['+
                        square_ex.text[other_start+1:other_end]+']'+
                        square_ex.text[other_end+1:])
                assert both[square_ex.open_index] == '['
                assert both[square_ex.close_index] == ']'
                counts['earlier_other_pair_no_effect_checks'] += 1

        original_first.append(first_words(source, 100))
        square_first.append(first_words(square_ex.text, 100))
        before = around_close(source, round_ex.close_index, 100)
        after = around_close(square_ex.text, square_ex.close_index, 100)
        if before and after:
            original_site.append(before)
            square_site.append(after)
    return dict(config=dict(seed=seed, n_windows=len(rows),
                            source='frozen FineWeb GPT-2 token windows',
                            source_sha256=hashlib.sha256(
                                raw.tobytes()).hexdigest()),
                counts=dict(counts),
                gap_words=dict(min=min(gaps), max=max(gaps),
                               mean=sum(gaps)/len(gaps)),
                eligible_row_indices=rows_seen,
                style_sweep={str(rate): dict(setting)
                             for rate, setting in style_sweep.items()},
                first100_form=dict(original=structure(original_first),
                                   square=structure(square_first)),
                target100_form=(dict(original=structure(original_site),
                                     square=structure(square_site))
                                if original_site else None),
                caveat='An empirical natural-text carrier with a known '
                       'delimiter edit, not a symbolic full-text oracle. '
                       'Window rows can overlap source documents; no '
                       'source prose is saved.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['counts'])
    for arm, panel in result['first100_form'].items():
        print(arm, panel['word_types'],
              panel['repeated_across_docs_trigram_occurrence_share'])


if __name__ == '__main__':
    main()
