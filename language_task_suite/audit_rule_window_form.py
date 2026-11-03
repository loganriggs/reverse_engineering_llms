"""Compare task-bearing synthetic windows to frozen natural-text windows.

This is a surface-form diagnostic. Natural windows contain no matched task
labels; each synthetic family uses at most one site per sampled document.
Only aggregate statistics and site counts are written.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

import torch
from tokenizers import Tokenizer

from audit_document_structure_vs_fineweb import structure
from audit_fineweb_surfaces import DEFAULT_ROWS, DEFAULT_TOKENIZER
from audit_surface_repetition import WORD
from event_frame_language_v2 import EventFrameLanguageV2, render as render_v2
from surface_mixture_v6 import render as render_v14
from surface_mixture_v14_conditional import SurfaceMixtureV14Conditional
from train_v13_attributes import COMMON, REFERENCE


FAMILIES = ('mixed_verb', 'mixed_object', 'attribute_object',
            'bracket_close', 'source_answer', 'conditional_action')


def word_window(text, target_word, width):
    matches = list(WORD.finditer(text))
    if len(matches) < width:
        return None
    start = max(0, min(target_word-width//2, len(matches)-width))
    return text[matches[start].start():matches[start+width-1].end()]


def synthetic_panel(kind, seed, n, length, width):
    gen = (SurfaceMixtureV14Conditional(
        seed=seed, **COMMON, **REFERENCE,
        conditional_continuation_rate=.35)
           if kind == 'v14' else EventFrameLanguageV2(seed))
    render = render_v14 if kind == 'v14' else render_v2
    allowed = FAMILIES if kind == 'v14' else FAMILIES[:2]
    rng = random.Random(seed+991)
    windows = defaultdict(list)
    site_count = Counter()
    for _ in range(n):
        ids, labels, _ = gen.sample_sequence(length)
        words = [gen.id2token[i] for i in ids]
        text = render(words)
        locations = defaultdict(list)
        for i, family in enumerate(labels):
            if family in allowed:
                site_count[family] += 1
                locations[family].append(i)
        for family, positions in locations.items():
            i = rng.choice(positions)
            target_word = len(WORD.findall(render(words[:i])))
            crop = word_window(text, target_word, width)
            if crop:
                windows[family].append(crop)
    return {'site_counts': dict(site_count),
            'window_counts': {k: len(windows[k]) for k in allowed},
            'form': {k: structure(windows[k]) for k in allowed
                     if windows[k]}}


def natural_panel(rows_path, tokenizer_path, seed, size_by_family, width):
    rows = torch.load(rows_path, map_location='cpu', weights_only=True)
    assert rows.ndim == 2 and rows.shape[1] == 513
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    texts = [tokenizer.decode(row.tolist(), skip_special_tokens=False)
             for row in rows]
    source_hash = hashlib.sha256(rows.numpy().tobytes()).hexdigest()
    out = {}
    for index, (family, n) in enumerate(size_by_family.items()):
        rng = random.Random(seed+index)
        eligible = [i for i, text in enumerate(texts)
                    if len(WORD.findall(text)) >= width]
        assert len(eligible) >= n
        chosen = rng.sample(eligible, n)
        windows = []
        for i in chosen:
            word_count = len(WORD.findall(texts[i]))
            center = rng.randrange(width//2, word_count-width//2)
            crop = word_window(texts[i], center, width)
            assert crop is not None
            windows.append(crop)
        out[family] = structure(windows)
    return {'source_sha256': source_hash, 'n_source_windows': len(texts),
            'form': out}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=128)
    p.add_argument('--length', type=int, default=256)
    p.add_argument('--width', type=int, default=41)
    p.add_argument('--seed', type=int, default=53401)
    p.add_argument('--rows', type=Path, default=DEFAULT_ROWS)
    p.add_argument('--tokenizer', type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    panels = {f'{kind}_seed_{i}': synthetic_panel(
        kind, a.seed+i+1000*(kind == 'v2'), a.n, a.length, a.width)
        for kind in ('v14', 'v2') for i in range(2)}
    sizes = {f'{name}/{family}': count
             for name, panel in panels.items()
             for family, count in panel['window_counts'].items()
             if count}
    natural = natural_panel(a.rows, a.tokenizer, a.seed+5000, sizes, a.width)
    result = {'config': {'n': a.n, 'length': a.length, 'width_words': a.width,
                         'seed': a.seed, 'source': 'frozen FineWeb GPT-2 '
                         '512-token windows'},
              'synthetic': panels, 'natural': natural,
              'caveat': 'Natural windows are randomly located, not '
                        'mechanism-matched. Synthetic panels use at most '
                        'one randomly selected site per document/family. '
                        'This audit measures form, not truth conditions '
                        'or task prevalence in natural text. No raw '
                        'source passages are exported.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    for key, panel in panels.items():
        for family, form in panel['form'].items():
            ref = natural['form'][key+'/'+family]
            print(key, family, panel['window_counts'][family],
                  'trigram', round(form[
                      'repeated_across_docs_trigram_occurrence_share'], 3),
                  'natural', round(ref[
                      'repeated_across_docs_trigram_occurrence_share'], 3),
                  flush=True)


if __name__ == '__main__':
    main()
