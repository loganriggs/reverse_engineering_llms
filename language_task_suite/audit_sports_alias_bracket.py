"""Four-arm semantic, alignment, and natural-form audit for joint rules."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from tokenizers import Tokenizer

from audit_document_structure_vs_fineweb import structure
from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from compare_v12_fineweb_fresh import first_words
from sports_alias_bracket_dgp import (
    realize, sample_plan, visible_bracket_answer,
)
from sports_entity_alias_dgp import _render_tokens


def _window(text, index):
    return ' '.join(text[:index].split()[-25:]+text[index:].split()[:16])


def _aligned(document, tokenizer, n_ctx):
    encoded = tokenizer.encode(document.text, add_special_tokens=False)
    for site in document.sites:
        matches = [i for i, (start, end) in enumerate(encoded.offsets)
                   if start <= site.start < end]
        if len(matches) != 1 or matches[0] < 1 or matches[0] >= n_ctx or \
                encoded.offsets[matches[0]][1] != site.end:
            return False
    return True


def audit(group, n, form_n, seed, tokenizer, n_ctx):
    rng = random.Random(seed)
    counts = Counter()
    seen = set()
    hashes = []
    originals, aliases, squares = [], [], []
    original_windows = {'alias': [], 'bracket': []}
    edited_windows = {'alias': [], 'bracket': []}
    manual = []
    attempts = 0
    while len(seen) < n:
        attempts += 1
        if attempts > n*500:
            raise RuntimeError('Too few unique joint-rule reports')
        plan = sample_plan(rng, group)
        if plan.alias.game_key in seen:
            continue
        seen.add(plan.alias.game_key)
        hashes.append(hashlib.sha256(repr(plan.alias.game_key).encode())
                      .hexdigest())
        docs = {(a, b): realize(plan, a, b)
                for a in ('a', 'b') for b in ('round', 'square')}
        for (a, b), doc in docs.items():
            alias, bracket = doc.sites
            assert alias.kind == 'alias' and bracket.kind == 'bracket'
            assert doc.text[alias.start:alias.end] == alias.answer
            assert doc.text[bracket.start:bracket.end] == bracket.answer
            assert doc.text[alias.source_start:alias.source_end] == \
                plan.alias.first+' '+alias.answer
            assert doc.text[bracket.source_start:bracket.source_end] == \
                ('(' if b == 'round' else '[')
            assert visible_bracket_answer(doc.text[:bracket.start]) == \
                bracket.answer
            counts['visible_site_checks'] += 2
            counts['both_sites_single_bpe_in_context'] += \
                _aligned(doc, tokenizer, n_ctx)
        assert docs['a', 'round'].sites[0].answer != \
            docs['b', 'round'].sites[0].answer
        assert docs['a', 'round'].sites[1].answer == \
            docs['b', 'round'].sites[1].answer
        assert docs['a', 'round'].sites[0].answer == \
            docs['a', 'square'].sites[0].answer
        assert docs['a', 'round'].sites[1].answer != \
            docs['a', 'square'].sites[1].answer
        counts['independent_edit_factorials'] += 1
        base = plan.alias
        original_text, spans = _render_tokens(base.source_tokens)
        original_idx = {'alias': spans[base.target_token][0],
                        'bracket': spans[plan.closer_token][0]}
        a = docs['a', 'round']
        square = docs['a', 'square']
        originals.append(first_words(original_text, 100))
        aliases.append(first_words(a.text, 100))
        squares.append(first_words(square.text, 100))
        if len(original_windows['alias']) < form_n:
            for kind, site in zip(('alias', 'bracket'), a.sites):
                original_windows[kind].append(_window(
                    original_text, original_idx[kind]))
                edited_windows[kind].append(_window(a.text, site.start))
            if len(manual) < 4:
                manual.append({'original': original_text[:500],
                               'joint': square.text[:500]})
    return {'counts': dict(counts), 'draw_attempts': attempts,
            'game_sha256': hashes,
            'first100': {'original': structure(originals[:form_n]),
                         'alias': structure(aliases[:form_n]),
                         'alias_square': structure(squares[:form_n])},
            'windows': {'original': {k: structure(v)
                                     for k, v in original_windows.items()},
                        'alias': {k: structure(v)
                                  for k, v in edited_windows.items()}},
            'manual': manual}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--form-n', type=int, default=64)
    p.add_argument('--seed', type=int, default=98101)
    p.add_argument('--n-ctx', type=int, default=320)
    args = p.parse_args()
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    result = {'config': {**vars(args), 'output': str(args.output)},
              'groups': {},
              'caveat': 'Original task windows are human prose. Square '
                        'brackets and aliases are controlled interventions. '
                        'The article has no full next-token oracle; alias '
                        'coreference is surface-screened. Manual examples '
                        'are printed locally and removed from JSON.'}
    for group in ('train', 'heldout'):
        row = audit(group, args.n, args.form_n,
                    args.seed+(group == 'heldout')*100, tokenizer,
                    args.n_ctx)
        manual = row.pop('manual')
        result['groups'][group] = row
        print(group, row['counts'], 'manual', manual, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
