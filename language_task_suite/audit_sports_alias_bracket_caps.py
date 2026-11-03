"""Eight-arm semantic, BPE-alignment, and form audit for three rules."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random

from tokenizers import Tokenizer

from audit_document_structure_vs_fineweb import structure
from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from compare_v12_fineweb_fresh import first_words
from sports_alias_bracket_caps_dgp import (
    caps_candidates, realize, sample_plan, visible_caps_answer)
from sports_alias_bracket_dgp import visible_bracket_answer
from sports_entity_alias_dgp import _render_tokens


ARMS = tuple(itertools.product(('a', 'b'), ('round', 'square'),
                               ('period', 'semicolon')))


def _window(text, index):
    return ' '.join(text[:index].split()[-25:]+text[index:].split()[:16])


def aligned(doc, tokenizer, n_ctx):
    enc = tokenizer.encode(doc.text, add_special_tokens=False)
    positions = []
    for site in doc.sites:
        matches = [i for i, (start, end) in enumerate(enc.offsets)
                   if start <= site.start < end]
        if (len(matches) != 1 or matches[0] < 1 or
                matches[0] >= n_ctx or
                enc.offsets[matches[0]][1] != site.end):
            return None
        positions.append(matches[0])
    return positions


def audit(group, n, seed, n_ctx, tokenizer):
    rng = random.Random(seed)
    counts = Counter()
    seen = set()
    game_hashes = []
    originals, full_edits = [], []
    windows = {'original': {k: [] for k in
                            ('alias', 'bracket', 'capitalization')},
               'edited': {k: [] for k in
                          ('alias', 'bracket', 'capitalization')}}
    manual = []
    attempts = 0
    while len(seen) < n:
        attempts += 1
        assert attempts <= 500*n
        plan = sample_plan(rng, group)
        key = plan.pair.alias.game_key
        if key in seen:
            continue
        seen.add(key)
        game_hashes.append(hashlib.sha256(repr(key).encode()).hexdigest())
        counts['cap_word/'+plan.pair.alias.source_tokens[
            plan.boundary_token+1]] += 1
        counts['cap_before_alias'] += (
            plan.boundary_token+1 < plan.pair.alias.target_token)
        counts['cap_before_bracket'] += (
            plan.boundary_token+1 < plan.pair.closer_token)
        docs = {arm: realize(plan, *arm) for arm in ARMS}
        original_text, original_spans = _render_tokens(
            plan.pair.alias.source_tokens)
        original_positions = {
            'alias': original_spans[plan.pair.alias.target_token][0],
            'bracket': original_spans[plan.pair.closer_token][0],
            'capitalization': original_spans[plan.boundary_token+1][0],
        }
        assert caps_candidates(plan.pair)
        all_aligned = True
        for arm, doc in docs.items():
            for site in doc.sites:
                assert doc.text[site.start:site.end] == site.answer
                counts['visible_site_checks'] += 1
            alias, bracket, caps = doc.sites
            assert doc.text[alias.source_start:alias.source_end] == \
                plan.pair.alias.first+' '+alias.answer
            assert visible_bracket_answer(doc.text[:bracket.start]) == \
                bracket.answer
            capital = plan.pair.alias.source_tokens[plan.boundary_token+1]
            assert visible_caps_answer(doc.text[:caps.start],
                                       (capital, capital.lower())) == caps.answer
            in_context = aligned(doc, tokenizer, n_ctx) is not None
            counts['all_three_single_bpe_in_context_arms'] += in_context
            all_aligned &= in_context
        counts['complete_aligned_factorials'] += all_aligned
        for left, right in itertools.combinations(ARMS, 2):
            a, b = docs[left].sites, docs[right].sites
            for index, (x, y) in enumerate(zip(a, b)):
                changed = left[index] != right[index]
                assert (x.answer != y.answer) == changed
        counts['independent_factorials'] += 1
        originals.append(first_words(original_text, 100))
        full = docs[('a', 'square', 'semicolon')]
        full_edits.append(first_words(full.text, 100))
        if len(windows['original']['alias']) < 64:
            for site in docs[('a', 'round', 'period')].sites:
                windows['original'][site.kind].append(_window(
                    original_text, original_positions[site.kind]))
            for site in full.sites:
                windows['edited'][site.kind].append(_window(full.text,
                                                           site.start))
        if len(manual) < 12:
            i = plan.boundary_token
            manual.append({
                'source_context': ' '.join(plan.pair.alias.source_tokens[
                    max(0, i-14):i+16]),
                'edited_context': full.text[max(0, full.sites[2].start-90):
                                            full.sites[2].end+95]})
    form_n = min(64, n)
    return {'counts': dict(counts), 'draw_attempts': attempts,
            'game_sha256': game_hashes,
            'first100_form': {
                'original': structure(originals[:form_n]),
                'fully_edited': structure(full_edits[:form_n])},
            'site41_form': {arm: {kind: structure(rows)
                                  for kind, rows in groups.items()}
                            for arm, groups in windows.items()},
            'manual': manual}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--seed', type=int, default=55201)
    p.add_argument('--n-ctx', type=int, default=320)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    result = {'config': {'n': a.n, 'seed': a.seed,
                         'n_ctx': a.n_ctx, 'arms': ARMS},
              'groups': {},
              'caveat': 'The report carrier is natural text with no full '
                        'next-token oracle. The semicolon/lowercase arm is '
                        'an orthographic clause-joining intervention; '
                        'candidate sentence semantics need manual review. '
                        'Only aggregate metrics and game hashes are saved.'}
    for group in ('train', 'heldout'):
        row = audit(group, a.n, a.seed+(group == 'heldout')*100,
                    a.n_ctx, tokenizer)
        print(group, row['counts'], 'manual', row.pop('manual'), flush=True)
        result['groups'][group] = row
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
