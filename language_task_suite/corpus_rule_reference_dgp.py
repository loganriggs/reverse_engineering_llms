"""Source-unconditioned FineWeb rule stream with ordinary token chunks.

Only selected local sites have symbolic answers. Natural background tokens do
not have an exact symbolic next-token oracle. The transformations preserve
text length, so several eligible rule families can coexist in one document.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

import torch

from corpus_numbered_list_dgp import candidates as list_candidates
from corpus_numbered_list_dgp import visible_next_number
from corpus_nonce_copy_dgp import candidates as nonce_candidates, draw_pair
from corpus_transformed_delimiters import candidates as bracket_candidates


PAD = 50256


@dataclass(frozen=True)
class RuleSite:
    kind: str
    source: tuple[int, int]
    target: tuple[int, int]
    answer: str


@dataclass(frozen=True)
class Document:
    source_text: str
    text: str
    sites: tuple[RuleSite, ...]
    eligible: tuple[str, ...]
    applied: tuple[str, ...]


def sample_document(source_text, rng: random.Random, nonce_pool,
                    list_rate=1., bracket_rate=1., nonce_rate=0.,
                    list_shift_rate=0., square_rate=0., max_chars=8000):
    rates = (list_rate, bracket_rate, nonce_rate, list_shift_rate,
             square_rate)
    if any(not 0 <= rate <= 1 for rate in rates):
        raise ValueError('Every rate must be in [0,1]')
    source_text = source_text[:max_chars]
    edits, sites, eligible, applied = {}, [], [], []

    lists = list_candidates(source_text)
    if lists:
        eligible.append('list')
        if rng.random() < list_rate:
            markers = rng.choice(lists)
            shift = int(rng.random() < list_shift_rate)
            for start, end, number in markers:
                assert end == start + 1
                edits[start] = str(number + shift)
            sites.append(RuleSite('list', markers[1][:2], markers[2][:2],
                                  str(markers[2][2] + shift)))
            applied.append('list')

    brackets = bracket_candidates(source_text)
    if brackets:
        eligible.append('bracket')
        if rng.random() < bracket_rate:
            start, end, _ = rng.choice(brackets)
            is_square = rng.random() < square_rate
            edits[start] = '[' if is_square else '('
            edits[end] = ']' if is_square else ')'
            sites.append(RuleSite('bracket', (start, start + 1),
                                  (end, end + 1), edits[end]))
            applied.append('bracket')

    nonces = list(nonce_candidates(source_text, entity_only=True))
    if nonces:
        eligible.append('nonce')
        if rng.random() < nonce_rate:
            plan = rng.choice(nonces)
            pair = draw_pair(plan, nonce_pool, rng)
            if pair is not None:
                alias = pair[0]
                for start, end in alias.spans:
                    assert end == start + 3
                    edits.update({start + i: letter
                                  for i, letter in enumerate(alias.code)})
                sites.append(RuleSite('nonce', alias.previous_span,
                                      alias.answer_span, alias.code))
                applied.append('nonce')

    chars = list(source_text)
    for index, char in edits.items():
        assert len(char) == 1
        chars[index] = char
    text = ''.join(chars)
    document = Document(source_text, text, tuple(sites), tuple(eligible),
                        tuple(applied))
    check_document(document)
    return document


def check_document(document):
    assert len(document.text) == len(document.source_text)
    assert len(document.sites) == len(set(site.kind for site in document.sites))
    for site in document.sites:
        a, b = site.target
        assert document.text[a:b] == site.answer
        assert site.source[1] <= a
        if site.kind == 'list':
            assert visible_next_number(document.text[:a]) == site.answer
        elif site.kind == 'bracket':
            opener = document.text[site.source[0]]
            assert (opener, site.answer) in (('(', ')'), ('[', ']'))
        elif site.kind == 'nonce':
            assert document.text[slice(*site.source)] == site.answer
        else:
            raise AssertionError(site.kind)
    return True


def _token_span(encoding, char_span):
    a, b = char_span
    indices = [i for i, (start, end) in enumerate(encoding.offsets)
               if start < b and end > a]
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        return None
    return tuple(indices)


def chunk_document(document, tokenizer, n_ctx=256, max_tokens=2048):
    """Nonoverlapping next-token chunks; site labels only with visible source."""
    encoding = tokenizer.encode(document.text, add_special_tokens=False)
    ids = encoding.ids[:max_tokens + 1]
    token_sites = []
    for site in document.sites:
        source = _token_span(encoding, site.source)
        target = _token_span(encoding, site.target)
        if source is None or target is None or source[-1] >= target[0]:
            continue
        first, last = encoding.offsets[target[0]], encoding.offsets[target[-1]]
        if site.kind == 'nonce':
            if (len(source) != 3 or len(target) != 3 or
                    last[1] != site.target[1]):
                continue
        elif site.kind == 'list':
            if len(target) != 1 or first != site.target:
                continue
        elif site.kind == 'bracket':
            if len(target) != 1 or first[0] != site.target[0]:
                continue
        token_sites.append((site, source, target))
    chunks = []
    for start in range(0, min(len(ids) - 1, max_tokens), n_ctx):
        window = ids[start:start + n_ctx + 1]
        if len(window) < 2:
            continue
        x = torch.full((n_ctx,), PAD, dtype=torch.long)
        y = torch.full((n_ctx,), -100, dtype=torch.long)
        x[:len(window) - 1] = torch.tensor(window[:-1])
        y[:len(window) - 1] = torch.tensor(window[1:])
        local_sites = []
        for site, source, target in token_sites:
            if (source[0] < start or target[0] <= start or
                    target[-1] >= start + len(window)):
                continue
            positions = tuple(index - start - 1 for index in target)
            assert all(y[pos].item() == ids[index]
                       for pos, index in zip(positions, target, strict=True))
            local_sites.append({'kind': site.kind,
                                'positions': positions,
                                'target_ids': tuple(ids[index] for index in target),
                                'source_token': source[0] - start})
        chunks.append({'x': x, 'y': y, 'sites': local_sites,
                       'start_token': start})
    return chunks
