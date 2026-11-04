"""Matched old/new-year prediction under a visible table-cell intervention."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import zipfile

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from audit_totto_table_description_dgp import DEFAULT_TOKENIZER
from build_totto_table_description_cache import TRAIN, encode, page_split
from totto_table_description_dgp import extract
from train_corpus_delimiter_bilinear import _model


def context_and_value(tokenizer, text, span):
    enc = tokenizer.encode(text)
    lo, hi = span
    hit = [i for i, (a, b) in enumerate(enc.offsets) if a < hi and b > lo]
    if not hit:
        return None
    a, b = hit[0], hit[-1]
    if a == 0 or b >= 256:
        return None
    if text[enc.offsets[a][0]:lo].strip() or text[hi:enc.offsets[b][1]].strip():
        return None
    return enc.ids[:a], enc.ids[a:b+1]


def make_pair(row, tokenizer, rng):
    example = extract(row)
    if example is None:
        return None
    if page_split(row.get('table_webpage_url') or example.page) != 'test':
        return None
    if encode(tokenizer, example) is None:
        return None
    sites = [site for site in example.sites
             if site.column.casefold() == 'year' and
             re.fullmatch(r'\d{4}', site.value)]
    if len(sites) != 1:
        return None
    site = sites[0]
    old = site.value
    original = example.text
    description = original.index('\nDescription: ')
    value_start = original.rfind('\n| ', 0, description) + 1
    value_line = original[value_start:description]
    hits = list(re.finditer(r'(?<!\d)' + old + r'(?!\d)', value_line))
    if len(hits) != 1 or old in original[:value_start]:
        return None
    alternatives = [str(int(old) + offset) for offset in
                    (1, -1, 2, -2, 3, -3, 5, -5, 10, -10)
                    if 1600 <= int(old) + offset <= 2026 and
                    str(int(old) + offset) not in original]
    if not alternatives:
        return None
    new = rng.choice(alternatives)
    source_lo = value_start + hits[0].start()
    source_hi = source_lo + 4
    target_lo, target_hi = site.span
    if source_hi >= target_lo:
        return None
    source_changed = original[:source_lo] + new + original[source_hi:]
    target_changed = original[:target_lo] + new + original[target_hi:]
    both_changed = source_changed[:target_lo] + new + source_changed[target_hi:]
    original_old = context_and_value(tokenizer, original, site.span)
    original_new = context_and_value(tokenizer, target_changed, site.span)
    changed_old = context_and_value(tokenizer, source_changed, site.span)
    changed_new = context_and_value(tokenizer, both_changed, site.span)
    if any(item is None for item in (original_old, original_new,
                                     changed_old, changed_new)):
        return None
    if (original_old[0] != original_new[0] or
            changed_old[0] != changed_new[0]):
        return None
    # Both arms get the same target candidates; only the table context moves.
    if original_old[1] != changed_old[1] or original_new[1] != changed_new[1]:
        return None
    return {'original_context': original_old[0],
            'changed_context': changed_old[0],
            'old_ids': original_old[1], 'new_ids': original_new[1],
            'old': old, 'new': new,
            'combo': {'title', 'year'} <= {
                s.column.casefold() for s in example.sites}}


@torch.no_grad()
def continuation_logprob(model, context, continuation, device):
    ids = context + continuation[:-1]
    logits = model(torch.tensor(ids, device=device).long()[None, :])[0]
    start = len(context)-1
    wanted = torch.tensor(continuation, device=device).long()
    return F.log_softmax(logits[start:start+len(wanted)].float(), -1)[
        torch.arange(len(wanted), device=device), wanted].sum().item()


def probe(source, cache, state_path, *, limit=128, seed=74901,
          device='cuda'):
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    state = torch.load(state_path, map_location='cpu', weights_only=True)
    cache_data = torch.load(cache, map_location='cpu', weights_only=True)
    identity = state['identity']
    model = _model(cache_data['vocab_size'], cache_data['n_ctx'],
                   identity[1], identity[2], identity[3], device, 'rms')
    model.load_state_dict(state['model'])
    model.eval()
    rng = random.Random(seed + 729)
    cases = []
    counts = Counter()
    with zipfile.ZipFile(source) as archive, archive.open(TRAIN) as stream:
        for line in stream:
            counts['records'] += 1
            case = make_pair(json.loads(line), tokenizer, rng)
            if case is None:
                continue
            counts['eligible_pairs'] += 1
            if len(cases) < limit:
                cases.append(case)
            else:
                j = rng.randrange(counts['eligible_pairs'])
                if j < limit:
                    cases[j] = case
    records = []
    for case in cases:
        old, new = case['old_ids'], case['new_ids']
        original_margin = (continuation_logprob(
            model, case['original_context'], old, device) -
            continuation_logprob(model, case['original_context'], new,
                                 device))
        changed_margin = (continuation_logprob(
            model, case['changed_context'], old, device) -
            continuation_logprob(model, case['changed_context'], new,
                                 device))
        records.append({'old': case['old'], 'new': case['new'],
                        'combo': case['combo'],
                        'original_margin_old_minus_new': original_margin,
                        'changed_margin_old_minus_new': changed_margin,
                        'shift_toward_new': original_margin-changed_margin})
    n = len(records)
    if n == 0:
        raise ValueError('No aligned year substitutions')
    return {'checkpoint_step': state['step'], 'n': n,
            'eligible_pairs': counts['eligible_pairs'],
            'mean_shift_toward_new': sum(x['shift_toward_new'] for x in records)/n,
            'fraction_shift_toward_new': sum(x['shift_toward_new'] > 0
                                             for x in records)/n,
            'fraction_old_preferred_original': sum(
                x['original_margin_old_minus_new'] > 0 for x in records)/n,
            'fraction_new_preferred_changed': sum(
                x['changed_margin_old_minus_new'] < 0 for x in records)/n,
            'fraction_correct_both_arms': sum(
                x['original_margin_old_minus_new'] > 0 and
                x['changed_margin_old_minus_new'] < 0 for x in records)/n,
            'examples': records[:12],
            'caveat': 'Four-digit Year cells only. The surrounding human '
                      'sentence is held fixed; only its target value is '
                      'counterfactually scored. Correctness is local to '
                      'this value, not a proof all text remains consistent.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=128)
    parser.add_argument('--seed', type=int, default=74901)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = probe(args.source, args.cache, args.state, limit=args.limit,
                   seed=args.seed, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'examples'},
                     indent=2))


if __name__ == '__main__':
    main()
