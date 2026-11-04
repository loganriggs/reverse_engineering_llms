"""Paired natural-code source-dependence gate on fresh FineWeb documents."""

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
import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from corpus_uppercase_copy_dgp import candidates, codebook, realize
from corpus_uppercase_copy_validation import _token_index
from eval_corpus_rule_reference_transfer import one_kind
from train_corpus_delimiter_bilinear import _model


PAD = 50256


def _chunk(encoding, start, n_ctx):
    window = encoding.ids[start:start+n_ctx+1]
    x = torch.full((n_ctx,), PAD, dtype=torch.long)
    y = torch.full((n_ctx,), -100, dtype=torch.long)
    x[:len(window)-1] = torch.tensor(window[:-1])
    y[:len(window)-1] = torch.tensor(window[1:])
    return x, y


def make_pair(plan, tokenizer, rng, n_ctx=256, max_tokens=2048):
    base = tokenizer.encode(plan.source_text, add_special_tokens=False)
    source_span = plan.spans[plan.target_occurrence-1]
    target_span = plan.spans[plan.target_occurrence]
    source = _token_index(base, plan.source_text, source_span)
    target = _token_index(base, plan.source_text, target_span)
    if source is None or target is None or not (source < target <= max_tokens):
        return None, 'not_one_token_or_limit'
    start = ((target-1)//n_ctx)*n_ctx
    if source < start or base.ids[source] != base.ids[target]:
        return None, 'not_local_exact_copy'
    codes = [code for code in codebook(tokenizer)
             if code != plan.source_code and code not in plan.source_text]
    rng.shuffle(codes)
    for code in codes:
        changed = realize(plan, code)
        edited = tokenizer.encode(changed.text, add_special_tokens=False)
        s2 = _token_index(edited, changed.text, changed.previous_span)
        t2 = _token_index(edited, changed.text, changed.answer_span)
        if s2 != source or t2 != target or edited.ids[s2] != edited.ids[t2]:
            continue
        if edited.ids[target] == base.ids[target]:
            continue
        if base.offsets[:target+1] != edited.offsets[:target+1]:
            continue
        expected = set()
        valid = True
        for span in plan.spans[:plan.target_occurrence]:
            token = _token_index(base, plan.source_text, span)
            other = _token_index(edited, changed.text, span)
            if token is None or token != other:
                valid = False
                break
            if start <= token < target:
                expected.add(token-start)
        if not valid or source-start not in expected:
            continue
        natural_x, natural_y = _chunk(base, start, n_ctx)
        edit_x, edit_y = _chunk(edited, start, n_ctx)
        actual = set((natural_x[:target-start] !=
                      edit_x[:target-start]).nonzero().flatten().tolist())
        if actual != expected:
            continue
        pos = target-start-1
        assert natural_y[pos].item() == base.ids[target]
        assert edit_y[pos].item() == edited.ids[target]
        return {'kind': 'literal_copy', 'source_code': plan.source_code,
                'edit_code': code, 'x_natural': natural_x,
                'y_natural': natural_y, 'x_edit': edit_x, 'y_edit': edit_y,
                'pos_natural': pos, 'pos_edit': pos,
                'token_natural': base.ids[target],
                'token_edit': edited.ids[target],
                'gap_words': plan.gap_words,
                'changed_prefix_tokens': len(expected)}, 'ok'
    return None, 'no_aligned_alternative'


def build(a):
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    source = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                          split='train', streaming=True)
    stream = iter(source)
    counts, records = Counter(), []
    try:
        for offset, row in enumerate(islice(stream, a.start, a.start+a.n),
                                     a.start):
            counts['source_documents'] += 1
            plans = list(candidates(row['text'][:a.max_chars]))
            counts['documents_with_candidate'] += bool(plans)
            for plan in plans:
                pair, status = make_pair(plan, tokenizer,
                                         random.Random(83000+offset),
                                         a.n_ctx, a.max_tokens)
                counts[status] += 1
                if pair is not None:
                    pair['offset'] = offset
                    records.append(pair)
                    break  # One independently sampled source document per row.
            if counts['source_documents'] % 1000 == 0:
                print('scanned', counts['source_documents'], flush=True)
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    assert counts['source_documents'] == a.n
    counts['paired_documents'] = len(records)
    a.cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'records': records, 'counts': dict(counts),
                'start': a.start, 'n': a.n, 'n_ctx': a.n_ctx}, a.cache)
    print('paired', len(records), flush=True)
    sys.stdout.flush()
    os._exit(0)  # HF streaming shutdown can otherwise abort after saving.


@torch.no_grad()
def evaluate(a):
    cache = torch.load(a.cache, map_location='cpu', weights_only=True)
    records = cache['records']
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    report = {'config': {'start': cache['start'], 'n': cache['n'],
                         'n_ctx': cache['n_ctx']},
              'counts': cache['counts'],
              'site_summary': {'distinct_source_codes': len(set(
                  row['source_code'] for row in records)),
                               'distinct_edit_codes': len(set(
                  row['edit_code'] for row in records)),
                               'mean_gap_words': sum(row['gap_words']
                                                     for row in records)/len(records)},
              'models': {},
              'scope': 'Fresh unedited FineWeb reference documents with a '
                       'consistent one-token acronym substitution in the '
                       'paired diagnostic arm. All and only prior same-code '
                       'tokens within the model prefix change. The later '
                       'answer is scored under both worlds; the edited '
                       'world need not be semantically natural. Selection '
                       'uses the actual later repetition and so is not an '
                       'unconditioned next-token accuracy estimate. A '
                       'source-independent static code prior has zero '
                       'strict paired two-way accuracy by construction. '
                       'Cache token IDs reconstruct prose; keep local.'}
    for path in a.checkpoints:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        model = _model(saved['vocab_size'], saved['n_ctx'], saved['depth'],
                       saved['d_model'], saved['heads'], a.device,
                       saved['norm'])
        model.load_state_dict(saved['state_dict'])
        model.eval()
        score = one_kind(model, records, a.device, a.batch_size, tokenizer)
        report['models'][path.stem] = score
        print(path.stem, score, flush=True)
        del model
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print('saved', a.output, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('build', 'evaluate'), required=True)
    p.add_argument('--start', type=int, default=383000)
    p.add_argument('--n', type=int, default=5000)
    p.add_argument('--max-chars', type=int, default=8000)
    p.add_argument('--n-ctx', type=int, default=256)
    p.add_argument('--max-tokens', type=int, default=2048)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--checkpoints', type=Path, nargs='*', default=[])
    p.add_argument('--output', type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(4)
    if a.mode == 'build':
        build(a)
    else:
        assert a.checkpoints and a.output
        evaluate(a)


if __name__ == '__main__':
    main()
