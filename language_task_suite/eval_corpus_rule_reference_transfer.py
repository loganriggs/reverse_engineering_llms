"""Fresh-source, ordinary-chunk paired list/bracket transfer test.

Build cache contains reconstructable token IDs: keep it local. JSON is aggregate.
"""

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
import torch.nn.functional as F
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from corpus_rule_reference_dgp import chunk_document, sample_document
from train_corpus_delimiter_bilinear import _model


KINDS = ('list', 'bracket')


def pair_for_text(text, tokenizer, offset, kind, n_ctx, max_tokens):
    seed = 81000 + offset
    kwargs = dict(list_rate=int(kind == 'list'),
                  bracket_rate=int(kind == 'bracket'), nonce_rate=0.,
                  max_chars=8000)
    natural = sample_document(text, random.Random(seed), (),
                              list_shift_rate=0., square_rate=0., **kwargs)
    edited = sample_document(text, random.Random(seed), (),
                             list_shift_rate=float(kind == 'list'),
                             square_rate=float(kind == 'bracket'), **kwargs)
    if kind not in natural.applied:
        return None, 'ineligible'
    assert kind in edited.applied and len(natural.sites) == len(edited.sites) == 1
    assert natural.sites[0].source == edited.sites[0].source
    assert natural.sites[0].target == edited.sites[0].target
    if natural.sites[0].answer == edited.sites[0].answer:
        return None, 'same_answer'
    a = chunk_document(natural, tokenizer, n_ctx, max_tokens)
    b = chunk_document(edited, tokenizer, n_ctx, max_tokens)
    candidates = []
    for left, right in zip(a, b):
        ls = [site for site in left['sites'] if site['kind'] == kind]
        rs = [site for site in right['sites'] if site['kind'] == kind]
        if ls and rs and left['start_token'] == right['start_token']:
            candidates.append((left, right, ls[0], rs[0]))
    if not candidates:
        return None, 'outside_local_chunk_or_bpe'
    left, right, ls, rs = candidates[0]
    if len(ls['target_ids']) != 1 or len(rs['target_ids']) != 1:
        return None, 'multitoken_answer'
    original, changed = ls['target_ids'][0], rs['target_ids'][0]
    if original == changed:
        return None, 'same_token_answer'
    return {'kind': kind, 'offset': offset,
            'x_natural': left['x'], 'y_natural': left['y'],
            'x_edit': right['x'], 'y_edit': right['y'],
            'pos_natural': ls['positions'][0],
            'pos_edit': rs['positions'][0],
            'token_natural': original, 'token_edit': changed}, 'ok'


def build(a):
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    source = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                          split='train', streaming=True)
    stream = iter(source)
    records, counts = [], Counter()
    try:
        for offset, row in enumerate(islice(stream, a.start, a.start + a.n),
                                     a.start):
            counts['source_documents'] += 1
            for kind in KINDS:
                pair, reason = pair_for_text(row['text'], tokenizer, offset,
                                             kind, a.n_ctx, a.max_tokens)
                counts[f'{kind}_{reason}'] += 1
                if pair is not None:
                    records.append(pair)
            if counts['source_documents'] % 1000 == 0:
                print('scanned', counts['source_documents'], flush=True)
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    assert counts['source_documents'] == a.n
    a.cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'records': records, 'counts': dict(counts),
                'start': a.start, 'n': a.n, 'n_ctx': a.n_ctx,
                'max_tokens': a.max_tokens}, a.cache)
    print('paired', {kind: counts[f'{kind}_ok'] for kind in KINDS}, flush=True)
    sys.stdout.flush()
    os._exit(0)  # HF streaming shutdown can otherwise abort after saving.


def one_kind(model, records, device, batch_size, tokenizer):
    sums = Counter()
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        n = len(batch)
        ix = torch.arange(n, device=device)
        xn = torch.stack([r['x_natural'] for r in batch]).to(device)
        xe = torch.stack([r['x_edit'] for r in batch]).to(device)
        yn = torch.stack([r['y_natural'] for r in batch]).to(device)
        pn = torch.tensor([r['pos_natural'] for r in batch], device=device)
        pe = torch.tensor([r['pos_edit'] for r in batch], device=device)
        tn = torch.tensor([r['token_natural'] for r in batch], device=device)
        te = torch.tensor([r['token_edit'] for r in batch], device=device)
        logits = model(torch.cat((xn, xe))).float()
        ln, le = logits[:n], logits[n:]
        sn, se = ln[ix, pn], le[ix, pe]
        pred_n, pred_e = sn.argmax(-1), se.argmax(-1)
        mn = sn[ix, tn] - sn[ix, te]
        me = se[ix, tn] - se[ix, te]
        sums['n'] += n
        sums['natural_top1'] += (pred_n == tn).sum().item()
        sums['edited_top1'] += (pred_e == te).sum().item()
        sums['paired_top1'] += ((pred_n == tn) & (pred_e == te)).sum().item()
        sums['natural_two_way'] += (mn > 0).sum().item()
        sums['edited_two_way'] += (me < 0).sum().item()
        sums['paired_two_way'] += ((mn > 0) & (me < 0)).sum().item()
        sums['positive_edit_response'] += (mn > me).sum().item()
        sums['sum_edit_response'] += (mn - me).sum().item()
        if batch[0]['kind'] == 'bracket':
            natural_char = [')' in tokenizer.decode([int(token)])
                            for token in pred_n.tolist()]
            edited_char = [']' in tokenizer.decode([int(token)])
                           for token in pred_e.tolist()]
            sums['natural_closer_character'] += sum(natural_char)
            sums['edited_closer_character'] += sum(edited_char)
            sums['paired_closer_character'] += sum(
                left and right for left, right in zip(natural_char, edited_char))
        # Natural background CE is conditional on a selected rule-bearing chunk.
        loss = F.cross_entropy(ln.flatten(0, 1), yn.flatten(),
                               ignore_index=-100, reduction='none').view_as(yn)
        mask = yn != -100
        mask[ix, pn] = False
        sums['background_loss'] += loss[mask].sum().item()
        sums['background_tokens'] += mask.sum().item()
    n = sums['n']
    metrics = ({key: int(sums[key]) for key in ('n', 'background_tokens')} |
            {key + '_fraction': sums[key] / n for key in (
                'natural_top1', 'edited_top1', 'paired_top1',
                'natural_two_way', 'edited_two_way', 'paired_two_way',
                'positive_edit_response')} |
            {'mean_edit_response_logit_margin': sums['sum_edit_response'] / n,
             'selected_chunk_background_ce': sums['background_loss'] /
             sums['background_tokens']})
    if records[0]['kind'] == 'bracket':
        metrics.update({key + '_fraction': sums[key] / n for key in (
            'natural_closer_character', 'edited_closer_character',
            'paired_closer_character')})
    return metrics


@torch.no_grad()
def evaluate(a):
    data = torch.load(a.cache, map_location='cpu', weights_only=True)
    records = data['records']
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    report = {'config': {'start': data['start'], 'n': data['n'],
                         'n_ctx': data['n_ctx'],
                         'max_tokens': data['max_tokens']},
              'counts': data['counts'], 'models': {},
              'scope': 'Uniform fresh source offsets and ordinary '
                       'nonoverlapping chunks. Natural source text paired '
                       'with a same-source controlled shift or bracket-style '
                       'edit. Only BPE-aligned, one-token, context-local '
                       'answers scored. Background CE is restricted to '
                       'selected natural rule-bearing chunks, not all '
                       'source text. No cross-split near-duplicate screen. '
                       'Token cache is local and reconstructable.'}
    for path in a.checkpoints:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        model = _model(50257, data['n_ctx'], 2, 64, 4, a.device, 'none')
        model.load_state_dict(saved['state_dict'])
        model.eval()
        by_kind = {}
        for kind in KINDS:
            chosen = [record for record in records if record['kind'] == kind]
            if chosen:
                by_kind[kind] = one_kind(model, chosen, a.device,
                                         a.batch_size, tokenizer)
                print(path.stem, kind, by_kind[kind], flush=True)
        report['models'][path.stem] = by_kind
        del model
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print('saved', a.output, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('build', 'evaluate'), required=True)
    p.add_argument('--start', type=int, default=363000)
    p.add_argument('--n', type=int, default=5000)
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
        assert a.output and a.checkpoints
        evaluate(a)


if __name__ == '__main__':
    main()
