"""Behavioral rule gate for V8 models across surface and delay shifts."""

import argparse
import json
from pathlib import Path

import torch

from audit_document_world_v5_clause_realizations import _edited_world
from document_world_v8_discourse import answer_from_visible_prefix, realize
from eval_document_world_v5_clause_causal import (
    _nuisance_source_world, _source_is_delayed, score)
from train_document_world_v5_clause import DeepModel, SPECS, encode, evaluate
from train_document_world_v8_discourse import sample_documents


def collect(n, seed, kwargs, n_ctx, max_cases):
    from collections import defaultdict
    import random

    docs = sample_documents(n, seed, **kwargs)
    cases = defaultdict(list)
    skipped = defaultdict(int)
    for doc in docs:
        for index, (site, plan) in enumerate(zip(
                doc.sites, doc.latent['recap_plan'], strict=True)):
            changed = realize(_edited_world(doc.latent, plan),
                              **{key: value for key, value in kwargs.items()
                                 if key not in ('source_delay_rate',
                                                'extra_source_recap_rate')})
            altered = changed.sites[index]
            assert altered.family == site.family
            assert altered.answer != site.answer
            assert answer_from_visible_prefix(doc.tokens[:site.index]) == site.answer
            assert answer_from_visible_prefix(
                changed.tokens[:altered.index]) == altered.answer
            if site.index > n_ctx or altered.index > n_ctx:
                skipped[site.family + '_length'] += 1
                continue
            row = (doc, site, changed, altered, altered.answer)
            cases[site.family].append(row)
            if site.family == 'source_attribution':
                suffix = ('delayed' if _source_is_delayed(doc.latent, plan)
                          else 'immediate')
                cases['source_' + suffix].append(row)
                nuisance = _nuisance_source_world(doc.latent, plan)
                if nuisance is not None:
                    other = realize(nuisance,
                                    **{key: value for key, value in kwargs.items()
                                       if key not in ('source_delay_rate',
                                                      'extra_source_recap_rate')})
                    other_site = other.sites[index]
                    assert other_site.answer == site.answer
                    assert answer_from_visible_prefix(
                        other.tokens[:other_site.index]) == site.answer
                    if other_site.index <= n_ctx:
                        nuisance_row = (doc, site, other, other_site,
                                        site.answer)
                        cases['source_nuisance_' + suffix].append(nuisance_row)
    rng = random.Random(seed + 991)
    selected = {}
    for family, rows in cases.items():
        rng.shuffle(rows)
        selected[family] = rows[:max_cases]
    return docs, selected, dict(skipped)


def load_model(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    vocab, n_ctx, depth = (saved[key] for key in
                           ('vocabulary', 'n_ctx', 'depth'))
    args = saved['model_kwargs']
    model = DeepModel(len(vocab), args['d_model'], args['n_head'],
                      SPECS[f'block{depth}'], n_ctx, norm=args['norm'],
                      residual=args['residual'],
                      attention=args['attention']).to(device)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    return model, saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=349991)
    parser.add_argument('--streams', type=int, default=500)
    parser.add_argument('--max-cases', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, saved = load_model(args.checkpoint, args.device)
    vocab, n_ctx = saved['vocabulary'], saved['n_ctx']
    common = dict(saved['generator_kwargs'])
    results = {}
    for syntax in (0., 1.):
        for delay in (0., 1.):
            kwargs = dict(common, syntax_entropy=syntax,
                          source_delay_rate=delay)
            docs, cases, skipped = collect(args.streams, args.seed, kwargs,
                                           n_ctx, args.max_cases)
            unknown = {token for doc in docs for token in doc.tokens} - set(vocab)
            assert not unknown, unknown
            data = encode(docs, vocab, n_ctx)
            results[f'syntax{syntax:g}_delay{delay:g}'] = dict(
                full_ce=evaluate(model, docs, data, vocab, args.device),
                edits=score(model, cases, vocab, n_ctx,
                            args.device, args.batch_size),
                skipped=skipped)
            print(syntax, delay,
                  results[f'syntax{syntax:g}_delay{delay:g}']['full_ce'][
                      'by_site'].get('source_attribution'),
                  flush=True)
    output = dict(config={**vars(args), 'checkpoint': str(args.checkpoint),
                          'output': str(args.output)},
                  train_generator=saved['generator_kwargs'],
                  panels=results,
                  caveat='Rule-specific visible-prefix edits and held-out '
                         'full-CE scores. A correct rule response is a '
                         'behavioral gate, not a recovered weight circuit.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
