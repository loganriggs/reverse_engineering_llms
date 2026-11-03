"""Weight-only versus data-informed local bond bases on learned V8 source copy."""

import argparse
import json
import random
from pathlib import Path

import torch

from eval_document_world_v8_paired import collect, load_model
from eval_document_world_v5_clause_bond_bases import (
    project_hook, query_vectors, unlabeled_gram)
from eval_document_world_v5_clause_causal import score
from train_document_world_v5_clause import encode, evaluate
from train_document_world_v8_discourse import sample_documents


def eigenbasis(gram):
    _, vectors = torch.linalg.eigh(gram.double())
    return vectors.flip(1).float().contiguous()


def make_bases(weight, no_source, all_data, contrast, seed):
    d = weight.shape[1]
    weight_gram = weight.double().T @ weight.double()
    normalized_weight = weight_gram / weight_gram.trace()
    normalized_data = no_source / no_source.trace()
    grams = dict(weight_only=weight_gram,
                 data_no_source_sites=no_source,
                 data_all_sites=all_data,
                 source_edit_supervised=contrast,
                 hybrid_weight_data=normalized_weight + normalized_data)
    bases = {name: eigenbasis(gram) for name, gram in grams.items()}
    bases['physical'] = torch.eye(d, device=weight.device)
    generator = torch.Generator(device=weight.device).manual_seed(seed)
    bases['random'] = torch.linalg.qr(torch.randn(
        (d, d), device=weight.device, generator=generator)).Q
    for name, basis in bases.items():
        assert (basis.T @ basis - torch.eye(d, device=weight.device)
                ).abs().max() < 1e-4, name
    return bases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--fit-streams', type=int, default=2000)
    parser.add_argument('--eval-streams', type=int, default=700)
    parser.add_argument('--fit-seed', type=int, default=351001)
    parser.add_argument('--eval-seed', type=int, default=352001)
    parser.add_argument('--eval-documents', type=int, default=200)
    parser.add_argument('--max-cases', type=int, default=128)
    parser.add_argument('--ranks', default='8,16,32,64')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, saved = load_model(args.checkpoint, args.device)
    assert saved['depth'] == 2 and args.layer in (0, 2)
    vocab, n_ctx = saved['vocabulary'], saved['n_ctx']
    bond = model.layers[args.layer].o
    native_kwargs = dict(saved['generator_kwargs'])
    assert native_kwargs['extra_source_recap_rate'] > 0
    assert native_kwargs['source_delay_rate'] == 0

    # Fit unlabeled activations on documents with *no scored source site*.
    # The reports remain present in prose; only source-query sites are held out.
    reference_kwargs = dict(native_kwargs, extra_source_recap_rate=0.)
    fit_docs = [doc for doc in sample_documents(
        args.fit_streams, args.fit_seed, **reference_kwargs)
        if doc.genre != 'summary' and
        not any(site.family == 'source_attribution' for site in doc.sites)
        and len(doc.tokens)-1 <= n_ctx]
    all_docs = [doc for doc in sample_documents(
        args.fit_streams, args.fit_seed+1, **reference_kwargs)
        if doc.genre != 'summary' and len(doc.tokens)-1 <= n_ctx]
    assert len(fit_docs) >= 100 and len(all_docs) >= 100
    no_source_gram, n_vectors = unlabeled_gram(
        model, bond, fit_docs, vocab, n_ctx, args.device, args.batch_size)
    all_gram, n_all_vectors = unlabeled_gram(
        model, bond, all_docs, vocab, n_ctx, args.device, args.batch_size)

    _, fit_cases, _ = collect(args.fit_streams, args.fit_seed+1000,
                              native_kwargs, n_ctx, 10000)
    source_fit = [row for row in fit_cases['source_immediate']
                  if row[0].genre != 'summary' and
                  len(row[0].tokens)-1 <= n_ctx and
                  len(row[2].tokens)-1 <= n_ctx][:256]
    assert len(source_fit) >= 128
    original = query_vectors(model, bond,
                             [(row[0], row[1]) for row in source_fit],
                             vocab, n_ctx, args.device, args.batch_size)
    edited = query_vectors(model, bond,
                           [(row[2], row[3]) for row in source_fit],
                           vocab, n_ctx, args.device, args.batch_size)
    delta = edited - original
    bases = make_bases(bond.weight.detach(), no_source_gram, all_gram,
                       delta.T @ delta, args.fit_seed+4000)
    ranks = sorted({int(value) for value in args.ranks.split(',')})
    assert ranks[-1] == bond.in_features and ranks[0] >= 1

    results = {}
    for syntax in (0., 1.):
        eval_kwargs = dict(native_kwargs, syntax_entropy=syntax)
        docs, cases, skipped = collect(args.eval_streams,
                                       args.eval_seed, eval_kwargs,
                                       n_ctx, 10000)
        test_docs = [doc for doc in docs if doc.genre == 'summary' and
                     len(doc.tokens)-1 <= n_ctx][:args.eval_documents]
        rng = random.Random(args.eval_seed+4000)
        selected = {}
        for family in ('source_immediate', 'source_nuisance_immediate'):
            rows = [row for row in cases[family]
                    if row[0].genre == 'summary' and
                    len(row[0].tokens)-1 <= n_ctx and
                    len(row[2].tokens)-1 <= n_ctx]
            rng.shuffle(rows)
            selected[family] = rows[:args.max_cases]
            assert len(selected[family]) >= 32, family
        data = encode(test_docs, vocab, n_ctx)
        native_eval = evaluate(model, test_docs, data, vocab, args.device)
        native_edits = score(model, selected, vocab, n_ctx,
                             args.device, args.batch_size)
        rows = []
        for name, basis in bases.items():
            for rank in ranks:
                hook = bond.register_forward_pre_hook(project_hook(basis, rank))
                try:
                    measured = evaluate(model, test_docs, data,
                                        vocab, args.device)
                    edits = score(model, selected, vocab, n_ctx,
                                  args.device, args.batch_size)
                finally:
                    hook.remove()
                rows.append(dict(basis=name, rank=rank,
                                 token_nll=measured['token_nll'],
                                 source=measured['by_site']['source_attribution'],
                                 source_edit=edits['source_immediate'],
                                 nuisance_edit=edits[
                                     'source_nuisance_immediate']))
                if rank == bond.in_features:
                    assert abs(measured['token_nll'] -
                               native_eval['token_nll']) < 1e-5
                    assert abs(edits['source_immediate'][
                        'mean_model_response'] - native_edits[
                        'source_immediate']['mean_model_response']) < 1e-5
                print('syntax', syntax, name, rank,
                      round(measured['token_nll'], 3),
                      round(edits['source_immediate'][
                          'mean_model_response'], 3), flush=True)
        results[f'syntax{syntax:g}'] = dict(
            native=dict(evaluation=native_eval, edits=native_edits),
            n_test_docs=len(test_docs), skipped=skipped,
            n_cases={name: len(value) for name, value in selected.items()},
            rows=rows)
    output = dict(config={**vars(args), 'checkpoint': str(args.checkpoint),
                          'output': str(args.output)},
                  fit=dict(n_no_source_documents=len(fit_docs),
                           n_all_documents=len(all_docs),
                           n_no_source_vectors=n_vectors,
                           n_all_vectors=n_all_vectors,
                           n_source_edit_pairs=len(source_fit),
                           heldout_fit_genre='summary'),
                  results=results,
                  caveat='Local pre-W_O rank projection, not full tensor '
                         'canonicalization. Source questions excluded from '
                         'unlabeled basis fit, but source reports are present; '
                         'summary held out from basis fit only. Supervised '
                         'edit basis is an explicitly labeled upper bound. '
                         'Only immediate source copy is behaviorally learned.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
