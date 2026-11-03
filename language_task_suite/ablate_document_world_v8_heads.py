"""Test immediate-source head necessity in dense and sparse V8 models."""

import argparse
import json
from pathlib import Path

import torch

from eval_document_world_v5_clause_causal import score
from eval_document_world_v8_paired import collect, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--seed', type=int, default=349991)
    parser.add_argument('--streams', type=int, default=500)
    parser.add_argument('--max-cases', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    models = [load_model(path, args.device) for path in args.checkpoints]
    reference = models[0][1]
    assert all(saved['vocabulary'] == reference['vocabulary'] and
               saved['generator_kwargs'] == reference['generator_kwargs'] and
               saved['n_ctx'] == reference['n_ctx'] for _, saved in models)
    kwargs = dict(reference['generator_kwargs'], source_delay_rate=0.)
    _, all_cases, skipped = collect(args.streams, args.seed, kwargs,
                                    reference['n_ctx'], args.max_cases)
    cases = {name: all_cases[name] for name in (
        'source_immediate', 'source_nuisance_immediate')}
    assert all(len(rows) >= 64 for rows in cases.values())
    rows = {}
    for path, (model, saved) in zip(args.checkpoints, models, strict=True):
        native = score(model, cases, reference['vocabulary'],
                       reference['n_ctx'], args.device, args.batch_size)
        heads = []
        for layer_index, layer in enumerate(model.layers):
            if not hasattr(layer, 'o'):
                continue
            for head_index in range(layer.n_head):
                width = layer.d_head
                sl = slice(head_index*width, (head_index+1)*width)
                with torch.no_grad():
                    original = layer.o.weight[:, sl].clone()
                    layer.o.weight[:, sl] = 0.
                try:
                    ablated = score(model, cases, reference['vocabulary'],
                                    reference['n_ctx'], args.device,
                                    args.batch_size)
                finally:
                    with torch.no_grad():
                        layer.o.weight[:, sl] = original
                heads.append(dict(layer=layer_index, head=head_index,
                                  source_response=ablated['source_immediate'][
                                      'mean_model_response'],
                                  source_response_drop=native[
                                      'source_immediate'][
                                      'mean_model_response'] - ablated[
                                      'source_immediate'][
                                      'mean_model_response'],
                                  nuisance_response=ablated[
                                      'source_nuisance_immediate'][
                                      'mean_model_response']))
        rows[str(path)] = dict(native=native, head_ablations=heads)
        print(path, 'native', round(native['source_immediate'][
            'mean_model_response'], 3), 'largest_drops', sorted(
            ((row['layer'], row['head'], round(row['source_response_drop'], 3))
             for row in heads), key=lambda row: -row[2])[:4], flush=True)
    output = dict(config={**vars(args),
                          'checkpoints': [str(path) for path in args.checkpoints],
                          'output': str(args.output)},
                  n_cases={name: len(value) for name, value in cases.items()},
                  skipped=skipped, models=rows,
                  caveat='Zeroing one head output tests its causal importance '
                         'for immediate-source edits, not a complete QK/OV '
                         'path or unique algorithm. Sparse models retain '
                         'dense embedding, unembedding and MLP weights.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
