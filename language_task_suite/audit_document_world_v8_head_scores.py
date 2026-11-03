"""Compare a task-blind head weight norm with causal ablation rankings."""

import argparse
import json
from pathlib import Path

import torch

from eval_document_world_v8_paired import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--ablations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    model, _ = load_model(args.checkpoint, 'cpu')
    ablations = json.loads(args.ablations.read_text())['models'][
        str(args.checkpoint)]['head_ablations']
    rows = []
    for layer_index, layer in enumerate(model.layers):
        if not hasattr(layer, 'q1'):
            continue
        for head_index in range(layer.n_head):
            sl = slice(head_index*layer.d_head,
                       (head_index+1)*layer.d_head)
            q1, k1 = layer.q1.weight[sl], layer.k1.weight[sl]
            q2, k2 = layer.q2.weight[sl], layer.k2.weight[sl]
            value, out = layer.v.weight[sl], layer.o.weight[:, sl]
            weight_score = (out.norm() * value.norm() *
                            (q1.T @ k1).norm() *
                            (q2.T @ k2).norm()).item()
            causal = next(row for row in ablations
                          if row['layer'] == layer_index and
                          row['head'] == head_index)
            rows.append(dict(layer=layer_index, head=head_index,
                             task_blind_weight_score=weight_score,
                             output_weight_nonzeros=int((out != 0).sum()),
                             source_response_drop=causal['source_response_drop']))
    ranked = sorted(rows, key=lambda row: -row['task_blind_weight_score'])
    output = dict(checkpoint=str(args.checkpoint),
                  ablations=str(args.ablations), ranked_heads=ranked,
                  caveat='Product of four Frobenius norms is a simple '
                         'task-blind, gauge-sensitive weight heuristic. '
                         'It is not a canonical tensor norm. Causal drops '
                         'use labeled source edits only for evaluation.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print([(row['layer'], row['head'],
            round(row['task_blind_weight_score'], 2),
            round(row['source_response_drop'], 3)) for row in ranked],
          flush=True)


if __name__ == '__main__':
    main()
