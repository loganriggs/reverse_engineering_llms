"""Test whether a model follows a supported title-topic genre cue edit."""

import argparse
import json
from pathlib import Path

import torch

from genre_document_v3 import GENRES, TOKEN2ID, next_distribution, sample_sequence
from train_joint_document_v1 import DeepModel, SPECS


def _vector(q, device):
    out = torch.zeros(len(TOKEN2ID), device=device)
    for word, p in q.items():
        out[TOKEN2ID[word]] = p
    return out


@torch.no_grad()
def run(checkpoint, n, seed, device):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    depth, cfg = saved['depth'], saved['model_kwargs']
    assert saved['vocabulary'] == TOKEN2ID
    model = DeepModel(len(TOKEN2ID), cfg['d_model'], cfg['n_head'],
                      SPECS[f'block{depth}'], saved['n_ctx'],
                      norm=cfg['norm'], residual=cfg['residual'],
                      attention=cfg['attention']).to(device).eval()
    model.load_state_dict(saved['state_dict'])
    rows = []
    for i in range(n):
        words, _, _, genre = sample_sequence(seed+i, 4)
        assert words[-1] == '.'
        edited = list(words)
        edited[2] = 'arithmetic' if words[2] == 'evidence' else 'evidence'
        assert next_distribution(words[:2]).get(edited[2], 0.) > 0
        q0 = _vector(next_distribution(words), device)
        q1 = _vector(next_distribution(edited), device)
        x = torch.tensor([[TOKEN2ID[w] for w in words],
                          [TOKEN2ID[w] for w in edited]], device=device)
        p0, p1 = model(x)[:, -1].softmax(dim=-1)
        rows.append(dict(genre=genre,
                         oracle_tv=.5*(q0-q1).abs().sum().item(),
                         model_tv=.5*(p0-p1).abs().sum().item(),
                         effect_l1=((p1-p0)-(q1-q0)).abs().sum().item(),
                         original_oracle_kl=(q0*(q0.clamp_min(1e-30).log()-
                                                  p0.clamp_min(1e-30).log())).sum().item()))
    keys = ('oracle_tv', 'model_tv', 'effect_l1', 'original_oracle_kl')
    means = {key: sum(row[key] for row in rows)/len(rows) for key in keys}
    return dict(checkpoint=str(checkpoint), depth=depth, n=n, seed=seed,
                means=means,
                by_hidden_genre={g: dict(n=sum(row['genre'] == g for row in rows),
                                         means={key: sum(row[key] for row in rows
                                                         if row['genre'] == g)/
                                                sum(row['genre'] == g for row in rows)
                                                for key in keys})
                                 for g in GENRES},
                caveat='Supported single-topic edit at the first sentence '
                       'boundary; this tests genre cue use, not source '
                       'binding or later document-state inference.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--n', type=int, default=500)
    p.add_argument('--seed', type=int, default=49001)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = run(args.checkpoint, args.n, args.seed, args.device)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['means'], flush=True)
