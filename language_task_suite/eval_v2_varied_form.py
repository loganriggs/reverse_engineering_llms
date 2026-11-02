"""Held-out oracle KL at determiners and parenthetical closure sites."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from surface_mixture_v2_varied import SurfaceMixtureV2Varied


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', default='language_task_suite/surface_mixture_v2_varied_soft_models')
    p.add_argument('--n', type=int, default=1024)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--seed', type=int, default=20261122)
    p.add_argument('--parenthetical-mass', type=float, default=None)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    saved = json.loads((Path(a.model_dir)/'results.json').read_text())
    assert saved['config']['generator'] == 'v2_varied'
    kwargs = dict(saved['generator_kwargs'])
    if a.parenthetical_mass is not None:
        kwargs['parenthetical_mass'] = a.parenthetical_mass
    gen = SurfaceMixtureV2Varied(seed=a.seed, **kwargs)
    ids, labels, qs = gen.sample_batch(a.n, a.length+1)
    x = torch.tensor([row[:-1] for row in ids], dtype=torch.long)
    q = np.zeros((a.n, a.length, len(gen.vocab)), dtype=np.float32)
    kinds = []
    for b, (row, row_labels, row_qs) in enumerate(zip(ids, labels, qs)):
        kinds.append(row_labels[1:])
        for t in range(1, len(row)):
            for token, mass in row_qs[t].items():
                q[b, t-1, gen.token2id[token]] = mass
    assert np.allclose(q.sum(-1), 1)
    q = torch.from_numpy(q)
    results = {}
    for record in saved['records']:
        model = BilinearGPT(Config(**record['model_config']))
        model.load_state_dict(torch.load(
            Path(a.model_dir)/f"model_depth{record['depth']}.pt",
            map_location='cpu', weights_only=True))
        model.eval()
        logits = torch.cat([model(batch)[0].float()
                            for batch in x.split(32)])
        logp = F.log_softmax(logits, -1)
        entropy = -(q*q.clamp_min(1e-30).log()).sum(-1)
        kl = -(q*logp).sum(-1)-entropy
        by_kind = {}
        for kind in ('mixed_verb', 'determiner', 'parenthesis_close', 'frame'):
            mask = torch.tensor([[label == kind for label in row]
                                 for row in kinds], dtype=torch.bool)
            by_kind[kind] = dict(n=int(mask.sum()),
                                 kl=float(kl[mask].mean()) if mask.any() else None,
                                 oracle_entropy=float(entropy[mask].mean())
                                 if mask.any() else None)
        results[str(record['depth'])] = by_kind
    output = dict(config={**vars(a), 'output': str(a.output)},
                  effective_generator_kwargs=kwargs,
                  label_counts=dict(Counter(label for row in kinds
                                            for label in row)),
                  results=results,
                  caveat='Sitewise KL compares the model with the exact '
                         'visible-prefix oracle. Parenthesis-close targets '
                         'include a 0.6/0.4 close/continue choice and a '
                         'deterministic final close; low KL does not prove '
                         'a general nested-bracket algorithm.')
    a.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
