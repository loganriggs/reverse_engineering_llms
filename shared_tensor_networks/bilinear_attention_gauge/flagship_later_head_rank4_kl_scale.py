"""Paired held-out natural-text KL for frozen later-head rank-four cuts."""

import json

import numpy as np
import torch
from torch.nn import functional as F

from flagship_later_head_gauge_capture import TARGETS
from flagship_later_head_rank_pilot import HeadCut, contexts, final_logits
from flagship_full_model_eval import GPT, GPTConfig
from flagship_matrix_free import CORPUS, SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_transfer import ART


SEED = 20261116
PANELS = ((32, 128), (64, 64))
METHODS = ('weight', 'corpus', 'task_delta', 'mixed', 'random')


def run():
    torch.set_num_threads(4)
    rng = np.random.default_rng(SEED)
    config = json.loads((SNAPSHOT / 'config.json').read_text())
    config.pop('step', None)
    model = GPT(GPTConfig(**config)).eval()
    state = torch.load(SNAPSHOT / 'pytorch_model.bin', map_location='cpu',
                       weights_only=True, mmap=True)
    model.load_state_dict(state)
    del state
    model.requires_grad_(False)
    probes = {layer: HeadCut(model.transformer.h[layer].attn)
              for layer in sorted({layer for layer, _ in TARGETS.values()})}
    bases = np.load(ART / 'later_head_rank_pilot_bases.npz')
    cases = []
    for length, count in PANELS:
        for corpus_name, path, docs in [('fineweb_test', CORPUS, range(17500, 20000))] + [
            (name, path, range(0, 1200)) for name, path in DATASETS.items()]:
            x = contexts(path, rng, count, docs, length)
            assert len(x) == count
            cases.append(dict(corpus=corpus_name, length=length, inputs=x))
    all_records = []
    with torch.no_grad():
        for case in cases:
            x = case.pop('inputs')
            native = torch.cat([final_logits(model, batch) for batch in x.split(8)])
            log_native = F.log_softmax(native.double(), dim=-1)
            prob_native = log_native.exp()
            original_argmax = native.argmax(-1)
            for target, (layer, head) in TARGETS.items():
                probe = probes[layer]
                probe.head = head
                for method in METHODS:
                    raw = bases[f'{target}__{method}'][:, :4]
                    assert np.linalg.norm(raw.T @ raw - np.eye(4)) < 1e-12
                    probe.basis = torch.tensor(raw.copy(), dtype=torch.float32)
                    candidate = torch.cat([final_logits(model, batch) for batch in x.split(8)])
                    probe.basis = None
                    log_candidate = F.log_softmax(candidate.double(), dim=-1)
                    kl = (prob_native * (log_native - log_candidate)).sum(-1).numpy()
                    assert np.isfinite(kl).all() and (kl > -1e-8).all()
                    flips = (candidate.argmax(-1) != original_argmax).numpy()
                    all_records.append(dict(corpus=case['corpus'], length=case['length'],
                        count=len(x), target=target, layer=layer, head=head, rank=4,
                        method=method, kl=kl.tolist(), top1_flips=flips.astype(int).tolist(),
                        mean_kl=float(kl.mean()), median_kl=float(np.median(kl)),
                        p90_kl=float(np.quantile(kl, .9)), flip_count=int(flips.sum())))
                    print(case['corpus'],case['length'],target,method,
                          'KL',round(float(kl.mean()),7),'flips',int(flips.sum()),flush=True)
    result = dict(description='Independent held-out natural-text KL and top-one flips for frozen rank-four final-position head cuts, paired by context across methods; old fit bases, new seeded context sample',
        seed=SEED, panels=[dict(length=l,count=n) for l,n in PANELS],
        measures=METHODS, targets={k:list(v) for k,v in TARGETS.items()},
        records=all_records)
    (ART / 'later_head_rank4_kl_scale.json').write_text(json.dumps(result,indent=2) + '\n')


if __name__ == '__main__':
    run()
