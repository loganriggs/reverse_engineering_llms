"""Fit unlabeled rank-four head cuts for output-weighted reconstruction."""

import json

import numpy as np
import torch

from flagship_full_model_eval import GPT, GPTConfig
from flagship_later_head_gauge_capture import TARGETS
from flagship_later_head_rank_pilot import HeadCut, contexts, final_logits
from flagship_matrix_free import CORPUS, SNAPSHOT
from flagship_transfer import ART


SEED = 20261118
RANK = 4
TRAIN_PER_LENGTH = 1024
VALIDATION_PER_LENGTH = 128


def top_eigenvectors(moment, rank):
    eigenvalues, eigenvectors = np.linalg.eigh((moment + moment.T) / 2)
    return eigenvectors[:, -rank:][:, ::-1].copy()


def objective(covariance, gram, basis):
    projector = basis @ basis.T
    missing = torch.eye(basis.shape[0], dtype=basis.dtype) - projector
    return torch.trace(gram @ missing @ covariance @ missing)


def optimize(covariance, gram, initial, steps=350):
    x = torch.nn.Parameter(torch.tensor(initial.copy(), dtype=torch.float64))
    optimizer = torch.optim.Adam([x], lr=.015)
    scale = torch.trace(gram @ covariance).detach().clamp_min(1e-30)
    best_loss = float('inf')
    best_basis = None
    for step in range(steps):
        optimizer.zero_grad()
        basis, _ = torch.linalg.qr(x, mode='reduced')
        loss = objective(covariance, gram, basis) / scale
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite weighted reconstruction loss')
        if float(loss.detach()) < best_loss:
            best_loss = float(loss.detach())
            best_basis = basis.detach().numpy().copy()
        loss.backward()
        optimizer.step()
    return best_basis, best_loss


def run():
    torch.set_num_threads(4)
    rng = np.random.default_rng(SEED)
    config = json.loads((SNAPSHOT / 'config.json').read_text())
    config.pop('step', None)
    model = GPT(GPTConfig(**config)).eval()
    weights = torch.load(SNAPSHOT / 'pytorch_model.bin', map_location='cpu',
                         weights_only=True, mmap=True)
    model.load_state_dict(weights)
    del weights
    model.requires_grad_(False)
    probes = {layer: HeadCut(model.transformer.h[layer].attn)
              for layer in sorted({layer for layer,_ in TARGETS.values()})}
    samples = {'train': {target: [] for target in TARGETS},
               'validation': {target: [] for target in TARGETS}}
    with torch.no_grad():
        for panel, count, doc_range in (
            ('train', TRAIN_PER_LENGTH, range(0, 16000)),
            ('validation', VALIDATION_PER_LENGTH, range(16000, 17500))):
            for length in (32, 64):
                x = contexts(CORPUS, rng, count, doc_range, length)
                for batch in x.split(8):
                    final_logits(model, batch, stop_at=max(probes))
                    for target, (layer, head) in TARGETS.items():
                        p = probes[layer]
                        start = head * p.head_dim
                        samples[panel][target].append(
                            p.last_input[:, -1, start:start + p.head_dim].numpy())
                print(panel, length, count, 'head states captured',flush=True)
    old = np.load(ART / 'later_head_rank_pilot_bases.npz')
    bases = {}
    records = []
    for target, (layer, head) in TARGETS.items():
        train = np.concatenate(samples['train'][target]).astype(np.float64)
        validation = np.concatenate(samples['validation'][target]).astype(np.float64)
        assert len(train) == 2 * TRAIN_PER_LENGTH
        assert len(validation) == 2 * VALIDATION_PER_LENGTH
        train_cov = train.T @ train / len(train)
        val_cov = validation.T @ validation / len(validation)
        p = probes[layer]
        w = p.native.__self__.weight.detach().numpy()[:,
            head*p.head_dim:(head+1)*p.head_dim].astype(np.float64)
        gram = w.T @ w
        train_tensor = torch.tensor(train_cov)
        val_tensor = torch.tensor(val_cov)
        gram_tensor = torch.tensor(gram)
        prior_weight = old[f'{target}__weight'][:, :RANK].copy()
        prior_corpus = old[f'{target}__corpus'][:, :RANK].copy()
        large_corpus = top_eigenvectors(train_cov, RANK)
        candidates = []
        for name, start in (('weight', prior_weight),
                            ('old_corpus', prior_corpus),
                            ('large_corpus', large_corpus)):
            fit, loss = optimize(train_tensor, gram_tensor, start)
            candidates.append((loss, name, fit))
            print(target,'start',name,'fit loss',round(loss,7),flush=True)
        candidates.sort(key=lambda x:x[0])
        options = {'weight': prior_weight, 'corpus_256': prior_corpus,
                   'corpus_2048': large_corpus,
                   'output_aware_2048': candidates[0][2]}
        for name, basis in options.items():
            assert np.linalg.norm(basis.T @ basis - np.eye(RANK)) < 1e-10
            bases[f'{target}__{name}'] = basis
            b = torch.tensor(basis)
            train_loss = float(objective(train_tensor,gram_tensor,b)/
                               torch.trace(gram_tensor@train_tensor))
            val_loss = float(objective(val_tensor,gram_tensor,b)/
                             torch.trace(gram_tensor@val_tensor))
            raw_train = float(objective(train_tensor,torch.eye(128,dtype=torch.float64),b)/
                              torch.trace(train_tensor))
            raw_val = float(objective(val_tensor,torch.eye(128,dtype=torch.float64),b)/
                            torch.trace(val_tensor))
            records.append(dict(target=target,method=name,
                normalized_output_mse_train=train_loss,
                normalized_output_mse_validation=val_loss,
                normalized_head_mse_train=raw_train,
                normalized_head_mse_validation=raw_val))
            print(target,name,'output train/val',round(train_loss,6),
                  round(val_loss,6),'head val',round(raw_val,6),flush=True)
    result=dict(description='Rank-four orthogonal head projectors: old weight and 256-context PCA, 2048-context uncentered PCA, and 2048-context output-weighted orthogonal projector optimized with QR from three starts',
        seed=SEED,rank=RANK,train_contexts_per_length=TRAIN_PER_LENGTH,
        validation_contexts_per_length=VALIDATION_PER_LENGTH,
        lengths=[32,64],fit_fineweb_docs=[0,16000],validation_fineweb_docs=[16000,17500],
        objective='E[||W(I-UU^T)z||^2] / E[||Wz||^2]',
        output_aware_starts=['weight','old_corpus','large_corpus'],
        records=records)
    np.savez_compressed(ART/'output_aware_head_fit_bases.npz',**bases)
    (ART/'output_aware_head_fit.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':run()
