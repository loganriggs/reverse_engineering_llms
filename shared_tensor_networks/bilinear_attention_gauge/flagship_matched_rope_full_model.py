"""Propagate matched pre- and post-RoPE rank-96 cuts through the 546M model."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.nn import functional as F

from flagship_full_model_eval import BondEdit, GPT, GPTConfig, logits, metrics
from flagship_matrix_free import CORPUS, HD, SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_prerope_cut_transfer import row_rotation
from flagship_transfer import ART


class PositionAwareBondEdit(BondEdit):
    def __init__(self, model):
        super().__init__(model)
        self.pre_cut = None
        self.pre_matrices = None

    def set_cut(self, geometry, matrix, length):
        self.cut = None
        self.pre_cut = None
        self.pre_matrices = None
        if geometry == "post":
            self.cut = torch.from_numpy(matrix).to(torch.float32)
        elif geometry == "pre":
            self.pre_cut = matrix
            self.pre_matrices = torch.from_numpy(np.stack([
                np.linalg.solve(row_rotation(p), matrix @ row_rotation(p))
                for p in range(length)
            ])).to(torch.float32)
        else:
            raise ValueError(geometry)

    def forward(self, q, k, v, q2, k2):
        if self.pre_cut is None:
            return super().forward(q, k, v, q2, k2)
        native = self.native(q, k, v, q2, k2)
        d = q.shape[-1]
        matrices = self.pre_matrices.to(device=q.device, dtype=q.dtype)
        qx = torch.einsum("btd,tde->bte", q[:, :, 0, :], matrices)
        score1 = qx @ k[:, :, 0, :].transpose(-1, -2) / d
        score2 = (q2[:, :, 0, :] @ k2[:, :, 0, :].transpose(-1, -2)) / d
        result = native.clone()
        result[:, 0] = (score1 * score2).tril() @ v[:, :, 0, :]
        return result


def run(n_docs=8, length=128, seeds=(23, 29), model_seeds=(41, 43, 47),
        batch_size=4, threads=4, corpora=None, factorial=False):
    torch.set_num_threads(threads)
    config = json.loads((SNAPSHOT / "config.json").read_text())
    config.pop("step", None)
    model = GPT(GPTConfig(**config)).eval()
    state = torch.load(SNAPSHOT / "pytorch_model.bin", map_location="cpu",
                       weights_only=True, mmap=True)
    model.load_state_dict(state)
    del state
    editor = PositionAwareBondEdit(model)
    cuts = {}
    for seed in model_seeds:
        path = (ART / f"head0_rope_metric_factorial_seed{seed}.npz" if factorial
                else ART / f"head0_matched_rope_fit_n32768_s0_q128_k8192_seed{seed}.npz")
        arrays = np.load(path)
        priors = ("uniform", "mixed", "corpus") if factorial else ("wtg_joint",)
        for prior in priors:
            for geometry in ("post", "pre"):
                name = (f"seed{seed}/{prior}/{geometry}" if factorial else
                        f"seed{seed}/{geometry}")
                cuts[name] = (geometry, arrays[f"{prior}__{geometry}"])
    datasets = {"fineweb_test": (CORPUS, 17500, 20000),
                **{name: (path, 0, 1200) for name, path in DATASETS.items()}}
    if corpora is not None:
        datasets = {name: datasets[name] for name in corpora}
    records = []
    identity_gates = {}
    with torch.no_grad():
        for corpus_name, (path, begin, end) in datasets.items():
            corpus = np.load(path, mmap_mode="r")
            for sample_seed in seeds:
                rng = np.random.default_rng(sample_seed + length * 1000)
                docs = rng.choice(np.arange(begin, end), size=n_docs,
                                  replace=False)
                totals = {name: dict(kl_sum=0., ce_original_sum=0.,
                    ce_edited_sum=0., argmax_flips=0, token_count=0,
                    max_abs_logit=0.) for name in cuts}
                for offset in range(0, n_docs, batch_size):
                    spans = []
                    for doc in docs[offset:offset + batch_size]:
                        start = int(rng.integers(0, corpus.shape[1] - length))
                        spans.append(np.asarray(
                            corpus[doc, start:start + length + 1], dtype=np.int64))
                    batch = np.stack(spans)
                    idx = torch.from_numpy(batch[:, :-1].copy())
                    target = torch.from_numpy(batch[:, 1:].copy())
                    editor.cut = None
                    editor.pre_cut = None
                    reference = logits(model, idx)
                    native_ce = model(idx, target)
                    gate = float(abs(native_ce - F.cross_entropy(
                        reference.reshape(-1, reference.shape[-1]),
                        target.ravel())))
                    assert gate < 1e-5, gate
                    if corpus_name not in identity_gates:
                        editor.set_cut("pre", np.eye(HD), length)
                        identity_gates[corpus_name] = float(
                            (logits(model, idx) - reference).abs().max())
                        assert identity_gates[corpus_name] < 1e-4
                    for name, (geometry, matrix) in cuts.items():
                        editor.set_cut(geometry, matrix, length)
                        candidate = logits(model, idx)
                        row = metrics(reference, candidate, target)
                        for key in ("kl_sum", "ce_original_sum", "ce_edited_sum",
                                    "argmax_flips", "token_count"):
                            totals[name][key] += row[key]
                        totals[name]["max_abs_logit"] = max(
                            totals[name]["max_abs_logit"], row["max_abs_logit"])
                for name, values in totals.items():
                    records.append(dict(corpus=corpus_name,
                        sample_seed=sample_seed, n_docs=n_docs, length=length,
                        cut=name, **values,
                        kl_per_token=values["kl_sum"] / values["token_count"],
                        ce_delta_per_token=(values["ce_edited_sum"] -
                            values["ce_original_sum"]) / values["token_count"],
                        flip_fraction=(values["argmax_flips"] /
                                       values["token_count"])))
                print(corpus_name, sample_seed, flush=True)
    result = dict(layer=0, head=0, rank=96, length=length,
        n_docs_per_seed=n_docs, sample_seeds=list(seeds),
        model_seeds=list(model_seeds), batch_size=batch_size,factorial=factorial,
        identity_pre_cut_max_abs_logit=identity_gates,
        fineweb_test_documents=[17500, 20000],
        scope="full 18-layer model; rank-96 QK1 cut on layer-0 head-0; original-to-edited KL and CE",
        records=records)
    suffix = f"_n{n_docs}_s{'-'.join(map(str, seeds))}"
    if factorial:
        suffix += "_factorial"
    if corpora is not None:
        suffix += "_" + "-".join(corpora)
    (ART / f"head0_matched_rope_full_model{suffix}.json").write_text(
        json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-docs", type=int, default=8)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--seeds", type=int, nargs="+", default=[23, 29])
    p.add_argument("--model-seeds", type=int, nargs="+", default=[41, 43, 47])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--corpora", nargs="+", default=None)
    p.add_argument("--factorial",action="store_true")
    a = p.parse_args()
    run(a.n_docs, a.length, tuple(a.seeds), tuple(a.model_seeds),
        a.batch_size, a.threads,
        None if a.corpora is None else tuple(a.corpora),a.factorial)
