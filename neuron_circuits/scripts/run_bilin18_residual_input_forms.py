#!/usr/bin/env python3
"""Contract frozen residual output directions into executable scalar quadratics."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_qkov_causal_edges import attention_details, load_full_model

ROOT = Path(__file__).resolve().parents[1]
STEM = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_residual_input_forms"


def quadratic(x, q):
    return ((x @ q) * x).sum(-1)


def contract(left, right, down, direction):
    q = left.T @ ((direction @ down)[:, None] * right)
    return (q + q.T) / 2


def score(target, prediction, train_mean):
    return float(1 - (target - prediction).square().sum() / (target - train_mean).square().sum().clamp_min(1e-30))


def sorted_eigh(q):
    values, vectors = torch.linalg.eigh(q)
    order = values.abs().argsort(descending=True)
    return values[order], vectors[:, order]


def fit_forms(train, test, q, ranks, floor_fraction):
    """Keep exact constant/linear response; approximate only centered quadratic."""
    mean = train.mean(0)
    d, v = train - mean, test - mean
    yd, yv = quadratic(train, q), quadratic(test, q)
    qd, qv = quadratic(d, q), quadratic(v, q)
    linear_d = 2 * d @ q @ mean
    linear_v = 2 * v @ q @ mean
    # Mean of omitted quadratic terms is estimated ONLY on discovery.
    baseline = yd.mean() + linear_v - linear_d.mean()
    cov = d.T @ d / len(d)
    ev, eu = torch.linalg.eigh(cov)
    ev = ev.clamp_min(ev.max() * floor_fraction)
    square = (eu * ev.sqrt()) @ eu.T
    inverse = (eu * ev.rsqrt()) @ eu.T
    output = {"linear_r2": score(yv, baseline, yd.mean()), "curves": {}}
    for name, form, dc, vc in (("raw", q, d, v), ("covariance", square @ q @ square, d @ inverse, v @ inverse)):
        values, vectors = sorted_eigh(form)
        dt, vt = dc @ vectors, vc @ vectors
        cells = {}
        for rank in ranks:
            pd = (dt[:, :rank].square() * values[:rank]).sum(-1)
            pv = (vt[:, :rank].square() * values[:rank]).sum(-1)
            pred = baseline + pv - pd.mean()
            cells[str(rank)] = {
                "r2": score(yv, pred, yd.mean()),
                "quadratic_variation_r2": score(qv, pv + qd.mean() - pd.mean(), qd.mean()),
                "spectral_energy": float(values[:rank].square().sum() / values.square().sum())
            }
        output["curves"][name] = cells
    return output


@torch.inference_mode()
def collect(model, corpus, start, config):
    result = {layer: {"inputs": [], "writes": []} for layer in config["layers"]}
    dim = model.config.n_embd
    for offset in range(0, config["rows_per_panel"], config["batch_size"]):
        raw = np.array(corpus[start + offset:start + offset + config["batch_size"], :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        x0 = F.rms_norm(model.embedding(ids), (dim,)); x = x0; bus = None
        for layer, block in enumerate(model.blocks):
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            details = attention_details(block.attn, F.rms_norm(mixed, (dim,)), bus, model.config, model.tt)
            bus = details["value_bus"]
            pre = mixed + details["write"]
            normalized = F.rms_norm(pre, (dim,))
            write = block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized))
            if layer in result:
                for key, value in (("inputs", normalized), ("writes", write)):
                    result[layer][key].append(value[:, config["positions"]].flatten(0, 1).cpu())
            x = pre + write + block.mlp.Down_bias
        if (offset + config["batch_size"]) % 64 == 0:
            print(f"panel {start}: {offset + config['batch_size']} rows", flush=True)
    return {layer: {key: torch.cat(value).double() for key, value in rows.items()} for layer, rows in result.items()}


@torch.inference_mode()
def main():
    torch.set_num_threads(6)
    config = json.loads((ROOT / "configs/bilin18_residual_input_forms.json").read_text())
    torch.manual_seed(config["seed"])
    axes = np.load(ROOT / config["source_axes"])
    corpus = np.load(ROOT / config["corpus"], mmap_mode="r")
    model = load_full_model("cuda")
    assert not model.config.gated
    model.embedding.float(); model.blocks.float()
    discovery = collect(model, corpus, config["discovery_start"], config)
    validation = collect(model, corpus, config["validation_start"], config)
    cache = {}
    results = {}
    for layer in config["layers"]:
        block = model.blocks[layer]
        left, right, down = [m.weight.detach().cpu().double() for m in (block.mlp.Left, block.mlp.Right, block.mlp.Down)]
        directions = torch.from_numpy(axes[f"residual_basis_L{layer}"][:config["axes_per_layer"]]).double()
        random = torch.linalg.qr(torch.randn(len(down), config["random_axes"], dtype=torch.float64)).Q.T
        rows = []
        for kind, basis in (("residual", directions), ("random", random)):
            for index, direction in enumerate(basis):
                q = contract(left, right, down, direction)
                train, test = discovery[layer]["inputs"], validation[layer]["inputs"]
                native = validation[layer]["writes"] @ direction
                exact = quadratic(test, q)
                relative_error = float((native - exact).square().sum().sqrt() / native.square().sum().sqrt().clamp_min(1e-30))
                row = {"kind": kind, "axis": index, "relative_rms_error": relative_error, **fit_forms(train, test, q, config["ranks"], config["covariance_floor_relative_max"])}
                rows.append(row)
                print(f"L{layer} {kind}{index}: linear={row['linear_r2']:.3f} raw16={row['curves']['raw']['16']['r2']:.3f} cov16={row['curves']['covariance']['16']['r2']:.3f}", flush=True)
        results[str(layer)] = rows
        for panel, values in (("discovery", discovery), ("validation", validation)):
            for key, value in values[layer].items(): cache[f"{panel}_{key}_L{layer}"] = value.float().numpy()
        cache[f"directions_L{layer}"] = directions.numpy()
    median = lambda rows, key: float(np.median([key(row) for row in rows]))
    compact = {}; nonlinear = {}; specific = {}
    for layer in (1, 8, 12):
        residual = [r for r in results[str(layer)] if r["kind"] == "residual"]
        random = [r for r in results[str(layer)] if r["kind"] == "random"]
        compact[str(layer)] = sum(r["curves"]["covariance"]["16"]["r2"] >= .9 for r in residual)
        nonlinear[str(layer)] = median(residual, lambda r: r["curves"]["covariance"]["16"]["r2"] - r["linear_r2"])
        specific[str(layer)] = median(residual, lambda r: r["curves"]["covariance"]["16"]["r2"]) - median(random, lambda r: r["curves"]["covariance"]["16"]["r2"])
    gates = {"exact": all(r["relative_rms_error"] < 1e-4 for rows in results.values() for r in rows), "compact": all(n >= 3 for n in compact.values()), "nonlinear": sum(v >= .1 for v in nonlinear.values()) >= 2, "specific": sum(v >= .1 for v in specific.values()) >= 2}
    payload = {"config": config, "gates": gates, "compact_counts": compact, "nonlinear_gains": nonlinear, "specificity_gains": specific, "results": results}
    STEM.with_suffix(".json").write_text(json.dumps(payload, indent=2, allow_nan=False))
    np.savez_compressed(STEM.with_suffix(".npz"), **cache)
    lines = ["# Residual-direction input quadratic pilot", "", config["scope"], "", "A scalar write is exactly xᵀQx for normalized pre-MLP x. Both spectra preserve the exact mean-state linear term and restore the discovery mean of omitted quadratic terms. Covariance coordinates use discovery inputs only. R2 is centered using the discovery scalar mean; it is not raw mean-dominated fidelity.", "", f"Locked gates: `{gates}`", "", "| Layer | Kind/axis | Linear R2 | Raw16 R2 | Cov16 R2 | Cov64 R2 | Cov16 quadratic-only R2 |", "|---|---|---:|---:|---:|---:|---:|"]
    for layer, rows in results.items():
        for r in rows:
            raw, cov = r["curves"]["raw"], r["curves"]["covariance"]
            lines.append(f"| {layer} | {r['kind']}/{r['axis']} | {r['linear_r2']:.3f} | {raw['16']['r2']:.3f} | {cov['16']['r2']:.3f} | {cov['64']['r2']:.3f} | {cov['16']['quadratic_variation_r2']:.3f} |")
    lines += ["", "This pilot does not establish semantic meaning, necessity, finite suffix fidelity, or compact upstream generation of the input coordinates. Four random axes are a descriptive control, not a calibrated tail-probability test. The linear term is a dense input covector and is explicitly charged as one additional read direction."]
    STEM.with_name("BILIN18_RESIDUAL_INPUT_FORMS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(gates), flush=True)


if __name__ == "__main__":
    main()
