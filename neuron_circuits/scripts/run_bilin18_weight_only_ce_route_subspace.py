#!/usr/bin/env python3
"""Can symmetry-invariant QK/OV weight geometry predict CE-selected route spaces?"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bilin18_ce_adjoint_residual_atlas import fit_basis, reconstruction_fidelity, subspace_cosine
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_unsupervised_weight_circuit_atlas import rotary_at_distance


CONFIG = Path("configs/bilin18_weight_only_ce_route_subspace.json")
OUTPUT = TENSOR_ROOT / "bilin18_weight_only_ce_route_subspace.json"
ARRAYS = TENSOR_ROOT / "bilin18_weight_only_ce_route_subspace.npz"
REPORT = TENSOR_ROOT / "BILIN18_WEIGHT_ONLY_CE_ROUTE_SUBSPACE.md"
METHODS = ("qk", "ov", "combined", "scaled_combined")


@torch.inference_mode()
def weight_signatures(model, probes, seed, distances):
    device = model.embedding.weight.device; dimension = model.config.n_embd
    heads = model.config.n_head; head_dim = dimension // heads
    generator = torch.Generator(device=device).manual_seed(seed)
    query, key, value, bus = [F.normalize(torch.randn(probes, dimension, generator=generator, device=device), dim=1) * math.sqrt(dimension) for _ in range(4)]
    readers = F.normalize(torch.randn(probes, dimension, generator=generator, device=device), dim=1)
    distance = torch.as_tensor(np.random.default_rng(seed).choice(distances, probes), device=device)
    bus_value = model.blocks[0].attn.c_v(bus).float().reshape(probes, heads, head_dim)
    results = {}
    for layer, block in enumerate(model.blocks):
        attention = block.attn
        def project(module, state):
            return F.rms_norm(module(state).float().reshape(probes, heads, head_dim), (head_dim,))
        q1 = rotary_at_distance(project(attention.c_q, query), distance); q2 = rotary_at_distance(project(attention.c_q2, query), distance)
        k1 = project(attention.c_k, key); k2 = project(attention.c_k2, key)
        route = ((q1 * k1).sum(-1) * (q2 * k2).sum(-1) / (head_dim * head_dim)).T
        current = attention.c_v(value).float().reshape(probes, heads, head_dim)
        mixed = (1 - attention.lamb.float()) * current + attention.lamb.float() * bus_value
        projection = attention.c_proj.weight.float().T.reshape(heads, head_dim, dimension)
        output_reader = torch.einsum("pd,hkd->phk", readers, projection)
        write = (mixed * output_reader).sum(-1).T
        route_norm = route.norm(dim=1); write_norm = write.norm(dim=1)
        qk = F.normalize(route, dim=1); ov = F.normalize(write, dim=1)
        combined = torch.cat((qk, ov), dim=1) / math.sqrt(2)
        scale = (route_norm * write_norm).sqrt()
        scaled = combined * (scale / scale.mean().clamp_min(1e-30))[:, None]
        results[layer] = {"qk": qk.cpu().numpy(), "ov": ov.cpu().numpy(), "combined": combined.cpu().numpy(), "scaled_combined": scaled.cpu().numpy(), "scale": scale.cpu().numpy()}
    return results


def operator_basis(signatures, rank):
    signatures = np.asarray(signatures, dtype=np.float64)
    gram = signatures @ signatures.T
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][:rank]
    return vectors[:, order].T, values[order]


def random_basis(dimension, rank, rng):
    q, _ = np.linalg.qr(rng.standard_normal((dimension, rank)))
    return q.T


def main():
    config = json.loads(CONFIG.read_text()); residual = np.load(config["residual_arrays"]); generic = np.load(config["generic_route_arrays"])
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    first = weight_signatures(model, config["signature_probes"], config["seed_a"], config["relative_distances"])
    second = weight_signatures(model, config["signature_probes"], config["seed_b"], config["relative_distances"])
    rng = np.random.default_rng(config["seed_a"] + config["seed_b"]); rows = []; saved = {}
    for layer in range(model.config.n_layer):
        ce_discovery = residual[f"discovery_residual_routes_L{layer}"]; ce_replication = residual[f"replication_residual_routes_L{layer}"]
        ce_basis = fit_basis(ce_discovery, config["rank"]); generic_basis = fit_basis(generic[f"discovery_routes_L{layer}"], config["rank"])
        ce_fidelity = reconstruction_fidelity(ce_replication, (ce_replication @ ce_basis.T) @ ce_basis)
        generic_fidelity = reconstruction_fidelity(ce_replication, (ce_replication @ generic_basis.T) @ generic_basis)
        methods = {}
        # Four weight views are searched below, so the matched null also takes
        # the best of four independent rank-matched random subspaces.
        null_alignment = np.asarray([
            max(subspace_cosine(ce_basis, random_basis(model.config.n_head, config["rank"], rng)) for _ in METHODS)
            for _ in range(config["null_replicates"])
        ])
        for method in METHODS:
            basis_a, spectrum_a = operator_basis(first[layer][method], config["rank"]); basis_b, spectrum_b = operator_basis(second[layer][method], config["rank"])
            alignment = subspace_cosine(ce_basis, basis_a)
            fidelity = reconstruction_fidelity(ce_replication, (ce_replication @ basis_a.T) @ basis_a)
            methods[method] = {
                "seed_stability": subspace_cosine(basis_a, basis_b), "ce_subspace_cosine": alignment,
                "replication_fidelity": fidelity, "spectrum_a": spectrum_a.tolist(), "spectrum_b": spectrum_b.tolist(),
                "random_percentile": float(np.mean(null_alignment <= alignment)),
            }
            saved[f"{method}_basis_L{layer}"] = basis_a.astype(np.float32)
        best_method = max(METHODS, key=lambda method: methods[method]["ce_subspace_cosine"])
        rows.append({
            "layer": layer, "ce_fidelity": ce_fidelity, "generic_fidelity": generic_fidelity,
            "random_alignment_p95": float(np.quantile(null_alignment, .95)), "methods": methods,
            "best_method": best_method, "best_alignment": methods[best_method]["ce_subspace_cosine"],
            "best_fidelity_method": best_method, "best_weight_fidelity": methods[best_method]["replication_fidelity"],
        })
        print(f"weight CE L{layer} best={best_method} cos={methods[best_method]['ce_subspace_cosine']:.3f} weight/CEfid={methods[best_method]['replication_fidelity']:.3f}/{ce_fidelity:.3f}", flush=True)
    stable = [row["layer"] for row in rows if row["methods"]["scaled_combined"]["seed_stability"] >= .90]
    above_null = [row["layer"] for row in rows if row["best_alignment"] > row["random_alignment_p95"]]
    scale_wins = [row["layer"] for row in rows if row["methods"]["scaled_combined"]["ce_subspace_cosine"] > row["methods"]["combined"]["ce_subspace_cosine"]]
    material = [row["layer"] for row in rows if row["best_weight_fidelity"] >= .5 * row["ce_fidelity"]]
    early = [row["layer"] for row in rows if row["layer"] in config["early_novel_layers"] and row["best_alignment"] >= .80]
    separated = [row["layer"] for row in rows if abs(row["methods"]["qk"]["ce_subspace_cosine"] - row["methods"]["ov"]["ce_subspace_cosine"]) >= .10]
    hypotheses = {
        "W0_weight_bases_are_sketch_stable": {"supported": len(stable) >= 12, "layers": stable},
        "W1_weight_geometry_predicts_ce_subspaces_above_random": {"supported": len(above_null) >= 12, "layers": above_null},
        "W2_scale_information_improves_weight_prediction": {"supported": len(scale_wins) >= 10, "layers": scale_wins},
        "W3_weight_basis_recovers_material_ce_residual_geometry": {"supported": len(material) >= 12, "layers": material},
        "W4_early_ce_novelty_is_visible_in_weights": {"supported": len(early) >= 3, "layers": early},
        "W5_qk_and_ov_roles_are_distinguishable": {"supported": len(separated) >= 8, "layers": separated},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "layers": rows}, indent=2, allow_nan=False)); np.savez_compressed(ARRAYS, **saved)
    lines = ["# Weight-only prediction of CE-selected route subspaces", "", "Each head is evaluated as a symmetry-invariant random function of its two QK factors and OV map. The leading eigenvectors of the nine-head operator Gram matrix define a static weight-only basis in the same coordinate space as the CE-selected routes. Two independent sketches test Monte Carlo stability.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| L | CE/generic fidelity | QK cos/fid | OV cos/fid | Combined cos/fid | Scaled cos/fid/stability | Best vs random p95 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        q, o, c, s = (row["methods"][name] for name in METHODS)
        lines.append(f"| {row['layer']} | {row['ce_fidelity']:.3f}/{row['generic_fidelity']:.3f} | {q['ce_subspace_cosine']:.3f}/{q['replication_fidelity']:.3f} | {o['ce_subspace_cosine']:.3f}/{o['replication_fidelity']:.3f} | {c['ce_subspace_cosine']:.3f}/{c['replication_fidelity']:.3f} | {s['ce_subspace_cosine']:.3f}/{s['replication_fidelity']:.3f}/{s['seed_stability']:.3f} | {row['best_alignment']:.3f}/{row['random_alignment_p95']:.3f} |")
    lines += ["", "## Guardrails", "", "The random-function basis is fully weight-only and symmetry invariant, but its isotropic input distribution need not match FineWeb states or the next-token loss. High canonical cosine would show that static operator geometry constrains the executable CE basis; low cosine would not make the CE basis uninterpretable, but would establish that context/frequency/loss weighting selects a different coordinate system. Choosing the best of four weight views receives an explicit layer-matched random-subspace comparison."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
