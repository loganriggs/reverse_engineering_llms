#!/usr/bin/env python3
"""Task-blind census of native bilinear MLP channels across natural strata."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_ce_adjoint_residual_atlas import differentiable_attention_parts
from run_bilin18_layer01_supervised_feature_census import vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_stratified_natural_route_census import token_class_from_flags


CONFIG = Path("configs/bilin18_stratified_natural_mlp_census.json")
OUTPUT = TENSOR_ROOT / "bilin18_stratified_natural_mlp_census.json"
ARRAYS = TENSOR_ROOT / "bilin18_stratified_natural_mlp_census.npz"
REPORT = TENSOR_ROOT / "BILIN18_STRATIFIED_NATURAL_MLP_CENSUS.md"


def forward_with_product_taps(model, ids):
    dimension = model.config.n_embd
    embedded = model.embedding(ids).detach().requires_grad_(True)
    x0 = F.rms_norm(embedded, (dimension,)); x = x0; bus = None; products = []
    for block in model.blocks:
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        pattern, mixed_value, bus = differentiable_attention_parts(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt)
        attended = torch.einsum("bhqk,bkhd->bhqd", pattern, mixed_value).transpose(1, 2).contiguous().view(ids.shape[0], ids.shape[1], dimension)
        pre = mixed + block.attn.c_proj(attended); normalized = F.rms_norm(pre, (dimension,))
        product = block.mlp.Left(normalized) * block.mlp.Right(normalized); product.retain_grad(); products.append(product)
        x = pre + block.mlp.Down(product) + block.mlp.Down_bias
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, -1]).float()
    return 30 * torch.tanh(raw / 30), products


def loss_labels(discovery_loss, values):
    boundaries = np.quantile(discovery_loss, [.25, .50, .75])
    return np.asarray(["easy", "medium-low", "medium-high", "hard"])[np.digitize(values, boundaries)]


def site_strata(panel, discovery_loss):
    count = len(panel["loss"]); result = {"all": np.ones(count, dtype=bool)}
    families = {"current": panel["current_class"], "next": panel["next_class"], "loss": loss_labels(discovery_loss, panel["loss"]), "position": np.asarray([f"p{value}" for value in panel["position"]])}
    for family, labels in families.items():
        for label in np.unique(labels): result[f"{family}:{label}"] = labels == label
    return result


def bank(contributions, mask, size):
    importance = np.square(contributions[mask].astype(np.float64)).mean(0)
    return np.argsort(-importance)[:size]


def scalar_fidelity(contributions, mask, selected):
    target = contributions[mask].astype(np.float64).sum(1)
    prediction = contributions[mask][:, selected].astype(np.float64).sum(1)
    return float(1 - np.square(target - prediction).sum() / max(np.square(target).sum(), 1e-30))


def energy_capture(contributions, mask, selected):
    values = contributions[mask].astype(np.float64)
    return float(np.square(values[:, selected]).sum() / max(np.square(values).sum(), 1e-30))


def write_fidelity(products, mask, selected, down_weight):
    values = torch.as_tensor(products[mask].astype(np.float32), device="cuda")
    weight = down_weight.float()
    full = values @ weight.T; approximation = values[:, torch.as_tensor(selected, device="cuda")] @ weight[:, torch.as_tensor(selected, device="cuda")].T
    return float(1 - (full - approximation).double().square().sum() / full.double().square().sum().clamp_min(1e-30))


def jaccard(first, second):
    a, b = set(map(int, first)), set(map(int, second))
    return len(a & b) / max(len(a | b), 1)


def oracle_k90(contributions):
    absolute = np.abs(contributions.astype(np.float32)); order = np.sort(absolute, axis=1)[:, ::-1]; cumulative = np.cumsum(order, axis=1); target = .9 * cumulative[:, -1:]
    return np.argmax(cumulative >= target, axis=1) + 1


def collect_panel(model, corpus, global_start, config, token_classes):
    layers = config["layers"]
    store = {layer: {"products": [], "contributions": []} for layer in layers}
    metadata = {key: [] for key in ("global_row", "position", "current_id", "next_id", "loss")}
    for position in config["positions"]:
        length = position + 1
        for start in range(0, len(corpus), config["batch_size"]):
            raw = np.asarray(corpus[start:start + config["batch_size"], :length + 1], dtype=np.int64)
            ids = torch.as_tensor(raw[:, :length], device="cuda"); targets = torch.as_tensor(raw[:, length], device="cuda")
            logits, products = forward_with_product_taps(model, ids); losses = F.cross_entropy(logits, targets, reduction="none"); losses.sum().backward()
            for layer in layers:
                product = products[layer][:, position].detach().float(); gradient = products[layer].grad[:, position].detach().float()
                store[layer]["products"].append(product.cpu().to(torch.float16).numpy()); store[layer]["contributions"].append((product * gradient).cpu().to(torch.float16).numpy())
            batch = len(raw); metadata["global_row"].extend((global_start + start + np.arange(batch)).tolist()); metadata["position"].extend([position] * batch); metadata["current_id"].extend(raw[:, position].tolist()); metadata["next_id"].extend(raw[:, length].tolist()); metadata["loss"].extend(losses.detach().cpu().tolist())
        print(f"natural MLP panel={global_start} position={position}", flush=True)
    output = {layer: {key: np.concatenate(value) for key, value in rows.items()} for layer, rows in store.items()}
    panel = {key: np.asarray(value) for key, value in metadata.items()}; panel["current_class"] = token_classes[panel["current_id"]]; panel["next_class"] = token_classes[panel["next_id"]]
    return output, panel


def main():
    config = json.loads(CONFIG.read_text()); tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]); token_classes = token_class_from_flags(flags)
    source = np.load(config["corpus"], mmap_mode="r"); model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    panels = {}
    for name, offset in (("discovery", config["discovery_start"]), ("replication", config["replication_start"])):
        corpus = source[offset:offset + config["rows_per_panel"]]; values, metadata = collect_panel(model, corpus, offset, config, token_classes); panels[name] = {"values": values, "metadata": metadata}
    dmeta = panels["discovery"]["metadata"]; rmeta = panels["replication"]["metadata"]; d_masks = site_strata(dmeta, dmeta["loss"]); r_masks = site_strata(rmeta, dmeta["loss"])
    rng = np.random.default_rng(config["seed"]); results = []; random_jaccard_means = []; oracle = {}; arrays = {}
    for key, value in dmeta.items(): arrays[f"discovery_meta_{key}"] = value
    for key, value in rmeta.items(): arrays[f"replication_meta_{key}"] = value
    for layer in config["layers"]:
        discovery = panels["discovery"]["values"][layer]; replication = panels["replication"]["values"][layer]
        arrays[f"discovery_products_L{layer}"] = discovery["products"]; arrays[f"discovery_contributions_L{layer}"] = discovery["contributions"]
        arrays[f"replication_products_L{layer}"] = replication["products"]; arrays[f"replication_contributions_L{layer}"] = replication["contributions"]
        oracle_values = oracle_k90(replication["contributions"]); oracle[str(layer)] = {"median_k90": float(np.median(oracle_values)), "q10": float(np.quantile(oracle_values, .1)), "q90": float(np.quantile(oracle_values, .9))}
        down = model.blocks[layer].mlp.Down.weight.detach()
        for size in config["bank_sizes"]:
            global_bank = bank(discovery["contributions"], d_masks["all"], size)
            for stratum, dmask in d_masks.items():
                rmask = r_masks.get(stratum)
                if stratum == "all" or rmask is None or dmask.sum() < config["minimum_stratum_sites"] or rmask.sum() < config["minimum_stratum_sites"]: continue
                local_bank = bank(discovery["contributions"], dmask, size); replication_bank = bank(replication["contributions"], rmask, size)
                local_ce = scalar_fidelity(replication["contributions"], rmask, local_bank); global_ce = scalar_fidelity(replication["contributions"], rmask, global_bank)
                local_write = write_fidelity(replication["products"], rmask, local_bank, down); global_write = write_fidelity(replication["products"], rmask, global_bank, down)
                random_ce = []
                for _ in range(config["random_replicates"]): random_ce.append(scalar_fidelity(replication["contributions"], rmask, rng.choice(replication["contributions"].shape[1], size=size, replace=False)))
                results.append({"layer": layer, "size": size, "stratum": stratum, "discovery_count": int(dmask.sum()), "replication_count": int(rmask.sum()), "local_ce_fidelity": local_ce, "global_ce_fidelity": global_ce, "ce_advantage": local_ce - global_ce, "local_write_fidelity": local_write, "global_write_fidelity": global_write, "write_advantage": local_write - global_write, "local_energy_capture": energy_capture(replication["contributions"], rmask, local_bank), "global_energy_capture": energy_capture(replication["contributions"], rmask, global_bank), "bank_jaccard": jaccard(local_bank, replication_bank), "random_ce_mean": float(np.mean(random_ce)), "random_ce_p95": float(np.quantile(random_ce, .95)), "local_bank": local_bank.tolist()})
        size = 128; cells = [row for row in results if row["layer"] == layer and row["size"] == size]
        for _ in range(config["random_replicates"]):
            overlaps = []
            for _cell in cells:
                first = rng.choice(discovery["contributions"].shape[1], size=size, replace=False); second = rng.choice(discovery["contributions"].shape[1], size=size, replace=False); overlaps.append(jaccard(first, second))
            if overlaps: random_jaccard_means.append(float(np.mean(overlaps)))
    np.savez_compressed(ARRAYS, **arrays)
    primary = [row for row in results if row["size"] == 128]; specialized = [row for row in primary if row["ce_advantage"] >= .05]; random_wins = [row for row in primary if row["local_ce_fidelity"] > row["random_ce_p95"]]
    observed_jaccard = float(np.mean([row["bank_jaccard"] for row in primary])); null_jaccard_p95 = float(np.quantile(random_jaccard_means, .95)); opposite = [row for row in primary if np.sign(row["ce_advantage"]) != np.sign(row["write_advantage"])]; broad = [int(layer) for layer, row in oracle.items() if row["median_k90"] > 128]
    hypotheses = {
        "M0_panels_are_fresh_and_task_blind": {"supported": config["discovery_start"] >= 8704 and config["discovery_start"] + config["rows_per_panel"] <= config["replication_start"], "ranges": [[config["discovery_start"], config["discovery_start"] + config["rows_per_panel"] - 1], [config["replication_start"], config["replication_start"] + config["rows_per_panel"] - 1]]},
        "M1_natural_strata_select_specialized_channel_banks": {"supported": len(specialized) >= .25 * len(primary), "passing": len(specialized), "eligible": len(primary), "fraction": len(specialized) / max(len(primary), 1)},
        "M2_selected_banks_beat_random_channels": {"supported": len(random_wins) >= .75 * len(primary), "passing": len(random_wins), "eligible": len(primary), "fraction": len(random_wins) / max(len(primary), 1)},
        "M3_channel_banks_recur": {"supported": observed_jaccard > null_jaccard_p95, "observed": observed_jaccard, "null_p95": null_jaccard_p95},
        "M4_ce_and_write_ontologies_differ": {"supported": len(opposite) >= .25 * len(primary), "opposite": len(opposite), "eligible": len(primary), "fraction": len(opposite) / max(len(primary), 1)},
        "M5_examplewise_native_channel_support_is_broad": {"supported": len(broad) >= 6, "layers": broad, "oracle": oracle},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "oracle_support": oracle, "results": results}, indent=2, allow_nan=False))
    lines = ["# Stratified natural bilinear-MLP channel census", "", "Exact gradients of every native post-product channel are collected at four ordinary FineWeb prediction positions. Fixed channel banks are selected without hand-authored task labels from current-token form, next-token form, loss quartile, or position strata. CE contribution and local MLP-write reconstruction are evaluated separately.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "## Size-128 summary", "", "| Stratum | Cells | Local/global CE | CE advantage | Local/global write | Write advantage | Jaccard | Beats random p95 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for stratum in sorted(set(row["stratum"] for row in primary)):
        cells = [row for row in primary if row["stratum"] == stratum]
        lines.append(f"| {stratum} | {len(cells)} | {np.mean([row['local_ce_fidelity'] for row in cells]):.3f}/{np.mean([row['global_ce_fidelity'] for row in cells]):.3f} | {np.mean([row['ce_advantage'] for row in cells]):+.3f} | {np.mean([row['local_write_fidelity'] for row in cells]):.3f}/{np.mean([row['global_write_fidelity'] for row in cells]):.3f} | {np.mean([row['write_advantage'] for row in cells]):+.3f} | {np.mean([row['bank_jaccard'] for row in cells]):.3f} | {np.mean([row['local_ce_fidelity'] > row['random_ce_p95'] for row in cells]):.2f} |")
    lines += ["", "## Examplewise oracle breadth", "", "| Layer | Median K90 | q10--q90 |", "|---:|---:|---:|"]
    for layer in config["layers"]: lines.append(f"| {layer} | {oracle[str(layer)]['median_k90']:.0f} | {oracle[str(layer)]['q10']:.0f}--{oracle[str(layer)]['q90']:.0f} |")
    lines += ["", "## Guardrails", "", "CE banks are selected from activation-times-total-gradient contributions and are prediction-objective coordinates, not monosemantic neurons. Local-write fidelity asks a different question and uses the same CE-selected bank without refitting. Examplewise K90 is an oracle descriptive breadth statistic based on absolute contribution mass; it is not a prospective sufficient circuit size. Finite suffix replacement is required before treating local scalar fidelity as causal sufficiency."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__": main()
