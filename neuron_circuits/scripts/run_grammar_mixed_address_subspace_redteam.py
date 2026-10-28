#!/usr/bin/env python3
"""Red-team whether the compact mixed address code is status-specific or shared."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from run_grammar_final_interface_redteam import load_model
from run_grammar_mixed_address_router import collect, prototypes, apply_prototypes, score, select_panel
from run_grammar_mixed_weight_canonical_address_subspace import pca, project, tensor_write


CONFIG = Path("configs/grammar_mixed_address_subspace_redteam.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_mixed_address_subspace_redteam.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_MIXED_ADDRESS_SUBSPACE_REDTEAM.md")


def prepare(spec, config, device):
    root = Path(spec["root"])
    manifest = load_manifest(root / "manifest.npz")
    model = load_model(root / f"rho_{config['rho']:g}/depth_4/checkpoint.pt", device)
    facts = np.asarray(manifest["train_facts"])
    labels, selected, changed = labels_for_rho(manifest, "train", config["rho"])
    eligible = np.ones(len(facts), bool)
    for template in config["evaluation_templates"]:
        prompts = encode_prompts(manifest, facts, template)
        for start in range(0, len(prompts), 1024):
            tokens = torch.as_tensor(prompts[start:start + 1024].astype(np.int64), device=device)
            prediction = model(tokens)[:, -1, list(VOCAB.answers)].argmax(-1).cpu().numpy()
            eligible[start:start + len(prediction)] &= prediction == labels[start:start + len(prediction)]
    panel = select_panel(labels, selected, changed, eligible, facts, config, config["seed"] + spec["seed"])
    discovery = {}
    for flag, status in ((False, "unselected"), (True, "selected")):
        idx = panel[flag]["discovery"]
        data, _, _ = collect(model, encode_prompts(manifest, facts[idx], config["discovery_template"]), device)
        discovery[status] = {"idx": idx, "state": data["premlp"], "write": data["write"]}
    pooled = torch.cat([discovery["unselected"]["state"], discovery["selected"]["state"]])
    bases = {}
    for status in ("unselected", "selected"):
        mean, basis = pca(discovery[status]["state"])
        bases[status] = {"mean": mean, "basis": basis}
    pooled_mean, pooled_basis = pca(pooled)
    bases["pooled"] = {"mean": pooled_mean, "basis": pooled_basis}
    return {"spec": spec, "manifest": manifest, "model": model, "facts": facts, "labels": labels,
            "panel": panel, "discovery": discovery, "bases": bases}


def evaluate(prepared, all_prepared, config):
    model = prepared["model"]
    facts = prepared["facts"]
    labels = prepared["labels"]
    manifest = prepared["manifest"]
    seed = prepared["spec"]["seed"]
    output = {}
    foreign = next(item for item in all_prepared if item["spec"]["seed"] != seed)
    for flag, status in ((False, "unselected"), (True, "selected")):
        other = "selected" if status == "unselected" else "unselected"
        eval_idx = prepared["panel"][flag]["evaluation"]
        train_idx = prepared["discovery"][status]["idx"]
        relation_proto = prototypes(prepared["discovery"][status]["write"], facts[train_idx, 2], labels[train_idx], True)
        output[status] = {}
        for template in config["evaluation_templates"]:
            data, _, _ = collect(model, encode_prompts(manifest, facts[eval_idx], template), next(model.parameters()).device)
            proto = apply_prototypes(relation_proto, facts[eval_idx, 2], labels[eval_idx], True)
            methods = {}
            native = tensor_write(model, data["premlp"])
            methods["native"] = score(native, data["write"], data["after"], data["zero_logits"], data["logits"], labels[eval_idx], proto, model, config["minimum_fisher_effect_norm"])
            for rank in config["ranks"]:
                own = prepared["bases"][status]
                opposite = prepared["bases"][other]
                pooled = prepared["bases"]["pooled"]
                foreign_basis = foreign["bases"][status]["basis"]
                candidates = {
                    f"own_top_{rank}": project(data["premlp"], own["mean"], own["basis"][:rank]),
                    f"opposite_basis_target_mean_{rank}": project(data["premlp"], own["mean"], opposite["basis"][:rank]),
                    f"pooled_top_{rank}": project(data["premlp"], pooled["mean"], pooled["basis"][:rank]),
                    f"own_bottom_{rank}": project(data["premlp"], own["mean"], own["basis"][-rank:]),
                    f"foreign_basis_target_mean_{rank}": project(data["premlp"], own["mean"], foreign_basis[:rank]),
                }
                for name, state in candidates.items():
                    write = tensor_write(model, state)
                    methods[name] = score(write, data["write"], data["after"], data["zero_logits"], data["logits"], labels[eval_idx], proto, model, config["minimum_fisher_effect_norm"])
            output[status][str(template)] = methods
    return {"seed": seed, "foreign_seed": foreign["spec"]["seed"], "results": output}


def main():
    config = json.loads(CONFIG.read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prepared = [prepare(spec, config, device) for spec in config["models"]]
    runs = [evaluate(item, prepared, config) for item in prepared]
    first, held = map(str, config["evaluation_templates"])
    def val(run, status, template, method, key="median_effect_fidelity"):
        return run["results"][status][template][method][key]
    r0 = all(val(run, status, template, "native") >= .9999 for run in runs for status in ("selected", "unselected") for template in (first, held))
    r1 = all(val(run, status, held, "own_top_48") >= .90 for run in runs for status in ("selected", "unselected"))
    r2 = all(val(run, "selected", held, "own_top_48") - val(run, "selected", held, "opposite_basis_target_mean_48") >= .15 for run in runs)
    r3 = all(val(run, status, held, "pooled_top_48") >= .90 for run in runs for status in ("selected", "unselected"))
    r4 = all(val(run, status, held, "own_top_48") - val(run, status, held, "own_bottom_48") >= .20 for run in runs for status in ("selected", "unselected"))
    r5 = all(val(run, status, held, "own_top_48") - val(run, status, held, "foreign_basis_target_mean_48") >= .20 for run in runs for status in ("selected", "unselected"))
    r6 = all(val(run, status, held, "pooled_top_32") >= .75 for run in runs for status in ("selected", "unselected"))
    r7 = all(abs(val(run, status, first, "pooled_top_48") - val(run, status, held, "pooled_top_48")) <= .10 for run in runs for status in ("selected", "unselected"))
    r8 = all(row["effectful_examples"] >= 64 for run in runs for status in ("selected", "unselected") for template in (first, held) for row in run["results"][status][template].values())
    hypotheses = {name: {"supported": bool(value)} for name, value in zip(config["hypotheses"], (r0, r1, r2, r3, r4, r5, r6, r7, r8))}
    payload = {"config": config, "hypotheses": hypotheses, "runs": runs}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = ["# Mixed address-subspace red team", "", "All projections are finite interventions executed through the original final bilinear tensor. Opposite- and foreign-basis controls retain the target model/status mean so that only subspace orientation changes.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Seed | Status | own48 | opposite48 | pooled48 | bottom48 | foreign48 | pooled32 |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for run in runs:
        for status in ("unselected", "selected"):
            lines.append(f"| {run['seed']} | {status} | {val(run,status,held,'own_top_48'):.3f} | {val(run,status,held,'opposite_basis_target_mean_48'):.3f} | {val(run,status,held,'pooled_top_48'):.3f} | {val(run,status,held,'own_bottom_48'):.3f} | {val(run,status,held,'foreign_basis_target_mean_48'):.3f} | {val(run,status,held,'pooled_top_32'):.3f} |")
    lines += ["", "A successful pooled/opposite-status basis means rule and arbitrary facts share a causal state manifold; it does not mean that their individual addresses or upstream construction are the same. A failed foreign basis is a coordinate-system control, not evidence against an invariant abstract computation."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
