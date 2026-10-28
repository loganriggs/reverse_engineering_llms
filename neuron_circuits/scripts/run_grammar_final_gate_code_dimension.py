#!/usr/bin/env python3
"""Causal affine dimension of the final native bilinear gate code."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from run_grammar_final_interface_redteam import load_model
from run_grammar_mixed_address_router import collect, prototypes, apply_prototypes, score, select_panel


CONFIG = Path("configs/grammar_final_gate_code_dimension.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_final_gate_code_dimension.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_FINAL_GATE_CODE_DIMENSION.md")


def gate(model, state):
    block = model.blocks[3]
    return (state @ block.left.weight.detach().cpu().T) * (state @ block.right.weight.detach().cpu().T)


def pca(values):
    mean = values.mean(0)
    _, _, basis = torch.linalg.svd(values - mean, full_matrices=False)
    return mean, basis


def project(values, mean, basis):
    return mean + (values - mean) @ basis.T @ basis


def eligible_mask(model, manifest, facts, labels, templates, device):
    eligible = np.ones(len(facts), bool)
    for template in templates:
        prompts = encode_prompts(manifest, facts, template)
        for start in range(0, len(prompts), 1024):
            tokens = torch.as_tensor(prompts[start:start + 1024].astype(np.int64), device=device)
            pred = model(tokens)[:, -1, list(VOCAB.answers)].argmax(-1).cpu().numpy()
            eligible[start:start + len(pred)] &= pred == labels[start:start + len(pred)]
    return eligible


def pure_panel(labels, eligible, facts, config, seed):
    rng = np.random.default_rng(seed)
    panel = {"discovery": [], "validation": [], "evaluation": []}
    for relation in range(8):
        for answer in range(8):
            idx = np.flatnonzero(eligible & (facts[:, 2] == relation) & (labels == answer))
            rng.shuffle(idx)
            a = config["discovery_per_relation_answer_status"]
            b = a + config["validation_per_relation_answer_status"]
            c = b + config["evaluation_per_relation_answer_status"]
            if len(idx) < c:
                raise RuntimeError((relation, answer, len(idx), c))
            panel["discovery"].extend(idx[:a]); panel["validation"].extend(idx[a:b]); panel["evaluation"].extend(idx[b:c])
    return {name: np.asarray(rows) for name, rows in panel.items()}


def analyze(spec, config, device):
    root = Path(spec["root"])
    manifest = load_manifest(root / "manifest.npz")
    model = load_model(root / f"rho_{spec['rho']:g}/depth_4/checkpoint.pt", device)
    facts = np.asarray(manifest["train_facts"])
    labels, selected, changed = labels_for_rho(manifest, "train", spec["rho"])
    eligible = eligible_mask(model, manifest, facts, labels, config["evaluation_templates"], device)
    if spec["regime"] == "pure":
        panels = {"all": pure_panel(labels, eligible, facts, config, config["seed"] + spec["seed"])}
    else:
        raw = select_panel(labels, selected, changed, eligible, facts, config, config["seed"] + spec["seed"])
        panels = {"unselected": raw[False], "selected": raw[True]}
    output = {}
    down = model.blocks[3].down.weight.detach().cpu()
    for status, panel in panels.items():
        train_idx = panel["discovery"]
        eval_idx = panel["evaluation"]
        train, _, _ = collect(model, encode_prompts(manifest, facts[train_idx], config["discovery_template"]), device)
        train_gate = gate(model, train["premlp"])
        mean, basis = pca(train_gate)
        importance = train_gate.var(0) * down.square().sum(0)
        coordinate_order = importance.argsort(descending=True)
        write_table = prototypes(train["write"], facts[train_idx, 2], labels[train_idx], True)
        output[status] = {}
        for template in config["evaluation_templates"]:
            data, _, _ = collect(model, encode_prompts(manifest, facts[eval_idx], template), device)
            native_gate = gate(model, data["premlp"])
            oracle_write = apply_prototypes(write_table, facts[eval_idx, 2], labels[eval_idx], True)
            methods = {}
            for rank in config["ranks"]:
                dense = project(native_gate, mean, basis[:rank])
                coordinate = mean.expand_as(native_gate).clone()
                chosen = coordinate_order[:rank]
                coordinate[:, chosen] = native_gate[:, chosen]
                for name, candidate in ((f"pca_rank_{rank}", dense), (f"coordinate_rank_{rank}", coordinate)):
                    methods[name] = score(candidate @ down.T, data["write"], data["after"], data["zero_logits"], data["logits"], labels[eval_idx], oracle_write, model, config["minimum_fisher_effect_norm"])
            generator = torch.Generator().manual_seed(config["seed"] + spec["seed"] * 100 + template + (1 if status == "selected" else 0))
            random_fidelity = []
            for _ in range(config["random_subspaces"]):
                random_basis = torch.linalg.qr(torch.randn(192, 64, generator=generator)).Q.T
                candidate = project(native_gate, mean, random_basis)
                row = score(candidate @ down.T, data["write"], data["after"], data["zero_logits"], data["logits"], labels[eval_idx], oracle_write, model, config["minimum_fisher_effect_norm"])
                random_fidelity.append(row["median_effect_fidelity"])
            methods["random_rank_64"] = {"median_fidelity_mean": float(np.mean(random_fidelity)), "median_fidelity_p95": float(np.quantile(random_fidelity, .95)), "replicates": len(random_fidelity)}
            output[status][str(template)] = methods
    return {"regime": spec["regime"], "seed": spec["seed"], "results": output}


def main():
    config = json.loads(CONFIG.read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = [analyze(spec, config, device) for spec in config["models"]]
    first, held = map(str, config["evaluation_templates"])
    pure = [run for run in runs if run["regime"] == "pure"]
    mixed = [run for run in runs if run["regime"] == "mixed"]
    def val(run, status, template, method, key="median_effect_fidelity"):
        return run["results"][status][template][method][key]
    g0 = all(val(run,status,template,"pca_rank_192") >= .9999 for run in runs for status in run["results"] for template in (first,held))
    g1 = all(val(run,"all",held,"pca_rank_32") >= .90 for run in pure)
    g2 = all(val(run,"selected",held,"pca_rank_32") < .90 for run in mixed)
    g3 = all(val(run,"selected",held,"pca_rank_64") >= .85 for run in mixed)
    g4 = all(val(run,"selected",held,"pca_rank_32") - val(run,"selected",held,"coordinate_rank_32") >= .15 for run in mixed)
    g5 = all(val(run,"unselected",held,"pca_rank_32") - val(run,"selected",held,"pca_rank_32") >= .05 for run in mixed)
    g6 = all(val(run,"selected",held,"pca_rank_64") >= run["results"]["selected"][held]["random_rank_64"]["median_fidelity_p95"] + .15 for run in mixed)
    g7 = all(abs(val(run,"selected",first,"pca_rank_64") - val(run,"selected",held,"pca_rank_64")) <= .10 for run in mixed)
    g8 = all(row["effectful_examples"] >= 64 for run in runs for status in run["results"] for template in (first,held) for name,row in run["results"][status][template].items() if name != "random_rank_64")
    hypotheses={name:{"supported":bool(value)} for name,value in zip(config["hypotheses"],(g0,g1,g2,g3,g4,g5,g6,g7,g8))}
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"runs":runs},indent=2,allow_nan=False))
    lines=["# Causal dimension of the final bilinear gate code","","PCA and coordinate subspaces are fit on discovery-template native post-product gate activations. Held-out gates are projected around the discovery mean, passed through the original down projection, and executed through final RMSNorm/unembedding.","","## Locked hypotheses",""]
    for name,row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","| Regime | Seed | Status | PCA r16/r32/r64/r96 | coordinate r32/r64 | random64 mean/p95 |","|---|---:|---|---:|---:|---:|"]
    for run in runs:
        for status in run["results"]:
            random=run["results"][status][held]["random_rank_64"]
            lines.append(f"| {run['regime']} | {run['seed']} | {status} | {val(run,status,held,'pca_rank_16'):.3f}/{val(run,status,held,'pca_rank_32'):.3f}/{val(run,status,held,'pca_rank_64'):.3f}/{val(run,status,held,'pca_rank_96'):.3f} | {val(run,status,held,'coordinate_rank_32'):.3f}/{val(run,status,held,'coordinate_rank_64'):.3f} | {random['median_fidelity_mean']:.3f}/{random['median_fidelity_p95']:.3f} |")
    lines += ["","This is an affine variation dimension, not a count of sufficient neurons: all methods retain the discovery mean gate. The coordinate comparison asks whether native channel axes describe variations more efficiently than a dense learned basis."]
    REPORT.write_text("\n".join(lines)+"\n")
    print("wrote",OUTPUT,REPORT,flush=True)


if __name__=="__main__": main()
