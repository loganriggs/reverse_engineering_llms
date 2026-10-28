#!/usr/bin/env python3
"""Finite interventions on final attention as an arbitrary-fact address correction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from run_grammar_final_interface_redteam import load_model
from run_grammar_mixed_address_router import collect, fisher_rows, select_panel


CONFIG = Path("configs/grammar_mixed_attention_address_intervention.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_mixed_attention_address_intervention.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_MIXED_ATTENTION_ADDRESS_INTERVENTION.md")


def prototype_table(values, relations, labels, mode):
    output = {}
    relations_to_use = range(8) if "relation" in mode else (None,)
    labels_to_use = range(8) if "answer" in mode else (None,)
    for relation in relations_to_use:
        for label in labels_to_use:
            keep = np.ones(len(values), bool)
            if relation is not None: keep &= relations == relation
            if label is not None: keep &= labels == label
            output[(relation, label)] = values[torch.as_tensor(keep)].mean(0)
    return output


def apply_table(table, relations, labels, mode):
    return torch.stack([table[(int(r) if "relation" in mode else None,
                               int(y) if "answer" in mode else None)] for r, y in zip(relations, labels)])


@torch.inference_mode()
def candidate(model, boundary, attention):
    block = model.blocks[3]; device = next(model.parameters()).device
    boundary, attention = boundary.to(device), attention.to(device)
    after = boundary + attention; normalized = block.mlp_norm(after)
    activation = block.left(normalized) * block.right(normalized); write = activation @ block.down.weight.T
    logits = model.unembedding(model.final_norm(after + write))[:, list(VOCAB.answers)].float().cpu()
    return attention.cpu(), write.cpu(), logits


@torch.inference_mode()
def metrics(data, candidate_attention, candidate_write, candidate_logits, model, threshold):
    device = next(model.parameters()).device
    # Decoder metric holds the native post-attention residual fixed and swaps only
    # the exact MLP write induced by the candidate attention state.
    decoder_logits = model.unembedding(model.final_norm((data["after"] + candidate_write).to(device)))[:, list(VOCAB.answers)].float().cpu()
    decoder_energy, decoder_cosine, decoder_fidelity = fisher_rows(
        data["zero_logits"], data["logits"] - data["zero_logits"], decoder_logits - data["zero_logits"])
    pre_logits = model.unembedding(model.final_norm(data["boundary3_final"].to(device)))[:, list(VOCAB.answers)].float().cpu()
    route_energy, route_cosine, route_fidelity = fisher_rows(
        pre_logits, data["logits"] - pre_logits, candidate_logits - pre_logits)
    keep = (decoder_energy >= threshold) & (route_energy >= threshold)
    return {"effectful_examples": int(keep.sum()),
            "decoder_median_fidelity": float(decoder_fidelity[keep].median()),
            "decoder_median_cosine": float(decoder_cosine[keep].median()),
            "route_median_fidelity": float(route_fidelity[keep].median()),
            "route_median_cosine": float(route_cosine[keep].median()),
            "answer_accuracy": float((candidate_logits[keep].argmax(-1) == data["labels"][keep]).float().mean()),
            "attention_to_boundary_norm": float((candidate_attention.norm(dim=1) / data["boundary3_final"].norm(dim=1).clamp_min(1e-30)).median())}


def analyze(spec, config, device):
    root = Path(spec["root"]); manifest = load_manifest(root / "manifest.npz")
    model = load_model(root / f"rho_{config['rho']:g}/depth_4/checkpoint.pt", device)
    facts = np.asarray(manifest["train_facts"]); labels, selected, changed = labels_for_rho(manifest, "train", config["rho"])
    eligible = np.ones(len(facts), bool)
    for template in (config["discovery_template"], config["evaluation_template"]):
        prompts = encode_prompts(manifest, facts, template)
        for start in range(0, len(prompts), 1024):
            tokens = torch.as_tensor(prompts[start:start + 1024].astype(np.int64), device=device)
            prediction = model(tokens)[:, -1, list(VOCAB.answers)].argmax(-1).cpu().numpy()
            eligible[start:start + len(prediction)] &= prediction == labels[start:start + len(prediction)]
    panel = select_panel(labels, selected, changed, eligible, facts, config, config["seed"] + spec["seed"])
    output = {}
    maximum_native_error = 0.0
    for status_bool, status in ((False, "unselected"), (True, "selected")):
        train_indices = panel[status_bool]["discovery"]; eval_indices = panel[status_bool]["evaluation"]
        train, _, _ = collect(model, encode_prompts(manifest, facts[train_indices], config["discovery_template"]), device)
        evaluation, _, _ = collect(model, encode_prompts(manifest, facts[eval_indices], config["evaluation_template"]), device)
        evaluation["labels"] = torch.as_tensor(labels[eval_indices], dtype=torch.long)
        train_rel, train_lab = facts[train_indices, 2], labels[train_indices]
        eval_rel, eval_lab = facts[eval_indices, 2], labels[eval_indices]
        candidates = {"native": evaluation["attn_write"],
                      "delete_attention": torch.zeros_like(evaluation["attn_write"])}
        for mode in ("global", "relation", "answer", "relation_answer"):
            table = prototype_table(train["attn_write"], train_rel, train_lab, mode)
            candidates[f"{mode}_prototype"] = apply_table(table, eval_rel, eval_lab, mode)
        generator = torch.Generator().manual_seed(config["seed"] + spec["seed"] * 10 + int(status_bool))
        candidates["shuffled_attention"] = evaluation["attn_write"][torch.randperm(len(eval_indices), generator=generator)]
        rows = {}
        for name, attention in candidates.items():
            candidate_attention, candidate_write, logits = candidate(model, evaluation["boundary3_final"], attention)
            if name == "native": maximum_native_error = max(maximum_native_error, float((logits - evaluation["logits"]).abs().max()))
            rows[name] = metrics(evaluation, candidate_attention, candidate_write, logits, model,
                                 config["minimum_fisher_effect_norm"])
        output[status] = rows
    return {"seed": spec["seed"], "maximum_native_error": maximum_native_error, "results": output}


def main():
    config = json.loads(CONFIG.read_text()); device = "cuda" if torch.cuda.is_available() else "cpu"; runs = []
    for spec in config["models"]:
        print("intervene", spec["seed"], flush=True); runs.append(analyze(spec, config, device))
    def get(run, status, method, key): return run["results"][status][method][key]
    c0 = all(run["maximum_native_error"] <= 1e-5 and get(run,"selected","native","route_median_fidelity") >= .9999 for run in runs)
    c1 = all(get(run,"selected","delete_attention","decoder_median_fidelity") >= .7 for run in runs)
    c2 = all(get(run,"selected","delete_attention","route_median_fidelity") < .7 for run in runs)
    c3 = all((1-get(run,"selected","delete_attention","route_median_fidelity")) >=
             (1-get(run,"unselected","delete_attention","route_median_fidelity")) + .15 for run in runs)
    c4 = all(get(run,"selected","relation_prototype","route_median_fidelity") >= .7 for run in runs)
    c5 = all(get(run,"selected","relation_answer_prototype","route_median_fidelity") >= .8 for run in runs)
    c6 = all(get(run,"selected","shuffled_attention","route_median_fidelity") < .5 for run in runs)
    c7 = all(get(run,"selected","native","attention_to_boundary_norm") < .2 and
             get(run,"selected","delete_attention","route_median_fidelity") <= .8 for run in runs)
    c8 = all(get(run,status,method,"effectful_examples") >=64 for run in runs for status in ("selected","unselected") for method in run["results"][status])
    hypotheses = {name:{"supported":bool(value)} for name,value in zip(config["hypotheses"],(c0,c1,c2,c3,c4,c5,c6,c7,c8))}
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"runs":runs},indent=2,allow_nan=False))
    lines=["# Finite final-attention interventions on the mixed address code","",
           "Each candidate replaces the final attention residual at held-out template-6 facts. The exact bilinear MLP is recomputed. Decoder fidelity holds the native post-attention residual fixed and compares only the induced MLP write; complete-route fidelity includes the replaced attention residual itself. Answer prototypes are explicitly leaky diagnostics.","","## Locked hypotheses",""]
    for name,row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    methods=("native","delete_attention","global_prototype","relation_prototype","answer_prototype","relation_answer_prototype","shuffled_attention")
    lines += ["","| Seed | Status | " + " | ".join(methods) + " |","|---:|---|"+"---:|"*len(methods)]
    for run in runs:
        for status in ("unselected","selected"):
            lines.append(f"| {run['seed']} | {status} | "+" | ".join(f"{get(run,status,m,'route_median_fidelity'):.3f}" for m in methods)+" |")
    lines += ["","Cells report complete-route Fisher fidelity. Decoder-only fidelity is retained in JSON to distinguish whether attention changes the bilinear address versus contributing a direct residual write."]
    REPORT.write_text("\n".join(lines)+"\n"); print("wrote",OUTPUT,REPORT,flush=True)


if __name__ == "__main__": main()
