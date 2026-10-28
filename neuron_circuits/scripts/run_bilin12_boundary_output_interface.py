#!/usr/bin/env python3
"""Compare paired boundary writes with a generic natural newline output interface."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from run_bilin12_late_address_program_audit import load_rows
from run_bilin12_layer11_boundary_upstream_decomposition import prefix
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_boundary_output_interface.json")
OUTPUT = TENSOR_ROOT / "bilin12_boundary_output_interface.json"
REPORT = TENSOR_ROOT / "BILIN12_BOUNDARY_OUTPUT_INTERFACE.md"


def token_id(tokenizer, text):
    values = tokenizer.encode(text, add_special_tokens=False)
    if len(values) != 1:
        raise RuntimeError(f"{text!r} is not one token: {values}")
    return values[0]


@torch.inference_mode()
def final_components(model, ids, layer):
    dimension = model.config.n_embd
    x0, x, bus = prefix(model, ids, layer)
    block = model.blocks[layer]
    mixed = block.lambdas[0] * x + block.lambdas[1] * x0
    attention, _ = block.attn(F.rms_norm(mixed, (dimension,)), bus)
    pre = mixed[:, -1] + attention[:, -1]
    state = F.rms_norm(pre, (dimension,))
    write = block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
    logits = logits_from_hidden(model, pre + write)
    return pre.float(), write.float(), logits.float()


@torch.inference_mode()
def logits_from_hidden(model, hidden):
    raw = model.lm_head(F.rms_norm(hidden, (model.config.n_embd,))).float()
    return 30 * torch.tanh(raw / 30)


@torch.inference_mode()
def paired_panel(model, spec, layer, newline, batch_size):
    rows = load_rows(spec)
    base_pre, base_write, base_logits, changed_write = [], [], [], []
    for start in range(0, len(rows), batch_size):
        raw = np.asarray(rows[start:start + batch_size, :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        keep = ids[:, -1] != newline
        if not bool(keep.any()):
            continue
        ids = ids[keep]
        changed = ids.clone(); changed[:, -1] = newline
        pre, write, logits = final_components(model, ids, layer)
        _, write_changed, _ = final_components(model, changed, layer)
        base_pre.append(pre); base_write.append(write); base_logits.append(logits)
        changed_write.append(write_changed)
    return tuple(torch.cat(values) for values in
                 (base_pre, base_write, base_logits, changed_write))


def natural_prototype(model, spec, layer, positions, target_token, batch_size):
    _, writes = collect_states_writes(model, layer, spec, positions, batch_size)
    rows = load_rows(spec)
    writes = writes.reshape(len(rows), len(positions), -1)
    ids = rows[:, positions]
    weighted = torch.zeros(writes.shape[-1], device=writes.device)
    count = 0
    for position_index in range(len(positions)):
        mask = torch.as_tensor(ids[:, position_index] == target_token, device=writes.device)
        if int(mask.sum()) == 0 or int((~mask).sum()) == 0:
            continue
        delta = writes[:, position_index][mask].mean(0) - writes[:, position_index][~mask].mean(0)
        weighted += delta * int(mask.sum()); count += int(mask.sum())
    if count < 32:
        raise RuntimeError(f"only {count} natural examples for token {target_token}")
    return weighted / count, count


def effect_similarity(base_logits, target, predicted):
    probability = F.softmax(base_logits.float(), -1)
    target = target - (probability * target).sum(-1, keepdim=True)
    predicted = predicted - (probability * predicted).sum(-1, keepdim=True)
    target_energy = (probability * target.square()).sum(-1).clamp_min(1e-30)
    predicted_energy = (probability * predicted.square()).sum(-1).clamp_min(1e-30)
    cosine = (probability * target * predicted).sum(-1) / torch.sqrt(
        target_energy * predicted_energy).clamp_min(1e-30)
    fidelity = 1 - (probability * (predicted - target).square()).sum(-1) / target_energy
    return float(cosine.median()), float(fidelity.median())


@torch.inference_mode()
def evaluate_panel(model, panel, semantic, generic, control, generic_scale, control_scale):
    pre, write, base_logits, changed_write = panel
    oracle_delta = changed_write - write
    oracle_logits = logits_from_hidden(model, pre + write + oracle_delta)
    oracle_effect = oracle_logits - base_logits
    methods = {}
    for name, delta in (("semantic", semantic),
                        ("generic", generic * generic_scale),
                        ("control", control * control_scale)):
        logits = logits_from_hidden(model, pre + write + delta)
        cosine, fidelity = effect_similarity(base_logits, oracle_effect, logits - base_logits)
        methods[name] = {"cosine": cosine, "fidelity": fidelity}
    probability = F.softmax(base_logits, -1)
    centered = oracle_effect - (probability * oracle_effect).sum(-1, keepdim=True)
    methods["oracle_fisher_energy"] = float((probability * centered.square()).sum(-1).median())
    methods["examples"] = len(pre)
    return methods


@torch.inference_mode()
def choose_scale(model, panel, delta, grid):
    pre, write, base_logits, changed_write = panel
    oracle_logits = logits_from_hidden(model, pre + changed_write)
    oracle_effect = oracle_logits - base_logits
    candidates = []
    for scale in grid:
        logits = logits_from_hidden(model, pre + write + delta * scale)
        cosine, fidelity = effect_similarity(base_logits, oracle_effect, logits - base_logits)
        candidates.append((fidelity, cosine, scale))
    fidelity, cosine, scale = max(candidates)
    return {"scale": scale, "fidelity": fidelity, "cosine": cosine}


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    newline = token_id(tokenizer, "\n")
    period = token_id(tokenizer, ".")
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    semantic_panel = paired_panel(
        model, config["semantic_discovery"], config["layer"], newline, config["batch_size"],
    )
    semantic = (semantic_panel[3] - semantic_panel[1]).mean(0)
    generic, newline_count = natural_prototype(
        model, config["prototype"], config["layer"], config["positions"], newline,
        config["batch_size"],
    )
    control, period_count = natural_prototype(
        model, config["prototype"], config["layer"], config["positions"], period,
        config["batch_size"],
    )
    calibration_panel = paired_panel(
        model, config["calibration"], config["layer"], newline, config["batch_size"],
    )
    generic_calibration = choose_scale(model, calibration_panel, generic, config["scale_grid"])
    control_calibration = choose_scale(model, calibration_panel, control, config["scale_grid"])
    results = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        panel = paired_panel(model, spec, config["layer"], newline, config["batch_size"])
        results[corpus] = evaluate_panel(
            model, panel, semantic, generic, control,
            generic_calibration["scale"], control_calibration["scale"],
        )
    n0 = all(row["semantic"]["fidelity"] >= .50 for row in results.values())
    n1 = all(row["generic"]["cosine"] >= .80 and row["generic"]["fidelity"] >= .50
             for row in results.values())
    n2 = all(row["generic"]["fidelity"] >= row["semantic"]["fidelity"] - .10
             for row in results.values())
    n3 = all(row["generic"]["fidelity"] >= row["control"]["fidelity"] + .20
             for row in results.values())
    n4 = all(results[corpus]["generic"]["fidelity"]
             >= generic_calibration["fidelity"] - .20
             for corpus in ("wikitext", "tinystories"))
    hypotheses = {
        "N0_mean_semantic_write_is_reusable": {"supported": n0},
        "N1_generic_newline_interface_transfers": {"supported": n1},
        "N2_generic_interface_matches_paired_semantics": {"supported": n2},
        "N3_newline_semantics_beats_punctuation_control": {"supported": n3},
        "N4_scalar_calibration_generalizes_ood": {"supported": n4},
    }
    write_cosine = float(F.cosine_similarity(semantic, generic, dim=0))
    OUTPUT.write_text(json.dumps({
        "config": config, "tokens": {"newline": newline, "period": period},
        "prototype_counts": {"newline": newline_count, "period": period_count},
        "write_cosine": write_cosine, "generic_calibration": generic_calibration,
        "control_calibration": control_calibration, "hypotheses": hypotheses,
        "results": results,
    }, indent=2))
    lines = ["# Natural boundary computation versus generic output interface", "",
             "The exact per-example Layer-11 MLP write effect of changing the current token to newline is predicted by a paired mean semantic edit, an independently estimated natural newline prototype, and a calibrated period-token control.", "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", f"Natural prototype counts: newline `{newline_count}`, period `{period_count}`. Semantic/generic write cosine: `{write_cosine:.3f}`. Generic scale/fidelity on calibration: `{generic_calibration['scale']:.3f}` / `{generic_calibration['fidelity']:.3f}`; period scale/fidelity: `{control_calibration['scale']:.3f}` / `{control_calibration['fidelity']:.3f}`.", "",
              "| Corpus | Examples | Semantic cosine/fidelity | Generic newline cosine/fidelity | Period control cosine/fidelity | Oracle Fisher energy |",
              "|---|---:|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        lines.append(
            f"| {corpus} | {row['examples']} | {row['semantic']['cosine']:.3f}/{row['semantic']['fidelity']:.3f} | "
            f"{row['generic']['cosine']:.3f}/{row['generic']['fidelity']:.3f} | "
            f"{row['control']['cosine']:.3f}/{row['control']['fidelity']:.3f} | "
            f"{row['oracle_fisher_energy']:.3e} |"
        )
    lines += ["", "The oracle is only the final-MLP write path of the token edit, realized on the original recipient state. No claim is made that this path reproduces the edit's upstream residual or attention changes."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
