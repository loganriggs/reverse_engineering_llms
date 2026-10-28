#!/usr/bin/env python3
"""Compare edited-source effects with every equal-count source subset."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bilin18_mlp_discriminators import TOKENIZER
from redteam_bilin18_broad_regime_construction import reconstruct_selected
from run_bilin18_broad_regime_construction import TENSOR_ROOT, contrasts, repeat_batch, suffix, trace
from run_bilin18_qkov_causal_edges import load_full_model


CONFIG = Path("configs/bilin18_broad_regime_source_position_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_broad_regime_source_position_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_BROAD_REGIME_SOURCE_POSITION_REDTEAM.md"


def program_position_deltas(model, layer, details, program):
    heads = model.config.n_head; head_dim = model.config.n_embd // heads
    pattern = details["pattern"][:, program, -1]
    values = details["mixed_value"].permute(0, 2, 1, 3)[:, program]
    projection = model.blocks[layer].attn.c_proj.weight.T.reshape(heads, head_dim, model.config.n_embd)[program]
    base = torch.einsum("bhsd,hdk->bhsk", pattern[:1, ..., None] * values[:1], projection)
    changed = torch.einsum("bhsd,hdk->bhsk", pattern[1:, ..., None] * values[1:], projection)
    return (changed - base).sum(1)[0]


@torch.inference_mode()
def evaluate_pair(model, tokenizer, pair, layer, program):
    ids = torch.as_tensor([pair["base_ids"], pair["source_ids"]], device="cuda")
    contrast = contrasts(tokenizer, [pair], ids.device, model.config.vocab_size)[0]
    x0, traces, _ = trace(model, ids); row = traces[layer]
    base_state = row["output"][:1]
    carried = row["mixed"][1:, -1] - row["mixed"][:1, -1]
    mlp = row["mlp"][1:, -1] - row["mlp"][:1, -1]
    no_attention = base_state.clone(); no_attention[:, -1] = base_state[:, -1] + carried + mlp
    deltas = program_position_deltas(model, layer, row["details"], program)
    edited = tuple(index for index, (base, source) in enumerate(zip(pair["base_ids"], pair["source_ids"])) if base != source and index != len(pair["base_ids"]) - 1)
    candidates = list(itertools.combinations(range(len(pair["base_ids"]) - 1), len(edited)))
    if not edited or edited not in candidates:
        return None
    programs = candidates + [tuple(range(len(pair["base_ids"])))]
    effects = []
    for start in range(0, len(programs), 32):
        chunk = programs[start:start + 32]; states = []
        for positions in chunk:
            state = no_attention.clone(); state[:, -1] += deltas[list(positions)].sum(0); states.append(state)
        logits = suffix(model, layer, torch.cat(states), repeat_batch(x0[:1], len(states)), repeat_batch(row["bus"][:1], len(states)))
        effects.extend((logits @ contrast).sub((suffix(model, layer, no_attention, x0[:1], row["bus"][:1]) @ contrast)).cpu().tolist())
    target = effects[-1]; subset_effects = np.asarray(effects[:-1]); actual_index = candidates.index(edited); actual = float(subset_effects[actual_index])
    denominator = max(target * target, 1e-30)
    fidelities = 1 - (target - subset_effects) ** 2 / denominator
    actual_fidelity = float(fidelities[actual_index])
    percentile = float(np.mean(fidelities <= actual_fidelity))
    return {
        "base_id": pair["base"].example_id, "source_id": pair["source"].example_id,
        "tokens": [tokenizer.decode([token]) for token in pair["base_ids"]],
        "source_tokens": [tokenizer.decode([token]) for token in pair["source_ids"]],
        "edited_positions": list(edited), "candidate_subsets": len(candidates),
        "full_program_target": target, "actual_effect": actual,
        "actual_projection_share": actual / target if abs(target) > 1e-30 else None,
        "actual_energy_fidelity": actual_fidelity, "percentile": percentile,
        "null_median_fidelity": float(np.median(fidelities)),
        "null_p90_fidelity": float(np.quantile(fidelities, .9)),
        "best_subset": list(candidates[int(np.argmax(fidelities))]),
        "best_fidelity": float(fidelities.max()),
    }


def main():
    config=json.loads(CONFIG.read_text()); source=json.loads(Path(config["source"]).read_text()); materiality=json.loads(Path(config["materiality_source"]).read_text())
    tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True); selected=reconstruct_selected(tokenizer,source)
    model=load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding,model.blocks,model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    results={}
    for family,row in materiality["results"].items():
        if not row["material_attention"]: continue
        layer=source["head_summary"][family]["site"]; program=source["head_summary"][family]["head_order"][:3]
        examples=[]
        for pair in selected[family]["heldout"]:
            result=evaluate_pair(model,tokenizer,pair,layer,program)
            if result is not None: examples.append(result)
        results[family]={
            "layer":layer,"program":program,"examples":examples,
            "median_percentile":float(np.median([item["percentile"] for item in examples])) if examples else None,
            "median_projection_share":float(np.median([item["actual_projection_share"] for item in examples])) if examples else None,
            "median_fidelity":float(np.median([item["actual_energy_fidelity"] for item in examples])) if examples else None,
        }
        print(f"position null {family} percentile={results[family]['median_percentile']}",flush=True)
    high=[family for family,row in results.items() if row["median_percentile"] is not None and row["median_percentile"]>=.9]
    low=[family for family,row in results.items() if row["median_percentile"] is not None and row["median_percentile"]<.5]
    hypotheses={
        "P0_edited_sources_are_exceptional_for_direct_read_programs":{"supported":len(high)>=3,"families":high},
        "P1_direct_read_and_relay_regimes_separate":{"supported":bool(low) and len(high)>=3,"direct":high,"relay":low},
        "P2_retrieval_payload_positions_are_specific":{"supported":results["repeated_sequence"]["median_percentile"]>=.9,"observed":results["repeated_sequence"]["median_percentile"]},
        "P3_factual_recall_cue_is_specific":{"supported":results["capital_recall"]["median_percentile"]>=.9,"observed":results["capital_recall"]["median_percentile"]},
        "P4_concrete_position_evidence_is_saved":{"supported":all(all({"tokens","source_tokens","edited_positions","actual_effect","full_program_target","null_median_fidelity"}<=set(item) for item in row["examples"]) for row in results.values())},
    }
    for row in hypotheses.values(): row["supported"]=bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"results":results},indent=2,allow_nan=False))
    lines=["# Edited-source equal-count subset red team","","For each held-out pair, the finite effect of the actual edited source positions is compared with every equal-count subset of non-destination source positions. The output metric is fidelity to the complete frozen top-three program effect.","","## Locked results",""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name,row in hypotheses.items())
    lines += ["","| Family | Site/program | Examples | Edited projection | Edited fidelity | Equal-count percentile |","|---|---|---:|---:|---:|---:|"]
    for family,row in results.items(): lines.append(f"| {family} | L{row['layer']} " + ",".join(f"H{h}" for h in row["program"]) + f" | {len(row['examples'])} | {row['median_projection_share']:.3f} | {row['median_fidelity']:.3f} | {row['median_percentile']:.3f} |")
    lines += ["","## Concrete held-out examples",""]
    for family,row in results.items():
        lines += [f"### {family}",""]
        for item in row["examples"][:3]:
            before="".join(item["tokens"]); after="".join(item["source_tokens"])
            lines.append(f"- `{before}` → `{after}`; edited={item['edited_positions']}; effect={item['actual_effect']:.3f}/{item['full_program_target']:.3f}; percentile={item['percentile']:.3f}; best subset={item['best_subset']}.")
        lines.append("")
    lines += ["## Guardrail","","A high percentile shows that the edited source locations are unusually sufficient for this frozen program, not that their internal vectors have a unique semantic label. A low percentile can indicate a relayed representation rather than failure of the overall computation."]
    REPORT.write_text("\n".join(lines)+"\n"); print(json.dumps(hypotheses,indent=2)); print(REPORT)


if __name__=="__main__": main()
