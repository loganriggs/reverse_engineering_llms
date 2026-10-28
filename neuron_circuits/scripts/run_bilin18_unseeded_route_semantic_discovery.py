#!/usr/bin/env python3
"""Decode fresh natural examples for strictly selected unseeded route modes."""

from __future__ import annotations

import heapq
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_natural_operator_invocation_atlas import clean, context_windows
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_unseeded_route_semantic_discovery.json")
OUTPUT = TENSOR_ROOT / "bilin18_unseeded_route_semantic_discovery.json"
REPORT = TENSOR_ROOT / "BILIN18_UNSEEDED_ROUTE_SEMANTIC_DISCOVERY.md"


def select_candidates(geometry, redteam, config):
    separated = set(redteam["hypotheses"]["R1_k4_centers_are_well_separated"]["layers"])
    rows = []
    for layer, candidates in geometry["clusters"].items():
        for row in candidates:
            if (int(layer) in separated
                    and row["center_cosine"] >= config["minimum_center_cosine"]
                    and min(row["discovery_size"], row["replication_size"]) >= config["minimum_cluster_size"]
                    and min(row["discovery_member_cosine"], row["replication_member_cosine"]) >= config["minimum_member_cosine"]
                    and min(row["discovery_margin"], row["replication_margin"]) >= config["minimum_margin"]):
                value = dict(row)
                value["selection_score"] = value["center_cosine"] * min(value["discovery_member_cosine"], value["replication_member_cosine"]) * (1 + min(value["discovery_margin"], value["replication_margin"]))
                rows.append(value)
    rows.sort(key=lambda row: row["selection_score"], reverse=True)
    return rows[:config["maximum_candidates"]]


def push(heap, score, serial, record, limit):
    item = (float(score), serial, record)
    if len(heap) < limit: heapq.heappush(heap, item)
    elif score > heap[0][0]: heapq.heapreplace(heap, item)


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text()); geometry = json.loads(Path(config["geometry_source"]).read_text()); redteam = json.loads(Path(config["redteam_source"]).read_text())
    selected = select_candidates(geometry, redteam, config)
    by_layer = {}
    for index, row in enumerate(selected):
        candidate = {**row, "candidate": index, "name": f"L{row['layer']}C{row['cluster']}"}
        by_layer.setdefault(row["layer"], []).append(candidate)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = np.load(config["corpus"], mmap_mode="r")[config["start_row"]:config["start_row"] + config["rows"], :config["sequence_length"]]
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    heaps = {candidate["name"]: [] for rows in by_layer.values() for candidate in rows}; serial = 0
    dimension = model.config.n_embd; heads = model.config.n_head; head_dim = dimension // heads
    destinations = torch.as_tensor(config["destinations"], device="cuda")
    for start in range(0, len(corpus), config["batch_size"]):
        raw = np.asarray(corpus[start:start + config["batch_size"]], dtype=np.int64); ids = torch.as_tensor(raw, device="cuda")
        x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
        for layer, block in enumerate(model.blocks):
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt); bus = details["value_bus"]
            if layer in by_layer:
                pattern = details["pattern"][:, :, destinations].float().permute(0, 2, 3, 1)
                routes = F.normalize(pattern, dim=-1)
                projection = block.attn.c_proj.weight.float().T.reshape(heads, head_dim, dimension)
                source_writes = torch.einsum("bshd,hdk->bshk", details["mixed_value"].float(), projection)
                write_norm = source_writes.norm(dim=-1)
                approximate = (pattern.square() * write_norm[:, None].square()).sum(-1).sqrt()
                causal = torch.arange(raw.shape[1], device="cuda")[None, None, :] <= destinations[None, :, None]
                for candidate in by_layer[layer]:
                    center = torch.as_tensor(candidate["center"], device="cuda", dtype=torch.float32)
                    cosine = (routes * center).sum(-1)
                    score = (cosine.clamp_min(0) * approximate).masked_fill(~causal, -torch.inf)
                    amount = min(config["batch_candidates_per_operator"], score.numel())
                    values, indices = torch.topk(score.reshape(-1), amount)
                    for match_score, flat_index in zip(values.cpu().tolist(), indices.cpu().tolist()):
                        batch_index, remainder = divmod(flat_index, len(config["destinations"]) * raw.shape[1])
                        destination_index, source = divmod(remainder, raw.shape[1]); destination = config["destinations"][destination_index]
                        route = pattern[batch_index, destination_index, source]
                        edge = torch.einsum("h,hk->k", route, source_writes[batch_index, source])
                        output_score = F.normalize(model.lm_head.weight[:tokenizer.vocab_size].float(), dim=1) @ F.normalize(edge, dim=0)
                        positive = torch.topk(output_score, config["decode_tokens"]).indices.cpu().tolist(); negative = torch.topk(-output_score, config["decode_tokens"]).indices.cpu().tolist()
                        source_text, destination_text = context_windows(raw[batch_index].tolist(), source, destination, tokenizer)
                        record = {
                            "global_row": config["start_row"] + start + batch_index, "source": source, "destination": destination, "distance": destination - source,
                            "source_token_id": int(raw[batch_index, source]), "source_token": clean(tokenizer, raw[batch_index, source]),
                            "destination_token_id": int(raw[batch_index, destination]), "destination_token": clean(tokenizer, raw[batch_index, destination]),
                            "next_token_id": int(raw[batch_index, destination + 1]), "next_token": clean(tokenizer, raw[batch_index, destination + 1]),
                            "source_equals_next": bool(raw[batch_index, source] == raw[batch_index, destination + 1]),
                            "source_equals_destination": bool(raw[batch_index, source] == raw[batch_index, destination]),
                            "route_cosine": float(cosine[batch_index, destination_index, source]), "match_score": match_score, "edge_norm": float(edge.norm()),
                            "output_positive": [clean(tokenizer, token) for token in positive], "output_negative": [clean(tokenizer, token) for token in negative],
                            "source_context": source_text, "destination_context": destination_text,
                        }
                        push(heaps[candidate["name"]], match_score, serial, record, config["examples_per_operator"]); serial += 1
            pre = mixed + details["write"]; normalized = F.rms_norm(pre, (dimension,))
            x = pre + block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
        if start % 64 == 0: print(f"unseeded semantic rows {start}/{len(corpus)}", flush=True)
    results = []
    selected_by_name = {f"L{row['layer']}C{row['cluster']}": row for row in selected}
    for name, heap in heaps.items():
        examples = [item[-1] for item in sorted(heap, reverse=True)]; row = selected_by_name[name]
        results.append({
            "name": name, "layer": row["layer"], "cluster": row["cluster"], "center": row["center"],
            "geometry": {key: row[key] for key in ("center_cosine", "discovery_size", "replication_size", "discovery_member_cosine", "replication_member_cosine", "effective_support", "discovery_margin", "replication_margin")},
            "examples": examples, "distinct_source_tokens": len(set(item["source_token_id"] for item in examples)),
            "median_distance": float(np.median([item["distance"] for item in examples])),
            "copy_rate": float(np.mean([item["source_equals_next"] for item in examples])),
            "self_rate": float(np.mean([item["source_equals_destination"] for item in examples])),
            "median_route_cosine": float(np.median([item["route_cosine"] for item in examples])),
        })
    results.sort(key=lambda row: next(i for i, item in enumerate(selected) if item["layer"] == row["layer"] and item["cluster"] == row["cluster"]))
    broad_layers = len(set(row["layer"] for row in results))
    complete = all(all({"source_context", "destination_context", "route_cosine", "match_score", "output_positive", "output_negative"} <= set(item) for item in row["examples"]) for row in results)
    hypotheses = {
        "D0_selection_is_label_blind": {"supported": True, "selection_fields": ["center_cosine", "cluster_size", "member_cosine", "assignment_margin", "selection_score"]},
        "D1_candidate_set_is_broad": {"supported": len(results) >= 12 and broad_layers >= 6, "candidates": len(results), "layers": broad_layers},
        "D2_natural_examples_are_diverse": {"supported": all(row["distinct_source_tokens"] >= 16 for row in results), "distinct": {row["name"]: row["distinct_source_tokens"] for row in results}},
        "D3_examples_support_prospective_labels": {"supported": complete},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Task-agnostic semantic discovery for natural route modes", "", "Candidates were frozen from two token-blind geometry panels and a discreteness red team. Only high-margin, well-separated modes enter this fresh semantic discovery panel. The examples below may motivate labels, but those labels require another disjoint validation panel.", "", "## Locked gates", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Candidate | Effective heads | Margin D/R | Median route cosine | Median distance | Copy | Self | Distinct sources |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        g = row["geometry"]
        lines.append(f"| {row['name']} | {g['effective_support']:.2f} | {g['discovery_margin']:.3f}/{g['replication_margin']:.3f} | {row['median_route_cosine']:.3f} | {row['median_distance']:.1f} | {row['copy_rate']:.3f} | {row['self_rate']:.3f} | {row['distinct_source_tokens']} |")
    lines += ["", "## Representative discovery examples", ""]
    for row in results:
        lines += [f"### {row['name']}", ""]
        for example in row["examples"][:6]:
            context = example["source_context"] if example["source_context"] == example["destination_context"] else example["source_context"] + " … " + example["destination_context"]
            outputs = ", ".join(f"`{token}`" for token in example["output_positive"][:4])
            lines.append(f"- d={example['distance']}, cos={example['route_cosine']:.3f}: `{context}`; source `{example['source_token']}` → next `{example['next_token']}`; outputs {outputs}.")
        lines.append("")
    lines += ["## Guardrail", "", "A candidate can be a coherent route mode without admitting a short human semantic label. Unexplained candidates remain valid geometric objects; broad labels will be accepted only if frozen here and replicated on fresh rows."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
