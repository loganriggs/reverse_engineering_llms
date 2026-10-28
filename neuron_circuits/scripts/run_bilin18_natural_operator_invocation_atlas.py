#!/usr/bin/env python3
"""Search unlabeled FineWeb for invocations of frozen typed read operators."""

from __future__ import annotations

import heapq
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_natural_operator_invocation_atlas.json")
OUTPUT = TENSOR_ROOT / "bilin18_natural_operator_invocation_atlas.json"
REPORT = TENSOR_ROOT / "BILIN18_NATURAL_OPERATOR_INVOCATION_ATLAS.md"


def cosine_rows(routes, template):
    template = template / template.norm().clamp_min(1e-30)
    return (routes * template).sum(-1) / routes.norm(dim=-1).clamp_min(1e-30)


def push(heap, score, record, limit):
    item = (float(score), int(record["global_row"]), int(record["destination"]), int(record["source"]), record)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item[:4] > heap[0][:4]:
        heapq.heapreplace(heap, item)


def clean(tokenizer, token):
    return tokenizer.decode([int(token)]).replace("\n", "\\n").replace("\t", "\\t")


def context_windows(tokens, source, destination, tokenizer):
    if destination - source <= 30:
        text = tokenizer.decode(tokens[max(0, source - 6):min(len(tokens), destination + 3)]).replace("\n", "\\n")
        return text, text
    source_text = tokenizer.decode(tokens[max(0, source - 6):min(len(tokens), source + 7)]).replace("\n", "\\n")
    destination_text = tokenizer.decode(tokens[max(0, destination - 8):min(len(tokens), destination + 3)]).replace("\n", "\\n")
    return source_text, destination_text


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text()); operator = json.loads(Path(config["operator_source"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = np.load(config["corpus"], mmap_mode="r")[config["start_row"]:config["start_row"] + config["rows"], :config["sequence_length"]]
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    families = list(operator["results"])
    specifications = {}
    by_layer = {}
    for family in families:
        row = operator["results"][family]
        route = .5 * (np.asarray(row["route0"]) + np.asarray(row["route1"]))
        specifications[family] = {"layer": row["layer"], "program": row["program"], "route": torch.as_tensor(route, device="cuda", dtype=torch.float32)}
        by_layer[row["layer"]] = family
    heaps = {(family, role): [] for family, spec in specifications.items() for role in range(spec["route"].shape[1])}
    control_heaps = {(family, role, permutation): [] for family, spec in specifications.items() for role in range(spec["route"].shape[1]) for permutation in range(5)}
    permutations = {}
    for family, spec in specifications.items():
        permutations[family] = [perm for perm in itertools.permutations(range(3)) if perm != (0, 1, 2)]

    dimensions = model.config.n_embd; heads = model.config.n_head; head_dim = dimensions // heads
    destinations = torch.as_tensor(config["destinations"], device="cuda")
    for start in range(0, len(corpus), config["batch_size"]):
        raw = np.asarray(corpus[start:start + config["batch_size"]], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        x0 = F.rms_norm(model.embedding(ids), (dimensions,)); x, bus = x0, None
        for layer, block in enumerate(model.blocks):
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            details = attention_details(block.attn, F.rms_norm(mixed, (dimensions,)), bus, model.config, model.tt)
            bus = details["value_bus"]
            if layer in by_layer:
                family = by_layer[layer]; spec = specifications[family]; program = spec["program"]
                pattern = details["pattern"][:, program][:, :, destinations].float().permute(0, 2, 3, 1)
                values = details["mixed_value"][:, :, program].float()
                projection = block.attn.c_proj.weight.float().T.reshape(heads, head_dim, dimensions)[program]
                source_writes = torch.einsum("bshd,hdk->bshk", values, projection)
                write_norm = source_writes.norm(dim=-1)
                causal = torch.arange(raw.shape[1], device="cuda")[None, None, :] <= destinations[None, :, None]
                for role in range(spec["route"].shape[1]):
                    template = spec["route"][:, role]
                    cosine = cosine_rows(pattern, template)
                    approximate_norm = (pattern.square() * write_norm[:, None].square()).sum(-1).sqrt()
                    score = cosine.clamp_min(0) * approximate_norm
                    score = score.masked_fill(~causal, -torch.inf)
                    flat = score.reshape(-1); amount = min(config["batch_candidates_per_role"], flat.numel())
                    values_top, indices = torch.topk(flat, amount)
                    for value_score, flat_index in zip(values_top.cpu().tolist(), indices.cpu().tolist()):
                        batch_index, remainder = divmod(flat_index, len(config["destinations"]) * raw.shape[1])
                        destination_index, source = divmod(remainder, raw.shape[1]); destination = config["destinations"][destination_index]
                        route_vector = pattern[batch_index, destination_index, source]
                        edge = torch.einsum("h,hk->k", route_vector, source_writes[batch_index, source])
                        output_score = F.normalize(model.lm_head.weight[:tokenizer.vocab_size].float(), dim=1) @ F.normalize(edge, dim=0)
                        positive = torch.topk(output_score, config["decode_tokens"]).indices.cpu().tolist()
                        negative = torch.topk(-output_score, config["decode_tokens"]).indices.cpu().tolist()
                        source_text, destination_text = context_windows(raw[batch_index].tolist(), source, destination, tokenizer)
                        record = {
                            "global_row": config["start_row"] + start + batch_index,
                            "source": source, "destination": destination, "distance": destination - source,
                            "source_token_id": int(raw[batch_index, source]), "source_token": clean(tokenizer, raw[batch_index, source]),
                            "destination_token": clean(tokenizer, raw[batch_index, destination]),
                            "next_token_id": int(raw[batch_index, destination + 1]), "next_token": clean(tokenizer, raw[batch_index, destination + 1]),
                            "source_equals_next": bool(raw[batch_index, source] == raw[batch_index, destination + 1]),
                            "route": route_vector.cpu().tolist(), "route_cosine": float(cosine[batch_index, destination_index, source]),
                            "match_score": value_score, "edge_norm": float(edge.norm()),
                            "output_positive": [clean(tokenizer, token) for token in positive],
                            "output_negative": [clean(tokenizer, token) for token in negative],
                            "source_context": source_text, "destination_context": destination_text,
                        }
                        push(heaps[(family, role)], value_score, record, config["top_examples_per_role"])
                    for permutation_index, permutation in enumerate(permutations[family]):
                        control_template = template[torch.as_tensor(permutation, device="cuda")]
                        control_cosine = cosine_rows(pattern, control_template)
                        control_score = (control_cosine.clamp_min(0) * approximate_norm).masked_fill(~causal, -torch.inf)
                        control_top = torch.topk(control_score.reshape(-1), amount).values.cpu().tolist()
                        for value_score in control_top:
                            control_heap = control_heaps[(family, role, permutation_index)]
                            if len(control_heap) < config["top_examples_per_role"]:
                                heapq.heappush(control_heap, value_score)
                            elif value_score > control_heap[0]:
                                heapq.heapreplace(control_heap, value_score)
            pre = mixed + details["write"]
            normalized = F.rms_norm(pre, (dimensions,))
            x = pre + block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
        if start % 64 == 0: print(f"natural operator rows {start}/{len(corpus)}", flush=True)

    results = {}; all_nonrepeat_copy = []
    for family, spec in specifications.items():
        role_rows = []
        for role in range(spec["route"].shape[1]):
            examples = [item[-1] for item in sorted(heaps[(family, role)], reverse=True)]
            scores = np.asarray([row["match_score"] for row in examples])
            cosines = np.asarray([row["route_cosine"] for row in examples])
            controls = []
            for permutation_index in range(len(permutations[family])):
                values_control = sorted(control_heaps[(family, role, permutation_index)], reverse=True)
                controls.append(float(np.median(values_control)))
            copy_rate = float(np.mean([row["source_equals_next"] for row in examples]))
            if family != "repeated_sequence": all_nonrepeat_copy.extend(row["source_equals_next"] for row in examples)
            role_rows.append({
                "role": role, "examples": examples,
                "median_score": float(np.median(scores)), "median_cosine": float(np.median(cosines)),
                "distinct_source_tokens": len(set(row["source_token_id"] for row in examples)),
                "copy_rate": copy_rate,
                "best_permuted_median_score": max(controls),
                "native_over_best_permuted": float(np.median(scores) / max(max(controls), 1e-30)),
            })
        results[family] = {"layer": spec["layer"], "program": spec["program"], "roles": role_rows}
    all_roles = [role for row in results.values() for role in row["roles"]]
    natural_match = all(len(role["examples"]) >= 32 and role["median_cosine"] >= .9 for role in all_roles)
    typed_families = [family for family, row in results.items() if min(role["native_over_best_permuted"] for role in row["roles"]) >= 1.05]
    diverse = all(role["distinct_source_tokens"] >= 16 for role in all_roles)
    repeat_copy = float(np.mean([role["copy_rate"] for role in results["repeated_sequence"]["roles"]]))
    nonrepeat_copy = float(np.mean(all_nonrepeat_copy))
    complete = all(all({"source_context", "destination_context", "route", "match_score", "output_positive", "output_negative"} <= set(example) for example in role["examples"]) for role in all_roles)
    hypotheses = {
        "N0_natural_matches_exist": {"supported": natural_match, "median_cosines": {f"{family}:{role['role']}": role["median_cosine"] for family, row in results.items() for role in row["roles"]}},
        "N1_native_head_typing_beats_permuted_templates": {"supported": len(typed_families) >= 3, "families": typed_families, "ratios": {f"{family}:{role['role']}": role["native_over_best_permuted"] for family, row in results.items() for role in row["roles"]}},
        "N2_natural_invocations_are_not_one_token_artifacts": {"supported": diverse, "distinct": {f"{family}:{role['role']}": role["distinct_source_tokens"] for family, row in results.items() for role in row["roles"]}},
        "N3_repetition_template_enriches_copy_relations": {"supported": repeat_copy > nonrepeat_copy, "repetition_copy_rate": repeat_copy, "other_copy_rate": nonrepeat_copy},
        "N4_examples_are_saved_for_semantic_redteam": {"supported": complete},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Natural FineWeb invocation atlas for typed read operators", "", "Frozen route templates from four controlled behaviors are scanned over unlabeled FineWeb edges. Ranking uses route-template cosine times a weight-aware edge-magnitude proxy. No task labels or answer contrasts enter the search. Every nonidentity permutation of route coefficients across the same three heads is a matched control.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Template | Site/program | Role | Median cosine | Native/permuted score | Distinct sources | Copy-next rate |", "|---|---|---:|---:|---:|---:|---:|"]
    for family, row in results.items():
        program = ",".join(f"H{x}" for x in row["program"])
        for role in row["roles"]:
            lines.append(f"| {family} | L{row['layer']} {program} | {role['role']} | {role['median_cosine']:.3f} | {role['native_over_best_permuted']:.3f} | {role['distinct_source_tokens']} | {role['copy_rate']:.3f} |")
    lines += ["", "## Representative unlabeled matches", ""]
    for family, row in results.items():
        lines += [f"### {family}", ""]
        for role in row["roles"]:
            for example in role["examples"][:5]:
                context = example["source_context"] if example["source_context"] == example["destination_context"] else example["source_context"] + " … " + example["destination_context"]
                outputs = ", ".join(f"`{token}`" for token in example["output_positive"][:4])
                lines.append(f"- role {role['role']}, d={example['distance']}, cos={example['route_cosine']:.3f}: `{context}`; source `{example['source_token']}` → next `{example['next_token']}`; direct outputs {outputs}.")
        lines.append("")
    lines += ["## Guardrail", "", "A high route-template match establishes recurrent tensor geometry, not the original task semantics. The saved examples are the discovery set for later semantic grouping; any labels inferred from them must be frozen and tested on disjoint corpus rows."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
