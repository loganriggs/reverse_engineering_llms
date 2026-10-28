#!/usr/bin/env python3
"""Distinguish contrast decodability from operational task competence."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_broad_task_competence import (
    TENSOR_ROOT, continuation_id, declared_probes, final_logits,
)
from run_bilin18_qkov_causal_edges import load_full_model


CONFIG = Path("configs/bilin18_broad_task_competence_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_broad_task_competence_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_BROAD_TASK_COMPETENCE_REDTEAM.md"


SENTIMENT_CLASSES = {
    " positive": (" positive", " good", " Good", " great", " Great", " excellent", " favorable"),
    " negative": (" negative", " bad", " Bad", " poor", " Poor", " terrible", " unfavorable"),
}
AGREEMENT_CLASSES = {
    " is": (" is", " was", " has", " does"),
    " are": (" are", " were", " have", " do"),
}


def semantic_classes(item):
    if item.family in {"sentiment_classification", "negated_sentiment"}:
        return [SENTIMENT_CLASSES[candidate] for candidate in item.candidates]
    if item.family in {"agreement_local", "agreement_attractor"}:
        return [AGREEMENT_CLASSES[candidate] for candidate in item.candidates]
    return [(candidate,) for candidate in item.candidates]


def class_token_ids(tokenizer, item):
    groups = []
    for group in semantic_classes(item):
        ids = []
        for continuation in group:
            token_id = continuation_id(tokenizer, item.prompt, continuation)
            if token_id is not None and token_id not in ids:
                ids.append(token_id)
        groups.append(ids)
    return groups


def permutation_p(scores, labels, draws, seed):
    observed = float(np.mean(np.argmax(scores, axis=1) == labels))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(draws):
        permuted = rng.permutation(labels)
        exceed += float(np.mean(np.argmax(scores, axis=1) == permuted)) >= observed
    return (exceed + 1) / (draws + 1)


def main():
    config = json.loads(CONFIG.read_text())
    source = json.loads(Path(config["source"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    source_examples = {row["example_id"]: row for row in source["examples"]}
    items = [item for item in declared_probes() if item.example_id in source_examples]

    model = load_full_model("cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    results = []
    by_length = defaultdict(list)
    for item in items:
        prompt_ids = tokenizer.encode(item.prompt, add_special_tokens=False)
        groups = class_token_ids(tokenizer, item)
        if any(not group for group in groups):
            continue
        by_length[len(prompt_ids)].append((item, prompt_ids, groups))
    for length, rows in sorted(by_length.items()):
        for start in range(0, len(rows), 16):
            batch = rows[start:start + 16]
            ids = torch.as_tensor([row[1] for row in batch], device="cuda")
            logits = final_logits(model, ids).cpu()
            for (item, prompt_ids, groups), values in zip(batch, logits):
                class_scores = torch.stack([torch.logsumexp(values[group], dim=0) for group in groups])
                correct = item.candidates.index(item.answer)
                other = torch.cat((class_scores[:correct], class_scores[correct + 1:]))
                best_correct_token = float(values[groups[correct]].max())
                predicted = int(values.argmax())
                results.append({
                    "example_id": item.example_id, "family": item.family,
                    "regime": item.regime, "answer": item.answer,
                    "class_scores": class_scores.tolist(), "answer_index": correct,
                    "semantic_choice_correct": int(class_scores.argmax()) == correct,
                    "semantic_margin": float(class_scores[correct] - other.max()),
                    "semantic_top1_correct": predicted in groups[correct],
                    "semantic_rank": int((values > best_correct_token).sum()) + 1,
                    "predicted_text": tokenizer.decode([predicted]),
                    "literal_top1_correct": source_examples[item.example_id]["full_vocab_top1"],
                })

    summaries = {}
    for family in sorted(source["summary"]):
        rows = [row for row in results if row["family"] == family]
        if not rows:
            summaries[family] = {"examples": 0, "operationally_competent": False}
            continue
        scores = np.asarray([row["class_scores"] for row in rows])
        labels = np.asarray([row["answer_index"] for row in rows])
        accuracy = float(np.mean([row["semantic_choice_correct"] for row in rows]))
        chance = 1 / scores.shape[1]
        normalized = (accuracy - chance) / (1 - chance)
        top1 = float(np.mean([row["semantic_top1_correct"] for row in rows]))
        literal_top1 = float(np.mean([row["literal_top1_correct"] for row in rows]))
        rank = float(np.median([row["semantic_rank"] for row in rows]))
        margin = float(np.median([row["semantic_margin"] for row in rows]))
        operational = (
            len(rows) >= config["minimum_examples"]
            and normalized >= config["normalized_choice_gate"]
            and margin > 0
            and (top1 >= config["semantic_top1_gate"] or rank <= config["median_semantic_rank_gate"])
        )
        summaries[family] = {
            "examples": len(rows), "regime": rows[0]["regime"],
            "candidate_gate": source["summary"][family]["competent"],
            "semantic_choice_accuracy": accuracy,
            "normalized_semantic_accuracy": normalized,
            "median_semantic_margin": margin,
            "semantic_top1_accuracy": top1,
            "literal_top1_accuracy": literal_top1,
            "median_semantic_rank": rank,
            "permutation_p": permutation_p(scores, labels, config["permutation_draws"], config["seed"] + len(summaries)),
            "operationally_competent": operational,
        }

    operational = [family for family, row in summaries.items() if row.get("operationally_competent")]
    false_positives = [family for family, row in summaries.items() if row.get("candidate_gate") and not row.get("operationally_competent")]
    rescued = [family for family, row in summaries.items() if row.get("semantic_top1_accuracy", 0) >= .5 and row.get("literal_top1_accuracy", 1) < .1]
    regimes = {summaries[family]["regime"] for family in operational}
    hypotheses = {
        "R0_candidate_choice_overstates_operational_competence": {"supported": bool(false_positives), "families": false_positives},
        "R1_semantic_output_sets_rescue_valid_classifications": {"supported": any(family in {"sentiment_classification", "negated_sentiment"} for family in rescued), "families": rescued},
        "R2_exact_retrieval_requires_exact_output": {"supported": not summaries.get("in_context_key_value", {}).get("operationally_competent", False), "observed": summaries.get("in_context_key_value")},
        "R3_breadth_survives_output_redteam": {"supported": len(operational) >= 5 and len(regimes) >= 4, "families": operational, "regimes": sorted(regimes)},
        "R4_competent_results_beat_label_permutation": {"supported": all(summaries[family]["permutation_p"] <= .05 for family in operational), "p_values": {family: summaries[family]["permutation_p"] for family in operational}},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "summary": summaries, "examples": results}, indent=2))

    lines = [
        "# Broad task competence: semantic-output red team", "",
        "The original screen asked whether the intended answer beat a declared candidate set. This red team additionally requires the intended answer—or a preregistered semantic synonym—to rank near the top of the complete vocabulary. Candidate-only success is called contrast decodability, not operational competence.", "",
        "## Locked results", "",
    ]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**" for name, row in hypotheses.items())
    lines += ["", "| Family | Regime | Candidate gate | Semantic choice | Literal top-1 | Semantic top-1 | Semantic rank | Permutation p | Operational |", "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for family, row in summaries.items():
        if not row.get("examples"):
            lines.append(f"| {family} | — | — | — | — | — | — | — | no |")
            continue
        lines.append(f"| {family} | {row['regime']} | {'yes' if row['candidate_gate'] else 'no'} | {row['semantic_choice_accuracy']:.3f} | {row['literal_top1_accuracy']:.3f} | {row['semantic_top1_accuracy']:.3f} | {row['median_semantic_rank']:.1f} | {row['permutation_p']:.4f} | {'yes' if row['operationally_competent'] else 'no'} |")
    lines += ["", "## Consequence", "", "The next circuit phase may use operational families directly. Candidate-only families remain useful for studying latent contrasts, but cannot be described as tasks the model actually executes without an explicit output quotient."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUTPUT} and {REPORT}")


if __name__ == "__main__":
    main()
