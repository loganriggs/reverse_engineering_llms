#!/usr/bin/env python3
"""Screen a broad computation suite on the FineWeb-only Bilin18 checkpoint.

This is a competence and tokenization assay, not a circuit-localization assay.
Only families passing the frozen behavioral gate become eligible for later
mechanistic comparisons.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
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
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_broad_task_competence.json")
OUTPUT = TENSOR_ROOT / "bilin18_broad_task_competence.json"
REPORT = TENSOR_ROOT / "BILIN18_BROAD_TASK_COMPETENCE.md"


@dataclass(frozen=True)
class Probe:
    example_id: str
    family: str
    regime: str
    prompt: str
    answer: str
    candidates: tuple[str, ...]
    template: str


NOUNS = (
    ("key", "keys"), ("dog", "dogs"), ("cat", "cats"),
    ("report", "reports"), ("picture", "pictures"),
    ("bottle", "bottles"), ("tree", "trees"), ("book", "books"),
)
NAMES = ("John", "Mary", "Alice", "Bob", "Carol", "Dave", "Sarah", "Tom")
CAPITALS = (
    ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"),
    ("Japan", "Tokyo"), ("China", "Beijing"), ("Russia", "Moscow"),
    ("Spain", "Madrid"), ("Portugal", "Lisbon"),
    ("Austria", "Vienna"), ("Greece", "Athens"),
)
CURRENCIES = (
    ("Japan", "yen"), ("Britain", "pound"), ("Europe", "euro"),
    ("India", "rupee"), ("Russia", "ruble"), ("China", "yuan"),
    ("Korea", "won"), ("America", "dollar"),
)
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
COLORS = ("red", "blue", "green", "yellow", "black", "white", "orange", "purple")


def probe(example_id, family, regime, prompt, answer, candidates, template):
    rendered = tuple(f" {candidate}" for candidate in candidates)
    return Probe(example_id, family, regime, prompt, f" {answer}", rendered, template)


def declared_probes() -> list[Probe]:
    rows: list[Probe] = []
    agreement = ("is", "are")
    for singular, plural in NOUNS:
        rows.append(probe(f"agr-local-{singular}", "agreement_local", "lexical", f"The {singular}", "is", agreement, "bare_np"))
        rows.append(probe(f"agr-local-{plural}", "agreement_local", "lexical", f"The {plural}", "are", agreement, "bare_np"))
        for other_singular, other_plural in NOUNS:
            if singular == other_singular:
                continue
            rows.append(probe(f"agr-attr-{singular}-{other_plural}", "agreement_attractor", "binding", f"The {singular} near the {other_plural}", "is", agreement, "near_pp"))
            rows.append(probe(f"agr-attr-{plural}-{other_singular}", "agreement_attractor", "binding", f"The {plural} near the {other_singular}", "are", agreement, "near_pp"))

    for first_index, first in enumerate(NAMES):
        for second in NAMES[first_index + 1:]:
            candidates = (first, second)
            rows.append(probe(f"ioi-{first}-{second}-{first}", "ioi", "binding", f"{first} and {second} went to the store. {first} gave the book to", second, candidates, "gave"))
            rows.append(probe(f"ioi-{first}-{second}-{second}", "ioi", "binding", f"{first} and {second} went to the store. {second} gave the book to", first, candidates, "gave"))

    bracket_candidates = (")", "]", "}")
    for opener, closer in (("(", ")"), ("[", "]"), ("{", "}")):
        for inner in ("x", "x + y", "value", "[x]", "{x}", "(x)"):
            rows.append(probe(f"bracket-{ord(opener)}-{inner}", "bracket_instruction", "formal", f"Complete the unmatched bracket: {opener}{inner}", closer, bracket_candidates, "instruction"))

    digits = tuple(str(value) for value in range(10))
    for left in range(10):
        for right in range(10 - left):
            rows.append(probe(f"add-{left}-{right}", "arithmetic_add", "algorithmic", f"What is {left} + {right}? Answer:", str(left + right), digits, "question"))
    for values in ((2, 8, 5, 1), (7, 3, 9, 4), (1, 6, 4, 8), (9, 5, 2, 7), (3, 0, 8, 6), (4, 9, 1, 5), (6, 2, 7, 0), (8, 4, 3, 9), (5, 1, 9, 2), (0, 7, 3, 6), (9, 2, 5, 4), (1, 8, 6, 3)):
        text = ", ".join(map(str, values))
        rows.append(probe(f"max-{'-'.join(map(str, values))}", "list_maximum", "algorithmic", f"The maximum of {text} is", str(max(values)), digits, "declarative"))
        rows.append(probe(f"min-{'-'.join(map(str, values))}", "list_minimum", "algorithmic", f"The minimum of {text} is", str(min(values)), digits, "declarative"))

    capital_answers = tuple(value for _, value in CAPITALS)
    for country, capital in CAPITALS:
        rows.append(probe(f"capital-{country}", "capital_recall", "factual_memory", f"The capital of {country} is", capital, capital_answers, "declarative"))
    currency_answers = tuple(value for _, value in CURRENCIES)
    for country, currency in CURRENCIES:
        rows.append(probe(f"currency-{country}", "currency_recall", "factual_memory", f"The currency of {country} is the", currency, currency_answers, "declarative"))

    for index, day in enumerate(WEEKDAYS):
        rows.append(probe(f"weekday-{day}", "weekday_successor", "structured_recall", f"The day after {day} is", WEEKDAYS[(index + 1) % len(WEEKDAYS)], WEEKDAYS, "declarative"))
    for index, month in enumerate(MONTHS):
        rows.append(probe(f"month-{month}", "month_successor", "structured_recall", f"The month after {month} is", MONTHS[(index + 1) % len(MONTHS)], MONTHS, "declarative"))

    for first_index, key1 in enumerate(CAPITALS[:8]):
        key2 = CAPITALS[(first_index + 3) % 8]
        candidates = (key1[1], key2[1])
        rows.append(probe(f"kv-{key1[0]}-{key2[0]}-a", "in_context_key_value", "retrieval", f"{key1[0]}: {key1[1]}; {key2[0]}: {key2[1]}; {key1[0]}:", key1[1], candidates, "two_entry"))
        rows.append(probe(f"kv-{key1[0]}-{key2[0]}-b", "in_context_key_value", "retrieval", f"{key1[0]}: {key1[1]}; {key2[0]}: {key2[1]}; {key2[0]}:", key2[1], candidates, "two_entry"))

    for index, first in enumerate(COLORS):
        second = COLORS[(index + 1) % len(COLORS)]
        distractor = COLORS[(index + 3) % len(COLORS)]
        rows.append(probe(f"induction-{first}-{second}", "repeated_bigram", "retrieval", f"{first} {second}; {distractor} {first}; {first}", second, COLORS, "repeated_key"))
        rows.append(probe(f"copy-{first}-{second}", "repeated_sequence", "retrieval", f"{first} {second}; {first} {second}; {first}", second, COLORS, "repeated_pair"))

    sentiment = ("positive", "negative")
    positive = ("wonderful", "excellent", "delightful", "enjoyable", "great", "amazing")
    negative = ("terrible", "awful", "dreadful", "boring", "bad", "horrible")
    for word in positive:
        rows.append(probe(f"sent-pos-{word}", "sentiment_classification", "classification", f"Review: The film was {word}. Sentiment:", "positive", sentiment, "plain"))
        rows.append(probe(f"sent-not-pos-{word}", "negated_sentiment", "composition", f"Review: The film was not {word}. Sentiment:", "negative", sentiment, "negated"))
    for word in negative:
        rows.append(probe(f"sent-neg-{word}", "sentiment_classification", "classification", f"Review: The film was {word}. Sentiment:", "negative", sentiment, "plain"))
        rows.append(probe(f"sent-not-neg-{word}", "negated_sentiment", "composition", f"Review: The film was not {word}. Sentiment:", "positive", sentiment, "negated"))

    boolean = ("True", "False")
    rows.extend((
        probe("bool-not-true", "boolean_negation", "formal", "In Python, not True is", "False", boolean, "python"),
        probe("bool-not-false", "boolean_negation", "formal", "In Python, not False is", "True", boolean, "python"),
        probe("bool-true-eq", "boolean_negation", "formal", "In Boolean logic, the negation of true is", "False", boolean, "logic"),
        probe("bool-false-eq", "boolean_negation", "formal", "In Boolean logic, the negation of false is", "True", boolean, "logic"),
    ))
    return rows


def continuation_id(tokenizer, prompt: str, continuation: str) -> int | None:
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    combined = tokenizer.encode(prompt + continuation, add_special_tokens=False)
    if len(combined) != len(prefix) + 1 or combined[:len(prefix)] != prefix:
        return None
    return int(combined[-1])


def tokenize_probe(tokenizer, item: Probe):
    prompt_ids = tokenizer.encode(item.prompt, add_special_tokens=False)
    answer_ids = [continuation_id(tokenizer, item.prompt, candidate) for candidate in item.candidates]
    if any(value is None for value in answer_ids) or len(set(answer_ids)) != len(answer_ids):
        return None
    answer_index = item.candidates.index(item.answer)
    return prompt_ids, answer_ids, answer_index


@torch.inference_mode()
def final_logits(model, ids: torch.Tensor) -> torch.Tensor:
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,))
    x, value_bus = x0, None
    for block in model.blocks:
        x, value_bus = block(x, value_bus, x0)
    return 30 * torch.tanh(model.lm_head(F.rms_norm(x, (dimension,))[:, -1]).float() / 30)


def wilson_lower(successes: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 0.0
    p = successes / total
    denominator = 1 + z * z / total
    center = p + z * z / (2 * total)
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return float(np.clip((center - radius) / denominator, 0.0, 1.0))


def hamming_pair(first, second):
    if len(first["prompt_ids"]) != len(second["prompt_ids"]):
        return None
    return sum(a != b for a, b in zip(first["prompt_ids"], second["prompt_ids"]))


def main():
    config = json.loads(CONFIG.read_text())
    rng = random.Random(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    declared = declared_probes()
    usable = []
    tokenization = defaultdict(lambda: {"declared": 0, "usable": 0})
    for item in declared:
        tokenization[item.family]["declared"] += 1
        encoded = tokenize_probe(tokenizer, item)
        if encoded is None:
            continue
        prompt_ids, candidate_ids, answer_index = encoded
        tokenization[item.family]["usable"] += 1
        usable.append({
            **asdict(item), "prompt_ids": prompt_ids,
            "candidate_ids": candidate_ids, "answer_index": answer_index,
        })

    selected = []
    by_family = defaultdict(list)
    for item in usable:
        by_family[item["family"]].append(item)
    for family in sorted(by_family):
        rows = by_family[family]
        rng.shuffle(rows)
        selected.extend(rows[:config["maximum_examples_per_family"]])

    model = load_full_model("cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    evaluated = []
    by_length = defaultdict(list)
    for item in selected:
        by_length[len(item["prompt_ids"])].append(item)
    for length, rows in sorted(by_length.items()):
        for start in range(0, len(rows), 16):
            batch = rows[start:start + 16]
            ids = torch.as_tensor([row["prompt_ids"] for row in batch], device="cuda")
            logits = final_logits(model, ids).cpu()
            for row, values in zip(batch, logits):
                candidates = torch.as_tensor(row["candidate_ids"])
                candidate_logits = values[candidates]
                correct = row["answer_index"]
                other = torch.cat((candidate_logits[:correct], candidate_logits[correct + 1:]))
                margin = float(candidate_logits[correct] - other.max())
                candidate_log_probs = F.log_softmax(candidate_logits, dim=0)
                full_rank = int((values > candidate_logits[correct]).sum()) + 1
                evaluated.append({
                    **row,
                    "candidate_correct": int(candidate_logits.argmax()) == correct,
                    "candidate_probability": float(candidate_log_probs[correct].exp()),
                    "candidate_margin": margin,
                    "full_vocab_rank": full_rank,
                    "full_vocab_top1": full_rank == 1,
                    "predicted_text": tokenizer.decode([int(values.argmax())]),
                })
        print(f"evaluated length={length} n={len(rows)}", flush=True)

    summaries = {}
    counterfactuals = {}
    for family in sorted(tokenization):
        rows = [row for row in evaluated if row["family"] == family]
        total = len(rows)
        correct = sum(row["candidate_correct"] for row in rows)
        candidates = len(rows[0]["candidate_ids"]) if rows else 0
        chance = 1 / candidates if candidates else 1.0
        accuracy = correct / total if total else 0.0
        normalized = (accuracy - chance) / (1 - chance) if chance < 1 else 0.0
        margin = float(np.median([row["candidate_margin"] for row in rows])) if rows else None
        competent = total >= config["minimum_usable_examples"] and normalized >= config["normalized_accuracy_gate"] and margin is not None and margin > config["median_margin_gate"]
        summaries[family] = {
            **tokenization[family],
            "regime": rows[0]["regime"] if rows else next((item.regime for item in declared if item.family == family), None),
            "candidate_count": candidates,
            "chance_accuracy": chance,
            "candidate_accuracy": accuracy,
            "normalized_accuracy": normalized,
            "wilson_lower": wilson_lower(correct, total),
            "median_candidate_margin": margin,
            "mean_candidate_probability": float(np.mean([row["candidate_probability"] for row in rows])) if rows else None,
            "median_full_vocab_rank": float(np.median([row["full_vocab_rank"] for row in rows])) if rows else None,
            "full_vocab_top1_accuracy": float(np.mean([row["full_vocab_top1"] for row in rows])) if rows else None,
            "competent": competent,
        }
        pairs = []
        ordered = sorted(rows, key=lambda row: row["example_id"])
        for first_index, first in enumerate(ordered):
            for second in ordered[first_index + 1:]:
                if first["answer_index"] == second["answer_index"] or first["candidate_ids"] != second["candidate_ids"]:
                    continue
                edits = hamming_pair(first, second)
                if edits is None or edits > config["maximum_counterfactual_token_edits"]:
                    continue
                pairs.append({
                    "base_id": first["example_id"], "source_id": second["example_id"],
                    "token_edits": edits, "sequence_length": len(first["prompt_ids"]),
                    "base_answer": first["answer"], "source_answer": second["answer"],
                })
        pairs.sort(key=lambda row: (row["token_edits"], row["base_id"], row["source_id"]))
        counterfactuals[family] = pairs[:config["maximum_saved_pairs_per_family"]]
        summaries[family]["eligible_counterfactual_pairs"] = len(pairs)

    competent = [family for family, row in summaries.items() if row["competent"]]
    nonlexical = [family for family in competent if summaries[family]["regime"] not in {"lexical", "classification"}]
    ready = [family for family in competent if summaries[family]["eligible_counterfactual_pairs"] >= 8]
    hypotheses = {
        "C0_tokenization_coverage": {"supported": sum(row["usable"] >= config["minimum_usable_examples"] for row in summaries.values()) >= 12},
        "C1_broad_competence": {"supported": len(competent) >= 6, "competent": competent},
        "C2_nonlexical_competence": {"supported": len(nonlexical) >= 3, "competent": nonlexical},
        "C3_intervention_ready": {"supported": len(ready) >= 4, "families": ready},
        "C4_instruction_bias_is_not_assumed": {"supported": True, "reporting": "family and template separated; no failed family is interpreted mechanistically"},
    }
    payload = {
        "config": config, "hypotheses": hypotheses, "summary": summaries,
        "counterfactual_pairs": counterfactuals,
        "examples": [{key: value for key, value in row.items() if key not in {"prompt_ids"}} for row in evaluated],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    lines = [
        "# Broad task competence screen for Bilin18", "",
        "This is a behavioral eligibility screen for the FineWeb-only checkpoint. A family passes only if it has at least eight exactly tokenized examples, achieves at least halfway from candidate-set chance to perfect accuracy, and has positive median correct-versus-best-alternative margin. Failure is not mechanistic evidence.", "",
        "## Locked gates", "",
    ]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**" for name, row in hypotheses.items())
    lines += ["", "## Family results", "", "| Family | Regime | Usable | Choices | Accuracy | Normalized | Median margin | Median vocab rank | Top-1 | CF pairs | Eligible |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for family, row in summaries.items():
        margin_text = "—" if row["median_candidate_margin"] is None else f"{row['median_candidate_margin']:.3f}"
        rank_text = "—" if row["median_full_vocab_rank"] is None else f"{row['median_full_vocab_rank']:.1f}"
        top1_text = "—" if row["full_vocab_top1_accuracy"] is None else f"{row['full_vocab_top1_accuracy']:.3f}"
        lines.append(f"| {family} | {row['regime']} | {row['usable']} | {row['candidate_count']} | {row['candidate_accuracy']:.3f} | {row['normalized_accuracy']:.3f} | {margin_text} | {rank_text} | {top1_text} | {row['eligible_counterfactual_pairs']} | {'yes' if row['competent'] else 'no'} |")
    lines += ["", "## Interpretation rule", "", "Only competent families are eligible for later circuit discovery. Candidate-set success without full-vocabulary top-1 is retained as graded evidence, but both quantities are reported. Instruction-like failures are prompt failures unless a naturalistic version of the same operation also fails."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUTPUT} and {REPORT}")


if __name__ == "__main__":
    main()
