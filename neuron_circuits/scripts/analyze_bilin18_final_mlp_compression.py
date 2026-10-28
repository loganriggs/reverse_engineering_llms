#!/usr/bin/env python3
"""Compare and report the two final-MLP compression splits."""

from __future__ import annotations

import json
from pathlib import Path


RUNS = (
    Path("artifacts/bilin18_final_mlp_compression/results.json"),
    Path("artifacts/bilin18_final_mlp_compression_replicate/results.json"),
)
OUTPUT = Path("artifacts/bilin18_final_mlp_compression/analysis.json")


def curves(run, replacement="mean"):
    return {
        row["k"]: row for row in run["conditions"]
        if row["selection"] == "ranked" and row["replacement"] == replacement
    }


def main():
    first, second = [json.loads(path.read_text()) for path in RUNS]
    a, b = curves(first), curves(second)
    ks = sorted(a)
    rows = []
    for k in ks:
        rows.append({
            "k": k,
            "fraction": k / 4608,
            "ce_delta_runs": [a[k]["cross_entropy_delta"], b[k]["cross_entropy_delta"]],
            "kl_runs": [a[k]["mean_kl_from_full"], b[k]["mean_kl_from_full"]],
            "top1_runs": [a[k]["top1_agreement"], b[k]["top1_agreement"]],
            "performance_recovery_runs": [
                1 - a[k]["cross_entropy_delta"] / a[0]["cross_entropy_delta"],
                1 - b[k]["cross_entropy_delta"] / b[0]["cross_entropy_delta"],
            ] if k else [0.0, 0.0],
        })
    first_order = [row["unit"] for row in first["top_units"]]
    second_order = second["ranking_order"]
    overlap = {
        str(k): len(set(first_order[:k]) & set(second_order[:k])) / k
        for k in (8, 16, 32, 64, 128, 256)
    }
    analysis = {
        "runs": [str(path) for path in RUNS],
        "eval_tokens": [first["baseline"]["tokens"], second["baseline"]["tokens"]],
        "baseline_cross_entropy": [first["baseline"]["cross_entropy"], second["baseline"]["cross_entropy"]],
        "rows": rows,
        "top_k_overlap": overlap,
        "replicate_score_effective_support": second["score_effective_support"],
        "replicate_score_fraction_top_k": second["score_fraction_top_k"],
    }
    OUTPUT.write_text(json.dumps(analysis, indent=2))
    lines = [
        "# Bilin18 final-MLP compression", "",
        "## Verdict", "",
        "A small fixed subset captures most of the final MLP's **next-token loss benefit**, "
        "but it does not faithfully reproduce the full output distribution. The final layer is "
        "therefore small in an operational performance sense and still broad in a mechanistic "
        "emulation sense.", "",
        "Channels were ranked on a disjoint discovery split by activation variance times the "
        "squared norm of their direct `Down→unembedding` direction. Omitted channels were replaced "
        "by their discovery mean, which can be folded into a replacement layer's bias. Results "
        "below are shown as initial / independent-replication values.", "",
        "|retained channels|fraction of layer|Δ cross-entropy|KL from full|top-1 agreement|MLP loss benefit recovered|",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["k"] not in (0, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 4608):
            continue
        lines.append(
            f"|{row['k']}|{row['fraction']:.1%}|"
            f"{row['ce_delta_runs'][0]:+.3f} / {row['ce_delta_runs'][1]:+.3f}|"
            f"{row['kl_runs'][0]:.3f} / {row['kl_runs'][1]:.3f}|"
            f"{row['top1_runs'][0]:.1%} / {row['top1_runs'][1]:.1%}|"
            f"{row['performance_recovery_runs'][0]:.1%} / {row['performance_recovery_runs'][1]:.1%}|"
        )
    lines.extend([
        "", "## What is concentrated", "",
        f"The activation-conditioned ranking has effective support {second['score_effective_support']:.1f} "
        "channels. Its top 32 channels hold "
        f"{second['score_fraction_top_k']['32']:.1%} of ranking mass and the top 64 hold "
        f"{second['score_fraction_top_k']['64']:.1%}. By contrast, raw `Down→unembedding` "
        "column energy has effective support about 4,485 channels. Concentration is therefore "
        "created by which channels vary on data, not by a visibly sparse output matrix.", "",
        "The ranking is stable across discovery samples: top-64 overlap is "
        f"{overlap['64']:.1%}, top-128 overlap is {overlap['128']:.1%}, and top-256 overlap is "
        f"{overlap['256']:.1%}.", "",
        "## Red-team interpretation", "",
        "- **Strongly supported:** 64 fixed channels (1.4% of the native layer) recover about "
        "69–71% of the mean-replacement loss benefit; 128 recover about 79%. Both greatly beat "
        "random subsets of the same size.",
        "- **Not supported:** a few dozen channels faithfully replace the final MLP. At 64 "
        "channels KL remains about 0.127 nats and top-1 agreement about 84–86%.",
        "- Roughly 1,024 channels are needed for KL below 0.04 nats on both runs; 4,096 are "
        "needed for near-exact KL around 0.002 and roughly 98% top-1 agreement.",
        "- Cross-entropy measures task performance and can hide compensating logit changes. KL "
        "to the intact model is the stricter replacement criterion.",
        "- The evaluation uses 400 and 1,600 held-out document-final positions at context length "
        "128. It should be replicated at other positions and longer contexts before pruning.",
        "- This is a fixed native-channel subset, not a learned low-rank refactorization. A newly "
        "trained smaller bilinear layer might compress the computation further.",
    ])
    (OUTPUT.parent / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
