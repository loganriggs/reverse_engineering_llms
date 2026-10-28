#!/usr/bin/env python3
"""Measure write-rank and native-channel requirements across four token edits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_boundary_bilinear_router import causal
from run_bilin12_boundary_output_interface import token_id
from run_bilin12_boundary_state_contraction import paired_panel
from run_bilin12_multi_operation_modules import activation_delta
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_edit_capacity_frontier.json")
OUTPUT = TENSOR_ROOT / "bilin12_edit_capacity_frontier.json"
REPORT = TENSOR_ROOT / "BILIN12_EDIT_CAPACITY_FRONTIER.md"


def first_passing(frontier, threshold):
    values = [int(key) for key, row in frontier.items() if row["fidelity"] >= threshold]
    return min(values) if values else None


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    pooled_config = json.loads(Path(config["parent_config"]).read_text())
    multi = json.loads(Path(pooled_config["parent_config"]).read_text())
    cooperation = json.loads(Path(multi["parent_config"]).read_text())
    analytic = json.loads(Path(cooperation["parent_config"]).read_text())
    subspace = json.loads(Path(analytic["parent_config"]).read_text())
    state_config = json.loads(Path(subspace["parent_config"]).read_text())
    parent = json.loads(Path(state_config["parent_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    tokens = {name: token_id(tokenizer, name) for name in multi["targets"]}
    model = load_snapshot_model(Path(parent["snapshot"]), "cuda")
    block = model.blocks[parent["layer"]]; down = block.mlp.Down.weight.float()
    all_deltas = []; importance_rows = []
    for name, target in tokens.items():
        panel = paired_panel(model, parent["semantic_discovery"], parent["layer"],
                             target, parent["batch_size"])
        all_deltas.append(panel[4] - panel[1])
        importance_rows.append(activation_delta(block, panel).abs().mean(0) * down.T.norm(dim=-1))
    _, _, vh = torch.linalg.svd(torch.cat(all_deltas).float(), full_matrices=False)
    basis = vh.T.contiguous()
    order = torch.stack(importance_rows).mean(0).argsort(descending=True)
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    random_basis = torch.linalg.qr(torch.randn(model.config.n_embd, model.config.n_embd,
                                               generator=generator, device="cuda")).Q
    random_order = torch.randperm(down.shape[1], generator=generator, device="cuda")
    results = {}
    for name, target in tokens.items():
        rows = {}
        for corpus, spec in parent["evaluation"].items():
            print("evaluate", repr(name), corpus, flush=True)
            panel = paired_panel(model, spec, parent["layer"], target, parent["batch_size"])
            delta = panel[4] - panel[1]; act = activation_delta(block, panel)
            rank_frontier = {}; random_rank_frontier = {}
            for rank in config["ranks"]:
                learned = basis[:, :rank]; random = random_basis[:, :rank]
                rank_frontier[str(rank)] = causal(model, panel, (delta @ learned) @ learned.T)
                random_rank_frontier[str(rank)] = causal(model, panel, (delta @ random) @ random.T)
            channel_frontier = {}; random_channel_frontier = {}
            for count in config["channel_counts"]:
                def write(indices):
                    selected = torch.zeros_like(act); selected[:, indices] = act[:, indices]
                    return block.mlp.Down(selected)
                channel_frontier[str(count)] = causal(model, panel, write(order[:count]))
                random_channel_frontier[str(count)] = causal(model, panel, write(random_order[:count]))
            rows[corpus] = {
                "rank": rank_frontier, "random_rank": random_rank_frontier,
                "channels": channel_frontier, "random_channels": random_channel_frontier,
                "minimum_rank_080": first_passing(rank_frontier, .80),
                "minimum_rank_090": first_passing(rank_frontier, .90),
                "minimum_channels_080": first_passing(channel_frontier, .80),
                "minimum_channels_090": first_passing(channel_frontier, .90),
            }
        results[name] = {"token_id": target, "corpora": rows}
    all_rows = [row for operation in results.values() for row in operation["corpora"].values()]
    f0 = all(row["rank"]["256"]["fidelity"] >= .80 for row in all_rows)
    f1 = all(row["rank"]["512"]["fidelity"] >= .90 for row in all_rows)
    f2 = all(row["channels"]["1024"]["fidelity"] >= .80 for row in all_rows)
    mean_minimum = []
    for name in tokens:
        values = [row["minimum_rank_080"] or 769 for row in results[name]["corpora"].values()]
        mean_minimum.append(float(np.mean(values)))
    f3 = all(left <= right for left, right in zip(mean_minimum, mean_minimum[1:]))
    f4 = all(row["rank"][str(rank)]["fidelity"]
             >= row["random_rank"][str(rank)]["fidelity"] + .20
             for row in all_rows for rank in (32, 64, 128, 256))
    hypotheses = {
        "F0_rank256_is_universally_sufficient": {"supported": f0},
        "F1_rank512_is_near_exact": {"supported": f1},
        "F2_top1024_native_channels_are_sufficient": {"supported": f2},
        "F3_operation_complexity_order_replicates": {"supported": f3,
                                                       "mean_minimum_ranks": mean_minimum},
        "F4_rank_frontier_beats_random_until_half_width": {"supported": f4},
    }
    OUTPUT.write_text(json.dumps({"config": config, "tokens": tokens,
                                  "hypotheses": hypotheses, "results": results}, indent=2))
    lines = ["# Capacity frontier of the shared late edit backbone", "",
             "One pooled write basis and one direct native-channel ranking are evaluated from compact to full size for four token edits and three corpora.", "", "## Locked hypotheses", ""]
    for key, item in hypotheses.items():
        lines.append(f"- **{key}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "| Target | Corpus | Min rank 0.80/0.90 | Min channels 0.80/0.90 | Rank 256/512 fidelity | Channels 1024/2048 fidelity |",
              "|---|---|---:|---:|---:|---:|"]
    for name, operation in results.items():
        for corpus, row in operation["corpora"].items():
            lines.append(
                f"| `{name.encode('unicode_escape').decode()}` | {corpus} | "
                f"{row['minimum_rank_080'] or 'none'}/{row['minimum_rank_090'] or 'none'} | "
                f"{row['minimum_channels_080'] or 'none'}/{row['minimum_channels_090'] or 'none'} | "
                f"{row['rank']['256']['fidelity']:.3f}/{row['rank']['512']['fidelity']:.3f} | "
                f"{row['channels']['1024']['fidelity']:.3f}/{row['channels']['2048']['fidelity']:.3f} |"
            )
    lines += ["", "The minimum is over the declared coarse grid, not an exact intrinsic dimension. Fidelity is behavior- and corpus-specific; there is no single model-wide number of necessary neurons."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
