#!/usr/bin/env python3
"""Test conditional compactness of the independent checkpoint's Layer-11 write.

Each expert retains the same weight-derived raw four-space plus a learned
16-dimensional tail. A linear router sees only a frozen whitening/PCA transform
of the upstream normalized state. The four-expert bank therefore has at most
the same 68-dimensional union as the fixed union control while hard execution
uses only 20 output directions per token.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer11_conditional_output_router.json")
SOURCE = TENSOR_ROOT / "bilin12_layer11_bottleneck_tail_replication.npz"
OUTPUT = TENSOR_ROOT / "bilin12_layer11_conditional_output_router.json"
CHECKPOINT = TENSOR_ROOT / "bilin12_layer11_conditional_output_router.pt"
REPORT = TENSOR_ROOT / "BILIN12_LAYER11_CONDITIONAL_OUTPUT_ROUTER.md"
FIGURE = TENSOR_ROOT / "bilin12_layer11_conditional_output_router.png"


class ConditionalOutputRouter(torch.nn.Module):
    def __init__(
        self,
        anchor: torch.Tensor,
        initial_tail: torch.Tensor,
        router_rank: int,
        seed: int,
    ) -> None:
        super().__init__()
        generator = torch.Generator(device=anchor.device).manual_seed(seed)
        self.register_buffer("anchor", anchor)
        self.expert_parameters = torch.nn.Parameter(
            initial_tail + 1e-3 * torch.randn(
                initial_tail.shape, generator=generator, device=anchor.device
            )
        )
        experts = initial_tail.shape[0]
        self.router_weight = torch.nn.Parameter(
            0.01 * torch.randn(
                router_rank, experts, generator=generator, device=anchor.device
            )
        )
        self.router_bias = torch.nn.Parameter(torch.zeros(experts, device=anchor.device))

    def expert_bases(self) -> torch.Tensor:
        anchor_coordinates = torch.einsum(
            "da,edr->ear", self.anchor, self.expert_parameters
        )
        anchor_projection = torch.einsum(
            "da,ear->edr", self.anchor, anchor_coordinates
        )
        residual = self.expert_parameters - anchor_projection
        tails = torch.linalg.qr(residual, mode="reduced").Q
        anchors = self.anchor[None].expand(len(tails), -1, -1)
        return torch.cat((anchors, tails), dim=2)

    def gates(
        self,
        features: torch.Tensor,
        temperature: float = 1.0,
        mode: str = "soft",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        soft = (features @ self.router_weight + self.router_bias).div(temperature).softmax(-1)
        hard = F.one_hot(soft.argmax(-1), soft.shape[1]).to(soft.dtype)
        if mode == "soft":
            selected = soft
        elif mode == "hard":
            selected = hard
        elif mode == "hard_st":
            selected = hard + soft - soft.detach()
        else:
            raise ValueError(f"unknown gate mode: {mode}")
        return selected, soft

    def expert_reconstructions(
        self,
        centered_write: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bases = self.expert_bases()
        coordinates = torch.einsum("bd,edr->ber", centered_write, bases)
        reconstructions = torch.einsum("ber,edr->bed", coordinates, bases)
        return reconstructions, bases

    def forward(
        self,
        features: torch.Tensor,
        centered_write: torch.Tensor,
        mean: torch.Tensor,
        temperature: float = 1.0,
        mode: str = "soft",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        experts, bases = self.expert_reconstructions(centered_write)
        gates, soft = self.gates(features, temperature, mode)
        reconstruction = torch.einsum("be,bed->bd", gates, experts)
        return mean + reconstruction, soft, bases


def route_summary(gates: torch.Tensor) -> dict[str, float | list[float]]:
    mean_gate = gates.float().mean(0)
    entropy = -(mean_gate * mean_gate.clamp_min(1e-12).log()).sum()
    site_entropy = -(gates * gates.clamp_min(1e-12).log()).sum(1).mean()
    return {
        "mean_gate": mean_gate.detach().cpu().tolist(),
        "effective_experts": float(entropy.exp()),
        "mean_site_entropy": float(site_entropy),
        "hard_shares": torch.bincount(
            gates.argmax(1), minlength=gates.shape[1]
        ).float().div(len(gates)).cpu().tolist(),
    }


def load_rows(spec: dict) -> np.ndarray:
    source = np.load(spec["path"], mmap_mode="r")
    return np.asarray(source[spec["start"] : spec["start"] + spec["rows"]])


@torch.inference_mode()
def collect_context(model, spec: dict, config: dict) -> dict[str, torch.Tensor]:
    rows = load_rows(spec)
    positions = config["positions"]
    dimension = model.config.n_embd
    layer = config["layer"]
    pre_rows, state_rows, write_rows, target_rows = [], [], [], []
    row_id_rows, position_rows = [], []
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start : start + config["batch_size"], :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        x0 = F.rms_norm(model.embedding(ids), (dimension,))
        x = x0
        value_bus = None
        for block in model.blocks[:layer]:
            x, value_bus = block(x, value_bus, x0)
        block = model.blocks[layer]
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
        pre = mixed + attention
        state = F.rms_norm(pre, (dimension,))
        write = block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
        pre_rows.append(pre[:, positions].float().flatten(0, 1))
        state_rows.append(state[:, positions].float().flatten(0, 1))
        write_rows.append(write[:, positions].float().flatten(0, 1))
        target_rows.append(torch.as_tensor(
            rows[start : start + len(raw), np.asarray(positions) + 1].reshape(-1),
            device="cuda", dtype=torch.long,
        ))
        row_id_rows.append(
            torch.arange(start, start + len(raw), device="cuda").repeat_interleave(len(positions))
        )
        position_rows.append(
            torch.arange(len(positions), device="cuda").repeat(len(raw))
        )
    return {
        "pre": torch.cat(pre_rows),
        "state": torch.cat(state_rows),
        "write": torch.cat(write_rows),
        "target": torch.cat(target_rows),
        "row_id": torch.cat(row_id_rows),
        "position": torch.cat(position_rows),
    }


def fit_router_transform(
    states: torch.Tensor,
    train_indices: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = states[train_indices].mean(0)
    centered = states[train_indices] - mean
    covariance = centered.T @ centered / len(centered)
    values, vectors = torch.linalg.eigh(covariance)
    values = values.flip(0)[:rank].clamp_min(1e-8)
    vectors = vectors.flip(1)[:, :rank]
    return mean, vectors, values.sqrt()


def router_features(
    states: torch.Tensor,
    mean: torch.Tensor,
    vectors: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return ((states - mean) @ vectors) / scale


def logits_from_write(model, pre: torch.Tensor, write: torch.Tensor) -> torch.Tensor:
    dimension = write.shape[1]
    raw = model.lm_head(F.rms_norm(pre + write, (dimension,))).float()
    return 30 * torch.tanh(raw / 30)


def exact_kl(reference_logits: torch.Tensor, changed_logits: torch.Tensor) -> torch.Tensor:
    reference_logprob = F.log_softmax(reference_logits, -1)
    probability = reference_logprob.exp()
    return (probability * (reference_logprob - F.log_softmax(changed_logits, -1))).sum(-1)


def router_loss(
    program: ConditionalOutputRouter,
    indices: torch.Tensor,
    data: dict[str, torch.Tensor],
    features: torch.Tensor,
    mean: torch.Tensor,
    model,
    temperature: float,
    config: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    centered = data["write"][indices] - mean
    changed, soft, _ = program(
        features[indices], centered, mean, temperature, mode="hard_st"
    )
    changed_logits = logits_from_write(model, data["pre"][indices], changed)
    with torch.no_grad():
        native_logits = logits_from_write(
            model, data["pre"][indices], data["write"][indices]
        )
    kl = exact_kl(native_logits, changed_logits).mean()
    mean_gate = soft.mean(0)
    balance = soft.shape[1] * (mean_gate - 1 / soft.shape[1]).square().sum()
    entropy = -(soft * soft.clamp_min(1e-12).log()).sum(1).mean()
    loss = kl + config["balance_weight"] * balance + config["entropy_weight"] * entropy
    return loss, {
        "kl": float(kl.detach()),
        "balance": float(balance.detach()),
        "entropy": float(entropy.detach()),
    }


@torch.inference_mode()
def validation_kl(
    program: ConditionalOutputRouter,
    indices: torch.Tensor,
    data: dict[str, torch.Tensor],
    features: torch.Tensor,
    mean: torch.Tensor,
    model,
    batch_size: int,
) -> float:
    values = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        changed, _, _ = program(
            features[selected], data["write"][selected] - mean,
            mean, 0.2, mode="hard",
        )
        native_logits = logits_from_write(model, data["pre"][selected], data["write"][selected])
        changed_logits = logits_from_write(model, data["pre"][selected], changed)
        values.extend(exact_kl(native_logits, changed_logits).cpu().tolist())
    return float(np.mean(values))


def fit_program(
    program: ConditionalOutputRouter,
    data: dict[str, torch.Tensor],
    features: torch.Tensor,
    mean: torch.Tensor,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    model,
    config: dict,
) -> tuple[list[dict], dict[str, torch.Tensor]]:
    optimizer = torch.optim.Adam(program.parameters(), lr=config["learning_rate"])
    generator = torch.Generator(device="cuda").manual_seed(config["seed"] + 1)
    initial_validation = validation_kl(
        program, validation_indices, data, features, mean, model,
        config["optimization_batch_size"],
    )
    best_value = initial_validation
    best_state = copy.deepcopy(program.state_dict())
    history = [{"step": -1, "validation_hard_kl": initial_validation}]
    for step in range(config["optimization_steps"]):
        sampled = train_indices[torch.randint(
            len(train_indices), (config["optimization_batch_size"],),
            generator=generator, device="cuda",
        )]
        progress = step / max(config["optimization_steps"] - 1, 1)
        temperature = config["temperature_start"] * (
            config["temperature_end"] / config["temperature_start"]
        ) ** progress
        loss, parts = router_loss(
            program, sampled, data, features, mean, model, temperature, config
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % config["validation_interval"] == 0 or step == config["optimization_steps"] - 1:
            value = validation_kl(
                program, validation_indices, data, features, mean, model,
                config["optimization_batch_size"],
            )
            if value < best_value:
                best_value = value
                best_state = copy.deepcopy(program.state_dict())
            with torch.no_grad():
                _, soft = program.gates(features[validation_indices], 0.2, "hard")
            history.append({
                "step": step,
                "temperature": temperature,
                "train_loss": float(loss.detach()),
                "validation_hard_kl": value,
                **parts,
                **{f"route_{key}": value for key, value in route_summary(soft).items()},
            })
    program.load_state_dict(best_state)
    return history, best_state


def summarize(records: dict[str, list[float]]) -> dict[str, float]:
    return {
        "mean_exact_kl": float(np.mean(records["kl"])),
        "median_exact_kl": float(np.median(records["kl"])),
        "p90_exact_kl": float(np.quantile(records["kl"], 0.9)),
        "mean_ce_delta": float(np.mean(records["ce_delta"])),
        "top1_agreement": float(np.mean(records["top1"])),
        "relative_write_mse": float(np.sum(records["write_error"]) / max(np.sum(records["write_energy"]), 1e-30)),
    }


@torch.inference_mode()
def evaluate(
    model,
    spec: dict,
    config: dict,
    program: ConditionalOutputRouter,
    mean: torch.Tensor,
    transform: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    fixed_tail16: torch.Tensor,
    fixed_tail64: torch.Tensor,
    position_hard: torch.Tensor,
    seed: int,
) -> dict:
    data = collect_context(model, spec, config)
    features = router_features(data["state"], *transform)
    centered = data["write"] - mean
    expert_centered, bases = program.expert_reconstructions(centered)
    soft, _ = program.gates(features, 0.2, "soft")
    hard, _ = program.gates(features, 0.2, "hard")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shuffled = hard[torch.randperm(len(hard), generator=generator, device="cuda")]
    positional = position_hard[data["position"]]
    uniform = torch.full_like(soft, 1 / config["experts"])

    trained_union = torch.linalg.qr(
        torch.cat((program.anchor, bases[:, :, program.anchor.shape[1] :].permute(1, 0, 2).flatten(1)), dim=1),
        mode="reduced",
    ).Q
    fixed_bases = {
        "fixed_tail16": fixed_tail16,
        "fixed_tail64": fixed_tail64,
        "trained_union68": trained_union,
    }
    gate_variants = {
        "soft": soft,
        "hard": hard,
        "shuffled_hard": shuffled,
        "position_hard": positional,
        "uniform": uniform,
    }
    names = list(fixed_bases) + list(gate_variants) + [f"expert_{i}" for i in range(config["experts"])] + ["oracle"]
    records = {
        name: {key: [] for key in ("kl", "ce_delta", "top1", "write_error", "write_energy")}
        for name in names
    }
    batch_size = config["optimization_batch_size"]
    for start in range(0, len(centered), batch_size):
        stop = start + batch_size
        selected = slice(start, stop)
        native_write = data["write"][selected]
        native_logits = logits_from_write(model, data["pre"][selected], native_write)
        native_logprob = F.log_softmax(native_logits, -1)
        native_probability = native_logprob.exp()
        native_ce = -native_logprob.gather(1, data["target"][selected, None]).squeeze(1)
        native_top1 = native_logits.argmax(-1)
        local_centered = centered[selected]
        local_experts = expert_centered[selected]
        candidates = {
            name: mean + (local_centered @ basis) @ basis.T
            for name, basis in fixed_bases.items()
        }
        candidates.update({
            name: mean + torch.einsum("be,bed->bd", gates[selected], local_experts)
            for name, gates in gate_variants.items()
        })
        expert_logits = []
        for expert in range(config["experts"]):
            changed = mean + local_experts[:, expert]
            candidates[f"expert_{expert}"] = changed
            expert_logits.append(logits_from_write(model, data["pre"][selected], changed))
        stacked_logits = torch.stack(expert_logits, 1)
        oracle_kl = torch.stack([
            (native_probability * (native_logprob - F.log_softmax(logits, -1))).sum(-1)
            for logits in expert_logits
        ], 1)
        oracle_index = oracle_kl.argmin(1)
        batch_index = torch.arange(len(oracle_index), device="cuda")
        candidates["oracle"] = mean + local_experts[batch_index, oracle_index]
        for name, changed in candidates.items():
            if name.startswith("expert_"):
                logits = expert_logits[int(name.split("_")[1])]
            elif name == "oracle":
                logits = stacked_logits[batch_index, oracle_index]
            else:
                logits = logits_from_write(model, data["pre"][selected], changed)
            logprob = F.log_softmax(logits, -1)
            kl = (native_probability * (native_logprob - logprob)).sum(-1)
            ce = -logprob.gather(1, data["target"][selected, None]).squeeze(1)
            records[name]["kl"].extend(kl.cpu().tolist())
            records[name]["ce_delta"].extend((ce - native_ce).cpu().tolist())
            records[name]["top1"].extend((logits.argmax(-1) == native_top1).cpu().tolist())
            records[name]["write_error"].extend((changed - native_write).square().sum(1).cpu().tolist())
            records[name]["write_energy"].extend(local_centered.square().sum(1).cpu().tolist())
    return {
        "behavioral": {name: summarize(row) for name, row in records.items()},
        "routing": {
            "soft": route_summary(soft),
            "hard": route_summary(hard),
            "position_hard": route_summary(positional),
        },
        "expert_overlap": [
            [float(torch.linalg.svdvals(first.T @ second).mean()) for second in bases]
            for first in bases
        ],
    }


def main() -> None:
    config = json.loads(CONFIG.read_text())
    source = np.load(SOURCE)
    raw4 = torch.from_numpy(source["raw_outlier_basis"]).cuda().float()
    tail = torch.from_numpy(source["tail_basis"]).cuda().float()
    mean = torch.from_numpy(source["post_bias_write_mean"]).cuda().float()
    initial_tail = torch.stack([
        tail[:, expert * config["tail_rank_per_expert"] : (expert + 1) * config["tail_rank_per_expert"]]
        for expert in range(config["experts"])
    ])
    fixed_tail16 = torch.cat((raw4, tail[:, :16]), 1)
    fixed_tail64 = torch.cat((raw4, tail[:, :64]), 1)

    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    print("collecting routed discovery panel", flush=True)
    discovery = collect_context(model, config["discovery"], config)
    train_indices = torch.nonzero(discovery["row_id"] % 2 == 0, as_tuple=False).flatten()
    all_validation = torch.nonzero(discovery["row_id"] % 2 == 1, as_tuple=False).flatten()
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    validation_indices = all_validation[torch.randperm(
        len(all_validation), generator=generator, device="cuda"
    )[: config["validation_examples"]]]
    transform = fit_router_transform(discovery["state"], train_indices, config["router_rank"])
    features = router_features(discovery["state"], *transform).detach()

    program = ConditionalOutputRouter(
        raw4, initial_tail, config["router_rank"], config["seed"]
    ).cuda()
    print("training hard conditional output router", flush=True)
    history, best_state = fit_program(
        program, discovery, features, mean, train_indices, validation_indices,
        model, config,
    )
    with torch.no_grad():
        discovery_hard, _ = program.gates(features, 0.2, "hard")
        position_mean = torch.stack([
            discovery_hard[discovery["position"] == position].float().mean(0)
            for position in range(len(config["positions"]))
        ])
        position_hard = F.one_hot(
            position_mean.argmax(1), config["experts"]
        ).float()

    results = {}
    for index, (corpus, spec) in enumerate(config["evaluation"].items()):
        print("conditional evaluation", corpus, flush=True)
        results[corpus] = evaluate(
            model, spec, config, program, mean, transform,
            fixed_tail16, fixed_tail64, position_hard,
            config["seed"] + index,
        )

    def behavior(corpus: str, name: str) -> dict[str, float]:
        return results[corpus]["behavioral"][name]

    hypotheses = {
        "C1_hard_routed20_matches_fixed68": {
            "supported": all(
                behavior(corpus, "hard")["mean_exact_kl"]
                <= 1.1 * behavior(corpus, "fixed_tail64")["mean_exact_kl"]
                for corpus in results
            ),
        },
        "C2_hard_routed20_beats_fixed20": {
            "supported": all(
                behavior(corpus, "hard")["mean_exact_kl"]
                <= 0.8 * behavior(corpus, "fixed_tail16")["mean_exact_kl"]
                for corpus in results
            ),
        },
        "C3_router_closes_oracle_gap": {
            "supported": all(
                (behavior(corpus, "fixed_tail16")["mean_exact_kl"] - behavior(corpus, "hard")["mean_exact_kl"])
                / max(behavior(corpus, "fixed_tail16")["mean_exact_kl"] - behavior(corpus, "oracle")["mean_exact_kl"], 1e-30)
                >= 0.70
                for corpus in results
            ),
        },
        "C4_hard_retains_soft_gain": {
            "supported": all(
                behavior(corpus, "fixed_tail16")["mean_exact_kl"] - behavior(corpus, "hard")["mean_exact_kl"]
                >= 0.8 * (
                    behavior(corpus, "fixed_tail16")["mean_exact_kl"] - behavior(corpus, "soft")["mean_exact_kl"]
                )
                for corpus in results
            ),
        },
        "C5_router_uses_multiple_experts": {
            "supported": all(results[corpus]["routing"]["hard"]["effective_experts"] >= 3 for corpus in results),
        },
        "C6_context_assignment_matters": {
            "supported": all(
                behavior(corpus, "shuffled_hard")["mean_exact_kl"]
                >= 1.25 * behavior(corpus, "hard")["mean_exact_kl"]
                for corpus in results
            ),
        },
        "C7_context_beats_position_only": {
            "supported": all(
                behavior(corpus, "hard")["mean_exact_kl"]
                <= 0.8 * behavior(corpus, "position_hard")["mean_exact_kl"]
                for corpus in results
            ),
        },
        "C8_oracle_experts_are_near_sufficient": {
            "supported": all(behavior(corpus, "oracle")["median_exact_kl"] <= 0.02 for corpus in results),
        },
        "C9_full_union_upper_bound": {
            "supported": all(
                behavior(corpus, "trained_union68")["mean_exact_kl"]
                < min(
                    behavior(corpus, name)["mean_exact_kl"]
                    for name in ("fixed_tail16", "hard", "soft", "oracle")
                )
                for corpus in results
            ),
        },
    }
    torch.save({
        "program": best_state,
        "router_mean": transform[0].cpu(),
        "router_vectors": transform[1].cpu(),
        "router_scale": transform[2].cpu(),
        "position_hard": position_hard.cpu(),
        "config": config,
    }, CHECKPOINT)
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "history": history,
        "discovery_routes": route_summary(discovery_hard),
        "position_mean_routes": position_mean.cpu().tolist(),
        "results": results,
    }, indent=2, allow_nan=False))

    lines = [
        "# Independent Layer-11 conditional output router",
        "",
        "Four experts share the weight-only raw four-space and each learn a 16-dimensional tail. The expert bank has at most a 68-dimensional union, while hard execution retains 20 directions per token. A linear router sees only 64 frozen whitened PCA coordinates of the upstream normalized state. Training and validation split by FineWeb row; all bases and routes are frozen before disjoint FineWeb, WikiText, and TinyStories evaluation.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Conditional router](bilin12_layer11_conditional_output_router.png)",
        "",
        "| Corpus | Fixed-20 | Hard-20 | Soft mixture | Oracle expert | Fixed PCA-68 | Trained union-68 | Shuffled | Position-only |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus, row in results.items():
        b = row["behavioral"]
        lines.append(
            f"| {corpus} | {b['fixed_tail16']['mean_exact_kl']:.5f} | {b['hard']['mean_exact_kl']:.5f} | "
            f"{b['soft']['mean_exact_kl']:.5f} | {b['oracle']['mean_exact_kl']:.5f} | "
            f"{b['fixed_tail64']['mean_exact_kl']:.5f} | {b['trained_union68']['mean_exact_kl']:.5f} | "
            f"{b['shuffled_hard']['mean_exact_kl']:.5f} | {b['position_hard']['mean_exact_kl']:.5f} |"
        )
        lines.append(
            f"| {corpus} route support | — | {row['routing']['hard']['effective_experts']:.2f} experts | "
            f"site entropy {row['routing']['soft']['mean_site_entropy']:.2f} | — | — | — | — | — |"
        )
    lines += [
        "",
        "The oracle asks whether the learned expert bank contains a compact per-example explanation; the router asks whether upstream state can select it. A soft-only improvement is an ensemble code, not sparse execution. A hard gain that disappears under shuffled and position-only routes is evidence for conditional computation. If the trained union succeeds while both oracle and router fail, the tail is simultaneous rather than routed at this granularity.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    corpora = list(results)
    display = ("fixed_tail16", "hard", "soft", "oracle", "fixed_tail64", "trained_union68", "shuffled_hard", "position_hard")
    labels = ("fixed20", "hard20", "soft", "oracle", "PCA68", "union68", "shuffled", "position")
    figure, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    x = np.arange(len(display))
    width = 0.25
    for index, corpus in enumerate(corpora):
        axes[0].bar(
            x + (index - 1) * width,
            [results[corpus]["behavioral"][name]["mean_exact_kl"] for name in display],
            width, label=corpus,
        )
    axes[0].set_xticks(x, labels, rotation=30, ha="right")
    axes[0].set_ylabel("mean exact KL")
    axes[0].set_title("Conditional sufficiency and controls")
    axes[0].legend(fontsize=8)
    for corpus in corpora:
        axes[1].plot(
            range(config["experts"]),
            results[corpus]["routing"]["hard"]["hard_shares"],
            marker="o", label=corpus,
        )
    axes[1].set_xticks(range(config["experts"]))
    axes[1].set_xlabel("expert")
    axes[1].set_ylabel("hard route share")
    axes[1].set_title("OOD route utilization")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, CHECKPOINT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
