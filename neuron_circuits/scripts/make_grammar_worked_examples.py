#!/usr/bin/env python3
"""Create a concrete neuron-patching walkthrough for the grammar experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from neuron_circuits.grammar_data import VOCAB, encode_prompts, load_manifest
from neuron_circuits.grammar_model import ActivationPatch, GrammarTransformer


BASE = Path("artifacts/grammar_regimes")
OUTPUT = BASE / "worked_examples"
CONDITIONS = {
    0.0: BASE / "easy_pilot" / "rho_0" / "depth_4",
    0.05: BASE / "interpolation_seed0_swiglu" / "rho_0.05" / "depth_4",
    0.2: BASE / "interpolation_seed0_swiglu" / "rho_0.2" / "depth_4",
    0.5: BASE / "easy_pilot" / "rho_0.5" / "depth_4",
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(path):
    checkpoint = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
    model = GrammarTransformer(VOCAB.size, checkpoint["width"], checkpoint["heads"], checkpoint["depth"], mlp_kind=checkpoint["mlp_kind"])
    model.load_state_dict(checkpoint["model"])
    return model.to(DEVICE).eval()


def entity_from_digits(digits):
    return (digits[:, 0] * 64 + digits[:, 1] * 8 + digits[:, 2]).astype(np.int32)


def transition_facts(seed, count):
    rng = np.random.default_rng(seed)
    subject = rng.integers(0, 8, size=(count, 3), dtype=np.int32)
    obj = rng.integers(0, 8, size=(count, 3), dtype=np.int32)
    before, after = subject.copy(), subject.copy()
    before[:, 0] = 2; after[:, 0] = 3
    relation = np.zeros(count, dtype=np.int32)
    base = np.column_stack((entity_from_digits(before), entity_from_digits(obj), relation))
    changed = np.column_stack((entity_from_digits(after), entity_from_digits(obj), relation))
    return base, changed


@torch.inference_mode()
def run(model, prompts, patch=None):
    tokens = torch.from_numpy(prompts.astype(np.int64)).to(DEVICE)
    logits, activations = model(tokens, patch=patch, return_activations=True)
    answer_logits = logits[:, -1, list(VOCAB.answers)].float()
    return answer_logits, activations


def describe(logits, old, new):
    probability = logits.softmax(-1)[0]
    return {
        "prediction": int(logits.argmax(1)[0]),
        "old_probability": float(probability[old]),
        "new_probability": float(probability[new]),
        "new_minus_old_margin": float(logits[0, new] - logits[0, old]),
    }


def tokens_to_text(tokens):
    return " ".join(VOCAB.names[int(token)] for token in tokens)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(BASE / "easy_pilot" / "manifest.npz")
    for root in (BASE / "interpolation_seed0_swiglu",):
        other = load_manifest(root / "manifest.npz")
        assert other["hash"] == manifest["hash"]
    old_answer = int(manifest["permutations"][0, 2]); new_answer = int(manifest["permutations"][0, 3])
    discovery_base, discovery_changed = transition_facts(91001, 256)
    evaluation_base, evaluation_changed = transition_facts(91002, 2048)
    discovery_prompts = (encode_prompts(manifest, discovery_base, 0), encode_prompts(manifest, discovery_changed, 0))
    evaluation_prompts = (encode_prompts(manifest, evaluation_base, 0), encode_prompts(manifest, evaluation_changed, 0))

    models, deltas, base_logits, changed_logits, base_acts, eligibilities = {}, {}, {}, {}, {}, {}
    for rho, path in CONDITIONS.items():
        model = load_model(path); models[rho] = model
        _, discovery_base_acts = run(model, discovery_prompts[0])
        _, discovery_changed_acts = run(model, discovery_prompts[1])
        deltas[rho] = (discovery_changed_acts[3] - discovery_base_acts[3]).mean(0)
        base_logits[rho], base_activations = run(model, evaluation_prompts[0])
        changed_logits[rho], _ = run(model, evaluation_prompts[1])
        base_acts[rho] = base_activations[3]
        eligible = ((base_logits[rho].argmax(1).cpu().numpy() == old_answer) &
                    (changed_logits[rho].argmax(1).cpu().numpy() == new_answer))
        eligibilities[rho] = eligible
    pure_indices = np.flatnonzero(eligibilities[0.0])
    if not len(pure_indices): raise RuntimeError("no naturally correct pure-rule example")
    # Choose an illustrative success for the already-established pure-rule effect.
    # Intermediate-load outcomes are not used. Population claims come from the full
    # audit, so this example is explicitly not an unbiased effect-size estimate.
    pure_delta = deltas[0.0]
    pure_top32 = torch.topk(pure_delta.abs(), 32).indices
    pure_native = torch.zeros_like(pure_delta); pure_native[pure_top32] = pure_delta[pure_top32]
    pure_prompts = evaluation_prompts[0][pure_indices]
    pure_patched, _ = run(models[0.0], pure_prompts, ActivationPatch(3, pure_native, additive=True))
    successful = pure_indices[pure_patched.argmax(1).cpu().numpy() == new_answer]
    if not len(successful): raise RuntimeError("no pure native-patch success in pure panel")
    index = int(successful[0])
    base_prompt = evaluation_prompts[0][index:index + 1]
    changed_prompt = evaluation_prompts[1][index:index + 1]
    records = []
    pure_top = torch.topk(deltas[0.0].abs(), 8).indices.cpu().numpy()
    channel_rows = []
    for rho, model in models.items():
        delta = deltas[rho]
        top32 = torch.topk(delta.abs(), 32).indices
        full_values = delta[None, :]
        native_values = torch.zeros_like(full_values); native_values[:, top32] = delta[top32]
        generator = torch.Generator().manual_seed(92000)
        rotation = torch.linalg.qr(torch.randn(model.channels, model.channels, generator=generator)).Q.to(DEVICE)
        rotated_delta = delta @ rotation
        retained = torch.zeros_like(rotated_delta)
        retained[torch.topk(rotated_delta.abs(), 32).indices] = rotated_delta[torch.topk(rotated_delta.abs(), 32).indices]
        rotated_values = (retained @ rotation.T)[None, :]
        intact_base, _ = run(model, base_prompt)
        intact_changed, _ = run(model, changed_prompt)
        interventions = {"base": intact_base, "natural_changed": intact_changed}
        for name, value in (("full_patch", full_values), ("native_top32_patch", native_values), ("rotated_top32_patch", rotated_values)):
            interventions[name], _ = run(model, base_prompt, ActivationPatch(3, value, additive=True))
        descriptions = {name: describe(logits, old_answer, new_answer) for name, logits in interventions.items()}
        energy = float((delta[top32] ** 2).sum() / (delta ** 2).sum().clamp_min(1e-12))
        records.append({"rho": rho, "intact_understands_pair": bool(eligibilities[rho][index]),
                        "top32_delta_energy_fraction": energy, "outcomes": descriptions})
        if rho == 0.0:
            axis = json.loads((CONDITIONS[rho] / "axis_audit.json").read_text())
            native = axis["layers"][3]["groups"]["all"]["native"]
            base_margin = descriptions["base"]["new_minus_old_margin"]
            for channel in pure_top:
                one = torch.zeros_like(full_values); one[0, channel] = delta[channel]
                one_logits, _ = run(model, base_prompt, ActivationPatch(3, one, additive=True))
                one_description = describe(one_logits, old_answer, new_answer)
                channel_rows.append({
                    "channel": int(channel), "discovery_activation_change": float(delta[channel]),
                    "best_simple_label": native["best_features"][int(channel)],
                    "heldout_label_r2": float(native["heldout_r2_by_channel"][int(channel)]),
                    "exact_target_margin_change": one_description["new_minus_old_margin"] - base_margin,
                })

    payload = {
        "selection_rule": "first fixed-RNG candidate naturally answered correctly in both forms by the pure-rule model and switched successfully under its native-32 patch; intermediate-load models and outcomes were not used",
        "pure_eligible_count": int(len(pure_indices)), "selected_index": index,
        "pure_native_success_count": int(len(successful)),
        "relation": "relation_0 returns a permuted label for the subject's first syllable",
        "change": "subject first syllable 2 -> 3", "old_answer": old_answer, "new_answer": new_answer,
        "old_answer_token": VOCAB.names[VOCAB.answers[old_answer]], "new_answer_token": VOCAB.names[VOCAB.answers[new_answer]],
        "base_prompt": tokens_to_text(base_prompt[0]), "changed_prompt": tokens_to_text(changed_prompt[0]),
        "records": records, "pure_rule_top_channels": channel_rows,
    }
    (OUTPUT / "worked_example.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    labels = ("0%", "5%", "20%", "50%")
    methods = (("base", "unchanged prompt", "#777777"), ("natural_changed", "text actually changed", "#2ca02c"),
               ("full_patch", "all 192-channel patch", "#2774ae"), ("native_top32_patch", "top-32 native patch", "#ff8c2a"),
               ("rotated_top32_patch", "top-32 rotated patch", "#9467bd"))
    x = np.arange(len(records)); width = .15
    for offset, (method, label, color) in enumerate(methods):
        axes[0, 0].bar(x + (offset - 2) * width, [r["outcomes"][method]["new_probability"] for r in records], width, label=label, color=color)
        axes[0, 1].bar(x + (offset - 2) * width, [r["outcomes"][method]["new_minus_old_margin"] for r in records], width, label=label, color=color)
    axes[0, 0].set(title="Probability of the intended new answer", ylabel="P(new answer)", xticks=x, xticklabels=labels, ylim=(0, 1))
    axes[0, 1].set(title="New-answer margin over old answer", ylabel="logit(new) − logit(old)", xticks=x, xticklabels=labels)
    channel_x = [row["discovery_activation_change"] for row in channel_rows]
    channel_y = [row["exact_target_margin_change"] for row in channel_rows]
    axes[1, 0].scatter(channel_x, channel_y, s=55, color="#2774ae")
    for row in channel_rows:
        axes[1, 0].annotate(str(row["channel"]),
                            (row["discovery_activation_change"], row["exact_target_margin_change"]),
                            xytext=(4, 4), textcoords="offset points", fontsize=9)
    axes[1, 0].axhline(0, color="black", linewidth=.8); axes[1, 0].axvline(0, color="black", linewidth=.8)
    axes[1, 0].set(title="Largest pure-rule neurons: activation and causal effect",
                   xlabel="mean activation change on discovery pairs",
                   ylabel="exact new-vs-old margin change on this prompt")
    axes[1, 1].plot(x, [r["top32_delta_energy_fraction"] for r in records], marker="o", label="fraction of activation-change energy in top 32")
    axes[1, 1].plot(x, [r["outcomes"]["native_top32_patch"]["prediction"] == new_answer for r in records], marker="s", label="worked-example patch succeeded")
    axes[1, 1].set(title="Compactness and this example's outcome", ylabel="fraction / binary success", xticks=x, xticklabels=labels, ylim=(-.05, 1.05))
    axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.grid(alpha=.18); axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("One fixed counterfactual: change the subject's first syllable from 2 to 3", fontsize=15)
    fig.savefig(OUTPUT / "worked_example.png", dpi=180); plt.close(fig)

    lines = ["# Worked neuron-patching example", "",
             f"**Rule:** {payload['relation']}.", "",
             f"**Counterfactual:** {payload['change']}.", "",
             f"**Old answer:** `{payload['old_answer_token']}`. **Intended new answer:** `{payload['new_answer_token']}`.", "",
             f"**Base prompt:** `{payload['base_prompt']}`", "",
             f"**Changed prompt:** `{payload['changed_prompt']}`", "",
             f"Selection rule: {payload['selection_rule']}. There were {payload['pure_eligible_count']} eligible pure-rule examples, of which {payload['pure_native_success_count']} were native-patch successes. This is deliberately an illustrative success, not an unbiased estimate of effect size.", "",
             "## What the interventions did", "",
             "The values below are probabilities assigned to the intended new answer. The unchanged base prompt should stay low; changing the text itself is the natural upper reference.", "",
             "| arbitrary fraction | intact model understands both prompts? | unchanged base | text changed naturally | all 192 channels patched | top 32 native patched | top 32 rotated patched | top-32 delta energy |",
             "|---:|:---:|---:|---:|---:|---:|---:|---:|"]
    for row in records:
        o = row["outcomes"]
        lines.append(f"| {row['rho']:.0%} | {'yes' if row['intact_understands_pair'] else 'no'} | {o['base']['new_probability']:.3f} | {o['natural_changed']['new_probability']:.3f} | {o['full_patch']['new_probability']:.3f} | {o['native_top32_patch']['new_probability']:.3f} | {o['rotated_top32_patch']['new_probability']:.3f} | {row['top32_delta_energy_fraction']:.3f} |")
    lines += ["", "## The largest pure-rule channels", "",
              "Each single-channel effect is recomputed through the downstream residual stream and final RMSNorm. Positive values push this recipient toward the intended new answer; negative values oppose it.", "",
              "| channel | mean activation change | best simple held-out label | label R² | exact target-margin change |",
              "|---:|---:|---|---:|---:|"]
    for row in channel_rows:
        lines.append(f"| {row['channel']} | {row['discovery_activation_change']:+.3f} | {row['best_simple_label']} | {row['heldout_label_r2']:.3f} | {row['exact_target_margin_change']:+.3f} |")
    lines += ["", "## How to read this example", "",
              "- `text changed naturally` shows that the intact model understands both prompts; it is not an intervention.",
              "- `all 192 channels patched` adds the discovery-set mean activation change without changing the words.",
              "- `top 32 native` asks whether the trained neuron basis contains a compact version of that transformation.",
              "- `top 32 rotated` keeps the same number of coordinates after an arbitrary basis change. A native advantage is evidence for neuron alignment.",
              "- The channel labels are restricted descriptive fits, not proof that a neuron has only that meaning. Exact single-channel margin effects are the causal measurement.",
              "- This single prompt is illustrative. The aggregate interpolation figures, not this example, support the population claim."]
    (OUTPUT / "worked_example.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT / "worked_example.md")


if __name__ == "__main__":
    main()
