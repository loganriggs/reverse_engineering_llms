#!/usr/bin/env python3
"""Visualize role-level address and payload decomposition."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/bilin18_tensor_decomposition"
BASE = ARTIFACT / "bilin18_transfer_state_signed_role_address.json"
HIGH = ARTIFACT / "bilin18_transfer_state_signed_role_address_highsample.json"
OUTPUT = ARTIFACT / "bilin18_signed_role_address.png"


def main() -> None:
    base = json.loads(BASE.read_text())
    high = json.loads(HIGH.read_text())
    boundaries = [6, 10, 14, 17]
    address_roles = ["mlp", "queries", "keys", "value", "query_key", "query_value", "key_value", "attention_no_mlp"]
    labels = ["MLP", "Q", "K", "V", "Q+K", "Q+V", "K+V", "attention/no MLP"]

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for role, label in zip(address_roles, labels):
        axes[0].plot(boundaries, [high["median_jaccard"][role][str(boundary)] for boundary in boundaries], marker="o", label=label)
    axes[0].set_xticks(boundaries); axes[0].set_ylim(0, 1.05)
    axes[0].set_xlabel("residual boundary"); axes[0].set_ylabel("top-8 address Jaccard")
    axes[0].set_title("Which roles recover exact read addresses?")
    axes[0].legend(fontsize=7, ncol=2); axes[0].grid(alpha=.25)

    targets = list(high["results"])
    address = []
    self_contained = []
    for role in address_roles:
        address.append(np.mean([
            high["results"][target]["6"][f"address_{role}"]["response_fidelity"]
            for target in targets
        ]))
        self_contained.append(np.mean([
            high["results"][target]["6"][f"self_{role}"]["response_fidelity"]
            for target in targets
        ]))
    x = np.arange(len(labels)); width = .38
    axes[1].bar(x - width / 2, address, width, label="addresses + exact payload basis")
    axes[1].bar(x + width / 2, self_contained, width, label="self-contained")
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].set_ylim(0, .8); axes[1].set_ylabel("mean boundary-6 causal fidelity")
    axes[1].set_title("Address and payload are different tasks")
    axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=.25)

    methods = ["address_full", "address_value", "address_key_value", "self_query_value", "self_key_value", "self_value"]
    display = ["exact address", "V address", "K+V address", "Q+V self", "K+V self", "V self"]
    medians = []; maxima = []
    for method in methods:
        differences = []
        for target in base["results"]:
            for boundary in base["results"][target]:
                differences.append(abs(
                    base["results"][target][boundary][method]["response_fidelity"]
                    - high["results"][target][boundary][method]["response_fidelity"]
                ))
        medians.append(np.median(differences)); maxima.append(np.max(differences))
    axes[2].bar(x[:len(methods)] - width / 2, medians, width, label="median |change|")
    axes[2].bar(x[:len(methods)] + width / 2, maxima, width, label="maximum |change|")
    axes[2].set_xticks(x[:len(methods)], display, rotation=35, ha="right")
    axes[2].set_ylim(0, .06); axes[2].set_ylabel("fidelity difference")
    axes[2].set_title("Independent higher-sample replication")
    axes[2].legend(fontsize=8); axes[2].grid(axis="y", alpha=.25)

    figure.suptitle("Keys address, queries interpret, values carry", fontsize=14)
    figure.savefig(OUTPUT, dpi=180)
    print(OUTPUT)


if __name__ == "__main__":
    main()
