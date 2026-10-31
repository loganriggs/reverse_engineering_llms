"""Compare a fixed post-RoPE bond cut with its pre-RoPE covariant relocation."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from flagship_full_model_eval import get_cuts
from flagship_matrix_free import HD, rotate, rotary
from flagship_rank96_function_grouping import ids_and_projections
from flagship_rank96_group_stability import CUTS
from flagship_transfer import ART
from flagship_matrix_free import SNAPSHOT


PAIRS = {
    "reference": (7, 8),
    "translated": (31, 32),
    "changed_gap": (32, 64),
}


def panel_response(query, key, value_sqrt, key_pos, query_pos, bond):
    cos, sin = rotary(max(key_pos, query_pos) + 1)
    q1 = rotate(query["c_q"], np.full(len(query["c_q"]), query_pos), cos, sin)
    q2 = rotate(query["c_q2"], np.full(len(query["c_q2"]), query_pos), cos, sin)
    k1 = rotate(key["c_k"], np.full(len(key["c_k"]), key_pos), cos, sin)
    k2 = rotate(key["c_k2"], np.full(len(key["c_k2"]), key_pos), cos, sin)
    score = q1 @ bond @ k1.T / HD
    other = q2 @ k2.T / HD
    value = key["c_v"] @ value_sqrt
    response = (score * other) @ value / len(k1)
    return score, response


def row_rotation(position):
    cos, sin = rotary(position + 1)
    return rotate(np.eye(HD), np.full(HD, position), cos, sin)


def pre_cut_as_post_cut(bond, query_pos, reference_pos=8):
    """Move a post cut to pre-RoPE at the reference and back at query_pos.

    Matrix inverses account for the stored, rounded rotary tables, which are
    close to orthogonal but not exactly orthogonal in float64.
    """
    if query_pos == reference_pos:
        return bond
    reference = row_rotation(reference_pos)
    current = row_rotation(query_pos)
    pre_cut = reference @ bond @ np.linalg.inv(reference)
    return np.linalg.solve(current, pre_cut @ current)


def relative_error(estimate, target):
    return float(np.sum((estimate - target) ** 2) / np.sum(target ** 2))


def run(n_queries=128, n_keys=8192):
    torch.set_num_threads(4)
    state = torch.load(SNAPSHOT / "pytorch_model.bin", map_location="cpu",
                       weights_only=True, mmap=True)
    query, key, value_sqrt, panel = ids_and_projections(state, n_queries, n_keys)
    del state
    cuts = get_cuts()
    records = []
    for split in ("fit", "test"):
        for law in ("uniform", "corpus"):
            for pair_name, (key_pos, query_pos) in PAIRS.items():
                for cut_name in CUTS:
                    exact_score, exact_response = panel_response(
                        query[split], key[split][law], value_sqrt,
                        key_pos, query_pos, np.eye(HD))
                    fixed = cuts[cut_name]
                    covariant = pre_cut_as_post_cut(fixed, query_pos)
                    fixed_score, fixed_response = panel_response(
                        query[split], key[split][law], value_sqrt,
                        key_pos, query_pos, fixed)
                    pre_score, pre_response = panel_response(
                        query[split], key[split][law], value_sqrt,
                        key_pos, query_pos, covariant)
                    records.append(dict(
                        split=split, law=law, pair=pair_name, cut=cut_name,
                        positions=[key_pos, query_pos],
                        post_score_relative_mse=relative_error(fixed_score, exact_score),
                        pre_score_relative_mse=relative_error(pre_score, exact_score),
                        post_response_relative_mse=relative_error(fixed_response, exact_response),
                        pre_response_relative_mse=relative_error(pre_response, exact_response),
                        pre_post_score_relative_mse=relative_error(pre_score, fixed_score),
                        pre_post_response_relative_mse=relative_error(pre_response, fixed_response),
                    ))
                print(split, law, pair_name, flush=True)
    result = dict(
        layer=0, head=0, rank=96, n_query_anchors_per_law=n_queries,
        n_key_anchors_per_law=n_keys, panel=panel, records=records,
        definition="For row rotary matrix R_p, a reference post-RoPE cut X at query position 8 is relocated to a pre-RoPE query cut Y=R_8 X inv(R_8); at query position p its equivalent post-RoPE matrix is inv(R_p) Y R_p. Inverses retain exactness with the stored rounded tables.",
        scope="first-layer, single-head, sampled-token pair response; not a full-model KL or branch-intervention result",
    )
    name=f"head0_prerope_cut_transfer_q{n_queries}_k{n_keys}.json"
    (ART / name).write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-queries", type=int, default=128)
    p.add_argument("--n-keys", type=int, default=8192)
    args = p.parse_args()
    run(args.n_queries, args.n_keys)
