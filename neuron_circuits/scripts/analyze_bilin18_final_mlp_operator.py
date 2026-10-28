#!/usr/bin/env python3
"""Exact Frobenius cosine between teacher/student bilinear operators.

For f(x) = sum_j d_j (l_j^T x)(r_j^T x), only the symmetrized input
matrix Q_j = (l_j r_j^T + r_j l_j^T) / 2 affects f.  The full operator is
T = sum_j d_j tensor Q_j.  Its inner products can be computed from factor
Gram matrices without materializing the 1152 x 1152 x 1152 tensor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from run_bilin18_qkov_causal_edges import load_full_model
from train_bilin18_final_mlp_student import NarrowBilinear


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-roots", nargs="+", type=Path, default=[
        Path("artifacts/bilin18_final_mlp_student"),
        Path("artifacts/bilin18_final_mlp_student_tiny"),
    ])
    parser.add_argument("--widths", default="8,16,32,64,128,256")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/bilin18_final_mlp_student/operator_cosine.json"))
    return parser.parse_args()


def q_inner(left_a, right_a, left_b, right_b):
    """All pairwise Frobenius products of sym(l r^T)."""
    return .5 * (
        (left_a @ left_b.T) * (right_a @ right_b.T)
        + (left_a @ right_b.T) * (right_a @ left_b.T)
    )


@torch.inference_mode()
def operator_norm_sq(left, right, down, chunk_size):
    total = torch.zeros((), device=left.device, dtype=torch.float64)
    for start in range(0, len(left), chunk_size):
        stop = min(start + chunk_size, len(left))
        qi = q_inner(left[start:stop], right[start:stop], left, right)
        di = down[:, start:stop].T @ down
        total += (qi.double() * di.double()).sum()
    return total


@torch.inference_mode()
def operator_inner(left_a, right_a, down_a, left_b, right_b, down_b, chunk_size):
    total = torch.zeros((), device=left_a.device, dtype=torch.float64)
    for start in range(0, len(left_a), chunk_size):
        stop = min(start + chunk_size, len(left_a))
        qi = q_inner(left_a[start:stop], right_a[start:stop], left_b, right_b)
        di = down_a[:, start:stop].T @ down_b
        total += (qi.double() * di.double()).sum()
    return total


def find_checkpoint(roots, width):
    for root in roots:
        path = root / f"student_k{width}.pt"
        if path.exists():
            return path
    raise FileNotFoundError(f"no checkpoint for width {width} under {roots}")


@torch.inference_mode()
def main():
    args = parse_args()
    model = load_full_model(args.device)
    teacher = model.blocks[-1].mlp
    tl = teacher.Left.weight.float()
    tr = teacher.Right.weight.float()
    td = teacher.Down.weight.float()
    teacher_sq = operator_norm_sq(tl, tr, td, args.chunk_size)
    results = {}
    for width in [int(value) for value in args.widths.split(",")]:
        path = find_checkpoint(args.student_roots, width)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        student = NarrowBilinear(saved["dimension"], saved["width"]).to(args.device)
        student.load_state_dict(saved["state_dict"])
        sl, sr, sd = student.balanced_factors()
        student_sq = operator_norm_sq(sl, sr, sd, args.chunk_size)
        inner = operator_inner(tl, tr, td, sl, sr, sd, args.chunk_size)
        cosine = inner / (teacher_sq * student_sq).sqrt().clamp_min(1e-30)
        projection_error = (1 - cosine.square()).clamp_min(0).sqrt()
        results[str(width)] = {
            "checkpoint": str(path),
            "operator_cosine": float(cosine),
            "best_rescaled_relative_frobenius_error": float(projection_error),
            "student_to_teacher_operator_norm": float((student_sq / teacher_sq).sqrt()),
        }
        print(width, results[str(width)], flush=True)
    output = {
        "definition": "cosine of T=sum_j d_j tensor sym(l_j r_j^T); biases excluded",
        "symmetries": "invariant to L/R swap, reciprocal scaling, paired signs, and channel permutation",
        "teacher_operator_norm_sq": float(teacher_sq),
        "students": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
