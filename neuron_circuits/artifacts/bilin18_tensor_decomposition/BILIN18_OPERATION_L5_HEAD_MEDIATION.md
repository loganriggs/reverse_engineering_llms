## H753 — L5's operation influence is distributed across heads and nonadditive

This sweep swaps each of nine L5 attention-head outputs before the output projection, at the matched final-query and source-key positions, between binding/copy contexts. An all-head swap replays the preceding whole-attention-output experiment. Each run uses free downstream Q2/K2 states, full native clamps, or the frozen rank-four operation/identity/wording/random spherical clamps. All240 H747 contexts remain; the table summarizes120 competent narratives. Head numbering is zero-based.

| L5 head | Full-node dependence RMS | Operation approximation error | Identity error | Wording error | Random error |
|---|---:|---:|---:|---:|---:|
| H0 | 0.0268 | 39.6% | 68.4% | 88.1% | 105.0% |
| H1 | 0.0226 | 60.9% | 68.3% | 104.1% | 99.5% |
| H2 | 0.0210 | 77.1% | 71.0% | 103.1% | 99.8% |
| H3 | 0.0234 | 26.8% | 75.7% | 89.0% | 99.8% |
| H4 | 0.0080 | 72.5% | 99.5% | 86.1% | 100.4% |
| H5 | 0.0390 | 34.4% | 53.7% | 91.9% | 99.1% |
| H6 | 0.0233 | 70.2% | 88.9% | 87.7% | 98.8% |
| H7 | 0.0216 | 51.4% | 86.8% | 90.8% | 96.5% |
| H8 | 0.0115 | 61.0% | 80.4% | 90.2% | 100.7% |
| All heads together | 0.1262 | 25.4% | 101.8% | 92.1% | 100.7% |

Full-node dependence is free-minus-full-clamped intended-margin change, measured in logits. Approximation error compares free-minus-compact-clamped change with that dependence and normalizes by its RMS. These are small selected-node effects within larger total head-swap effects, not percentages of task computation. All-head total margin-change RMS is1.775; individual totals range0.080–0.601.

H5 has the largest individual dependence magnitude; H3 has the lowest operation approximation error and a clearer separation from nuisance controls. H5's identity control also approximates some of its dependence (53.7% error), so it is useful as a comparator for overlapping identity/operation structure. Neither head alone accounts for the full layer's dependence. This does not establish that H3 contains one semantic feature or that H5 contains a particular manifold.

Nonadditivity is substantial: summing the nine isolated free-minus-full-clamped effects differs from the simultaneous all-head effect by48.6% relative RMS on competent narratives. For operation-coordinate dependence, the discrepancy is49.6%. These contrasts can interact through downstream computations and normalization; isolated effects cannot be interpreted as additive causal shares. Diagnostic list discrepancies are18.7%/18.8%, but those tasks still fail native competence and are not evidence of successful transfer.

Interpretation: the upstream route is distributed and context-dependent. L5H3 is a promising candidate for unsupervised recovery of a relatively operation-specific component, while L5H5 provides a contrasting case with stronger identity-related overlap. The all-head result must remain a reference because single-head explanations miss interactions. This is an exploratory selection on reused task data, not an independent confirmation of these head labels.

Audits:16,140 forwards =240 native captures +14,400 head/scope interventions +1,440 no-swap controls +60 self-swaps. Native logits replay exactly; all-head H752 margin replay error<=4.29e-6; no-swap/self-swap logit error<=1.34e-5; full raw clamps exact; compact coordinate restoration error<=6.35e-7; normalized norm error<=2.06e-6. Complement rescaling range0.903–1.062. All hashes unchanged, outputs finite; zero original reserved-test evaluations. Natural document count26,112 unchanged.

Next: collect task-blind natural activations for L5H3 and L5H5 using the established train/validation documents, then compare unsupervised scalar, grouped/overlapping and linear baselines against held-out causal targets. Keep all-head and diverse-task references, matched reconstruction/active dimension controls, and the distinction between recovery of this supervised reference and discovery of a meaningful manifold. Avoid inferring a privileged feature basis from compression alone.
