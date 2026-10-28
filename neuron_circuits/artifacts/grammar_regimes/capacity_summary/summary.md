# Fixed-task capacity sweep

The task is fixed at 50% selected arbitrary assignments. Width changes; depth remains four. S90 and N90 are exact intervention measurements after a first-order ranking, not counts inferred from activations alone.

## Behavioral results

| MLP | width | seeds | gate passes | exception recall | rule generalization | template transfer |
|---|---:|---:|---:|---:|---:|---:|
| swiglu | 64 | 3 | 2/3 | 0.977 | 0.486 | 0.942 |
| swiglu | 96 | 3 | 2/3 | 0.984 | 0.431 | 0.929 |
| swiglu | 192 | 3 | 1/3 | 0.991 | 0.408 | 0.898 |
| swiglu | 384 | 3 | 2/3 | 0.983 | 0.337 | 0.929 |
| bilinear | 64 | 3 | 1/3 | 0.990 | 0.440 | 0.940 |
| bilinear | 96 | 3 | 2/3 | 0.977 | 0.429 | 0.926 |
| bilinear | 192 | 3 | 1/3 | 0.991 | 0.414 | 0.893 |
| bilinear | 384 | 3 | 2/3 | 0.973 | 0.285 | 0.907 |

## Causal monosemanticity profile

Each S90 entry is `mean seed-level median count (fraction of all MLP channels)`. Reuse is the fraction of matched recipients recovering at least 90% of their own MLP-added margin.

| MLP | width | exception S90 | exception axis advantage | exception reuse | generalized-rule S90 | rule axis advantage | rule reuse | large reversal rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| swiglu | 64 | 39 (0.076) | -0.09 | 0.23 | 11 (0.022) | -0.06 | 0.25 | 0.88 |
| swiglu | 96 | 42 (0.054) | 0.18 | 0.17 | 34 (0.044) | 0.61 | 0.32 | 0.67 |
| swiglu | 192 | 116 (0.075) | 1.58 | 0.06 | 88 (0.057) | 1.80 | 0.36 | 0.46 |
| swiglu | 384 | 74 (0.024) | 3.20 | 0.13 | 32 (0.010) | 2.20 | 0.40 | 0.50 |
| bilinear | 64 | 70 (0.136) | -0.17 | 0.25 | 23 (0.046) | 0.66 | 0.31 | 0.79 |
| bilinear | 96 | 70 (0.092) | 0.53 | 0.29 | 32 (0.041) | 0.97 | 0.23 | 0.54 |
| bilinear | 192 | 283 (0.184) | 0.34 | 0.13 | 123 (0.080) | 0.28 | 0.28 | 0.51 |
| bilinear | 384 | 336 (0.109) | 1.19 | 0.11 | 149 (0.049) | 1.25 | 0.18 | 0.48 |

## Interpretation and hypothesis verdicts

- **More width did not restore the shared rule.** Exception recall stayed around 0.97–0.99, while new structured-pair accuracy fell from 0.486 to 0.337 for SwiGLU and from 0.440 to 0.285 for bilinear. This falsifies the simple prediction that spare capacity separates exceptions and recovers rule generalization under this training objective.
- **The learned neuron basis becomes more privileged with width.** Native S90 is increasingly smaller than random-rotation S90, especially for width-384 SwiGLU. This supports axis-aligned causal specialization.
- **Axis alignment is not the same as semantic monosemanticity.** Same-rule-and-answer reuse does not rise reliably with width, and matched-answer exception reuse is low. Wider models appear to develop sparse, privileged coordinates without those coordinates forming universally reusable rule circuits.
- **The constraint-web signature remains visible.** Nearly every ranked-prefix curve has some reversal; large reversals become less frequent with width. This is consistent with cooperation/cancellation becoming milder as capacity grows, but it is not a KKT proof for these multi-layer Adam-trained models.
- **Architecture matters.** Width-384 SwiGLU circuits become very sparse as a fraction of available channels; bilinear absolute circuit counts generally remain larger. A single scalar would conceal this distinction.

## Reading CMP-90

- Small S90 says a compact ranked channel set can reconstruct the example's MLP contribution.
- Small N90 says the contribution is causally concentrated rather than robustly redundant.
- Positive axis advantage says the learned neuron coordinates are more compact than a random rotation.
- High same-rule reuse is the semantic test: a compact circuit found on one example also works on another example governed by the same relation.
- Sparse S90 without reuse is compatible with a fact-specific lookup and is not called monosemantic.

## Limitations

- Circuit panels contain 12 naturally correct examples per group per run; neurons are correlated within a model, so seeds—not examples—are the replication units.
- Several runs failed the strict unseen-template fidelity gate. They remain visible as open graph markers and are exploratory, not matched-fidelity confirmation.
- S90/N90 are smallest crossings on a fixed dense ranked-prefix grid. Non-monotonicity makes them resolution-limited upper-bound estimates, not globally minimal circuits.
- The reuse test controls relation plus answer for structured facts and answer plus different relation for exceptions, but a broader family of semantic patch tests is still needed.

## Figures

- `behavior_by_width.png`: whether capacity improves memorization, rules, or both.
- `cmp90_*.png`: separate causal profiles for exceptions, trained structured facts, and generalized structured facts.
- `worked_circuit_curves.png`: exact native-versus-rotated interventions for two concrete examples.
- `nonmonotonicity_by_width.png`: how often nested ranked circuits reverse direction, revealing cooperation or cancellation.
