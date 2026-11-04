# Score-copy and bracket head ablations

Two independently trained, two-block bilinear models learned score copying and bracket closure from separate-rule RotoWire report edits. This check asks whether an OV head's weight size identifies its causal importance, and whether the two behaviors occupy distinct heads. The 28-game joint score×bracket panel is game-disjoint from training and contains four controlled edits per game; no joint edits were used for training or fitting activation statistics.

For each of eight attention heads, we zeroed that head's output columns in `O`, leaving all other weights frozen. We measured paired exact prediction at both rule sites and KL from the intact model. We also measured the Frobenius norm of `O_h V_h` and its input-covariance-weighted norm using separate-rule training reports.

| Seed | Native paired bracket / score | Strongest singleton effects on the 28-game panel |
| --- | --- | --- |
| 55801 | .964 / 1.000 | L1H3 ablation: bracket .000, score 1.000, bracket KL 18.11. L1H1: bracket .893, score .000, score KL 7.34. L1H2: bracket .964, score .429, score KL 2.16. |
| 55802 | .911 / .929 | L1H2: bracket .000, score .179, bracket KL 24.02, score KL 4.23. L1H1: bracket .893, score .250, score KL 4.42. |

The first seed has a notably bracket-specific head and at least two score-relevant heads. The second seed puts both behaviors through L1H2, so head-level separation is not stable across initialization. This is compatible with reusable or entangled lower-level representations, but singleton ablation cannot distinguish those explanations. First-layer head ablations had smaller paired-exact effects, though this does not mean the first layer is unimportant: it may be redundant at the head level, and later heads consume its outputs.

Weight size is a poor screening criterion here. In seed 55801, the heads with the largest score and bracket ablation effects have `||O_h V_h||_F` about 2.47 and 2.43, while several weak-effect first-layer heads are about 2.8–3.1. Activation-weighted norms show the same broad mismatch. Neither norm accounts for attention selection, downstream interactions, or cancellation.

This is a causal *head-output* test, not a full circuit decomposition or a proof of unique mechanisms. Exact prediction and teacher KL answer different questions; a head may shift probabilities without changing the top token. The next interpretability step should use the behaviorally competent two-rule benchmark as a validation target for methods, while DGP development broadens naturalistic sources, typed realizations, and heldout rule combinations. Do not extend this into an open-ended single-task head hunt.

Reproduce with `ablate_score_bracket_heads.py`. Aggregate results: `results/score_bracket_head_ablation_seed55801_n28.json` and `results/score_bracket_head_ablation_seed55802_n28.json`. The model checkpoints remain local.
