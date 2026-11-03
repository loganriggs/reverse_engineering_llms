# The shared closure head does not contain a single weight-SVD closure mode

The [complete-head intervention](CROSS_SOURCE_CLOSURE_HEAD_ABLATION.md) localized sports bracket and FineWeb delimiter closure to one late attention head per seed. This follow-up removes individual **weight-only OV singular modes** within that head and retains its top-ranked modes cumulatively. The result rejects the simplest “one singular direction is the shared closure circuit” account. Both behaviors survive substantial OV compression, but their dependence on individual modes differs.

For each critical head, form its complete value-to-output product (O_hV_h\in\mathbb R^{64\times64}), compute its SVD, and refactor the product into the original 16 value coordinates. This is a gauge change because bilinear attention multiplies the value vector by a scalar score per head before applying (O_h). Full-rank refactorization changes probe logits by at most (1.14\times10^{-5}) and (1.53\times10^{-5}) in the two seeds. A single-mode intervention removes one singular outer product and refactors the remainder. The decomposition itself uses **weights only**; no task examples fit or select its modes. Scores use the same 100-game heldout sports and 164-document fresh FineWeb panels as the head screen.

| Seed | Top OV rank retained | Weight Frobenius energy retained | Sports bracket paired | FineWeb delimiter paired | Sports alias paired |
|---|---:|---:|---:|---:|---:|
| 41701 | full 16 | 1.000 | 1.000 | .527 | .745 |
| 41701 | 1 | .198 | .765 | .314 | .745 |
| 41701 | 2 | .361 | .980 | .421 | .745 |
| 41701 | 4 | .606 | .990 | .500 | .745 |
| 41701 | 8 | .849 | 1.000 | .530 | .745 |
| 41702 | full 16 | 1.000 | .990 | .683 | .680 |
| 41702 | 1 | .214 | .665 | .037 | .670 |
| 41702 | 2 | .363 | .990 | .387 | .670 |
| 41702 | 4 | .572 | .990 | .384 | .660 |
| 41702 | 8 | .841 | .990 | .662 | .680 |

Removing only mode 0 lowers **sports bracket paired** accuracy by .000 and .160 across seeds, despite complete-head drops of 1.000 and .865. It lowers **FineWeb delimiter paired** accuracy by .155 and .604. In seed 41701, no single mode lowers sports bracket paired by more than .010, yet retaining just the top two modes preserves .980: that behavior is robust to one-mode removal and fragile to removing most of the head. Seed 41702 likewise needs at least two leading modes to restore .990 sports paired accuracy. FineWeb closure improves more gradually with rank and reaches near-full performance at rank 8. Alias copying remains largely unchanged throughout.

Thus the common necessary head contains a **low-rank closure-supporting subspace**, but the Euclidean weight-SVD axes do not isolate one reusable closure computation across sources. This may reflect redundant modes, distinct closure features within the same head, or dependence on the head's QK source selection and downstream nonlinearities. The SVD is canonical for the OV product under an ordinary Euclidean norm; it is not a canonicalization of the full attention function. The next comparison should fit an activation-weighted and a rule-effect-weighted basis on training-only examples, then evaluate their rank–fidelity and causal-specificity frontiers on whole-task heldouts. A selected mode must also survive a fresh heldout confirmation before being named a circuit.

The [runner](ablate_cross_source_ov_modes.py) and [aggregate result](results/cross_source_closure_ov_mode_ablation.json) contain every mode's singular value, energy, rule score, and full-rank gauge check. The source-token panels and model checkpoints remain local.
