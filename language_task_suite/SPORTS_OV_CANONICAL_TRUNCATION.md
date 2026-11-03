# A first OV gauge reduction on rule-verified natural reports

The sports alias×bracket bilinear models are a narrow natural-text bridge: both seeds learned bracket closure on heldout game reports, while alias copying remains partial. This experiment asks whether an exact **value–output gauge** can expose a smaller representation that preserves the learned bracket behavior. It does not claim a canonical decomposition of the whole transformer or a human-readable bracket circuit.

For one attention head, write its value–output map as `M_h = O_h V_h` (64×64, rank at most 16). At fixed bilinear QK scores, any factorization of the same `M_h` computes exactly the same head output. SVD of `M_h` yields balanced value and output factors; using all 16 directions is an exact gauge change. Using only rank `r` is a function-changing truncation. This identity remains exact even though the trained model uses RMSNorm; the norm is upstream of the value projection. The [reproducible evaluator](eval_sports_ov_canonical_truncation.py) changes all four heads in both attention layers, leaving QK, MLPs, embeddings, and unembedding intact.

Two rank priors are compared at matched `r`:

* **Weight-only:** best Frobenius rank-`r` approximation of each `M_h`.
* **Data-aware:** best rank-`r` approximation after right weighting by the square root of a regularized input covariance. The covariance uses normalized value inputs from 80 train-game worlds, all four controlled arms per world. This optimizes a local value-map error under the fit activations, not the end-to-end model or the bracket rule.

Both are refactored into the same 16-coordinate head shape, with the unused coordinates set to zero. At rank 8, half the V/O coordinates and their corresponding dense input/output connections remain; at rank 4, one quarter remain. Two independently trained width-64 depth-2 seeds are scored on the **same 40 heldout-game worlds** (160 controlled arms) selected by seed 41601. Fit and test game keys do not overlap. Alongside token NLL and paired rule behavior, the evaluator computes teacher-to-truncated-model KL at both task sites and a fixed sparse sample of background token positions.

| Trained seed | Method | Rank/head | Bracket paired exact | Bracket-site KL | Alias paired exact | Alias-site KL | Background KL |
|---|---|---:|---:|---:|---:|---:|---:|
| 98201 | Native / rank 16 | 16 | .925 | ~0 | .288 | ~0 | ~0 |
| 98201 | Weight | 4 | .900 | .043 | .075 | 1.792 | .268 |
| 98201 | Data | 4 | .850 | .077 | .000 | 2.205 | .159 |
| 98201 | Weight | 8 | .925 | .008 | .125 | .797 | .085 |
| 98201 | Data | 8 | .925 | .003 | .100 | .907 | .054 |
| 98202 | Native / rank 16 | 16 | .950 | ~0 | .338 | ~0 | ~0 |
| 98202 | Weight | 4 | .500 | .973 | .000 | 3.191 | .316 |
| 98202 | Data | 4 | .863 | .182 | .000 | 3.745 | .231 |
| 98202 | Weight | 8 | .763 | .362 | .087 | 1.661 | .130 |
| 98202 | Data | 8 | .950 | .006 | .075 | 1.720 | .100 |

The full-rank reparameterization changes probe logits by at most 2.7×10⁻⁵ and has near-zero teacher KL, verifying the implemented gauge within float32 numerical error. Rank-1 and rank-2 variants in the [first-seed aggregate](results/sports_ov_canonical_n80_n40.json) fail bracket paired exact entirely; rank 4 is a sharp transition for that seed. The corresponding [first-seed teacher-KL result](results/sports_ov_canonical_kl_seed98201_n80_n40.json) and [second-seed result](results/sports_ov_canonical_kl_seed98202_n80_n40.json) separate functional fidelity from next-token loss.

The robust contrast is **task-specific**. Data-aware rank 8 preserves bracket paired accuracy in both seeds, but substantially changes alias predictions. It also lowers background KL versus weight-only rank 8 in both seeds. At rank 4, the better bracket method switches across trained seeds: weight-only is better in seed 98201 and data-aware in seed 98202. So no universal prior wins this panel. Even the successful bracket compression identifies only a low-rank OV path sufficient for this behavior on these heldout games; it does not prove the directions are human-legible, independently causal, or reusable in other genres.

Next method comparison: optimize a task-conditional sparse or shared basis at matched active V/O connections, and require heldout bracket behavior **and** edit specificity while tracking alias degradation and teacher KL separately. Evaluate on a fresh report style before calling any recovered path portable. The wider sampled language DGP and multi-rule behavioral gates remain the main objective; this is a positive natural-form target for testing reduction methods.
