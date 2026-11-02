# First-layer attention-score sparsity: uniform versus observed token pairs

The [iterated weight L1/prune test](VARIED_ITERATED_SPARSITY_PATH_RESULTS.md) made the varied bilinear language model sparse in parameter coordinates without concentrating its token-pair QK score function or isolating the known rules. This experiment applies a penalty to that **function** instead. It tests both a data-free uniform-vocabulary prior and a distribution-aware prior on generated token pairs. The comparison includes ordinary 150-step finetuning with no functional penalty.

The exact first-layer fold maps each of the 317 token embeddings through the block's RMSNorm and QK1/QK2 projections, retains RoPE for relative lags 1, 4, and 16, then forms each head's product score table $S_h^{(\ell)}(q,k)=\mathrm{QK1}_h^{(\ell)}(q,k)\mathrm{QK2}_h^{(\ell)}(q,k)$. Its native-block replay error is about `1e-7` relative L2 for all three fitted models. The tables cover **all** token pairs, regardless of whether a pair occurs in generated text.

[`fit_varied_functional_sparsity.py`](fit_varied_functional_sparsity.py) minimizes the same next-token CE and exact-oracle verb soft-target loss as the earlier pruning experiment, plus a coefficient-0.1, scale-invariant concentration penalty. For each lag, the penalty is $(\mathbb E|S|/\sqrt{\mathbb E S^2})$ divided by that quantity in the original checkpoint, averaged over the three lags. The uniform arm averages over all vocabulary pairs and heads. The observed arm weights pairs by counts from 8,192 independently generated training streams; a separate 8,192-stream sample measures pair support. The no-penalty control has identical training examples, initialization, steps, and optimizer. Each arm runs 150 CPU updates with seed 3. This metric encourages a few large scores and many small ones; it does **not** enforce exact zeros or sparse OV paths.

| Model | Fresh overall KL | Fresh verb KL | Fresh induction-verb KL | Lag-1 top 1% score mass | Lag-4 | Lag-16 |
|---|---:|---:|---:|---:|---:|---:|
| Original checkpoint | .0503 | .0628 | .0770 | .322 | .323 | .310 |
| Finetune control | **.0442** | **.0523** | **.0676** | .324 | .327 | .316 |
| Uniform-vocabulary penalty | .0447 | .0541 | .0696 | **.562** | **.616** | **.665** |
| Observed-pair penalty | .0455 | .0551 | .0705 | .376 | .386 | .437 |

This is a real change in the score function: ordinary finetuning barely changes concentration, while the uniform penalty nearly doubles the top-1% mass at some lags. On separately generated actual causal token pairs, the lag-agnostic top-1% score-product mass is .402 for the original, .402 for the control, .493 for the uniform penalty, and **.524 for the observed-pair penalty**. The data-aware objective concentrates more on actual contexts even though its full-vocabulary table is less concentrated.

The **uniform** objective also exploits unused parts of the vocabulary table. On the independent 8,192-stream support sample, only **6.0%** of the uniform arm's top-1% lag-one score entries were seen, carrying **2.7%** of that top set's absolute mass. For the control, 11.2% of top entries were seen, carrying 8.9% of top mass. The observed-pair arm had 10.3% seen, carrying 12.2% of top mass. These finite-sample numbers do not classify unseen pairs as grammar-impossible, but they expose the tendency to concentrate score magnitude away from observed language. In the uniform arm, first-layer head 2's strongest score grew from about 5 to about 58 for object/`Sometimes` pairs, and its observed output-path RMS rose from .66 to 1.50. That is not by itself a legible causal rule.

On the same independent 512-stream causal-edit panel, the control, uniform, and observed arms have signed-effect L1 errors:

| Rule edit | Control | Uniform | Observed |
|---|---:|---:|---:|
| Lexical verb | .240 | .243 | .248 |
| Induction | .240 | .241 | .245 |
| Previous place | .214 | .214 | .219 |
| Cue | .210 | .212 | .213 |

Complete-head removal on a separate 256-stream panel barely changes which heads affect induction and previous-place edits. We therefore have evidence for **functional score concentration with modest predictive cost**, but not for a simpler causal decomposition or better task understanding. Distribution-aware weighting suppresses the off-support concentration failure; it does not yet improve rule specificity. A sensible next objective must act on complete QK×OV paths or intervention effects, compare at matched KL and causal fidelity, and explicitly evaluate held-out token combinations and task families. The current support audit covers a finite generator sample, not FineWeb or natural language.

Reproducible outputs: [`control`](surface_mixture_v2_varied_functional_shape_control/), [`uniform`](surface_mixture_v2_varied_functional_shape_p10/), [`observed`](surface_mixture_v2_varied_functional_observed_p10/), and the [held-out pair-support audit](functional_score_support_n8192.json). The fitter writes resumable checkpoints every 25 steps and final outputs atomically; interrupted preliminary runs with empty files were discarded.
