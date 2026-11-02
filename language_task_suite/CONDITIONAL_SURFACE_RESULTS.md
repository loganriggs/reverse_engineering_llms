# Readable joint-rule language with a visible strength intervention

This experiment extends the [original toy-language proposal](https://www.lesswrong.com/posts/GncmyuKmp526RJpmj/ambitious-mech-interp-w-tensor-transformers-on-toy-languages) toward the question it emphasizes: do trained circuits implement the known *interference* between language rules? `conditional_surface_language.py` uses the readable event stream, not the separate `Reports/Guide/FAQ/Policy` task mixture. At every verb, four proposals compete: a subject→verb lexical distribution, previous-subject induction when available, a lead-word proposal, and a previous-place proposal. The opening visible mode token changes their weights while leaving the body grammar and the support of every body prefix intact.

| Mode | Lexical | Induction | Lead | Place skip |
| --- | ---: | ---: | ---: | ---: |
| Balanced | .40 | .25 | .20 | .15 |
| Lexical | .70 | .10 | .12 | .08 |
| Recall | .20 | .55 | .15 | .10 |
| Setting | .20 | .15 | .15 | .50 |

The mode is an explicit diagnostic cue, not a claim about a naturally occurring register. All weights are positive. On a fixed visible body prefix, changing only the cue therefore defines a valid counterfactual conditional distribution. Frame-token distributions are exactly invariant to the cue.

## Generator checks and realism

`audit_conditional_surface_language.py` replayed every prefix of **3,000 × 72** generated streams, checked sampled-token support and normalization, and independently recalculated each verb distribution from named candidate rules and weights. It saw **13,139** verb slots, with **12,816** modal conflicts between candidate rules. Each of the 3,376 Balanced-mode verb prefixes changed under every alternate mode; mean oracle total-variation shifts were **0.196** to Lexical, **0.132** to Recall and **0.282** to Setting. Every checked frame-token distribution was invariant.

This is still a synthetic diagnostic, not a FineWeb match. On 480 windows of 72 tokens, it has **143** distinct alphabetic words versus **6,305** in the matched-length FineWeb prefixes; mean sentence length is **14.2** versus **16.3** words and p90 **16** versus **29**. The tokenizers differ. The surface is more readable than the original 26-token stream, but remains too rigid and lexically narrow to satisfy the realism objective.

## Model behavior versus the exact oracle

`train_conditional_surface.py` trained two- and three-layer width-64, four-head, attention-only bilinear transformers with RMSNorm for 4,000 AdamW steps, batch 64, on equal mode exposure. These are one-seed exploratory models. On 256 held-out streams **per mode**, induction-active verb KL to the exact oracle was:

| Layers | Balanced | Lexical | Recall | Setting |
| ---: | ---: | ---: | ---: | ---: |
| 2 | .164 | .100 | .318 | .196 |
| 3 | .144 | .098 | .234 | .162 |

The recall-heavy mode is hardest. The more decisive test is `eval_conditional_surface_shift.py`: sample body prefixes, then evaluate the **same prefix** with Balanced and each alternate cue. Among 1,024 Balanced-generated streams, 856 lexical-versus-induction conflict prefixes qualified. The table reports RMSE of the model's cue-induced lexical-minus-induction log-odds shift against the exact oracle. A constant prediction equal to the oracle's mean shift is a deliberately simple baseline.

| Change from Balanced | Oracle mean shift | Two-layer RMSE | Three-layer RMSE | Constant-mean RMSE |
| --- | ---: | ---: | ---: | ---: |
| Lexical | +1.175 | .482 | .491 | **.277** |
| Recall | −1.330 | .477 | .441 | **.147** |
| Setting | −.142 | .590 | .510 | **.377** |

The models get much of the **average** shift but fail to implement its context dependence: both are worse than the constant-mean baseline. The result repeats when prefixes come from the Recall-mode distribution (937 qualifying conflicts). On thousands of sampled frame positions where the oracle shift is exactly zero, the models' probability-change L1 is only about **0.004–0.006**. Thus they do use the cue selectively for verb predictions, but the detailed rule mixture remains inaccurate.

Upweighting sampled verb-token CE fivefold during otherwise matched two-layer training improves final induction-active verb KL on Balanced/Lexical/Recall/Setting to **.128/.074/.308/.142**, from **.164/.100/.318/.196**. The fixed-prefix shift result is mixed: RMSE improves for Lexical **.482→.377** and Setting **.590→.403**, but worsens for Recall **.477→.569**. Better ordinary verb prediction does not automatically recover the strongest induction intervention.

The exact prefix oracle permits a stronger diagnostic: at verb positions, replace sampled-token CE with expected CE under the **full oracle distribution**, retaining fivefold verb weighting. This removes label-sampling noise while leaving the architecture, visible inputs, training stream seeds, optimizer and 4,000-step count matched to the sampled-label weighted run. The two-layer model's final induction-active verb KL falls to **.041/.020/.109/.047**; its Lexical/Recall/Setting fixed-prefix shift RMSE is **.312/.354/.177**. A three-layer model with the same supervision reaches KL **.035/.015/.088/.035** and shift RMSE **.280/.294/.225**. This is a large gain in oracle fit, especially for the recall-heavy mode, but the Recall shift remains worse than the constant-mean RMSE **.147**. Depth and ordinary KL can move differently from intervention fidelity (the three-layer Setting shift is worse than the two-layer one).

## What compact bond bases actually show

`eval_conditional_surface_bond_mediation.py` fits a tag-contrast PCA basis only on **no-conflict** positions, then patches a Recall-mode donor into a Balanced-mode recipient on held-out induction conflicts. For any first-layer non-tag query, changing only the tag changes its pre-output-projection attention activation within the span of two fixed tag-value vectors per head. With four heads, that contrast has rank at most **8**. The empirical rank-eight check passes; the top four directions contain **98.9%** of first-bond contrast energy in the two-layer model.

At the sampled-label two-layer first bond, the RMSE of a four-coordinate contrast-PCA patch versus the **donor model's** log-odds shift is **.033**, compared with **.025** for full-width patching, **1.187** for four raw coordinates, and **1.186** with no patch. A 16-coordinate weight-only SVD patch reaches **.030**. This is a strong local compression and mediation result that transfers from no-conflict fitting positions to held-out conflicts. The rank bound follows from the architecture and tag edit; it is not by itself evidence of a learned language rule. Crucially, the donor model's shift has the oracle errors in the preceding table.

In the more competent two-layer **soft-target** model, the same four-coordinate contrast basis still nearly reaches the full-bond patch (**.073** versus **.051** donor-shift RMSE), while 16-coordinate weight-only SVD worsens to **.518**. The first output matrix's top-16 singular values contain **87.2%** of its squared weight energy, versus **97.8%** for the original sampled-label model. Improved oracle fidelity coincides with less compressibility under this particular weight-only truncation. That does not invalidate weight-only methods; it shows why compression results must be conditioned on model competence and distinguished from causal accuracy.

In the three-layer model, a full-width first-bond patch at non-tag positions still leaves shift RMSE **.961** versus the donor; all-position and tag-only controls show that patching that bond's tag position adds little. Full-width patches at the second and third attention bonds reduce the residual to **.155** and **.118** respectively. Local route and compressibility change with depth, so a first-bond low-rank contrast should not be mistaken for an end-to-end canonical computation.

## Decision

The `pair` regime nearly isolates lexical versus induction rules, assigning only .01 weight each to lead and place proposals while preserving shared prefix support. Its `Setting` mode has **exactly the same rule weights as Balanced**, providing a no-effect cue control. With the same two-layer, 4,000-step, fivefold verb-weighted exact-oracle supervision, induction-active KL on Balanced/Lexical/Recall/Setting is **.030/.016/.087/.026**, versus **.041/.020/.109/.047** in the full four-rule regime. Yet the pair-regime fixed-prefix shift RMSE is **.435** for Lexical and **.375** for Recall, while the oracle shift itself varies by only **.126** and **.046** nats respectively across the qualifying conflicts. The model's predicted shifts vary much more (SD **.520** and **.385**). On the no-effect Setting cue, the model's spurious shift RMSE is **.026** and frame-token effect L1 is **.001**. Thus removing four-way interference improves ordinary oracle KL modestly but does **not** solve the context-level induction effect; the bottleneck is present even in a mostly two-rule mixture.

Distance stratification points to a sharper failure mode. On **2,048** Balanced-generated test streams, the full-regime two-layer soft-target model's Recall-change RMSE is **.265** for induction lag 1–16 tokens (459 cases), **.298** for lag 17–32 (1,180), and **1.096** beyond 32 (56). In the pair regime, evaluated on 1,024 streams, the corresponding values are **.306 / .328 / 1.089** (134 / 321 / 16 cases). The longest-lag counts remain modest, but the model mean shift nearly vanishes there while the oracle remains near −1.3.

A matched two-layer, full-regime run gave **10× additional loss weight** to >32-token induction targets, still using exact oracle soft targets and fivefold ordinary verb weighting. On the same 2,048-stream test set, it improves the long-lag Recall-shift RMSE from **1.096 to .606** and the long-lag model mean shift from **−.256 to −.828** (oracle approximately −1.28). But it worsens short/middle-lag RMSE from **.265/.298 to .417/.428**, and overall Recall RMSE from **.347 to .432**. Its ordinary induction-active verb KL is also worse, **.079/.042/.135/.096** across Balanced/Lexical/Recall/Setting versus **.041/.020/.109/.047** without extra long-lag weight. The result identifies a real long-range weakness but shows that simple reweighting trades off the common regime rather than solving the full rule.

The bottleneck is now clear enough to guide the next experiment: broadcasting a mode cue is easy and low-dimensional; implementing the exact **context-dependent normalization, overlap, and longer-range induction** of rule proposals is not. Stop treating further first-bond compression improvements as mechanism recovery until fixed-prefix shift RMSE beats the constant-mean baseline. Then test held-out collision patterns and a sparse/shared-dictionary basis against both model fidelity and oracle-effect fidelity. Keep the natural-text realism work separate: this stream remains a controlled bridge, not a realistic language model.
