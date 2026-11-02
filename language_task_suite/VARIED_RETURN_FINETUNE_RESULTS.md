# Training distribution changes the long-range induction circuit

The [second-layer source assay](VARIED_SECOND_LAYER_SOURCE_CIRCUITS.md) found that the original varied-language model often locates an earlier verb in head 0 even when its long-range induction effect is much too weak. This experiment tests whether extra training on non-immediate subject returns strengthens that computation, with an extra-training control and a middle distribution. It does not claim that the toy text matches FineWeb; the earlier [dependency-proxy audit](DEPENDENCY_PROXY_REALISM_AND_REPEAT_TRADEOFF.md) found a more realistic repeat rate for the return setting but a persistent vocabulary and surface-form gap.

All three two-layer, four-head, width-64 bilinear models start from the **same trained varied-language checkpoint**. Each receives 600 additional CPU AdamW updates at learning rate .001, batch 64, length 72, with sampled next-token CE and exact-oracle verb soft-target weight 2. Seed 12, architecture, optimizer, and token budget are matched. Only the subject-selection distribution changes:

| Arm | Previous-subject repeat mass | Non-immediate earlier-subject return mass |
|---|---:|---:|
| Extra-training control | .40 | .00 |
| Return-focused | .10 | .30 |
| Mixed | .25 | .15 |

The mixed subject distribution is the pointwise arithmetic mean of the first two subject-selection distributions. The rest of the language generator is unchanged. This makes it a controlled curriculum/prevalence comparison, though the *sampled sequences* necessarily differ across arms.

[`eval_varied_return_finetunes.py`](eval_varied_return_finetunes.py) evaluates all checkpoints on the **same** 512-stream seed 20261209 in the original, return-training, and harder return-stress regimes. Induction-site exact-oracle KL (lower is better):

| Model | Original regime .40/.00 | Return-training regime .10/.30 | Return-stress regime .40/.50 |
|---|---:|---:|---:|
| Original checkpoint | .0805 | .2003 | .1374 |
| Extra-training control | **.0722** | .1999 | .1362 |
| Return-focused finetune | .1253 | **.1279** | .1206 |
| Mixed finetune | .0840 | .1474 | **.1124** |

Aggregate KL can obscure the causal tradeoff. On the same return-stress generator, grammar-valid earlier-verb edits were split at source-to-next-token distance 24. Two disjoint 512-stream edit panels had 550/572 short and 478/440 long edits. The exact oracle probability shift has L1 magnitude **.5**. The model's mean signed effect magnitudes were:

| Model | Short edit effect L1, panel 1 / 2 | Long edit effect L1, panel 1 / 2 | Long edit effect cosine with oracle, panel 1 / 2 |
|---|---:|---:|---:|
| Original checkpoint | .443 / .450 | .122 / .121 | .430 / .402 |
| Extra-training control | .411 / .416 | .108 / .108 | .338 / .293 |
| Return-focused finetune | .241 / .242 | **.264 / .257** | **.909 / .919** |
| Mixed finetune | .346 / .349 | .174 / .172 | .861 / .848 |

The return-focused arm recovers much of the *direction* of the long effect, but still undershoots its oracle magnitude and harms the short effect. The mixed arm occupies an intermediate causal point despite having the best aggregate induction KL on the stress regime. None is fully competent on both distance strata after 600 updates. This is evidence that the original failure was at least partly **learnability/training distribution**, not proof of an architectural impossibility. It is also not a proof of a permanent short/long capacity tradeoff: training remains brief and only one initialization is tested.

The mechanistic source test gives a corresponding change. At median long-source lag 35, second-block head 0 ranks the edited earlier verb as its top aligned source in .801/.805 of cases for the original model, .753/.761 for extra original-stream training, **.948/.932** for return-focused training, and .929/.909 for mixed training. The head assignment stays similar, but the retrieval is more consistently centered on the right source. A factor counterfactual on long edits (first panel) finds the head-0 edited-source full-patch probability effect grows from **.072 to .154 L1** from original to return-focused training; QK-routing-only grows .085→.167 and V-only .070→.142. Changing only QK2 grows .074→.157. These effects are nonlinear, contextual, and not additive shares, but show both routing and value strengthening rather than a mere change of head label.

The causal edit panels use a .40/.50 **stress** return distribution, which differs from all three finetuning distributions. Short/long pairs are measured on identical generated examples across models, so their comparison is valid, but the experiment cannot separate distance from all other consequences of changing subject returns. The source/factor patches remain hybrid-state interventions. The result supplies a useful **causal fidelity target** for future featurizers: preserve the sign, magnitude, source, and QK/V roles of edits across distance, instead of maximizing score concentration alone. Stop extending this single-task curriculum sweep until a path-level featurizer or canonicalization method is compared on that target alongside other held-out task families and language surfaces.

Reproduce with [`train_surface_mixture.py`](train_surface_mixture.py) using `--generator v2_varied --depths 2 --steps 600 --batch 64 --length 72 --width 64 --heads 4 --rmsnorm --lr .001 --verb-soft-weight 2 --seed 12 --init-checkpoint` pointing at the original model, plus the rates above. The [common KL record](varied_return_finetune_common_kl_n512.json) and the [`finetune_control_s12/`](surface_mixture_v2_varied_finetune_control_s12/), [`finetune_return01_03_s12/`](surface_mixture_v2_varied_finetune_return01_03_s12/), and [`finetune_mixed25_15_s12/`](surface_mixture_v2_varied_finetune_mixed25_15_s12/) directories contain checkpoints and short/long causal-source results for both edit seeds. The original model directory contains the matching baseline stress-panel records.
