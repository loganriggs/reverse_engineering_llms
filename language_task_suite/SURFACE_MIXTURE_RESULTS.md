# Readable mixed-rule language: scope and validation

The original 26-token [mixed-rule benchmark](EXISTING_GENERATOR_BRIDGE.md) gives an exact joint DGP but does not resemble natural text. `surface_mixture_language.py` is an intermediate test: 141 word/punctuation tokens, present-tense event sentences, variable visible tails, and four mechanisms competing at each verb. It is **not** presented as a FineWeb match. The existing [document world](V2_RESULTS.md) remains the separate benchmark for state updates and semantic binding.

## Generator and oracle

A sentence such as “Meanwhile, Frank holds the old book inside the school before the season while Kate watches.” comes from a fully specified autoregressive distribution. The subject has a 40% chance to repeat the prior sentence's subject, making local discourse and induction visible. At the verb slot, the next-token distribution is the weighted union of:

1. a 70/20/10 lexical subject→verb distribution;
2. the last verb used with that subject, if one exists;
3. a lead-word-conditioned verb proposal;
4. a proposal determined by the previous sentence's place.

Weights are 0.40/0.25/0.20/0.15, normalized over active mechanisms. Other slots have explicit distributions over people, objects, places, modifiers, and events, or deterministic punctuation/function words. The next-token oracle is computed from the **visible prefix**; the sentence frame and tail type are observable, with no latent bracket state. Unlike the priority-based older `generator_v3.py`, the rule proposals genuinely compete probabilistically. The model trains on all next tokens, while evaluation separates verb-slot KL from easy frame tokens.

`audit_surface_mixture.py` replayed every prefix of 3,000 × 70 generated tokens and found oracle normalization, actual-token support, and independent grammar checks pass. Across **13,118** verb slots, induction was active **36.1%** of the time. Two, three, and four mechanisms were active at 3,000, 5,378, and 4,740 slots respectively. Using *modal output disagreement* as the collision definition, **97.2%** of verb slots had competing proposals. In 200,000 independently generated four-token streams, 1,593 happened to begin `Today , Alice`; the empirical distribution of the next verb was within **0.026 total variation** of the oracle. This is a reachable-prefix check of the whole sampler, though still modest in sample size.

## Surface realism check

The synthetic 3,000 × 70-token sample has **139 distinct alphabetic words**, mean **14.0 words per sentence**, and p90 **16**. The frozen local FineWeb panel of 480 GPT-2-token windows has **17,932** distinct alphabetic words, mean **17.8 words per sentence**, and p90 **32** under the same crude regex. The units differ in length and are not parallel documents, so type/token ratios are not directly comparable. The lexical and syntactic-diversity gaps are plainly large. The new generator improves readability and supplies controlled rule collisions, but **does not pass the user's realism criterion yet**. Its rigid sentence skeleton and all-present-tense event style are especially unlike the web panel.

The [manual FineWeb cue review](fineweb_dependency_review.json) provides a second, qualitative calibration. It uses six deterministic sampled regex matches per cue, stores row coordinates and labels rather than article text, and is not a prevalence estimate. In those inspected examples, all six numbered-list hits were genuine enumerations but served different purposes (instructions, references, rankings); all six parentheticals were real but ranged from dates to glosses and citations. All six weekday hits were dates, schedules, or narrative time, not weekday-successor computations. All six `more than` hits were numeric/duration thresholds or an idiom, not the toy's transitive comparison chain. Of six state-change-verb hits, one described physical location, one stored information, one list membership, and three idiomatic or phrasal uses. Pronouns included specific, generic, and group reference. These distinctions matter when selecting tasks: surface-cue frequency cannot stand in for frequency of the exact latent dependency.

## Model and interpretation tests

`train_surface_mixture.py` trains attention-only bilinear transformers and reports expected KL to the exact DGP on 256 held-out 70-token prefixes, separately for all tokens, all verbs, and verbs with and without induction. Width is 64, with four heads, batch size 64, 5,000 AdamW updates, and one seed per depth. The 1–3-layer models have no normalization. The unnormalized 4-layer run produced a non-finite loss at step 1,708 and saved no checkpoint (`surface_mixture_models_v1/unnormalized_depth4_failure.json`). A separate 4-layer run with per-block RMSNorm completed. Because it changes the architecture, its gain cannot be attributed to depth alone.

| Model | All-token KL | Verb KL | Verb KL with induction |
| --- | ---: | ---: | ---: |
| 1 layer, no norm | 0.0181 | 0.1687 | 0.2156 |
| 2 layers, no norm | 0.0154 | 0.1259 | 0.1693 |
| 3 layers, no norm | 0.0143 | 0.1308 | 0.1648 |
| 4 layers, RMSNorm | 0.0092 | 0.0838 | 0.1016 |

All-token KL hides difficulty because most frame positions are easy or deterministic. Induction remains the hardest reported subset.

`eval_surface_mixture_edits.py` makes **grammar-valid, single-token changes** before the current verb. A changed lead alters only the lead proposal; a previous occurrence of the current subject receives a different supported verb to alter induction; a previous place receives a different supported place to alter carryover. Changing a previous adjective is a no-effect control: the exact current-verb oracle stays unchanged. Across 256 fresh generated sequences, the average L1 error between the model's and oracle's *probability changes* is:

| Model | Lead edit | Previous-place edit | Previous-verb induction edit | Irrelevant adjective |
| --- | ---: | ---: | ---: | ---: |
| 1 layer | 0.311 | 0.369 | 0.349 | 0.007 |
| 2 layers | 0.260 | 0.284 | 0.331 | 0.013 |
| 3 layers | 0.259 | 0.287 | 0.321 | 0.009 |
| 4 layers, RMSNorm | 0.235 | 0.222 | 0.278 | 0.011 |

The four-layer model improves each causal edit class while largely ignoring the nuisance edit, but it still does not implement induction exactly. An edit can change the probability of its *earlier* prefix under the DGP even though the edited prefix is valid; these metrics condition on the resulting visible prefix. The full results also report oracle/model total variation and effect-direction cosine.

`eval_surface_bond_gauges.py` tests local orthogonal bases of each attention-output bond. Full-width rotations reproduce logits within numerical tolerance; truncation is an intervention. At **32/64 coordinates** in the 4-layer model's final bond, the weight-only SVD basis has verb model-to-truncated KL below 0.001 on a separate test set, versus about **0.037** for the original coordinate order and **0.009** for activation PCA. At 16/64, SVD's model-to-truncated KL is 0.022, so a smaller basis is not lossless.

Crucially, causal-edit fidelity survives the 32-coordinate SVD truncation: induction effect L1 error is **0.278** both before and after truncation, while 32 raw axes increase it to **0.304**; previous-place error is **0.222** before/after SVD and **0.284** on raw axes. At 16 coordinates, weight SVD still gives previous-place/induction errors **0.246/0.272**, versus **0.370/0.321** on raw axes. These are local compression results on a **single** trained model. SVD does not itself label directions as lexical, induction, or discourse components, and a lower edit error after truncation can reflect accidental regularization. See `surface_mixture_models_v1_rmsnorm/bond_depth4_layer3.json` and its `paired_edits_layer3_*` results.

`eval_surface_edit_bases.py` probes a more interpretation-directed prior. It fits PCA to *bond activation differences* caused by one edit family on 256 fitting sequences, then tests patches on 256 separate sequences and the other edit families. At the third attention bond (layer index 2), patching the donor's **full sequence** of bond activations shows 16-coordinate family-specific bases. The table gives mean L1 error between each restricted patch's probability change and the **full-width bond patch**; lower means the local bond effect is better retained.

| Fitting basis, 16/64 coordinates | Lead edits | Induction edits | Place edits |
| --- | ---: | ---: | ---: |
| Physical axes | 0.104 | 0.070 | 0.045 |
| Activation PCA | 0.122 | 0.132 | 0.111 |
| Weight SVD | 0.064 | 0.064 | 0.042 |
| **Lead-edit contrast** | **0.013** | 0.089 | 0.065 |
| **Induction-edit contrast** | 0.084 | **0.020** | 0.055 |
| **Place-edit contrast** | 0.051 | 0.066 | **0.020** |

This is evidence that intervention labels can select different, compact **local response subspaces**. It is **not** recovery of complete rule circuits. Even a full-width patch at this one bond leaves average L1 residuals of **0.444/0.137/0.240** against the edited model's lead/induction/place effects, because other positions, residual paths, and layers remain. At the final bond, the full-patch residual is worse for all three; a basis that compresses that local patch can still move predictions *away* from the exact oracle. The experiment therefore preserves the distinction between local mediation, model-behavior fidelity, and DGP fidelity. The 256-sequence result is `surface_mixture_models_v1_rmsnorm/edit_basis_transfer_layer2_all_positions_full.json`; smaller positional and layer controls are adjacent.

The next mechanistic test should combine causal paths across bonds and residual streams, and evaluate **whole lexical-combination holdouts** rather than only fresh sequences. Sparse and shared-dictionary featurizers should be compared with these contrast bases at matched width, model fidelity, and oracle effect fidelity. The next *language* revision should add genuinely different sentence and discourse frames, richer vocabulary, and a manual naturalness evaluation; matching only average sentence length would not close the measured realism gap.
