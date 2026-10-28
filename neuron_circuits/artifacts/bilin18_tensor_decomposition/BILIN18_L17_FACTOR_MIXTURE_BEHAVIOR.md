

## H329 — Behavioral effect of the missing cross-response component

Reuse H327’s128 documents and payload realizations. Compute the exact native final residual state after each source intervention. Replace only the missing residual cross response y with its frozen prediction p, giving state_native−y+p. All other native terms remain exact by oracle construction. Apply final RMS normalization, native unembedding and30×tanh(logits/30). Measure KL(native edited distribution || approximated distribution), top1 agreement and response fidelity. The baseline removes y entirely. No fitting or selection. This is an isolated component mediation test, not a complete deployable MLP replacement or a fresh-document replication of this behavioral endpoint.

|Target|Predictor|Response fidelity|Mean KL (nats)|KL recovery [95%]|Top1 agreement|
|---|---|---:|---:|---:|---:|
|actual|omitted|0.00%|3.44039e-05|0.00% [0.00,0.00]|99.72%|
|actual|full|90.90%|2.05713e-06|94.02% [93.44,94.54]|99.96%|
|actual|inherited1024|30.16%|2.03425e-05|40.87% [38.60,43.13]|99.74%|
|actual|pooled64|7.77%|3.15674e-05|8.24% [7.04,9.41]|99.78%|
|actual|pooled256|31.57%|2.32931e-05|32.30% [29.76,34.52]|99.76%|
|actual|pooled512|55.86%|1.548e-05|55.01% [52.66,57.16]|99.78%|
|actual|random256|20.12%|2.69808e-05|21.58% [19.78,23.22]|99.72%|
|angle|omitted|0.00%|4.96239e-05|0.00% [0.00,0.00]|99.65%|
|angle|full|90.53%|3.13034e-06|93.69% [92.61,94.58]|99.91%|
|angle|inherited1024|29.33%|2.80976e-05|43.38% [40.86,45.93]|99.67%|
|angle|pooled64|7.32%|4.57734e-05|7.76% [6.22,9.20]|99.65%|
|angle|pooled256|30.71%|3.37419e-05|32.00% [29.31,34.44]|99.74%|
|angle|pooled512|54.82%|2.17414e-05|56.19% [53.39,58.62]|99.78%|
|angle|random256|20.14%|3.96809e-05|20.04% [17.29,22.55]|99.65%|

Locked gates: full_kl80: PASS, pooled256_kl80: FAIL, pooled256_behavior_bonus10: FAIL. Full and pooled256 KL gates require>=80% recovery in both families. Behavior-bonus gate requires pooled256 KL recovery exceed response fidelity by>=10pp in both. Bootstrap resamples documents conditional on the frozen models and angle realizations.

**Scale finding.** Removing this component entirely produces only3.44e-5 actual/4.96e-5 angle mean KL, retaining99.72%/99.65% top1 agreement. Full prediction recovers94.02%/93.69% of that small effect. Pooled256 recovers32.30%/32.00%, close to its31.57%/30.71% response fidelity. Thus broad response reconstruction does not itself establish substantial behavioral importance. The next priority is comparing this component’s effect with the full source intervention and other residual terms before further compression.

**Meaning and limits.** KL recovery is relative to removing this specific component, not total model error. Absolute baseline KL and top1 agreement are included to prevent a large relative recovery from obscuring a small effect. High response fidelity need not imply high KL recovery; behavior depends on direction, context, normalization and the output distribution. This tests final-layer next-token effects, not task-level behavior, intact replacement, or semantic meaning. Pooled mixtures still compute all4608 factors.

**Audit.** Full and inherited1024 response errors replay H327 across all18 groups, maximum relative discrepancy0. Frozen model, basis and corpus hashes checked. KL values checked nonnegative to tolerance. Native edited state includes the actual residual change, exact native bilinear write and Down_bias. Native weights unchanged.

**Next discriminator.** First compare this small component effect with the full source intervention and other residual terms. If materially important effects remain, learn a downstream-sensitive metric on discovery data and test transfer against energy-based mixtures under fixed rank. Keep exact-component mediation separate from deployable program testing. Broad research goal remains active.
