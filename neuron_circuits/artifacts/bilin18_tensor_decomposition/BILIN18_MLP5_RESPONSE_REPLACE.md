## H627 — MLP5 context affects downstream response, but later computation sustains the differences

The previous turn was progress: H626 analytically separated MLP5 normalization and bilinear context factors. H627 tests their downstream causal consequences. Add±0.02 times the frozen vector at final Attention5 output. In replacement runs, set final MLP5 output to its own cached native output plus alpha times a chosen response. Earlier token outputs, each recipient's native baseline, and every later layer remain live.

Six conditions are native live computation, native analytic response, capital-mean bilinear context with native normalization response, native context with capital-mean normalization response, both factors shared, and zero MLP5 perturbation response. These are tested at−0.02,0,+0.02 on72 prompts:1,296 forwards. The analytic native replacement checks whether local linearization or patching itself introduces a discrepancy. All alpha0 outputs match native exactly; analytic replacement changes final response by at most0.000721 relative weighted norm.

**The bilinear-context difference has a downstream causal effect.** On held-out language forms, replacing that context changes the final vocabulary response by0.379/0.913/0.319 relative to its native norm. Sharing only normalization changes it by0.009/0.044/0.059. This supports the local factor distinction from H626 at the output, rather than only in MLP write geometry. It does not assign additive fractions of behavior to either factor.

| Held-out cell | Shared-context response change | Shared-normalization response change | Both-shared response change | Zero-MLP-response change | Native answer-logprob derivative | Shared-context derivative | Both-shared derivative |
|---|---:|---:|---:|---:|---:|---:|---:|
|capital/0|0.0326|0.0028|0.0302|0.3875|0.5447|0.5471|0.5466|
|capital/1|0.1414|0.0177|0.1414|0.4668|2.8758|2.7880|2.7548|
|capital/2|0.4953|0.0270|0.4931|0.6470|0.4922|0.6655|0.6667|
|language/0|0.3793|0.0092|0.3745|0.4394|0.1149|0.0709|0.0723|
|language/1|0.9129|0.0438|0.8794|0.6253|-0.2353|-0.1106|-0.1113|
|language/2|0.3189|0.0595|0.3209|0.5103|0.0904|0.0577|0.0585|
|plural/0|0.2445|0.0111|0.2401|0.4533|0.2220|0.1935|0.1897|
|plural/1|0.9582|0.0172|0.9548|0.4507|0.3563|0.3432|0.3427|
|plural/2|0.5770|0.0361|0.5636|0.6578|0.1178|0.0875|0.0708|

Response change uses the full vocabulary, native-probability-weighted centered logit norm, normalized by the native response norm. It measures the difference between two local intervention effects, not a percentage of a semantic computation. An error near1 can involve gain, direction, or both. Intended-token derivatives are addition-signed local derivatives of log probability, not finite-dose probability changes or task accuracy.

**MLP5 is not the whole explanation.** For language “speak”, native intended-token derivative is−0.235; shared context changes it to−0.111, still negative. Shared normalization gives−0.225. Setting both factors to the same capital means gives−0.111. Under both-shared replacement, the MLP5 perturbation response—and therefore the total block5 perturbation response—is identical across prompts. Yet final responses remain strongly context-dependent. Native downstream states and their readouts are sufficient to transform that common response differently.

To assess remaining differences, compare each recipient with the capital-discovery mean response under the *same replacement mode*, allowing an oracle signed scalar gain. For language “speak”, relative shape error is0.975 natively and0.984 after both-sharing. For language questions it is0.851 versus0.848. Thus making this early response common does not yield a common downstream vocabulary-response shape. These are diagnostic fits to observed responses; the scalar gain is not predicted from text.

**Removing MLP5's response does not remove the overall intervention.** The injected residual vector still travels through the skip path and later computation. Zeroing the MLP5 perturbation response changes final response norm and shape, but all tested cell-mean response norms remain nonzero; the “speak” answer derivative stays negative at−0.104. This is compatible with distributed downstream transformation. It does not establish that MLP5 is unnecessary for native behavior, because its native output is preserved.

The experiment substitutes first-order writes, not native finite-dose MLP functions. Small symmetric alpha and the analytic control justify the local interpretation. At large dose, higher-order responses and their interactions would need separate testing. Shared bilinear factors are analytic operators and need not correspond to attainable native states. The panel and capital discovery examples are familiar; this is a mechanism test within an adaptive research sequence.

Twenty hypotheses after this causal check:

1. The injected direction has one context-independent semantic output effect — weakened by H623–H627.
2. It supports answer-emission modulation on the tested capital prompts — supported locally.
3. Its effect is solely a vocabulary bias — weakened by small-dose response-shape changes.
4. A common bias with a signed scalar gain is sufficient — weakened by oracle-gain errors.
5. A fixed vector can still be a useful local intervention — supported despite context dependence.
6. It primarily encodes a specific donor country — weakened by cross-country transfer and selectivity checks.
7. Capital identity probabilities never change — false; rank can persist while probability shifts.
8. The final position suffices for approximate output-distribution effects — supported on the earlier matched panel.
9. Final-only and full intervention are exactly decision-equivalent — false on some recipients.
10. Rank2 gives a useful compact output approximation — supported locally, not universal.
11. Raw context cosine measures functional equivalence well — weakened by H626.
12. Input RMS variation dominates the first response divergence — weakened for this vector at MLP5.
13. Bilinear operating context drives much of that local divergence — supported analytically and causally.
14. MLP5 alone creates the language sign reversal — weakened; replacement does not remove it.
15. Later computation can transform a shared block5 response differently — supported by both-shared/zero-response controls.
16. Final readout alone explains all response variation — weakened by H625.
17. Both intermediate transformations and readout matter — supported.
18. Small-dose response predicts all finite-dose behavior — weakened by H624's curvature tests.
19. High choice accuracy among an incomplete answer set proves task competence — contradicted by plural controls.
20. A compact context-conditioned operator is a better unit than a standalone feature vector — plausible research direction, not yet a validated general model.

The current evidence favors a conditional-computation description: a reusable direction is transformed by a sequence of context-dependent local operators. This is more informative than assigning the direction a universal semantic name, but it is not yet a predictive explanation of the full mechanism or a canonical form. The next experiment should test a concrete operator model rather than merely adding labels.

Next build and validate final-position tangent propagation through later blocks. Hold native states fixed while applying each block's local derivative to an incoming perturbation, then compare with end-to-end finite differences. Once validated, substitute reference-context Jacobians at selected stages while retaining the propagated incoming response. That preserves the distinction between changing a local operator and replacing its output with a fixed vector. It can test which contextual transformations are necessary for the remaining response variation without assigning nonadditive layer shares.

Audits: all source/manifest/vector/trace/factor/runner hashes unchanged;1,296 forwards; every mode's zero-alpha baseline exact; live±0.02 reproduces prior logits exactly; response-write replacement error≤5.36e-6; native analytic output-response error≤0.000721; finite metrics. No new natural-text documents; FineWeb remains26,112. Broad research goal active and incomplete.
