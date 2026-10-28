

## H284 — Intact-state gains do not transfer to improved source responses

H283 froze a(.5 linear,.1 quadratic,.4 outside) mixture with modest fresh intact-state KL gains. H284 evaluates its response changes, without refitting, for all nine source heads at removal scales.25 and1.0. Angle controls preserve each source payload norm and dot product with the pre-MLP residual, within the head output span. Constants cancel from response differences. The base correction uses stable polynomial differences; adapter differences use their saved maps and matrices. Native weights, compiler, adapter weights and mixture choices are unchanged.

The downstream prediction is changed residual + native intact MLP output + predicted MLP change + native bias. All models share the native intact output. Thus conditional KL measures response repair with an oracle intact anchor, not deployable intact replacement; it must not be compared numerically to H283 intact KL as if they were the same metric. The residual fidelity denominator is the native response minus the original compiler response. No CE repair claim is made in this conditional experiment.

Fresh128 documents exclude2880 prior unique complete-document hashes. Full document/window/file hashes and window contents independently pass. All heads and four positions are clustered within each bootstrapped document.2000 bootstrap draws,95% intervals conditional on frozen fits. Angle norm/dot maximum relative discrepancies3.34e-7/2.11e-7. Oracle reconstruction KL stays below2.13e-13, all saved metrics finite. Existing polynomial/feature-map tests supply prior validation; the full suite was not rerun. Float32 forward differences for the adapters are used at finite scales.25/1, not infinitesimal derivatives.

| Payload / scale | Base conditional KL | Mixture conditional KL | Base residual fidelity | Mixture residual fidelity |
|---|---:|---:|---:|---:|
|actual_s0.25|0.000007210|0.000007338|41.56%|40.74%|
|actual_s1.0|0.000115332|0.000118360|41.24%|40.32%|
|angle_s0.25|0.000010337|0.000010364|28.10%|27.53%|
|angle_s1.0|0.000179323|0.000180669|27.99%|27.36%|

- actual_s0.25, mixture_vs_base: -1.78% relative conditional-KL reduction, paired95% [-5.00,0.52]%.
- actual_s0.25, mixture_vs_shrink: 12.95% relative conditional-KL reduction, paired95% [11.37,14.65]%.
- actual_s1.0, mixture_vs_base: -2.63% relative conditional-KL reduction, paired95% [-6.12,-0.16]%.
- actual_s1.0, mixture_vs_shrink: 13.46% relative conditional-KL reduction, paired95% [11.81,15.26]%.
- angle_s0.25, mixture_vs_base: -0.26% relative conditional-KL reduction, paired95% [-1.52,1.03]%.
- angle_s0.25, mixture_vs_shrink: 5.18% relative conditional-KL reduction, paired95% [3.99,6.44]%.
- angle_s1.0, mixture_vs_base: -0.75% relative conditional-KL reduction, paired95% [-2.23,0.72]%.
- angle_s1.0, mixture_vs_shrink: 5.25% relative conditional-KL reduction, paired95% [3.82,6.75]%.

Both5% improvement gates FAIL: actual and angle responses do not improve over the base at either scale. Both retention gates PASS: mixture outperforms the shrunken single adapter, comfortably satisfying lower95% bound>−3%. These successes refer to retention relative to an adapter that itself worsens responses, not superiority to the base. At full actual removal, mixture KL is2.63% worse than base, with interval corresponding to.16–6.12% worsening. Other mixture/base intervals include zero. Top1 agreement is near99% because interventions are small; that does not establish high response fidelity or intact accuracy. Per-head results and raw document arrays are retained for heterogeneity analysis without selecting favorable heads.

**Hypothesis update.** There is a measured state/response tradeoff in this fitted family. H283 gains came from state-supervised corrections; H284 finds lower response fidelity than base and no conditional-KL improvement. Mixture averaging mitigates the stronger response damage from the single linear adapter but does not erase it. This reinforces the need to characterize units by intervention behavior, not merely reconstruction or intact-output fit. It does not prove state and response objectives are inherently incompatible, nor that the added coordinates lack causal relevance.

**Next discriminator.** Hold the feature maps fixed and fit a small, predeclared response-regularized readout sweep. Contrast intact KL only with intact KL plus an explicit native response-error loss on discovery interventions, using separate document validation for both metrics. Start with the existing linear64 adapter for tractable attribution, then test whether it admits a state/response Pareto improvement before widening the family. Use actual and angle discovery payloads, retain held-out heads as a transfer diagnostic, and compare to both zero adapter and H280 adapter. Freeze selection using both validation metrics before a new test. If regularization preserves state gains while restoring responses, loss mismatch is sufficient for this tradeoff; if state gains collapse, investigate estimation and geometry instead of asserting impossibility. This follow-up is not yet run.

All jobs terminal. Source weights/compiler unchanged. Research goal remains active.
