

## H359 — Matching residual direction as well as confidence

Reuse H356/357 native responses for heads0,1,6. At each position select24 nearest other documents by standardized entropy/top1 log odds, then choose the six with largest cosine to the recipient’s intact final residual direction. No response outcomes enter neighbor selection. Re-evaluate native response directional Fisher energy at those donor states; retain unnormalized, per-site geometric pairing statistic. Compare with prior six confidence-only neighbors.

|Head|Moment confidence-only|Moment residual-matched [joint95%]|Donor confidence-only|Donor residual-matched [joint95%]|
|---|---:|---:|---:|---:|
|0|0.887|0.959 [0.876, 1.049]|0.896|0.965 [0.882, 1.056]|
|1|1.102|1.061 [0.970, 1.161]|1.060|1.058 [0.967, 1.158]|
|6|1.365|1.203 [1.100, 1.317]|1.306|1.185 [1.083, 1.297]|

Head6 residual/confidence ratios: moment0.882 [0.810,0.959], donor0.907 [0.834,0.987], using separate joint2 bands. Both upper bounds are below1, but neither establishes at least10% attenuation at the frozen threshold. Head6 remains above1 under joint6 bands, but the stronger persistence criterion lower>1.2 fails. Heads0/1 bands include1; that does not establish equivalence.

Matching audit: mean residual cosine rises0.394→0.694, exceeding the frozen0.05 improvement threshold. Mean standardized confidence distance worsens0.208→0.451. The comparison changes a bundle of donor properties; attenuation cannot be attributed solely to residual direction. The geometry effect is plausible, but this is not exact confound control or a causal mediation estimate.

Frozen outcomes: {'head6_persists': False, 'head6_attenuation10': False, 'cosine_improvement05': True}. These constrain a strong residual-geometry-independent pairing explanation while leaving a weaker residual effect.

Audit: cached native responses and source hashes verified; donors exclude self and preserve token position; own-context rows reused identically. All new metric energies and neighbor identities saved. No native weight changes. Confidence-only head estimates replay H357 by construction.

Uncertainty:4000 joint recipient-document/control-realization/neighbor-rank resamples. Joint6 bands cover selected heads×families; joint2 bands cover head6 comparison. These are separate families, not one global band. Fixed matching and donor dependence not rebuilt; heads were selected after earlier outcomes; reused panel.

Next discriminator: separate residual-direction matching from confidence degradation using matched confidence-distance controls, or preserve the exact confidence neighbor set and test a continuous relation between donor residual similarity and pairing. Prioritize this comparison before interpreting the residual effect as semantic coordination. Broad research goal remains active.
