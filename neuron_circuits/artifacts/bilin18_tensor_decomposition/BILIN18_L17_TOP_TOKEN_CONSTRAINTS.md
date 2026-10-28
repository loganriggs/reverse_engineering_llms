

## H383 — Number of likely-token constraints in oracle fitting

Prespecified top-K ladder: 8, 32, 128, 512. At each recipient site, select tokens by intact native probability, condition and renormalize probabilities on that set, then fit the full target response in each frozen probability/random basis (ranks 16/64). Use the minimum coefficient-norm pseudoinverse solution, relative eigenvalue cutoff 1e-12, and omit the cap derivative as in H381–382. Keep native capped logits, full vocabulary finite KL evaluation, and the same eight-step target-dependent safeguard. Compare with the cached full-recipient uncapped-derivative fit. Reused 512-document panel; all 39 head/response conditions. This is a diagnostic oracle, not target-free prediction or model compression.

|Basis|Head|Family|Full64|Top8|Top32|Top128|Top512|
|---|---|---|---:|---:|---:|---:|---:|
|probability|0|actual|0.860|0.616|0.677|0.742|0.848|
|probability|0|moment|0.744|0.478|0.528|0.604|0.729|
|probability|0|donor|0.751|0.486|0.538|0.613|0.736|
|probability|1|actual|0.797|0.544|0.610|0.673|0.778|
|probability|1|moment|0.781|0.521|0.582|0.650|0.762|
|probability|1|donor|0.782|0.516|0.584|0.652|0.764|
|probability|6|actual|0.820|0.561|0.639|0.701|0.801|
|probability|6|moment|0.840|0.553|0.657|0.709|0.830|
|probability|6|donor|0.842|0.558|0.662|0.712|0.832|
|random|0|actual|0.809|0.526|0.657|0.727|0.796|
|random|0|moment|0.722|0.483|0.582|0.639|0.711|
|random|0|donor|0.728|0.488|0.588|0.646|0.716|
|random|1|actual|0.765|0.511|0.627|0.685|0.752|
|random|1|moment|0.746|0.505|0.606|0.664|0.733|
|random|1|donor|0.749|0.504|0.608|0.666|0.736|
|random|6|actual|0.775|0.480|0.607|0.687|0.758|
|random|6|moment|0.815|0.502|0.662|0.750|0.809|
|random|6|donor|0.816|0.504|0.662|0.751|0.811|

Frozen gates: {'top32_within5': False, 'top512_within5': False, 'rank_bound': True}. The two recovery gates require every full-minus-truncated simultaneous upper bound to be below 5 percentage points, across both bases, both ranks, and all nine head/family groups. Rank-bound audit is separate from the scientific recovery criteria.

|K|Mean retained probability mass|Rank64 recovery range|Full-minus-truncated point loss range (both ranks)|Largest joint upper loss|
|---:|---:|---:|---:|---:|
|8|0.6915|0.478–0.616|0.159–0.313|0.349|
|32|0.8127|0.528–0.677|0.109–0.235|0.271|
|128|0.8956|0.604–0.751|0.018–0.141|0.176|
|512|0.9502|0.711–0.848|0.000–0.019|0.055|

Rank deficiency is expected: a covariance over K tokens has rank at most K−1. The retained eigenvalue ranks obey that bound in both bases. Underconstrained fits use the minimum-norm coefficient solution; this choice affects unseen-token extrapolation. More available basis directions therefore need not act like more independent token constraints. All finite errors on omitted tokens remain included in scoring.

Audits: input artifact hashes verified; normal-equation relative residuals [5.608495026723444e-12, 7.320280629171928e-13]. Local solutions do not exceed Euclidean coefficient error under their own truncated fitting metric. Independent saved-array audit confirms grid-zero native baseline replay within 0; saved minima exactly match the grid; retained mass is nondecreasing with K. Full per-site ranks, coefficient norm ratios, retained mass, all finite grid scores and choices are saved. Nonworsening against baseline follows by construction from including step zero.

Uncertainty: 4,000 paired document/control-draw bootstrap resamples; simultaneous max-absolute-deviation bands over all 144 full-minus-truncated contrasts. The random orientation and empirical donor assignments are fixed. This is a reused discovery panel, not independent confirmation. Figure contains point estimates; the JSON retains all bands. Independent audit fields are added by work/audit_h383.py after the experiment runner.

Interpretation: the handful-of-token explanation is too small for this fitting procedure. Top 32 loses 10.88–23.52 percentage points across all tested settings; every loss remains positive in the simultaneous band. Top 128 still loses as much as 14.05 points. Top 512, retaining 95.02% of intact probability mass on average, is much closer: all 36 point losses are between 0.012 and 1.888 points. Its largest simultaneous upper bound is 5.482 points, so the frozen five-point criterion FAILS; do not relabel this as demonstrated equivalence. The shared max-deviation band across all 144 contrasts also reflects variability in the smaller-K conditions. Any narrower follow-up requires a new prespecified analysis and fresh data.

Rank-64 guarded recovery ranges: full 72.24–86.01%, top8 47.76–61.62%, top32 52.75–67.68%, top128 60.40–75.13%, top512 71.09–84.75%. The response can be approximated substantially using a context-selected vocabulary subset, but current evidence favors hundreds of constraints over a tiny likely-token set. This is about the metric needed to choose coefficients in arbitrary spans, not a count of latent semantic features or independent computations. K token constraints are correlated and their effective rank is at most min(K−1, span dimension).

The tail may matter through additional identities, through aggregate regularization of directions weakly constrained by high-probability tokens, or both. Normal-equation audits pass, so the loss is not attributable to an obvious solver failure. Large pre-safeguard coefficient norm ratios remain: maximum squared norm ratio 52.36 in the probability basis and 21.73 in the random basis. Excluded-token finite extrapolation and the minimum-norm convention are part of the mechanism being tested, not nuisance details to discard.

Updated priorities: (1) hundreds of context-specific token constraints suffice approximately; supported by top512 point estimates, awaiting confirmation; (2) a handful suffice; disfavored for this solver and panel; (3) tail constraints mainly stabilize weak directions rather than encode distinct semantic content; plausible, not separated; (4) output covariance is strongly context dependent even within likely-token subsets; supported by H382; (5) a privileged feature basis explains most recovery; weakened by similar random-span results; (6) response-dependent oracle fitting is the remaining source of apparent simplicity; still unresolved and central to the broader goal.

Next discriminator: freeze top512 and full-recipient fits on fresh complete documents, with an explicitly declared joint 36-contrast five-point noninferiority criterion. Keep both bases, both ranks, heads/families, and finite safeguard unchanged. Exclude all 5,312 previously used complete-document hashes. Confirm or reject the small loss without moving the threshold. Then separate tail identities from tail stabilization using top-K plus a prespecified isotropic/shared tail metric, and ultimately replace target-dependent coefficient selection with an input-derived predictor. Broader research goal remains active.
