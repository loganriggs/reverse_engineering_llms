

## H343 — Magnitude versus cancellation

Reuse H342 exact term Grams. With mean-linear response u and context-linear response v, tangent energy is a+b+2ρ√(ab), where a=||u||², b=||v||² and ρ is their Fisher cosine. Group a,b as magnitude and ρ as alignment. Attribute the actual-to-control energy difference by averaging both orders of replacing these two groups (a symmetric two-factor decomposition). This is algebraic accounting of aggregate geometry, not a realizable causal intervention.

|Modes|Control|Magnitude share (95% interval)|Alignment share (95% interval)|
|---|---|---:|---:|
|all|moment|41.60% [26.32, 54.03]|58.40% [45.97, 73.68]|
|all|donor|44.64% [33.37, 54.53]|55.36% [45.47, 66.63]|
|top4|moment|36.91% [18.50, 51.82]|63.09% [48.18, 81.50]|
|top4|donor|41.45% [26.78, 53.39]|58.55% [46.61, 73.22]|

Both magnitude and alignment contributions are positive across their pointwise intervals, but neither majority gate passes: both controls must have a full16 share lower confidence bound above50%. We cannot justify prioritizing a single dominant mechanism.

Audit: the two contributions sum to the tangent energy difference in every point estimate and all8000 scope/control bootstrap evaluations per scope. Source SHA256 verified. 4000 resamples jointly resample recipient documents and independently resample six global realizations per control family. Intervals are pointwise and conditional on frozen compiler, discovery mean, fixed donor assignments and sampled realizations. Donor recipients share source documents: this bootstrap does not reconstruct donor assignments or model all induced dependence. Top4 selection occurred on this panel; no heldout feature claim.

Next: compute the local compiler Jacobian restricted to source span, and measure actual/control normalized edit alignment with its sensitive input directions. This avoids dependence on the mean/context split and can distinguish low local sensitivity from an artifact of that split. Then freeze the analysis and repeat on fresh documents. Research goal remains active.
