

## H361 — RMS geometry carries the metric-pairing effect

Reuse H356 native head6 response vectors and confidence-matched contexts. Factor the directional probability metric into RMS-normalization derivative N(s) and downstream unembedding/cap/probability covariance L(s). Independently select recipient (O) or donor (D) state for each factor: OO, OD, DO, DD. Hybrids need not correspond to a realizable model state; they isolate components of the metric at fixed native response vectors.

Compute equal-site geometric control/actual Rayleigh ratios. At each site the scalar RMS factor cancels, so the normalization contrast principally tests its residual-direction projector. Attribute total log pairing logR(OO)−logR(DD) symmetrically between factors, averaging both orders of replacement. Shares are algebraic contributions to this log contrast, not causal percentages of behavior.

|Components (RMS, logits)|Moment control/actual|Donor control/actual|
|---|---:|---:|
|OO|2.197|1.970|
|OD|2.305|2.055|
|DO|1.651|1.544|
|DD|1.610|1.508|

|Contribution|Moment share [95% interval]|Donor share [95% interval]|
|---|---:|---:|
|normalization_share|103.58% [92.16, 121.59]|103.50% [90.24, 126.57]|
|logit_share|-3.58% [-21.59, 7.84]|-3.50% [-26.57, 9.76]|

Normalization-majority gate PASS; logit-majority FAIL. Each requires both lower95% contribution shares>50%. Shares slightly above100% occur because the downstream factor offsets part of the normalization contrast; its intervals include zero. The log interaction is negative at the point estimate, but its intervals also include zero.

This weakens a token-specific probability/cap explanation of the own-versus-matched native pairing effect. It does not show that token probabilities are irrelevant to absolute sensitivity, nor explain why the native response aligns with a particular residual projector. It is distinct from H349’s probability-versus-uniform comparison and uses a different response/statistic. Learned source coordination and generic residual geometry remain possible explanations of the upstream alignment.

Audit: cached OO/DD endpoints reproduce H356 geometric pairing within absolute error 2.22e-16; symmetric contributions sum exactly to total log contrast in estimates and all bootstrap draws; all Rayleigh energies positive; source hashes verified. Hybrid per-site values and bootstrap arrays saved.

Uncertainty:4000 paired document/control-realization/neighbor-rank resamples; pointwise conditional intervals, fixed confidence neighbors and donor dependence not rebuilt. Reused panel and head6 selected from earlier work.

Next: test whether a compact description of native response alignment to the residual projector predicts the head-specific pairing pattern, with an independent panel and explicit native response controls. This can turn the geometric finding into a falsifiable computational account without assigning semantic labels prematurely. Broad research goal remains active.
