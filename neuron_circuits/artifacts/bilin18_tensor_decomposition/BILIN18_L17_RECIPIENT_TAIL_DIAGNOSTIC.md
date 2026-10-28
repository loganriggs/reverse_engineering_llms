## H430 — Recipient sensitivity, normalization gain and source concentration

H429 was progress: its fresh test exposed an unusually concentrated quartic-energy tail. This post-selection diagnostic re-evaluates baseline states for all512 H429 recipients and1024 H425 recipients, then inspects saved intervention coefficients. No new documents, exclusions, fresh statistical confirmation or changes to H429 failed gates. Total unique documents remains11456.

Almost all of document199's quartic energy comes from token position63 (flattened site797):18.868% of natural-source and24.544% of Gaussian-source total quartic energy. The immediately preceding text ends “June 2, 2008: Kerry Kill”. This is context identification, not a semantic feature label or an explanation. Quartic energy is the coefficient of edit amplitude to the fourth power, not total behavioral effect.

| Baseline quantity at site797 | Value | Panel percentile |
|---|---:|---:|
| Input residual RMS |840.76|1.81%|
| Output residual RMS |865.01|0.93%|
| Actual head6 vector norm |3688.50|49.80%|
| Prediction entropy (nats) |3.080|49.85%|
| Largest token probability |0.405|58.45%|

Median input/output RMS values are1775.93/2563.14. Thus a large actual head vector or unusually concentrated baseline probabilities does not describe this case. Actual-head norm does not measure the norms of substituted donor/Gaussian signals, so those remain to be examined. Low residual scales are a plausible amplifier: input RMS differentiation contributes inverse input scale, the bilinear second-order response contains two such derivatives, and the Fisher output metric contributes squared inverse output scale.

Controlled algebraic diagnostic: retain each site's derivative radial projectors, response directions, baseline probabilities and all other operations, but replace input and/or output scalar inverse-RMS gains by their panel medians. Under these defined changes, quartic energy is multiplied by (input_scale/median_input)^4 and/or (output_scale/median_output)^2. Native float32 RMS epsilon is retained. This is an exact rescaling of the specified derivative computation, not a native finite model intervention or an independent causal mediation estimate.

| Scalar gains replaced | Site797 share, natural | Site797 share, Gaussian |
|---|---:|---:|
| None |18.87%|24.54%|
| Input only |7.05%|9.90%|
| Output only |8.31%|11.42%|
| Both |1.28%|1.87%|

The both-gain multiplier for this site is0.005721. Its share changes less than that multiplier because other sites are also rescaled. The previous panel's most Gaussian-sensitive position also has low input/output RMS (801.21/895.54); its Gaussian quartic share drops from6.87% to0.382% under the same defined operation. Across all positions, quartic energy's rank correlation with inverse input_scale^4*output_scale^2 is about0.52 for both source families on the current panel and0.51–0.56 on the previous panel. These are descriptive associations, not confidence-tested population estimates. Current source averages include reflection; previous ones do not, so panel differences are not controlled causal contrasts.

Source variation remains substantial. At site797, the largest draw contributes38.85% of natural-source and88.48% of Gaussian reflection-averaged quartic energy across12 draws. Reflection is not the whole origin of the tail: using only original sources, the site still accounts for14.29% natural/11.29% Gaussian quartic energy. A receiving context's gain and a source draw's direction or magnitude can therefore jointly matter; this analysis does not isolate those components.

Updated priorities: (1) low recipient normalization scales amplify sensitivity; (2) particular donor/Gaussian directions or magnitudes generate an additional tail; (3) output metric alignment can differ even with ordinary probability entropy; (4) finite-source sampling and recipient assignment can dominate uncertainty; (5) higher centered moments remain possible but are not established. No new evidence here establishes a semantic computational unit.

Next high-information experiment: freeze the selected source vectors and compare their responses across the tail recipient and ordinary recipients, alongside source-norm controls and the explicitly defined scalar-gain controls. Measure response magnitude before output scoring as well as Fisher energy, separating input amplification, bilinear direction alignment and output sensitivity. Then check whether the effect persists with exact input normalization and exact output KL at finite edit sizes. A polynomial/derivative tail could be attenuated by finite normalization; that would materially change its interpretation. Preserve the original full-panel statistics and do not delete high-sensitivity positions.
