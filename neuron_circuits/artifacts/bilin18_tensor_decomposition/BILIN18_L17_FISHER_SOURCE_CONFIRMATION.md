## H499 — Fresh receiver confirmation of probability-aware source coordinates

Freeze all nine rank-4 Fisher source maps from H498 and compare them with rank-4 PCA. Each map was derived from the old source calibration covariance and the local output Fisher metric on 16 training receivers. Nothing is refitted for confirmation. Use 512 fresh receiving contexts, each from a distinct complete document, disjoint from all 18,880 previously tracked documents. The 64 donor sources remain unchanged and are assigned cyclically to eight receivers each. Evaluate both signed unit interventions with native downstream MLP computation.

The maps preserve four source coordinates; Fisher uses an oblique read/write projection whereas PCA uses an orthogonal projection. This is a matched bottleneck-rank comparison, not a strict parameter-count match. The experiment measures native finite intervention response fidelity and retains the full downstream MLP. It is not a stand-alone model compression or observed-token CE comparison.

KL recovery = 1 − pooled KL(full native response || reconstructed-source native response) / pooled KL(full native response || intact baseline). Gains below are Fisher minus PCA recovery, in percentage points. The 95% intervals are simultaneous across all 18 head/sign comparisons, using 4,000 paired bootstrap resamples of receivers within donor assignments. They condition on the fixed calibration, metric fit, and donor pool. They do not capture uncertainty from retraining or drawing new sources.

| Head | Sign | PCA recovery | Fisher recovery | Gain (pp) | Simultaneous 95% gain interval (pp) |
|---|---|---:|---:|---:|---|
|0|negative|25.42%|27.11%|+1.69|[-12.01, +15.38]|
|0|positive|22.55%|20.26%|-2.29|[-15.61, +11.03]|
|1|negative|76.63%|84.26%|+7.62|[+4.07, +11.18]|
|1|positive|75.52%|82.83%|+7.31|[+3.80, +10.82]|
|2|negative|40.71%|52.02%|+11.32|[+1.59, +21.04]|
|2|positive|44.30%|50.62%|+6.32|[-1.97, +14.61]|
|3|negative|40.38%|50.40%|+10.02|[-7.07, +27.11]|
|3|positive|41.40%|49.69%|+8.29|[-11.08, +27.65]|
|4|negative|76.66%|78.59%|+1.94|[-3.06, +6.94]|
|4|positive|83.41%|82.58%|-0.83|[-3.08, +1.41]|
|5|negative|6.24%|56.63%|+50.39|[+32.78, +68.01]|
|5|positive|3.36%|56.64%|+53.28|[+29.60, +76.96]|
|6|negative|98.13%|98.50%|+0.37|[+0.07, +0.67]|
|6|positive|98.11%|98.55%|+0.44|[+0.23, +0.66]|
|7|negative|43.84%|55.40%|+11.56|[+3.23, +19.90]|
|7|positive|41.61%|54.47%|+12.85|[+3.85, +21.86]|
|8|negative|77.82%|83.66%|+5.84|[+2.59, +9.10]|
|8|positive|77.48%|83.54%|+6.06|[+2.96, +9.16]|

Eleven of the 18 simultaneous lower bounds exceed zero. Both signs pass for heads 1, 5, 6, 7, and 8; head 2 passes only for the negative sign. Heads 0, 3, and 4 have no passing sign. Failure to establish a gain is not evidence of equivalence or absence of a gain. Two point estimates regress (head 0 positive and head 4 positive); neither establishes a loss under the simultaneous intervals.

Head 5 gives the clearest illustration of the difference between raw-data and behavioral coordinates: PCA preserves only 3.36–6.24% of its native effect KL, while the frozen Fisher map preserves 56.63–56.64%. Its gains are 50.40 and 53.28 percentage points, with simultaneous lower bounds 32.78 and 29.60 points. Head 6's gains are much smaller (0.37–0.44 points), because its PCA projection already preserves about 98% on this receiver panel.

Absolute effect sizes matter. This is a ratio relative to each head's own intervention effect, not a ranking of total head importance. The JSON preserves the mean native effect KL and raw response recovery for every head/sign. Different heads and receivers can have substantially different effect magnitudes, and a few large effects can drive a pooled ratio. The intervals retain the paired design but do not make these ratios per-case guarantees.

The conclusion supported here is that a low-rank source representation chosen using a downstream probability metric can preserve substantially more finite behavior than variance-based PCA, and this advantage can transfer to new receiver documents. The evidence does not support universal improvement for every head, a unique semantic feature ontology, arbitrary intervention transfer, or fresh-source generalization. It also does not isolate objective weighting from the additional freedom of an oblique map.

Audits: the independent document audit rehashed complete documents and checked token windows against each panel's own source cache; 512 new documents have zero overlap with 18,880 prior documents. Total tracked allocation is 19,392. The cache is not asserted to be held out from model pretraining. Frozen model and source artifacts were hash-checked before execution. All per-case outputs are finite; aggregated gains replay. The bootstrap used one shared resampling index array across heads/signs/methods and a maximum-standardized-error critical value of about 2.967. Runner, construction, document builder, and audits are retained.

Next test fresh donor transfer with the maps and calibration mean still frozen. Use a new complete-document-disjoint donor pool, retain the H499 receiver panel and the prescribed rank 4 for every head, and report the same all-head signed comparison without selecting favorable heads. Reusing receivers isolates a different axis of generalization; no new fitting or tuning should use their outcomes. Measure source covariance/energy shift alongside finite KL so failures can distinguish receiver dependence from source-distribution dependence. Separately, an orthogonal-map control is needed to distinguish metric choice from oblique projection freedom. The broader questions of semantic units, stable canonical axes, and CE-oriented equivalence remain open.
