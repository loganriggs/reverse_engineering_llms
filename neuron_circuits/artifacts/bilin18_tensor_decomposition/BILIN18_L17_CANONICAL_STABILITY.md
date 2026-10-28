## H492 — Stable behavior versus unstable canonical axes

Freeze source4 calibration, exact cross operators, quadratic core, normalization and native baseline. Generate four reproducible complementary splits of the1024 training receivers, with16 of each32 donor's receivers in each512-receiver half. The halves within a split are disjoint and cover the original set; repetitions overlap and are not independent confirmations. Fit a cross-output64 SVD to each half's signed cross contributions. Do not change rank or select a favorable split after outcomes.

Compare subspaces by singular values of V_left V_right^T: principal angles are arccos of those values, and normalized projector overlap is their mean square. Compare individual axes by a maximum-total-absolute-cosine one-to-one assignment, then sign alignment. This corrects sign/permutation ambiguity but intentionally does not rotate axes arbitrarily: arbitrary rotations belong to the subspace comparison. An in-script Hungarian assignment solver is independently checked against all5! permutations of a small random test matrix.

| Split | Normalized projector overlap | Median principal angle | Median matched axis cosine | First three matched cosines |
|---|---:|---:|---:|---|
|0|0.483|47.66°|0.215|0.9990, 0.9981, 0.9957|
|1|0.476|47.55°|0.204|0.9999, 0.9991, 0.9954|
|2|0.476|47.89°|0.199|0.9995, 0.9967, 0.9877|
|3|0.481|47.16°|0.212|0.9998, 0.9992, 0.9852|

Geometric instability is substantial across much of the64-dimensional space, not merely sign flips or permutations. The leading few axes remain much more reproducible than later axes. Large leading spectral gaps and small rank64 boundary gaps are consistent with this pattern, but no causal stability theorem is claimed from these observations. Small spectral gaps permit rotations/exchanges in low-energy directions; finite-sample differences can also change the subspace itself.

Evaluate all eight half-data maps and the frozen full-data map on the same512 H491 receiving contexts, already used in earlier confirmation. All exact structural terms and targets are shared, so differences come from the fitted cross projector. This is a sampling-sensitivity diagnostic with reused receivers, not fresh confirmation.

| Fitted basis | Negative KL recovery | Positive KL recovery |
|---|---:|---:|
|full|96.037%|96.121%|
|0_left|96.036%|95.872%|
|0_right|95.722%|95.891%|
|1_left|95.710%|95.769%|
|1_right|95.944%|95.943%|
|2_left|95.791%|95.910%|
|2_right|95.940%|95.878%|
|3_left|95.992%|95.896%|
|3_right|95.744%|95.937%|

Half-data native fidelity remains close across splits:95.71–96.04% negative and95.77–95.94% positive, versus96.04/96.12% full-data. Pairwise prediction disagreement is not zero. Symmetric KL between the two half-model distributions is1.09–1.20% of the native effect for negative signs and1.22–1.28% for positive signs. Raw squared prediction disagreement is~.18–.21% of native response energy, or~.092–.110% of cross-contribution energy. Similar pooled accuracy therefore does not imply identical per-case functions or strict functional equivalence.

Each half's64 axes capture over99.9% of its training cross-response energy, yet normalized projector overlap is only about48%. Unweighted subspace geometry treats all retained directions equally; the functional metric weights how frequently and strongly the directions are used and how the readout responds. This mismatch explains why geometric and behavioral stability need not track one another. The geometry does not prove that the low-energy unstable directions are behaviorally irrelevant; a direct component test is still needed.

Interpretation for the canonical-form proposal: the data-defined factorization has a stable leading core and a much less reproducible tail of axes at this sample size. The same architecture/source conditioning admits multiple empirically fitted coordinate descriptions with similar, not identical, behavior. Thus naming all64 SVD axes as fundamental features is not justified. Conversely, these results do not invalidate a useful canonical equivalence class, prove exact parameter non-identifiability, or establish invariance to different distributions or source bases. Source calibration was held fixed and the four splits share the same training population.

Audits: balanced disjoint halves verified; source operators/normalization fixed; assignment independently brute-force checked; full-data prediction replay matches H491; all arrays finite. Split membership, bases, singular spectra, alignment permutations/signs, per-case errors and pairwise disagreement saved. No new documents, calibration or model training beyond these basis fits; total18880 tracked documents unchanged. No success gates or new confidence intervals are claimed. No process remains active.

Next isolate the stable shared core from the weakly shared remainder. Within each split, use principal-vector pairs from the training-basis comparison and a frozen cosine threshold (e.g..95) to define shared directions; evaluate each side's core-only and full64 cross projections with all exact structural terms retained. Also quantify the omitted cross component and its readout effect. The threshold must be fixed before evaluating these component outcomes, and all four splits retained. This can determine whether a reproducible small core suffices behaviorally or whether individually unstable directions collectively supply needed computation. Do not interpret positive component energy as additive KL contribution. Broader CE objectives, semantic interpretation, native-baseline removal and cross-head/layer transfer remain open.
