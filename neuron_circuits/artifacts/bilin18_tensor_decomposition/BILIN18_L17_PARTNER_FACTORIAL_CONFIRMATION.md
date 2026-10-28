

## H301 — Fresh rank gains are broad; refinement hides head-level regressions

Four candidates frozen before outcomes: proxy/refined partners at ranks32/64 from H299–300. New128 documents exclude3520 complete-document hashes across21 previous panels. Independent audit verifies source-document hashes, window slices and hashes, output hash,128 unique new documents and zero overlap. Cached corpus is not a model-training holdout or a new distribution. Four positions/document, nine heads, actual and angle-matched full removals. Float32 native trajectories, float64 cross-response evaluation with reorthogonalized frozen bases; maximum basis constraint error<1e-14. Source artifact hashes checked before evaluation. No checkpoint is selected on this panel.

| Candidate | Actual fidelity | Angle fidelity |
|---|---:|---:|
|proxy32|16.77%|14.68%|
|refined32|17.59%|18.25%|
|proxy64|24.92%|22.01%|
|refined64|26.62%|26.87%|

| Contrast | Family | Energy-weighted gain [paired95%] | Equal-document gain | Documents improved |
|---|---|---:|---:|---:|
|rank_gain|actual|9.02pp [8.77,9.29]|9.04pp|100.0%|
|rank_gain|angle|8.62pp [8.36,8.87]|8.70pp|100.0%|
|refinement32|actual|0.83pp [0.63,1.03]|0.84pp|82.0%|
|refinement32|angle|3.57pp [3.21,3.94]|3.53pp|100.0%|
|refinement64|actual|1.70pp [1.50,1.90]|1.74pp|95.3%|
|refinement64|angle|4.86pp [4.58,5.12]|4.85pp|100.0%|

Paired bootstrap uses2000 document resamples, preserving positions, heads and candidate pairing. Intervals condition on the frozen fitted candidates and sampled angle payloads; they do not capture training or new-distribution uncertainty. Rank gain is refined64−refined32; refinement compares each candidate with its own proxy.

Locked gates: rank>=5pp in both families with positive lower bounds PASSES. Refinement>=2pp in both families FAILS at each rank because actual gains are smaller. Compact80 FAILS. Positive refinement intervals establish small gains, not passage of the magnitude thresholds.

**Breadth versus pooling.** Rank gains are positive on all128 documents and all9 heads in both families. Per-head rank gains range6.80–9.51pp actual and5.89–10.71pp angle. Equal-document and energy-weighted effects closely agree. Maximum document energy shares are2.40% actual/3.47% angle; top5 shares8.21/10.83%. Leave-one-document-out rank gains remain8.97–9.05pp actual and8.58–8.70pp angle. This weakens a few-dominant-documents explanation for rank expansion.

Refinement is different: actual aggregate gains are positive for each head, but angle gains are negative for heads3,5,7,8 at both ranks. Rank64 angle refinement ranges−.70pp to+7.02pp across heads; rank32−.56pp to+6.25pp. Thus all documents can improve after summing heads while four head categories regress. The two-family common-descent algorithm guarantees neither per-head monotonicity nor per-head generalization. These subgroup results are descriptive, not multiplicity-adjusted confirmations.

**Inference.** The rank limitation and modest optimization gains transfer to new documents from this corpus. Validation overuse alone cannot explain these contrasts. A high-rank interaction family remains plausible, but no global optimum or irreducible rank has been established. Dense rank64 output coefficients cost7,077,888 before other program terms;26–27% cross fidelity is not an economical complete computational account. No semantic units, intact substitution, or downstream behavioral gains are established here.

**Next discriminator.** Audit training gradients separately for each head and payload family at the frozen rank64 proxy/refined bases, before another optimizer sweep. Measure head energy weights and pairwise gradient cosines; compare pooled-family, equal-head-normalized and worst-head directions on the existing training/validation split. This separates weighting from incompatible local demands. The fresh panel is now consumed and must not become a checkpoint selector. A richer representation comparison remains necessary if per-head balancing cannot improve broad coverage.

All jobs terminal; original model/compiler/base unchanged. Broad research goal remains active.
