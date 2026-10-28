## H496 — How broadly does the compact structured response transfer across heads?

Apply the same construction separately to all nine attention-head sources entering the layer-17 MLP: four source coordinates, a 64-dimensional output space for the bilinear cross term, and exact quadratic and normalization corrections. Each head has its own calibrated source basis and fitted cross-output basis. This tests the method and rank budget, not transfer of one frozen head-6 model. Calibration IDs, 32 training donors, 1,024 training receivers, 64 evaluation donors, and 512 evaluation receivers are matched across heads. Evaluation receivers were reused from H491; this is exploratory, with no fresh confirmation or confidence claim. The native intact baseline remains a dependency.

Recovery is one minus pooled prediction KL divided by pooled native-intervention KL; it is not token accuracy. Exact-projection recovery substitutes the four-coordinate source but leaves the MLP response computation native. Comparing that result with the compressed response helps locate errors, although KL errors are not an additive decomposition.

| Head | Calibration top-1 energy | Donor top-4 energy | Exact source-projection KL recovery −/+ | Compressed full-source KL recovery −/+ | Mean native effect KL −/+ |
|---|---:|---:|---|---|---|
|0|8.15%|22.59%|31.16% / 31.00%|9.21% / 8.52%|0.000186 / 0.000183|
|1|80.07%|82.23%|84.68% / 83.94%|75.44% / 75.18%|0.000516 / 0.000494|
|2|15.16%|36.07%|33.46% / 32.03%|16.66% / 16.38%|0.000614 / 0.000677|
|3|11.26%|19.28%|32.52% / 30.36%|26.00% / 23.70%|7.89e-05 / 7.63e-05|
|4|54.23%|51.70%|83.99% / 82.93%|73.25% / 71.99%|0.000206 / 0.000203|
|5|14.17%|10.15%|9.59% / 8.55%|4.56% / 3.42%|2.65e-05 / 2.71e-05|
|6|97.63%|97.94%|98.90% / 98.73%|96.04% / 96.12%|0.0029 / 0.0028|
|7|15.60%|23.54%|43.48% / 41.55%|41.75% / 40.67%|8.16e-05 / 8.22e-05|
|8|11.20%|30.37%|76.32% / 78.62%|72.65% / 74.94%|0.000236 / 0.000247|

Head 6 is exceptional at this budget: its leading source axis contains 97.63% of calibration energy, and the compressed program recovers about 96% of native effect KL. Other heads range from about 3% to 75% recovery. Their four-coordinate exact source projections already range from about 9% to 85%, so increasing only the cross-output rank cannot generally recover discarded source information. In heads 0 and 2, compression adds a substantial additional shortfall: exact projection recovers about 31–33%, whereas the compressed version recovers about 9–17%. These comparisons do not establish independent or additive causal error fractions.

Raw source energy alone is also insufficient to predict behavioral fidelity. Head 8 retains only 30.37% of donor source energy in four coordinates but recovers about 76–79% of effect KL with exact source projection. Head 4 retains 51.70% of source energy but recovers about 83–84% of effect KL. Head 5 retains 10.15% of source energy and recovers about 9% of effect KL. The important source directions depend on the downstream computation and chosen metric.

Native effects vary substantially in magnitude: head 6 has mean effect KL about 0.0028–0.0029, while head 5 is about 0.000027. Recovery ratios must be read alongside these denominators. This diagnostic uses centered source interventions of fixed signed unit scale; it does not rank the heads' overall importance on ordinary text.

The evidence favors a distribution- and objective-dependent description with heterogeneous rank requirements, rather than a universal four-coordinate source representation. It does not yet identify fundamental semantic units or show that the remaining heads are intrinsically incompressible. Source calibration coverage, source rank, and cross-output rank remain distinguishable explanations.

Audits: archived per-case KL and raw errors replay all reported compressed recoveries; head-6 source replay and aggregate H491 recovery checks pass; all arrays are finite. No new documents were consumed; tracked total remains 18,880.

Next H497: vary source rank through 4, 16, 64, and 128 for every head, retaining native downstream computation. Freeze this grid before outcomes, use the same calibration/donors/receivers, and include the full 128-dimensional head span as an endpoint audit. This isolates source-projection adequacy before paying for larger cross-map models. The experiment is exploratory and will report all heads and ranks. It should distinguish insufficient source rank from poor cross-term compression, and compare raw-energy retention with behavioral recovery. Matched-complexity comparisons of write/intervention/KL/observed-token-CE objectives remain a separate open experiment.
