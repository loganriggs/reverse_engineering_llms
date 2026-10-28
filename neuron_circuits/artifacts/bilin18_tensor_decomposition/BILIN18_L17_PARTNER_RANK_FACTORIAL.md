

## H300 — Rank expansion contributes more than short optimization, but coverage remains low

Complete the locked rank32/rank64 × proxy/refined comparison. Rank32 is reused from H299; rank64 uses the same executable, frozen core96, first768 discovery training documents, next256 validation documents, all9 heads, actual/angle payload seed, eight full-gradient iterations, and training-only Armijo grid. Equal numbers of full-gradient evaluations are not equal FLOPs or equal representation costs. Fixed step Frobenius norms also differ in per-direction scale across ranks. Proxy bases are nested; independently refined bases need not remain nested.

| Partner rank | State | Actual train | Angle train | Actual validation | Angle validation |
|---|---|---:|---:|---:|---:|
|32|proxy|15.377%|14.172%|16.595%|15.315%|
|32|refined|18.137%|19.224%|17.394%|18.763%|
|64|proxy|23.353%|21.244%|24.773%|22.537%|
|64|refined|27.974%|28.694%|26.491%|27.332%|

Rank64 improves selected validation fidelity over rank32 by9.10pp actual and8.57pp angle. Within rank64, refinement adds1.72pp actual and4.80pp angle; within rank32 it adds.80pp and3.45pp. The descriptive rank×refinement interaction is+.92pp actual/+1.35pp angle. This single trajectory per rank does not establish statistical significance of that interaction.

Locked comparison gates: rank gain>=5pp in BOTH families PASSES; rank64 optimization gain>=2pp in BOTH families FAILS; both rank64 validation fidelities>=80% FAILS. All accepted training steps satisfy the recorded Armijo conditions. Rank64 selects step8 and stops at budget, not convergence. Saved basis constraints pass at maximum absolute error<1e-5; configuration equality audited for all operative shared sampling and optimization settings.

**Hypothesis update.** Limited partner rank in this projection family is better supported than an explanation based solely on the failed Adam optimizer: even unrefined rank64 outperforms refined rank32 on both families. Optimization matters too, with measurable gains at both ranks. Overfitting alone does not explain this rank comparison because training and validation both improve with added directions. However, validation is repeatedly reused, initial spans were selected by a tensor/data proxy, and neither trajectory estimates a global optimum. Distributed cross interactions and a restrictive projection family remain live explanations; no conclusion about universal feature granularity follows.

**Cost.** Dense native cross output coefficients double from3,538,944 to7,077,888 (96×rank×1152), excluding reads and other program terms. Capturing about26–27% of this cross response at that cost is not an economical complete explanation. This remains native-coefficient oracle accounting, not a new compiled program with demonstrated intact behavioral fidelity.

**Next discriminator.** Freeze all four candidates now and test previously unused documents, with paired document bootstrap and both energy-weighted and equal-document summaries. Primary contrasts are refined64−refined32 and each refined−own-proxy. Retain >=5pp rank and >=2pp refinement thresholds for both families; require positive paired95% lower bounds for confirmation. This separates document-selection optimism from transferable rank/optimization gains. Report per-head effects and energy concentration to detect a small set of dominant cases. Do not choose a new checkpoint on that panel. If rank gains transfer, compare spectrum/cost or a different representation family before further optimizer sweeps.

No fresh-document or intact/downstream test in H300. Original model/compiler/base unchanged. All jobs terminal; broad research goal remains active.
