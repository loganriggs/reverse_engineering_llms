# Competing explanations after the finite-label scaling panels

Updated2026-10-29. These hypotheses are not mutually exclusive. The strongest current evidence distinguishes identification, local stability, optimization and transfer; it does not establish a single universal explanation.

| # | Hypothesis | Current evidence / discriminating test |
|---|---|---|
| 1 | Sharing forces tasks to agree on a reusable dictionary. | Supported by the fixed ten-output transfer control: shared beats both independent models5/5teacher blocks; optimization/parameter confounds remain. |
| 2 | More heads merely add readout parameters. | Hidden width is fixed but parameter count grows; independent equal-width and near-budget controls help expose this confound. |
| 3 | More heads identify previously unconstrained local directions. | Generic local rank is already full modulo gauges for one d6/h3head; cannot be the universal explanation. |
| 4 | More heads strengthen weak local directions. | Supported by profiled-information and local-noise tests; measure the same effect in finite-data regimes. |
| 5 | More heads remove distant incorrect decompositions. | Local rank does not settle global alternatives; exact alternative decompositions or global identification arguments would discriminate. |
| 6 | Diverse task losses make bad optimization basins harder to enter. | Supported more directly:13/15independent equal-width fits admit a copied shared solution with lower empirical loss, certifying suboptimal optimization outcomes. |
| 7 | Benefit is ordinary overfitting reduction through tying. | Supported by release control: removing sharing lowers empirical loss but worsens output-span transfer5/5teachers, even with teacher-initialized independent fits. |
| 8 | The benefit is mostly additional input coverage. | More heads actually reduce distinct inputs at fixed labels; coverage can explain the low-budget reversal, not every high-budget gain. |
| 9 | Sharing acts as an implicit low-rank constraint on task functions. | Algebraically true: outputs lie in hidden-product span; compare unconstrained polynomial and rank-constrained function-span baselines. |
| 10 | Sufficient source-task span, rather than hidden recovery, explains transfer. | Test equal-dimensional predicted-output features and independently measured component recovery. |
| 11 | Near-exact source fitting is enough to recover components generically. | Proven only under stated full-span assumptions; scalar counterexamples and approximate failures forbid a blanket claim. |
| 12 | Approximate identification is unstable near degenerate teachers. | Plausible; relate teacher-factor/readout conditioning to perturbation sensitivity without selecting teachers. |
| 13 | Reciprocal factor imbalance causes most failures. | Initial balancing helps some tied fits but not universally; already bounded, do not rescue-tune. |
| 14 | Occurrence multiplicity creates an effective layer learning-rate advantage. | Exact metric result and earlier SGD controls support the confound; not sufficient to explain current TRF results. |
| 15 | Curvature correction is the main missing ingredient. | GN dramatically improved earlier toys, yet currentTRFstill has failures; curvature alone is insufficient. |
| 16 | Alternating updates unlock shared structure. | Naive alternation and blockGNshow no consistent superiority; no reason for another broad order sweep. |
| 17 | Readout conditioning determines adaptation error more than hidden geometry. | Teacher output condition numbers60.8–746.8 and independent-fit span rotations support this mechanism; coherent shared dictionary errors have a different first-order structure. |
| 18 | Extra labels mainly average away observation noise. | Noiseless controls also exhibit failures; task-dependent information and optimization effects remain. |
| 19 | Apparent advantages are a few favorable teacher/initialization draws. | Five fresh teacher confirmations with all initialization means reduce this concern; larger-network universality remains untested. |
| 20 | Recovery relies on special teacher architecture or privileged normalization/interventions. | True scope limitation: Gaussian bilinear teachers, knownRMS and labeled ablations are controlled assumptions; transfer to mismatched architectures remains separate. |

Most actionable now:1,4,6,8,9,10. Local stability already supports4. Fixed-budget reversals support investigating8. The next control tests1/6/9/10with shared versus independently trained heads, retaining2as a parameter confound. Use equal-dimensional predicted-source-output transfer to avoid giving independent heads a larger union dictionary at adaptation.

Do not treat a small source residual as circuit recovery, a local rank certificate as global uniqueness, or a teacher-initialized fit as practical learnability. Stop each prespecified panel; preserve negative/mixed outcomes.
