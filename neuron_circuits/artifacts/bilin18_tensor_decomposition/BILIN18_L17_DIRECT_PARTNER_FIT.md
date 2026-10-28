

## H272 — Direct partner optimization improves response capture but leaves large residual error

H271 found specific but incomplete partner structure using spectral surrogates. H272 directly minimizes the exact cross-response reconstruction loss, keeping natural core32, partner rank32, native/compiled tensor coefficients, source family and population fixed. Only the partner projection is trained. Core-complement projection and differentiable QR enforce the constraint; Adam steps are followed by QR retraction. This is nonconvex optimization without a global-optimality or convergence certificate.

First768 discovery documents train; next256 select checkpoints. Four fixed runs: weighted or random0 initialization, learning rates.003/.01,400 steps, minibatches256, checkpoints0/100/200/300/400. All runs share the seeded minibatch sequence; step0 is eligible. Validation selects weighted/.003 at step300. Float32 optimization; final test evaluation uses float64 polynomial arithmetic and reorthogonalized projections. Test128 H270 documents are reused; no fresh replication. Training uses all nine full source removals; test includes scales0.1/1. No test selection of rank, learning rate or checkpoint.

| Partner32 model | Training cross fidelity | Validation cross fidelity | Full test cross fidelity | Full test total fidelity |
|---|---:|---:|---:|---:|
|tensor|43.67%|45.29%|42.83%|35.07%|
|weighted|44.98%|46.88%|44.33%|35.59%|
|random0|3.46%|3.46%|3.58%|18.17%|
|optimized|54.43%|54.45%|51.69%|39.11%|

Optimized-minus-weighted paired document-bootstrap95% contrasts:

- s0.1, cross: 7.31 points [5.59, 8.58].
- s0.1, total: 3.58 points [3.24, 3.92].
- s1.0, cross: 7.36 points [5.77, 8.54].
- s1.0, total: 3.52 points [3.18, 3.84].

**All three locked gates fail.** Cross improvement7.3–7.4 points is below10; total improvement3.5–3.6 below5; cross fidelity52% is below80%. Positive gains remain evidence of selection-method limitations; thresholds are not relaxed after outcomes. Random initialization improves from3.5% to53.4% validation, near the selected weighted start54.5%. This reduces dependence on the spectral initializer but does not exclude better local minima or an insufficient optimization budget. The higher learning rate deteriorates late on validation; without checkpoint-specific full training diagnostics, this is not identified as overfitting rather than optimization instability.

**What changed.** Optimizing actual responses helps substantially more than the earlier data-weighting surrogate (about7.4 versus1.5 points). At the selected checkpoint, training and validation are both54.4%, and test is51.7%; a large selected train–validation gap does not explain the remaining error. Optimized32 approximately matches H271's weighted64 cross fidelity52.1% using half as many partner directions and core-partner products. These are exact-coefficient diagnostics, not a fully compiled efficient model: full mean-linear response, within-core coefficients, partner reads and output coefficients still have costs.

**Interpretation.** Compactness estimates depend on the selection objective as well as the interaction family. There is useful shared cross-core structure, but this run does not show that32 partners suffice for most cross computation. Neither failed gates nor small train–validation gaps prove intrinsic dimension or irreducibility. The cross-term metric and full residual metric remain distinct, and no semantic or downstream-behavior claim follows.

**Next discriminator.** Freeze this optimization protocol and compare ranks16/32/64 with the same discovery split and compute budget, reporting learning curves and all checkpoint choices. Evaluate fixed selected models on a new document panel after selection. If optimized64 captures substantially more than optimized32, the tested shared representation is capacity-limited under this procedure; if rank helps little while training remains poor, investigate optimizer/objective limitations before invoking context-dependent partner spans. A conditional alternative should be compared at explicit total storage/read/routing cost. Once a useful representation is stable, compile its forward cross-term computation and test downstream effects rather than adding semantic labels from response energy alone.

**Audits.** Three relevant tests pass, including finite-difference checking of the gradient through QR and the exact cross-response expression, complement/orthogonality constraints, full-complement response identity and basis-rotation identities. Training projection overlap<=2.7e-8 and orthogonality error<=4.8e-7; evaluation reorthogonalizes in float64. Full project suite not rerun. The first launch failed before training because FullModel is a wrapper without parameters(); the unnecessary call was removed, and its startup-error log is preserved. Experiment settings were unchanged. Native weights and original compiler unchanged. All four runs and test evaluation complete. Figure inspected. Broad goal remains active.
