## H470 — Canonical forms fitted to changes rather than states

Hold H469 augmented64 empirical bilinear input coordinates, support768, source assignments and data splits fixed. Compare saved augmented-state rank16, response-only rank16, equal normalized state/response joint rank16, and response-only rank64. Contrasts concatenate both signed training changes relative to their intact states, without a response intercept. All fits use exact reduced-rank regression in the selected feature metric. The unrestricted rank64 response normal equations close to3.27e−14 relative residual; all fit Gram matrices retain rank64. Natural-state intercepts for new maps are anchored to their training means, and response-only training does not otherwise constrain intact writes. No validation selection or new corpus data.

State and response losses are divided by their separate training target energies. In the shared normalized units, state energy is1 and response energy0.0149975. This quantifies why whole-state fitting can put little emphasis on responses; it does not alone prove that weighting is the sole limitation.

| Map | Train raw-response recovery | Test negative KL / raw-response recovery | Test positive KL / raw-response recovery | Intact KL |
|---|---:|---|---|---:|
|state16|67.488%|-20.75% / 69.90%|-16.84% / 67.56%|0.125443|
|response16|96.585%|57.44% / 94.14%|61.59% / 93.33%|0.286429|
|joint16|92.330%|54.26% / 93.23%|55.46% / 93.03%|0.142229|
|response64|97.047%|57.90% / 93.95%|60.43% / 93.09%|0.304654|

| Frozen primary statistic | Estimate | Joint95% interval | Pass |
|---|---:|---|---|
|joint16_deltaKL_negative|+0.542594|[+0.278859, +0.806329]|False|
|joint16_deltaKL_positive|+0.554629|[+0.358175, +0.751082]|False|
|joint16_minus_state16_intactKL|+0.016785|[-0.000372, +0.033942]|False|
|response16_minus_state16_deltaKL_negative|+0.781962|[+0.015082, +1.548842]|False|
|response16_minus_state16_deltaKL_positive|+0.784325|[+0.226375, +1.342276]|True|

Prespecified gates require joint16 absolute delta-KL recovery>.8 at each sign, joint16 intact-KL increase<.01, and response16-minus-state16 delta-KL gains>.2 at each sign. Only the positive-sign gain clears its gate. Both gain intervals exclude zero, supporting improvement, but the negative-sign lower bound1.51pp does not exceed20pp. Joint16 absolute recovery54.26/55.46% falls well short of high fidelity on point estimates and fails both80% gates. Intact KL increases.016785 with interval[−.000372,+.033942], failing the.01 practical-loss gate without establishing a positive difference at this joint confidence level.4000 paired test-document bootstrap replicates, joint5 standardized intervals; calibration and source assignment fixed. Symmetric recovery intervals extending above100% are interval artifacts, not achievable recovery above100%.

The meaningful constructive result is the fixed-input-family response improvement. Response16 reaches57.44/61.59% KL recovery and94.14/93.33% raw recovery, while the prior state fit had negative KL point recovery. These are full learned response differences around a native baseline, without an exact retained-factor correction. The full forward state predictor remains worse: response16 intact KL.286429 versus state16.125443; joint16 gives.142229. The exact baseline in the delta score is an evaluation device and does not make the response-only map a faithful whole-state replacement.

The rank64 capacity witness raises training raw-response recovery from96.585% to97.047% but does not materially improve test point scores:57.90/60.43% KL and93.95/93.09% raw. No formal equivalence test between ranks was specified. It shows that increasing output rank alone is not a demonstrated remedy, and that the remaining2.953% training write-response error cannot be removed by any linear output map in these64 fixed input features under this least-squares objective. It is not a lower bound on KL error, a nonlinear decoder, or a richer input feature family.

Descriptive untrained-source-index response16 KL recoveries are66.00/65.26%, versus52.03/59.39% for trained indices. This gives no obvious point-estimate collapse on unseen donor indices, but fixed small strata and reused source pool do not establish general source transfer. No semantic axes are identified.

### Twenty explanations to discriminate in the canonical-form phase

These are competing or compatible hypotheses, not twenty findings. Priorities are qualitative, and selected outcomes on this reused panel are not independent confirmation.

| Explanation | Current assessment | Discriminator |
|---|---|---|
|1. State magnitude masks response differences in the loss.|Supported contributor by fixed-feature contrast fitting.|Sweep fixed normalized state/contrast weights and measure both behaviors.|
|2. Raw-write error weights the wrong output directions for probabilities.|Leading:93–94% raw versus57–62% KL recovery.|Fit finite response KL with input family held fixed.|
|3. The64 input coordinates omit important interaction directions.|Leading capacity limitation: rank64 retains training response error.|Expand input rank while fixing response objective and output budget.|
|4. Rank16 output bottleneck is the main remaining limitation.|Weakened by unrestricted rank64 witness.|Independent rank sweep and probability-weighted output selection.|
|5. Finite donor sampling inflates apparent structure.|Open; source strata are small.|Freeze maps and use new donors/receivers.|
|6. More training contexts are needed for the empirical quotient.|Open.|Independent training-size curves with fixed ranks.|
|7. Empirical kernel coordinates are less effective than native learned factors.|Open; current coordinates use a degree-two polynomial kernel.|Compare learned native multiplicative coordinates at matched budget.|
|8. State-dependent downstream sensitivity prevents one global output metric.|Plausible, supported by earlier research.|Compare global and state-conditioned response weighting.|
|9. Normalization curvature drives remaining errors.|Possible but included in exact targets.|Amplitude sweep and analytic normalization-aware contrast features.|
|10. Local derivatives are enough; finite effects create avoidable complexity.|Unresolved in this canonical family.|Train derivatives and test small/large signed amplitudes.|
|11. The useful source variable is scalar gain along a common write.|Strong source-energy evidence, behavior untested.|Rank1 source projection plus orthogonal-tail interventions.|
|12. Low-energy source directions dominate some probability effects.|Plausible.|Direct tail effects and output-sensitivity-weighted source coordinates.|
|13. Training source/context pairings limit transferable response fits.|Open.|Crossed versus paired source/context training at fixed examples.|
|14. Empirical canonical axes are unstable while their spans are stable.|Open.|Resample calibration; compare spectra, subspaces and behavior separately.|
|15. Multiple equivalent low-rank parameterizations obscure a stable function.|Algebraically compatible.|Audit functional equivalence after metric whitening and latent gauge changes.|
|16. CE needs far more independent labeled examples than distillation.|Plausible after CE validation/test mismatch.|Matched data-scaling of CE and teacher KL objectives.|
|17. Optimization failures masquerade as objective equivalence.|Supported by H467 small-step improvement.|Full-gradient finite-response loss with fixed trust steps.|
|18. An unconstrained offset separates state and contrast objectives.|Relevant limitation; contrast loss cancels constants.|Optimize/calibrate intercept separately and test state behavior.|
|19. Sparse exceptional contexts dominate pooled KL uncertainty.|Plausible given wide bands.|Report effect/error concentration, robust strata and independent replication.|
|20. Head6 compactness does not generalize to other computations.|Established scope concern.|Repeat frozen methodology on broader heads and layers.|

Next prioritize hypotheses2 and3, with17 as an implementation check: optimize finite response KL at fixed64 input/rank16 output starting from the joint map, with small full-training steps selected only on validation and with ordinary-state diagnostics. Preserve the exact least-squares capacity witness. If response KL improves materially, the output metric was a bottleneck; if it cannot improve training, audit gradient/step behavior before attributing failure to the input family. A subsequent input-rank expansion can distinguish limited representability from generalization. Do not abandon the user's CE objective comparison; return to it after achieving a reliable KL comparator. Native-source rank1/tail tests remain a separate pending explanation, not silently declared resolved.

Numerical normal-equation and baseline-replay gates pass. Supports, maps, intercepts, per-case errors and bootstrap samples saved. Total12992 corpus documents; broad goal remains active.
