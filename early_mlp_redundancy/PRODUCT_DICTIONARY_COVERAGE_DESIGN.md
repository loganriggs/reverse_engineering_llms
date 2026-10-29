# M98: does broader causal coverage improve transfer?

M97 showed that causal supervision improved the aggregate training objective but worsened some held-out source interventions. Those regressions were already predicted by local directional derivatives; late-path curvature was not needed to explain them. This motivates testing example coverage before making a capacity or representation claim.

## Discriminating hypotheses

1. **Coverage-limited learning:** gradients from a narrow set of contexts capture incidental regularities. Broadening contexts should improve held-out causal fidelity, particularly the source interventions that previously regressed.
2. **Objective tradeoffs:** the average objective rewards final-position improvements while tolerating source regressions. Broader coverage may improve the aggregate yet leave the same site-specific conflict.
3. **Representation limitation:** a fixed 128-dimensional output dictionary may not express the needed corrections, even though it retains all 4608 native bilinear products. Failure of this experiment would be consistent with this hypothesis, but would not distinguish it from optimization or inadequate coverage.

These are hypotheses about the learned replacement and its supervision. None alone establishes redundancy, cancellation, or semantic units in the native MLP.

## Frozen comparison

Two original initial states × two coverage conditions × two objectives = eight fits. Each uses 140 Adam updates: 70 natural-text updates alternating with 70 task-pair updates. Only the 589824 product-to-dictionary coefficients train. Dictionary, product factors, normalization and offset stay fixed. The causal weight and denominators remain those of M95.

The full pool visits all 70 training pairs once. The control visits every previously encountered pair at least once (35 or 41, depending on seed), then fills its remaining assignments by resampling within family. Both conditions have exactly the same task-family sequence and natural-document sequence. Training includes neither the 14 source-holdout pairs nor the seven whole-task families.

Independent input-level verification found 110 distinct source prefixes in the full pool versus 72/80 in the controls. Repeated identical prefixes have bit-identical cached normalized inputs. Thus the increase is genuine input coverage, not merely additional IDs.

The budget is matched in updates and task endpoints, not exact FLOPs. Full-pool assignments contain 1305 endpoint tokens; controls contain 1309/1298. Source-ineligible endpoint assignments are 2 in each full-pool schedule and 2/0 in the controls. These cases remain included under the existing loss rules. Family frequencies match between new conditions but differ from M95's older 66-task schedules.

## Interpretation and stopping rule

Evaluate all eight finals and both original initials on natural retention and both causal holdout cohorts. Report source, final-position and identity interventions separately, with all three background/change corners, absolute errors and equal-family/pooled summaries. The primary contrast is the change in joint-versus-KL performance when moving from the control to full coverage; it does not replace inspecting the absolute errors.

A positive result supports broader causal supervision for learning transferable replacements. It does not prove that dictionary coordinates are native computational units. That requires subsequent intervention specificity, reusable effects across tasks, and independent manifold validation. A negative result does not prove intrinsic impossibility or optimality.

Stop after this fixed comparison and synthesis. No optimizer, rank, or step sweep. Only two seeds are included; these historical evaluation cohorts are not pristine new-task discovery tests.

## Execution checkpoint

All eight fits completed: 1120 optimizer updates and 5022 native forwards. The structural audit checked every saved state, schedule assignment, loss arithmetic and Adam step count. Coefficients changed in all eight fits; all frozen buffers were preserved. Independent NumPy loss recomputation on the fixed first training example and natural document agreed within 7.36e-16 across 72 forwards. These checks establish arithmetic and saved-state consistency, not transfer or complete optimizer-trajectory replay.

Held-out evaluation remains pending. Retain all eight models and both original initial references. Results and audits are in `validation/product_dictionary_coverage_training_results.json`, `validation/product_dictionary_coverage_training_audit.json`, and `validation/product_dictionary_coverage_final_loss_audit.json`.
