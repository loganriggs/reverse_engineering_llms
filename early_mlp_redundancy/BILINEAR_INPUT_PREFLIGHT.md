# Executable bilinear input featurizers: correctness preflight

M81 local derivative preflight is complete. Original hard-gate crossings and a separately frozen smaller-step follow-up are both retained. The subsequent fixed warmups failed for seven of eight models; see [warmup diagnostic](BILINEAR_INPUT_WARMUP.md). No successful feature-discovery claim is made.

## Implemented representation

The predictor consumes normalized MLP1 inputs. Fixed training-only input/output means and global RMS scales connect those inputs to128signed bilinear coordinates, followed by a unit-column decoder. Dense, scalar, disjoint-block and shared-group variants have identical443,776trainable parameters and initial arrays within each ofseeds71/72. Gates use deterministic stable descending order with lower-index ties. Shared groups select a union of coordinates and decode each atom once. Group membership is fixed; this implementation does not learn a hierarchy or computational DAG.

All eight initial states are frozen with zero optimizer updates. Statistics use exactly4,092natural training positions, computed in float64 and storedfloat32. Independent NumPy statistics agree with the Torch calculation. No task or evaluation data enters normalization.

## Checks completed

- Independent NumPy polynomial evaluation, group scoring, stable sorting and set-union decoding agree for all eight models. Zero-score ties and parameter equality across variants are checked.
- All32 fixed full-parameter directional finite differences agree with architecture autograd under the predeclared tolerances and unchanged supports.
- An injected rank-one bilinear teacher is exactly represented by every architecture. In the shared-group case, two selected groups contain the active atom; its contribution is counted once. This is parameter-injection recovery, not evidence that optimization learns the teacher.
- All eight whole-MLP overrides exactly match instrumented callbacks while both native product branches are guarded. The native model is restored exactly. This uses18native forwards on a training document, with no inference access to native writes.
- The task loss implements all three background/change corners, each centered on its own background. It excludes identity from the active response mean and uses the fixed training-only family/region scales. Donor and recipient inputs are encoded separately.
- Independent NumPy scoring agrees on18fixed training cases, including both endpoints of an unequal-length pair. Identity losses are exactlyzero and ineligible source branches are omitted. Maximum metric discrepancy2.67e-15.
- Autograd reaches both predicted endpoints; all featurizer parameter gradients are finite, at least one is nonzero, and all native model gradients remain absent. This establishes connectivity, not native derivative accuracy.
- Providing true native donor/recipient writes recovers the teacher on four cases; maximum recovery loss2.03e-12. The native loss checks use142forwards, for160native forwards total in this stage so far.

## Original gates, now completed, and current next action

Before fitting, freeze and execute native endpoint and model-parameter directional finite-difference checks, with explicit perturbation scales, gate-crossing reporting and numerical tolerances. Check training-only natural KL arithmetic/gradients through the same input adapter. Verify that saved initial states and normalization are the exact ones used. Then freeze warmup/evaluation receipts before running the already specified eight warmup starts. Calibration occurs after warmup, using training data only, before the16matched downstream fits.

The existing comparison remains fixed: two supervision arms within each architecture, matched initial state, parameters, schedule and updates. No additional seeds, architecture tuning, checkpoint selection or evaluation-driven loss weights. All holdouts remain excluded. See validation/bilinear_input_featurizer_preflight_protocol.json for the complete planned comparison.

Files: bilinear_input_featurizer.py, bilinear_input_runtime.py, bilinear_input_causal_loss.py, check_bilinear_input_architecture.py, prepare_bilinear_input_starts.py, check_bilinear_input_override.py, check_bilinear_input_native_loss.py, and their hash-bound validation JSON. Large initial states and native-loss samples remain in the research artifact directory.

Update: the native finite-difference and natural-KL checks described above are now completed under validation/bilinear_input_preflight_completion.json. Eight warmups ran after their execution/evaluation protocols were frozen. Downstream calibration remains deferred because the audited warmup results are unstable. See BILINEAR_INPUT_WARMUP.md for the complete evidence and next diagnostic.
