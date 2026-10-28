

## H279 — Constant-only optimization has diminishing tested returns, not a proven optimum

H278 selected its last checkpoint, so remaining error could not be assigned solely to missing state dependence. H279 restarts from those selected constants with fresh Adam moments (the previous optimizer states were not saved). For each actual/mixture program, base rates.1/.3/1 run1200 updates, multiplied by1 for steps0–399,.3 for400–799,.1 thereafter. Minibatches64; first768 discovery documents train, next256 validate; checkpoints0/400/800/1200 include the previous constant. All nonconstant arrays remain frozen.

Both programs select base rate1 at step1200. Full-discovery mean gradients are accumulated in16-site chunks at the previous and selected constants. Fixed negative-gradient probes of displacement norm1/10/100 are evaluated on train/validation only; they are not selection candidates or test models. Fresh128 documents exclude2240 prior complete-document hashes; uniqueness, exclusions and file SHA256 verified. Not globally or training-set fresh. Nonconstant saved arrays are verified identical to H278 and stable responses remain bitwise unchanged.

| Program | Train KL previous→refined | Validation KL previous→refined | Fresh KL previous→refined | Fresh refined CE increase | Fresh refined top1 agreement |
|---|---:|---:|---:|---:|---:|
|actual|0.050230→0.048955|0.056693→0.056147|0.055948→0.055439|0.06702|85.94%|
|mixture|0.050576→0.049261|0.056599→0.056139|0.055118→0.054688|0.06990|86.52%|

Locked gates: material_restart: FAIL, repair_actual: FAIL, repair_mixture: FAIL.

- actual: fresh KL reduction 0.91%, paired document-bootstrap95% [-0.05,1.96]%.
- mixture: fresh KL reduction 0.78%, paired document-bootstrap95% [-0.17,1.82]%.

Both intervals include zero and effects are below the5% material-restart threshold. Refined KL recovery relative to the original compiler is about40–41%, still below80%; CE increase about.067–.070 nats and top1 agreement86% fail the intact criteria. Do not compare top1 values directly with H278's different test panel as an intervention effect.

**Gradient audit.** Full-discovery gradient L2 norms fall from2.70e-6→1.13e-6 (actual) and3.08e-6→1.14e-6 (mixture), in KL-per-output-coordinate units. Small absolute gradients alone do not certify convergence because coordinate scale matters. The dimensionless norm||g||||b||/loss remains about.105 in both models. Both selected checkpoints are again at the budget boundary.

At the refined actual constant, a norm100 negative-full-gradient step lowers training KL.048955→.048899 but raises validation KL.056147→.056223. The mixture counterpart lowers.049261→.049198 while raising.056139→.056211. Smaller1/10 probes also improve training and slightly worsen validation. These probes show a residual descent direction but no demonstrated validation benefit in that direction; they are not a global search or selected predictors.

**Interpretation.** The constant-only class yielded meaningful earlier gains, but these fixed restart schedules show diminishing out-of-sample returns. The results justify shifting experimental priority to a controlled state-dependent extension, not declaring all constants exhausted. Remaining alternatives include nonconvex optimization, sampling limits and missing input dependence. The unchanged response map still cannot identify the constant; intact and response evidence remain distinct.

**Next discriminator: use existing reads before inventing new features.** Keep the refined constant, input mean, core32 and partner32 fixed. Add a learned linear correction from those already stored64 coordinates to the1152-dimensional write, with discovery-only feature normalization and validation-selected regularization/checkpoints. This adds73,728 coefficients but no new input directions or quadratic products. Compare an output-MSE fit with a downstream-KL fit and the constant-only baseline on the same held-out documents; freeze all choices before independent confirmation. This tests whether the existing coordinates contain useful state information whose readout is poorly matched to behavior. A gain would not imply semantic identification, and a failure would not rule out other coordinates or interactions. Count the additional coefficients and require intact-behavior metrics, not just response energy.

**Audits/costs.** Three focused tests pass: full-mean gradient accumulation is invariant to chunk size including partial final chunks, native-to-prediction KL gradient, and anchor/response invariance. Full project suite not rerun. Six restarts and gradient probes complete. Previous constants remain available; refined exports are separate. Scalar inference storage unchanged at3,191,040 correction plus504,224 compiler. No native answer used in predictor inference. Native weights/original compiler unchanged; figure inspected. Broad research goal remains active.
