

## H278 — Behavior-focused constants improve KL while worsening residual-energy fidelity

H277's MSE-optimal constant gave only modest intact behavioral gains. H278 optimizes the same1152-dimensional constant for native-to-approximate next-token KL. The original compiler, mean, core/partner reads, dense linear and quadratic coefficients are all frozen. Exact native final RMS/head/softcap are used during training. Native logits are teacher targets only, not predictor inputs. All nonconstant saved arrays are verified byte-for-byte identical to H277, and stable response predictions remain bitwise unchanged.

First768 discovery documents train, next256 validate. Start from each MSE constant. Three fixed Adam rates0.3/3/30,400 updates, minibatches32, checkpoints0/100/200/300/400; step0 eligible. Both actual/mixture programs select rate3 at step400. KL uses float64 log probabilities from float32 logits, with the head frozen and gradients only through the constant. No test tuning. Fresh128 documents exclude2112 prior complete-document hashes; uniqueness, exclusions and file hash verified. This is not a global or model-training holdout.

| Model/constant | Residual fidelity | Mean KL | KL recovery vs compiler | CE increase | Native top1 agreement |
|---|---:|---:|---:|---:|---:|
|compiled|0.00%|0.08928|0.00%|0.09583|84.18%|
|actual_mse|49.46%|0.06590|26.19%|0.09766|86.13%|
|actual_kl|37.20%|0.05107|42.80%|0.06560|88.09%|
|mixture_mse|49.22%|0.06529|26.88%|0.09967|85.74%|
|mixture_kl|36.63%|0.04982|44.19%|0.06631|88.09%|

Locked gates: behavior_constant_gain: PASS, bias_dominant: FAIL, repair_actual: FAIL, repair_mixture: FAIL.

- actual, KL versus MSE constant: 22.51% relative KL reduction, paired document-bootstrap95% [15.81,29.14]%.
- mixture, KL versus MSE constant: 23.68% relative KL reduction, paired document-bootstrap95% [16.65,30.32]%.

The20% behavior-constant gain criterion passes for both programs. The50% bias-dominance and both intact repair criteria fail. Repair requires>=80% KL reduction relative to original compiler, CE increase<=.02 and top1>=.98. The calibrated programs still have CE increase about.066 nats and88.1% top1 agreement. These remain failed replacement candidates.

| Program | Training KL, MSE→KL constant | Validation KL, MSE→KL constant | Constant change norm |
|---|---:|---:|---:|
|actual|0.07163→0.05023|0.07684→0.05669|3549.54|
|mixture|0.07265→0.05058|0.07789→0.05660|3595.45|

**Information gained.** The same response map can correspond to meaningfully different intact behavior. Optimizing a constant for KL lowers test KL by22–24% relative to MSE calibration while lowering residual-energy fidelity from49% to37%. Thus energy-optimal calibration is not behavior-optimal. The mismatch is directly demonstrated while all state dependence stays fixed; it is not inferred from comparing unrelated model classes. Constants can therefore matter substantially to a proposed computational account even though finite differences cannot identify them.

**Limits.** Both selected checkpoints are at the400-step boundary. This experiment does not establish optimizer convergence or the best possible constant, so remaining error cannot yet be attributed exclusively to missing state-dependent computation. Rate30 is worse than rate3; simply taking larger steps is not evidence of an optimum. The objective is nonconvex through RMS/softcap. These results establish a useful correction and a metric mismatch, not an impossibility theorem, semantic features or a unique representation.

**Next discriminator.** Audit optimization within the constant-only class before adding another state-dependent component: restart from the selected constants with fresh optimizer moments, a fixed lower-rate schedule and additional checkpoints (optimizer state was not saved, so this is not a literal trajectory continuation), compute full-discovery gradient norms and compare train/validation progress. Retain the400-step candidates in selection. Freeze the resulting constants before new-document confirmation. If the boundary keeps moving, report that; if multiple starts/schedules stabilize far from the behavioral criteria, the case for missing state dependence strengthens. In every case verify nonconstant arrays and stable response maps remain identical. This is a targeted convergence check, not a claim that arbitrary extra optimization must succeed.

**Costs/audits.** Three focused tests pass: teacher-to-student KL direction and analytic gradient, finite-difference gradient check, anchor/response invariance and constant-fit identity. Stable-response bitwise checks pass for both real programs. Nonconstant arrays verified identical. Scalar storage remains3,191,040 correction plus504,224 compiler; only1152 values change, with no extra inference cost. Six fixed runs complete. Curves, all selected constants, fresh test errors and separate *_kl_program.npz exports retained. They can use the standalone intact_forward evaluator and still fail replacement criteria. Full project suite not rerun. Native weights/original compiler unchanged, figure inspected. Broad goal remains active.
