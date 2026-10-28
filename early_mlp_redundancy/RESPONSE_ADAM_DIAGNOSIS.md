# M22: Adam's carried first moment drives the diagnosed local damage

2026-10-28. Five fixed one-step counterfactuals distinguish current gradients from optimizer history in the M21 failure. At steps80–83, retaining only Adam's carried first moment reproduces most or more than all of the monitored damage. Current-gradient-only updates cause little harm or improve the target, including a magnitude-matched control. This is a local diagnosis, not yet a demonstrated training fix.

## Unchanged reference and controlled counterfactuals

Replay block71 M20 response treatment with the same120updates and post-selected capital-recall target198. Inspect the prespecified zero-based steps79–83, around the rise after80completed updates. At each selected step, split Adam's first moment into beta1 times its previous value and (1−beta1) times the current gradient. Use the same actual updated second moment, epsilon, learning rate and bias corrections for both pieces.

Evaluate three temporary parameter changes from the same preupdate state:

1. Carried first-moment contribution only.
2. Current-gradient contribution only.
3. Current-gradient contribution scaled to the actual raw Adam update norm.

Apply the same unit-column decoder normalization after each. The third matches norm before normalization; achieved postnormalization norms are recorded. Restore the actual parameters after each diagnostic; optimizer moments are checked unchanged. Every final tensor and every M21 reference trajectory field reproduces the original run. Counterfactuals do not become part of a new training trajectory.

## Measured response error

Lower is better. Each row starts from its own original preupdate state; the counterfactual rows are not chained into a trajectory.

| Step | Current input | Before | Actual Adam | Carried only | Current only | Current, raw-norm matched |
|---|---|---:|---:|---:|---:|---:|
| 79 | capital_recall | 0.011337 | 0.011005 | 0.011360 | 0.011031 | 0.013586 |
| 80 | natural text | 0.011005 | 0.043500 | 0.043489 | 0.011006 | 0.011016 |
| 81 | country_language | 0.043500 | 0.104462 | 0.104729 | 0.043317 | 0.042717 |
| 82 | natural text | 0.104462 | 0.190562 | 0.190532 | 0.104465 | 0.104488 |
| 83 | capital_recall | 0.190562 | 0.268247 | 0.296615 | 0.171053 | 0.146245 |

At step82, for example, actual Adam increases error.104462→.190562. The carried-only counterfactual reaches.190532; the norm-matched current contribution reaches.104488. Both actual and matched-current steps leave the monitored active support unchanged. Similar evidence holds at steps80,81and83. At83 the current contribution actually offsets some of the harm from history.

The matched-current postnormalization update norms are within0.02% of the actual norms over these five steps. Thus the near-zero damage at natural-text steps80/82 is not explained simply by making a much smaller step. This is a directional effect under the retained adaptive scaling.

Step79 is an important exception to any universal conclusion. Its norm-matched current-gradient-only step worsens error.011337→.013586 despite a locally negative first-order prediction, and changes one token's support. Finite-step and support effects remain possible. We do not claim that discarding momentum guarantees descent or solves every training failure.

## What the diagnosis supports

For this selected trajectory, optimizer history persists in a direction that damages an already trained response while the current gradients in subsequent examples would have caused much less harm. This is consistent with an outdated direction continuing after the target's update. The experiment isolates the carried first moment conditional on the same second-moment history; it does not isolate all optimizer history, or prove that one past example supplied the entire harmful direction.

The finding narrows the featurizer problem. Valid native causal targets and correct gradients were insufficient to preserve responses under this optimizer/data sequence. Neither feature labels nor stable TopK support alone protect a computational role from training-induced drift. This is still a diagnosis of learning the representation, not an ontology of the native MLP.

## Next fixed training comparison

Close the single-sentinel counterfactual inspection. Test the concrete optimizer hypothesis across all four starts and both objectives: Adam beta1=0 versus the existing beta1=.9 runs, with beta2=.999, learning rate.0003,120updates, identical schedules, starting weights, response weights, targets and decoder normalization. Train both intact-KL and response-supervised arms at beta1=0, eight new fits; reuse the audited M20 beta1=.9 results as references. Freeze this comparison before fitting and do not sweep rates or beta values.

This factorial comparison can distinguish a generic optimizer effect from an interaction with response supervision. The real beta1=0 algorithm changes first-moment bias correction and step magnitudes, so the local decomposition is a motivation, not a quantitative prediction of its outcome. Evaluate full training-target retention, the unchanged704-direction native benchmark, all output-fidelity quadrants, natural forgetting and family-level effects. Do not select or tune against target198. If training still fails, preserve that result rather than broadening the optimizer search automatically.

A successful whole-run result would make native-response supervision more credible. It would still not validate semantic manifolds, feature identity, new-task discovery or complete reverse engineering. The broad research goal remains active.

## Verification and reproduction

```bash
python probe_response_adam.py
python audit_response_adam.py
python analyze_response_adam.py
```

The replay used1,045 native forwards. All120reference steps and every final tensor match M21/M20. Independent NumPy audit checks five first/second-moment decompositions, direction formulas and all15counterfactual response scores, norms, gradient-dot predictions and support changes. Maximum moment-relative discrepancy3.86e-8; maximum direction-sum versus actual raw-update relative discrepancy1.29e-5, consistent with float32 update/subtraction rounding. Maximum response-score discrepancy2.09e-16. Adaptive scaling and decoder normalization are explicitly retained; the counterfactuals are not independently trained optimizers.

[Protocol](validation/response_adam_protocol.json), [replay and counterfactuals](validation/response_adam_replay.json), [audit](validation/response_adam_audit.json), [comparison](validation/response_adam_analysis.json).

Full tensors remain local at `artifacts/bilin18_tensor_decomposition/emlp_m22_response_adam_samples.pt`. Scripts and small results are bundled. No job remains running.
