# M20: the fixed response-supervised run fails to retain its training targets

2026-10-28. Eight frozen-protocol fits, native intervention evaluation and final training-target checks are complete. Response supervision does **not** establish better computational units here: it worsens its own final training response error in all four starts, while transfer is mixed and ordinary output fidelity generally costs more. Diagnose optimization before treating this as evidence against causal supervision itself.

## Exact experiment

Four M18 expanded-context starts × intact_KL versus intact_KL_plus_response. Both arms use identical expanded task/source pairs and natural documents,120fresh Adam updates at.0003,60natural/60task and six task updates per family. The treatment adds the M19 frozen initial-decoder response objective with fixed training-calibrated weights. Both reconstructed branches differentiate; native targets and deltas remain detached. All57 inactive training directions retain their KL updates. No coefficient/group/width search or checkpoint selection.

All native training inputs replayed exactly against the M19 target cache before optimization. Initial and eight final models are evaluated on the same M18 output inventory. Native validation retains the old M11-based704 perturbations, including all96 zero directions. All native effects/delta energies/no-ops replay, and new initial baselines match the previous M18 expanded results. This rules out changing the benchmark as an explanation of arm differences.

## Training-target failure comes before generalization

Final mean response error on all120 scheduled training endpoints per model, measured at the final checkpoint. The initial column is the M19 starting value. This is training fit, not transfer.

| Start | Initial R | Intact-KL control R | Response-trained R |
|---|---:|---:|---:|
| scalar71 | 0.004298 | 0.003648 | 0.021409 |
| scalar72 | 0.002317 | 0.002077 | 0.013399 |
| block71 | 0.002505 | 0.002342 | 0.018042 |
| block72 | 0.007959 | 0.004539 | 0.018687 |

The response-trained models also have larger final combined task objective K+lambda*R than both their starting values and the control checkpoints. Online loss samples are not a final fit estimate: they concern different targets at different times and can conceal later damage.

Post-result concentration accounting finds one endpoint supplies40–79% of final treatment response error per start. The largest endpoints were already much better before their scheduled updates than at the final checkpoint. For example scalar72 capital recall had initial R=.00961, online preupdate R=.00357 at step105, and final R=1.13703. That establishes deterioration between those observations; it does not identify which update, gate transition or optimizer mechanism caused it.

Paired omission of the single largest treatment-error endpoint still leaves treatment mean response error worse than control for all four starts. The failure is therefore not solely the inclusion of one outlier. This omission is a descriptive sensitivity, not a repaired model or independent confirmation.

## Native causal transfer remains mixed

Relative squared margin-response error on the unchanged benchmark (zero response predictor=1), plus the native-weighted logprob response metric. Lower is better.

| Model | Region | Margin: control → response-trained | Logprob response: control → response-trained |
|---|---|---:|---:|
| scalar71 | prefix | 0.708 → 0.737 | 0.782 → 0.854 |
| scalar71 | task | 2.636 → 2.269 | 0.689 → 0.647 |
| scalar72 | prefix | 1.602 → 1.399 | 1.217 → 1.212 |
| scalar72 | task | 0.382 → 0.343 | 0.417 → 0.463 |
| block71 | prefix | 0.360 → 0.560 | 0.739 → 1.213 |
| block71 | task | 1.414 → 1.224 | 0.789 → 0.797 |
| block72 | prefix | 1.049 → 1.381 | 0.903 → 0.924 |
| block72 | task | 0.688 → 0.981 | 0.630 → 0.755 |

Margin fidelity improves4/8 model/region aggregates, equal-family margin improves3/8, and the logprob-response metric being trained improves only2/8 aggregates. Restricting to both withheld lexical items and withheld templates, margin fidelity improves2/8 comparisons. This is not consistent evidence of successful causal transfer. Some individual improvements are real measurements but should not override failure of the aggregate training objective.

Fixed-direction response fidelity remains weaker than final-feature identity or semantic-manifold recovery. Different starting dictionaries also define different perturbation families across models, so within-start arm comparisons are cleaner than architecture rankings.

## Ordinary fidelity costs

All-row mean output KL on the joint lexical/template evaluation and natural-text evaluation:

| Start | Joint evaluation: control → treatment | Natural evaluation: control → treatment |
|---|---:|---:|
| scalar71 | 0.0605 → 0.0642 | 0.1191 → 0.1259 |
| scalar72 | 0.0563 → 0.0624 | 0.1130 → 0.1207 |
| block71 | 0.0779 → 0.0860 | 0.1436 → 0.1479 |
| block72 | 0.0705 → 0.0780 | 0.1538 → 0.1569 |

Joint output KL worsens allfour comparisons; natural KL costs about2–7%. Joint task top1 is unchanged for scalar71, drops for scalar72 and block72, and increases for block71. No single metric supports a universal quality ranking. Full family/region/quadrant results remain available.

## Next bounded diagnostic, not a hyperparameter sweep

Before more fitting or a distribution-shift diagnosis, replay one failed trajectory with instrumentation: block71 treatment, the largest proportional final combined-objective regression among these four starts. This choice is post-result diagnosis, not new confirmation. Keep the original data order, targets, optimizer and120updates, and require reproduction of the saved final weights/results.

Track its largest-error training endpoint before/after each update. Measure the actual parameter update's first-order prediction for that endpoint's response loss, changes in selected supports, and an evaluation using the new parameters but the endpoint's previous support held fixed. This can separate smooth update interference from discontinuous gate changes; it will not establish a universal cause from one trajectory. Do not modify the loss weight or learning rate during replay. Stop after identifying reproducible failure steps or reporting that the replay cannot yet explain them.

The featurizer deliverable is an optimization diagnosis needed before reliable native-response supervision. Finite-difference checks validated a local response gradient; they did not guarantee stability of Adam steps, changing TopK supports, or alternating natural/task updates. Keep those hypotheses separate from implementation errors, insufficient capacity and generalization. The failed M20 run remains preserved and is not replaced by a tuned success. Semantic units and new-task discovery remain unachieved; the broad goal remains active.

## Audit and reproduction

```bash
python train_response_supervision.py
python audit_response_training.py
python probe_response_native_fidelity.py
python audit_response_native_fidelity.py
python evaluate_response_training_targets.py
python audit_response_training_fit.py
python analyze_response_fidelity_comparison.py
python analyze_response_fit_concentration.py
```

Training/evaluation uses16,871forwards (including423 extra nonzero response branches), native validation3,004, and final training-target diagnosis1,646. Twelve checkpoints include four unmodified initial baselines. No jobs remain for this experiment.

Independent training audit checks all8 schedules/frozen loss components,12 hashes/initial tensor equality, all13,824 evaluation records and228 featurizer/output samples (maxdecoder2.14e-7, metric6.81e-8, zero tie exceptions). Native audit checks all704 old direction/teacher invariants,2,112 response records,336 summaries and72 decoded perturbation/output samples (maxmetric1.78e-15, decoder2.09e-7, zero ties). Final training-fit audit checks all960 target/arm records,88 summaries and16 sampled full objectives (maxmetric4.06e-16). These audits do not independently reimplement the native transformer or optimizer trajectory; that is why the next replay is needed.

[Training results](validation/response_training_results.json), [training audit](validation/response_training_audit.json), [native fidelity](validation/response_native_fidelity.json), [native audit](validation/response_native_fidelity_audit.json), [training-target fit](validation/response_training_fit.json), [fit audit](validation/response_training_fit_audit.json), [weighted/equal-family comparison](validation/response_fidelity_comparison.json), [post-result concentration](validation/response_fit_concentration.json).

Local model states: `artifacts/bilin18_tensor_decomposition/emlp_m20_response_training/`. Large tensor samples stay local; JSONs/scripts/reports are bundled in this packet.
