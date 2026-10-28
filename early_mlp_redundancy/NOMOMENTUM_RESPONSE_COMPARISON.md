# M23: removing first-moment memory does not rescue response supervision

2026-10-28. Eight fixed new fits, four M18 expanded-context starts (scalar/block, seeds71/72), two objectives, compared with the eight existing M20 fits. Adam beta1 changes from .9 to 0; beta2=.999, learning rate .0003, 120 updates, paired data order, initial states, fixed perturbations and training-only response weights remain unchanged. Actual first moments equal current gradients at every new update. This changes step magnitudes and bias correction; it is not a norm-matched ablation of the entire optimizer history.

## Training retention remains the bottleneck

R is the native-probability-weighted squared error in final log-probability changes under a fixed intervention. Lower is better. All scheduled endpoints, including inactive targets and repeats, are included.

| Start | beta1=.9 control R | beta1=.9 response R | beta1=0 control R | beta1=0 response R |
|---|---:|---:|---:|---:|
|scalar71|.003648|.021409|.003734|.024989|
|scalar72|.002077|.013399|.002396|.041879|
|block71|.002342|.018042|.003343|.010085|
|block72|.004539|.018687|.005054|.033553|

The new response arm is worse than its matched intact-KL control and starting response error in all four starts. Only block71 improves over the old response arm. Its combined task objective is still worse than its start. This is not evidence of a clean training-versus-generalization gap: the trained targets themselves are not retained.

## The unchanged causal benchmark does not show a rescue

All704 native directions, native effects, noops and denominators exactly match the old M18 benchmark. Directions remain defined by the M11 basis, distinct from training perturbations defined at M18 starts. Of eight model/region comparisons, response supervision at beta1=0 improves aggregate margin error in1/8, log-probability response error in1/8, equal-family margin error in5/8, and jointly withheld lexical/template margin error in3/8. The weighting-dependent results matter: equal-family improvements do not imply improved aggregate fidelity. The jointly withheld causal panel has only six pairs.

Ordinary joint lexical/template evaluation KL worsens in all four response-versus-control comparisons. Natural KL worsens in three; block72 improves slightly. All are familiar development data, not pristine validation. No whole-task exclusions changed, no new tasks were discovered, and no semantic-unit or manifold claim follows.

## Interpretation and stopping criterion

M22 correctly identified carried momentum as the dominant local damaging direction for a selected trajectory segment. M23 shows that removing it is not a sufficient full-run solution. Do not generalize that local diagnosis to the entire failure. Remaining explanations include sequential objective interference, finite-step sensitivity, target concentration, and representational constraints; this experiment does not identify which dominates.

Close the optimizer branch: no learning-rate, beta or loss-weight sweep. Next bounded diagnostic should measure compatibility of the ten task-family response gradients and intact-output gradients across all four starting featurizers, using the fixed training inventory and weights. Report family gradient norms, pairwise alignment, and alignment with the aggregate objective; verify selected directional derivatives through both response branches. Use this to distinguish immediate objective conflict from a failure emerging over sequential training. Gradient conflict alone is not proof that no jointly satisfactory representation exists. The deliverable is a training-objective compatibility map that feeds the joint featurizer comparison; stop after that map and choose a single mechanism-directed experiment rather than another sentinel detour.

## Validation and artifacts

Training:16871 forwards,12 checkpoint entries,13824 evaluation records. Audit checks eight schedules,12 model hashes,228 independent NumPy samples, frozen components and full inventory. Native benchmark:3004 forwards,2112 comparisons,96 inactive directions;72 sampled NumPy response/decoder checks and336 summary checks pass. Final training-target evaluation:1646 forwards,960 arm-target records;16 independently recomputed sample objectives and all88 summaries pass (maximum metric error3.23e-16). Audits do not independently reimplement the native transformer or optimizer/backpropagation.

[Fixed protocol](validation/nomomentum_protocol.json), [training audit](validation/nomomentum_training_audit.json), [native audit](validation/nomomentum_native_fidelity_audit.json), [training fit audit](validation/nomomentum_training_fit_audit.json), [full factorial results](validation/optimizer_factorial.json). Full model states and sampled tensors remain in the local artifact directory with hashes in result JSONs; the packet contains scripts and numerical results.
