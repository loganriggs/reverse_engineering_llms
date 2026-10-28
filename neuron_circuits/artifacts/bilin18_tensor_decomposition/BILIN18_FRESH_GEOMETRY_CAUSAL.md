## H825 — Fresh block interventions transfer context, but do not establish causal locality

Execute all frozen H824 pairs:56 receivers across14 early residual blocks, each with same-token/same-position near and far donors from document-disjoint natural examples. Six residual0 blocks contribute24 receivers; eight residual1 blocks contribute32. No selection changes after outcomes. Residual0 block104's earlier coverage failure remains excluded and documented.

### Intervention and controls

At the receiver's selected final prefix position, add a decoded donor-minus-receiver block contribution. For the common-norm arms, normalize both near and far block directions to the near donor's patch norm. Compare training-only global PCA16 and fixed random16 projections of the full donor residual difference, rescaled to the same norm. Separately run literal native-scale block contribution swaps; near native and common-norm block edits coincide by construction. A zero edit is evaluated for every receiver. Other positions, earlier prefix context, and the persistent value bus remain those of the receiver.

Measure receiver KL as KL(native receiver distribution || edited distribution), and donor gain as KL(donor || native receiver) minus KL(donor || edited). Positive gain moves the receiver distribution toward that particular donor; it is not task correctness or a fraction of computation recovered. Different near/far donors have different target distributions, so raw gain differences between these arms are not a clean proximity test. Comparisons with PCA/random use the identical donor and receiver.

### Equal-magnitude results

| Site | Direction | Mean receiver KL: block | Mean donor gain: block | PCA donor gain | Random donor gain |
|---|---|---:|---:|---:|---:|
| residual0 | near | .000377 | .01700 | .00255 | -.00145 |
| residual0 | far | .000268 | .00626 | .00445 | .00961 |
| residual1 | near | .001832 | .01503 | .00082 | .00262 |
| residual1 | far | .002719 | .03972 | .02170 | -.00761 |

Block edits often move outputs toward donors: positive gains occur on15/24 near and10/24 far residual0 edits, and22/32 near and23/32 far residual1 edits. Means are skewed: block median gains are .000729/-.000657 at residual0 and .002940/.008066 at residual1 for near/far. Per-example and per-block data are retained.

There is no consistent demonstrated near/far ordering of receiver-distribution change at matched norm. Block far-minus-near mean receiver KL is -.000109 at residual0, descriptive block-bootstrap95% interval[-.000248,.000002], and +.000888 at residual1[-.000221,.002684]. The direction reverses across sites and both intervals include zero. Residual1 block30 contributes much of the positive site difference. PCA and random near/far comparisons also have intervals spanning zero. Failure to establish an ordering is not proof of equality or absence of computational geometry: the sample is small and this tests output-change magnitude, not complete response-function similarity.

Block-minus-PCA donor-gain intervals span zero in all four site/arm comparisons. Thus this batch does not establish a reliable benefit over a flat subspace. Block-minus-random gains are descriptively positive for residual0 near (+.01844,[.00726,.03191]) and residual1 far (+.04733,[.01308,.10183]); the other two comparisons span zero. All6 block means are positive for the first comparison and all8 for the second, but the latter is strongly influenced by block91. These are exploratory summaries, not multiplicity-corrected significance claims. Blocks may share documents and there are only6/8 blocks; block bootstrap does not remove every dependence.

### Native-scale transplant contrast

Far native-scale swaps produce much larger mean receiver KL: .00580 at residual0 and .05989 at residual1, compared with .000268/.002719 after shrinking far patches to the near norm. Mean donor gains are .06526/.20627, with positive gains13/24 and26/32. Larger edits therefore create much stronger effects in this batch, but also change direction, dose and off-manifold risk together. These native-scale results must not be mistaken for a distance-specific effect at controlled magnitude. Near native-scale arms exactly reproduce the corresponding common-norm outputs.

### What we learned

H822–823 established non-Gaussian local geometry under their controls. H825 does not elevate that geometry to a smooth computational manifold. Some selected blocks carry transferable contextual information beyond random directions, while evidence against PCA is inconclusive and controlled proximity has no consistent effect-size ordering. Discrete context mixtures, recipient-dependent sensitivity and ordinary subspace structure remain viable explanations. There is no justification here for assigning semantic task labels or adding hierarchy/DAG structure.

The experiment separates two claims: a block can carry donor-context information without Euclidean proximity predicting its downstream effects. The latter requires a more direct response-alignment or response-prediction test. The current norm-matched far patches are direction probes, not literal far-state transplants.

### Verification

663 whole-model forwards:159 unique native endpoints plus504 edited forwards (448 nonzero interventions and56 zero controls). All hashes unchanged; all metrics finite. Native prefix states replay captures to relative error2.15e-6; zero-edit full logits differ by at most2.24e-5, below1e-4 tolerance. Common patch norms agree to relative error6.66e-16. Full edited/native logits, bases and patches are saved for independent reanalysis. PCA uses only original2048 training documents; random basis seed825+layer. No reserved-test evaluations or new documents; natural-document pool26,112.

Next use the saved full-logit changes to test directional response alignment against patch alignment and donor distance, controlling recipient and edit norm. This can distinguish differing sensitivity from merely different effect sizes without additional forwards. Then use a small fixed-direction recipient-crossing and dose-linearity test if alignment remains context dependent. Any predictive comparison should train/select on one subset and evaluate on independent occurrences; the present56 receivers should not become an endlessly reused confirmation set. Keep alternative flat/grouped featurizers and diverse competent tasks in the broader research plan.
