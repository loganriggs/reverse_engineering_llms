## H817 — Full-distribution KL supports the selected IOI gate, correcting the scalar-only interpretation

Repeat the same frozen H816 component interventions and evaluate KL(full uncompressed intervention output || candidate output) across the full vocabulary in float64. The full reference uses the same source perturbation, local baseline, and own/input-swap arm; it is not the native or zero-perturbation distribution. Reverse KL and total variation are secondary metrics. No gate is reselected using held-out outcomes. Full logits are saved for further analysis without another forward pass.

### Distributional reconstruction

| Components retained | Copy mean KL | IOI mean KL |
|---|---:|---:|
| Neither | .00090557 | .00001619 |
| Shared + selected private | .00005728 | .00000435 |
| Shared only | .00029377 | .00000772 |
| Selected private only | .00051579 | .00001512 |
| Shared + other private | .00022874 | .00000698 |

Relative to the zero-module-response control, selected shared/private mean KL is .0633 for copying and .2690 for IOI. These are ratios of mean divergences, not RMS errors, accuracy, or percentages of explained causal computation. Absolute divergences are small because these perturbations and the tested module's role are small. Mean total variation for selected shared/private is .00343 and .00108.

Removing shared coordinates worsens full-distribution fidelity in both groups for both tasks. Removing private coordinates also worsens aggregate KL on each task. Thus H816's operational shared-component usefulness survives moving beyond a named-token metric.

### The gate diagnosis changes with the readout

For IOI, H816 found that the other private block improved fixed-name log-probability fidelity overall. Full-distribution KL gives the opposite ordering: selected .00000435 versus other .00000698. Both learned groups favor the selected block in mean KL. The paired item-mean other-minus-selected KL difference is +2.63e-6, with descriptive three-item bootstrap interval [1.36e-6,3.46e-6]. Mean reverse KL and total variation agree with this ordering. The scalar-only observation must not be generalized into an IOI distributional gating failure.

Copying favors the selected private block overall (.00005728 versus .00022874), largely because group0 strongly benefits from it (.00007463 versus .00047025). In copying group1, the other private block has lower mean KL (.00002439 versus .00004260), matching the scalar result. However the item-mean difference interval [-5.00e-5,1.43e-5] spans zero on only three lexical items. The other block wins on 48.1% of group1 response variants, not uniformly across examples. This is a candidate local failure of the geometry gate, not evidence for replacing it globally.

Other-private is better on 34.4% of copying variants and 22.9% of IOI variants overall. Variants share lexical items, doses, local states, and arms; those percentages are not independent sample estimates. Different group objectives can prefer different representations, so future fidelity claims must name their readout explicitly.

### Verification and limits

960 model forwards completed. Every H816 fixed-token response replays exactly, as do the historical uncompressed references. Full-reference self-KL is exactly zero; all KL values are nonnegative and finite. Reverse KL and total variation are finite. Source directions match exactly, component sum error is 1.78e-15, basis orthogonality error is 3.11e-15, all tracked hashes unchanged, and lexical train/holdout sets remain disjoint. Reserved test untouched; natural-document pool 26,112.

Only three held-out lexical items per task support the results, and IOI group0 has two. The panels were examined before the internal fitting split. Saved full output logits support additional metrics, but trying further readouts on these same items does not make the validation independent. The gates' initialization sensitivity and lack of a universally better storage/fidelity tradeoff remain unresolved.

### Next research step

Freeze the current dictionary, membership matrix, and geometry gate before expanding validation to new lexical items on competent copying and IOI panels. Check tokenizer alignment and native competence independently; report all exclusions. Measure full-distribution KL, fixed-token effects, and gate stability rather than optimizing further on these three held-out examples. A larger fresh panel can determine whether shared-component usefulness and the possible copying-group1 gate mismatch survive. Do not claim a canonical manifold, semantic hierarchy, or computational DAG from the present operational reuse result.
