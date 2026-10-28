# M32: context changes the direction of native intervention effects

2026-10-28. Native-model diagnostic; no student fitting, candidate selection or new prompts.

## Question and scope

M30 transported exactly the same task-span intervention into two additional wrappers. M31
reencoded components in the new context, but this did not remove the two selected block
causal-fidelity crossovers. M32 asks whether the native model's response itself changes only
through a common log-probability shift, through scalar gain, or through relative-output direction.
MLP1 is zero-indexed block 1. Scalar/block labels identify the frozen featurizer that generated
the intervention, not different native language models.

Identical MLP1 interventions have context-dependent relative-logit effects in the native model. Removing the common log-probability shift leaves 57–91% of the response-change energy; 60–91% of that centered change is orthogonal to the reference response. These are descriptive aggregates on familiar tasks, not semantic-unit validation or a universal failure of context transfer.

## Fixed comparison and exact decomposition

All 240 scheduled M30 task-region directions are retained in reference, passage and sample
contexts: 720 native responses and 480 paired comparisons. Task suffix token IDs and float32
intervention vectors are held fixed; prefixes receive zero intervention. Position and wrapper
meaning are not separately controlled. There are 31 inactive directions in each context.
The replay uses 845 forwards; cached full final-token vectors are stored locally, hash linked.

For each triple, fix weights w to the reference intact native next-token probabilities. Let
r be perturbed minus intact log probabilities. All comparisons use this same metric:

    d = r_context - r_reference
    total_change = E_w[d²]
    centered_change = E_w[(d - E_w[d])²]
    constant_change = E_w[d]²
    total_change = centered_change + constant_change

Centering each response also removes the log-softmax normalization constant exactly: its
centered log-probability response equals its centered raw-logit response. This is an algebraic
separation, not a unique attribution of internal causal mechanisms. The constant term can
reflect meaningful probability changes and should not simply be called an artifact.

Post-result projection: let u and v be centered reference and context responses. Set
alpha = <u,v>_w / ||u||²_w. Then

    ||v-u||²_w = ||(alpha-1)u||²_w + ||v-alpha*u||²_w.

The first term is rescaling along the reference effect; the second cannot be explained by
any scalar rescaling. This descriptive projection uses the observed context response; it is
not a learned or held-out predictor. Zero cases remain in sums and have undefined angles.

## Results

Fractions are ratios of summed energies, not averages of per-example fractions. Cosines are
medians among nonzero responses (60 scalar directions, 40 block seed 71, 49 block seed 72).

| Generator | Context | Centered / total change | Orthogonal / centered change | Median centered cosine |
|---|---|---:|---:|---:|
| scalar 71 | passage | 78.0% | 90.1% | 0.870 |
| scalar 71 | sample | 73.8% | 81.2% | 0.905 |
| scalar 72 | passage | 86.2% | 63.3% | 0.881 |
| scalar 72 | sample | 90.8% | 59.8% | 0.910 |
| block 71 | passage | 57.3% | 69.9% | 0.885 |
| block 71 | sample | 58.2% | 90.9% | 0.953 |
| block 72 | passage | 66.4% | 81.2% | 0.853 |
| block 72 | sample | 72.8% | 78.9% | 0.917 |

Median angles remain fairly aligned; the claim concerns the composition of the *change*,
not that the entire response rotates dramatically. Centered change/reference centered energy
ranges from 0.103 to 0.265 across aggregates. No architectural ranking is justified by the
unequal active groups, intervention magnitudes and generator geometries.

Post-result family accounting: the largest family contributes 20–42% of centered-change energy.
Removing one family at a time leaves centered fractions ranging from 47–96% across all cases.
Thus a claim that centered effects always dominate would be too strong; the paired results
nevertheless cannot generally be reduced to a common shift. Family omissions are correlated
sensitivity checks, not fresh validation or confidence intervals.

## Implication for representation research

A geometrically stable component can have different downstream effects in different contexts.
That is compatible with a reusable conditional computation: downstream readers may use the
same information differently. We should not define a valid feature by an invariant final-output
vector. Conversely, stable component geometry alone does not validate a computational unit.
This native-only experiment does not establish why one trained featurizer beats another.

The user's capacity/robustness question reinforces the distinction between input ambiguity,
internal damage and context-conditioned use. T2–T4 address capacity and noise in trained toys;
M32 addresses the effect of the same internal damage across native contexts. It does not test
rare exceptions, unseen conjunctions or capacity overload in the language model.

## Next bounded discriminator and stopping criterion

Before fitting context-conditioned representations, distinguish local context sensitivity from
nonlinearity caused by a large removal. Use the same fixed directions and contexts, symmetric
perturbations at amplitudes 0.1 and 0.01, and centered central-difference responses under the
same reference weights. Preserve every family, zero direction and competence failure. Check
agreement between scales and floating-point resolution before interpreting a local derivative;
report nonconvergence rather than tuning amplitudes to obtain a preferred result.

Deliverable: an audited per-direction local-versus-full-removal response comparison. If direction
changes persist at converged small amplitudes, future causal supervision should include context
or reader state; if they largely disappear, finite-ablation curvature is the more direct target.
This is one diagnostic, not an open-ended amplitude sweep. Then return to the scalar/block/shared
comparison and diverse task coverage rather than accumulating further wrapper variants.

All data remain development data. Whole-task exclusions remain sequence_copy, spatial_inverse,
weekday_successor, role_binding and tense (past_tense is a distinct allowed family). No new task
discovery, semantic manifold validation or unseen-combination transfer is claimed.

## Reproduction and evidence

The native replay refuses to overwrite completed results and supports its existing per-generator
progress cache. Audits need the local 188 MiB vector cache whose absolute path/hash is in the result.
The shareable packet includes scripts and scalar results, not those vectors or model weights.

- [Frozen protocol](validation/native_geometry_protocol.json)
- [Native results](validation/native_response_geometry.json)
- [Full-vector NumPy audit](validation/native_response_geometry_audit.json): 720 responses,
  480 decompositions and 88 aggregates; maximum centered-identity error 3.54e-15.
- [Post-result projections and family accounting](validation/native_geometry_concentration.json)
- [Independent residual-vector projection audit](validation/native_geometry_projection_audit.json):
  all 480 projections and 8 aggregates; maximum energy discrepancy 9.03e-17.

Scripts: replay_native_response_geometry.py, audit_native_response_geometry.py,
analyze_native_response_geometry.py and audit_native_geometry_projection.py. Run with the
established environment, single-thread BLAS/OpenMP and cached pinned native checkpoint.
The audit independently scores vectors; it does not independently implement the transformer.
