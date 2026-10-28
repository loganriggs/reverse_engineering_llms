# EMLP-P10: a direction shared across word pairs steers behavior, but is not the native response

2026-10-28. Executed the fixed P9 protocol on previously inspected training data. This is
lexical cross-validation, not an untouched scientific holdout, new-task discovery, or a trained
featurizer. The original whole-task exclusions remain unused.

## Construction and exclusions

Agreement has four singular/plural pairs: dog/dogs, cat/cats, bird/birds, horse/horses.
Sentiment has wonderful/terrible, excellent/awful, great/bad, pleasant/unpleasant. For each held-out
pair and each receiver direction, construct a unit direction from the other three lexical pairs:
unit-normalize each original-context response, average within lexical pair, average the three
lexical pairs equally, then normalize. Repeated identical source prefixes are deduplicated.
The bank contains eight agreement and 32 sentiment directional source responses. The 16
family × lexical-pair × receiver-direction estimates exclude the test pair across all its suffixes,
preambles and both directions. Forward and reverse directions are estimated separately.

Test 40 prefixed pairs (80 directions): agreement under both competent preambles and sentiment
under the one competent preamble, exactly as selected by P8. Retain incorrect examples inside
passing cohorts. Every intervention has the native test response's norm, explicitly an **oracle
magnitude**. Compare native response, same-lexical original-context direction, held-out mean
direction, and four signed-permutation random controls. Family and edit orientation are supplied;
they are not inferred without supervision. No output labels optimize direction or gain.

## Results

Percent recovery of the prefixed donor-output KL gap; higher is stronger donor steering, not
necessarily a more faithful model of the intermediate computation.

| Cohort | Native response | Same-lexical direction | Held-out mean direction | Random | Mean cosine: shared/native |
|---|---:|---:|---:|---:|---:|
| Agreement A |18.62|16.43|30.91|.14|.567|
| Agreement B |19.67|16.68|30.43|−.05|.593|
| Sentiment B |18.72|16.98|29.44|.29|.373|

A = `Consider the following text. `; B = `Read this carefully: `.
Shared-direction task-margin recovery is17.58%,18.45%,18.98%, respectively, versus native
9.68%,10.32%,10.80%. Both metrics support behavioral transfer; neither establishes natural fidelity.

Every lexical fold has positive shared-direction KL recovery:

| Held-out pair | Agreement A | Agreement B | Sentiment B |
|---|---:|---:|---:|
| dog/dogs |14.06|13.73|—|
| cat/cats |38.44|40.46|—|
| bird/birds |36.15|36.65|—|
| horse/horses |35.12|31.34|—|
| wonderful/terrible |—|—|25.08|
| excellent/awful |—|—|33.35|
| great/bad |—|—|29.53|
| pleasant/unpleasant |—|—|29.91|

Dog/dogs performs worse than its native response in both contexts. The pooled advantage is not
uniform superiority. Folds share training pairs and suffixes share source contexts, so these are
not independent replications or a large-sample confidence result.

## What this changes

A same-word lookup is no longer the only useful candidate: directions estimated from other words
can drive the relevant behavior. Yet their average cosine with the native response is only.37–.59.
In particular, pleasant/unpleasant has cosine.174 but29.91% donor-KL recovery. Good steering and
good intermediate-vector reconstruction clearly differ here.

Three live explanations guide the next experiment:

1. A shared task-related component is mixed with lexical-specific components in native responses;
   averaging enriches the shared component.
2. The mean is a broadly effective steering direction, but is not an important component of the
   natural causal response. Oracle norm can amplify a small shared projection.
3. The native residual contains useful lexical detail, opposing effects, or both; removing it
   changes behavior in ways that donor-KL and one task margin do not fully characterize.

None is resolved by the current shared-vector success. We have not shown a necessary feature,
a naturally gated scalar, a task-independent manifold, or transferable computation across tasks.

## Next fixed discriminator: projection and residual

On the same 80 test directions, use the already fitted held-out unit direction u. Decompose the
native response r exactly into p = u(uᵀr) and e = r−p. Apply p alone, e alone and p+e to the
otherwise native recipient. Crucially, do not renormalize p or e: these arms measure their
actual contribution at native amplitude. Verify that p+e replays the native-response arm.
Report component energy, signed projection, donor KL, task margin, and the non-additive interaction
of the component effects. Do not call these unique causal shares. Include a cross-family mean
direction at native response norm as a specificity control (all available original training pairs
of the other family, separate receiver directions, no per-example selection).

If the actual shared projection carries useful recovery and the residual adds little or opposes
it, this supplies a candidate causal component for the shared-dictionary featurizer objective.
If only the renormalized mean steers well, retain it as a steering target rather than calling it
a faithful feature. The cross-family control tests a narrow specificity alternative, not general
semantic selectivity. No more prompt search, new gains, or exposure of whole-task holdouts.

## Reproduction and checks

```bash
python probe_lexical_direction_transfer.py --output validation/lexical_direction_transfer.json
python audit_lexical_direction_transfer.py
```

832 model forwards. All16 training exclusions are audited; all80 estimated directions independently
reconstructed in NumPy (max element error1.80e-8);12 saved output triplets independently checked
(max metric error4.44e-15). The native/same-lexical control metrics replay P8 within5.15e-6; the
same-lexical expression changes float32 operation order (unit-normalize then scale versus norm
ratio then multiply). This tolerance does not affect the reported effect sizes. Norm matching,
arm counts, and every cohort/fold KL and margin aggregate are checked. This is a sampled output
audit, not independent replay of every forward.

The runner writes `/tmp/emlp_p10_samples.pt`, required by the audit but not bundled with the
share packet. [Results](validation/lexical_direction_transfer.json),
[audit receipt](validation/lexical_direction_audit.json).
