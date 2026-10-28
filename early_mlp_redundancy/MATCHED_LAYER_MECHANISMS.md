# EMLP-M1: the first matched layer comparison rejects a simple strategy dichotomy

2026-10-28. Same160 opened training endpoints and native model, same masks for MLP0/1/2,
all downstream computation live. Four quarter deletions, two complementary half deletions, one
87.5% deletion (576 retained products), and all-products deletion. Bias remains in each arm.
Channel-index masks are identical across layers, not semantically aligned. No fit or ranking.
The extreme mask is one replicate; quarter/half masks come from one fixed partition.

## Channel implementation: measure interference, not only summed magnitudes

For product-channel writes c_i, excluding Down bias, define:

    E_channel = sum_positions sum_i ||c_i||²
    E_net     = sum_positions ||sum_i c_i||²
    energy ratio = E_channel / E_net

The signed pair-interference sum is E_net−E_channel. A ratio above1 means net destructive
interference; below1 means net constructive interference. Some pairs may cancel even when the
aggregate is constructive. This is native-channel implementation geometry, not basis-invariant
functional complexity. Aggregate energies before division; report zero-norm states separately.

| Layer | Channel/net energy ratio | (Net−channel)/channel energy | Contribution participation |
|---|---:|---:|---:|
| MLP0 |.229|+3.364|1056|
| MLP1 |.403|+1.482|1077|
| MLP2 |.520|+.924|1246|

All three are net constructive on this panel. MLP2 is less constructive, not aggregate
cancellation-dominated by this measure. Participation is the energy-weighted effective number
of nonzero contribution magnitudes, not a rank or count of independent features.

The sum-of-norms/net-norm ratios are15.75,21.65,25.79. Those large numbers alone cannot establish
cancellation: many orthogonal contributions also produce a large ratio. This qualifies the first
toy's “cancellation ratio” terminology. In that constructed toy it distinguishes two known cases;
in the real model it must be paired with signed interference. No near-zero net-write positions
were observed. All geometry includes the same prompt positions used by the deletion assay.

## Matched deletion behavior

Mean native||edited full-output KL across positions and endpoints; quarter and half entries
average their respective masks. These are descriptive averages, not uncertainty estimates.

| Removed fraction | MLP0 | MLP1 | MLP2 |
|---|---:|---:|---:|
|25%|.01068|.01222|.00887|
|50%|.03894|.03698|.02311|
|87.5%|.66621|.27227|.07118|
|100% of products|1.04122|.59793|.11907|

MLP2 is less disrupted in output KL despite larger local relative write errors:

| Removed fraction | MLP0 local error | MLP1 local error | MLP2 local error |
|---|---:|---:|---:|
|25%|.1057|.1365|.1604|
|50%|.3064|.3488|.3794|
|87.5%|.8340|.8037|.8370|

Local error is squared error divided by native write energy, including bias, computed per endpoint
then averaged. Different local effects and output effects demonstrate why channel geometry and
downstream consequences need separate axes. They do not prove active compensation in later layers:
input states, write magnitudes, directions and downstream readers all differ between layers.

The old MLP2 finding—retaining512 native channels was worse than deleting the layer—does not
reproduce in this one matched random mask on the task panel. This does not invalidate that study:
its natural-text data,512 versus576 retained products, mean compensation, selection procedures
and outcome metrics differ. It does invalidate treating its conclusion as an unconditional MLP2
identity or using it as the counterpart of D1's mild MLP1 deletions.

## An exact expectation explains much of the local deletion curve

For a uniformly sampled fixed-size deleted subset of m out of N channels, let p=m/N and
p2=m(m−1)/(N(N−1)). Then:

    E ||sum_deleted c_i||² = p2 ||sum_all c_i||² + (p−p2) sum_i ||c_i||².

This follows directly from singleton and pair inclusion probabilities; no fitted parameters.
A post-result check, independently verified by exhaustive enumeration on six toy vectors, gives:

| Layer |25% observed / expected error|50% observed / expected error|
|---|---:|---:|
|MLP0|.1057 /.1057|.3064 /.3075|
|MLP1|.1365 /.1378|.3488 /.3499|
|MLP2|.1604 /.1606|.3794 /.3806|

The observed finite partition closely tracks this generic expectation at these fractions. This is
not independent validation of a learned mechanism: the expression uses observed native contribution
energies. It shows why a deletion curve alone cannot identify duplicated semantic computation.
See [expectation receipt](validation/matched_layers_deletion_expectation.json) for the one-mask87.5%
comparison, which deviates more. No claim of statistically exceptional or ordinary masks is made.

## Updated mechanism hypothesis and next discriminator

The supported working picture is a continuum of distributed, partially aligned contributions
across all three layers, with MLP2 less collectively aligned on these inputs. Its downstream
consequences are also smaller here. None of the layers is assigned one exclusive strategy label.
The extreme cancelling toy is not a sufficient explanation of the current aggregate signatures.

Before training a large toy grid, add a known-ground-truth **shared-plus-private contribution**
model: each channel carries a common component plus a private component, with controllable signed
coordination. Require it to reproduce both energy/interference and local deletion signatures;
attach readers with controlled sensitivity to demonstrate how identical local errors can differ
in behavioral consequence. This is a falsifiable diagnostic toy, not a fit proving LLM mechanism.
Use held-out masks or inputs for any fitted toy parameters and report failures. Then test whether
training on token maps versus conjunction tasks changes these profiles. Natural-text provenance
and a fixed corpus check remain necessary before claiming the observed continuum is general.

## Reproduction and audit

```bash
python probe_matched_layers.py --output validation/matched_layers.json
python audit_matched_layers.py
python analyze_deletion_expectation.py
```

4,000 model forwards. All corresponding MLP1 metrics replay D1 exactly. All3840 intervention
rows and480 native geometry records saved; per-family summaries retained. Independent NumPy checks
cover12 full-sequence outputs (maximum metric error2.39e-8) and three native product/decoder captures
(max relative geometry error7.64e-9). Aggregates checked. Samples are generated at
`/tmp/emlp_m1_samples.pt`, not bundled. [Results](validation/matched_layers.json),
[audit](validation/matched_layers_audit.json).
