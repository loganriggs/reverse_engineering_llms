# EMLP-P5: distributed MLP1 response carries much of prefix-transfer recovery

Exploratory native-model result, 2026-10-28. Same opened context/noun panel as P2–P4.

## The unresolved question

P4 found that donor MLP0 writes across the prefix predict much of the natural MLP1 change and reduce
output discrepancy. But prefix replacement also changes residual pathways. Does recovery require
the induced MLP1 writes, or does the important signal mostly bypass MLP1?

## Four physical conditions

For each recipient, compute the prefix MLP0 edit and capture the resulting MLP1 writes at **all
positions**, including the continuation. Cross the prefix edit with those induced writes:

| Condition | MLP0 prefix | MLP1 writes |
|---|---|---|
| Baseline | Recipient | Recipient |
| Bypass only | Donor | Frozen recipient |
| Response only | Recipient | Captured induced writes |
| Joint | Donor | Captured induced writes |

The joint matches the ordinary live prefix intervention exactly. The baseline with frozen MLP1
matches intact recipient logits exactly. All persistent recipient embeddings/value-bus state is
preserved, attention remains live, and the rest of the model is recomputed.

Also test four random prefix edits per recipient. Each applies one signed feature permutation
shared across prefix positions, preserving the edit norm at every position, its position support,
and correlations under the orthogonal permutation. These are better matched to the prefix arm
than P4's local-only random controls, though they need not be on-manifold.

32 context-derived edits plus128 random controls. No fitting, task labels, or fresh validation.

## Donor-output recovery

Remaining discrepancy is pooled `KL(donor || intervention) / KL(donor || recipient)`; lower is better.

| Condition | At noun | Future-position mean |
|---|---:|---:|
| Baseline |1.0000|1.0000|
| Bypass only |.9747|.9859|
| Response only |.6638|.6606|
| Joint |.6203|.6037|

The joint recovers37.97%/39.63% of donor discrepancy. The response-only arm recovers33.62%/33.94%,
whereas bypass-only recovers2.53%/1.41%. Thus the changed distributed MLP1 writes carry much of
the observed recovery under these interventions. **These are contrasts, not additive causal shares.**

In absolute KL, the intact donor gap is .05673 at the noun and .14028 over the continuation.
The joint leaves .03519/.08469; response-only leaves .03766/.09266. These target-oriented gains are
not the same metric as the small recipient-output changes measured in the earlier local-edit pilots.

Random-prefix controls do not show comparable target recovery. Joint remaining discrepancy is
1.0083/.9988; response-only is1.0092/.9995. So generic amplification is insufficient: the direction
of the natural prefix edit matters for approaching this donor distribution.

## Context dependence and interaction

Joint future-gap recovery by context pair is50.75% (dog/dogs),2.11% (she/he),42.25% (enjoyed/disliked),
and48.65% (yesterday/today). The weak she/he result limits generalization even within this panel.
These labels identify textual contrasts, not validated behavioral task families.

Centered-logit factorial interaction RMS is11.89% of the total future joint-effect RMS for context
edits and3.79% for controls. This is appreciably larger than the earlier local/future write-response
factorials. Do not assume that near-additivity in P2/P3 makes this different decomposition additive.

## What we can now say

On this restricted panel, MLP1 is a direction-dependent transformation of information arriving
through distributed MLP0 writes. Its output response is causally relevant to donor-distribution
transfer. This is stronger than activation overlap or arbitrary sensitivity, but weaker than an
identified semantic circuit, a compact reusable representation, or a new-task discovery.

The response-only arm uses native induced writes computed by the original model. It is an oracle
intervention, not a small autonomous computation. No information about its parameter complexity,
feature sparsity or transfer to unseen tasks follows from its success.

## Next bounded localization

Split the induced MLP1 write change into the **one changed lexical position** and all other positions.
On an unchanged recipient MLP0 background, test each subset and their union. This distinguishes a
source-token relay from a genuinely distributed set of necessary response sites. It can define a
smaller causal target for later featurizer training, but needs fresh competent tasks before promotion.
Stop widening this same synthetic panel: this localization is the next discriminator, not another
round of prompt or architecture tuning.

## Verification

704 native forwards. Exact identity and joint replay. Independent NumPy checks of32 scalar metrics
from two saved direction examples differ by at most3.06e-16; all overall means and ratio arithmetic
checked. Previous P4 prefix-arm KL replays within1.96e-7 absolute: adding the difference here versus
directly assigning the donor write in P4 changes float32 arithmetic slightly. That difference is
reported, not called exact replay. No additional audit forwards.

[Results](validation/prefix_response_factorial.json) · [Audit](validation/prefix_response_factorial_audit.json)
· [Runner](probe_prefix_response_factorial.py) · [Metric audit](audit_prefix_response_factorial.py)

The runner writes sample logits to `/tmp/emlp_p5_logits.pt` for the audit. These are retained in the
main research workspace and omitted from the compact packet. All code/results are hash-bound.
No corpus-level inference, answer accuracy, unique mediation fraction, or learned featurizer claim.
