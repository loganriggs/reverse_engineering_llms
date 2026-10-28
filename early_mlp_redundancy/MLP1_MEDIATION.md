# EMLP-P4: much of the observed MLP1 context change transfers through prefix MLP0 writes

Exploratory finite-transfer assay, 2026-10-28. Same opened noun/context panel as P2/P3.

## Why this test

P3 showed that MLP1 amplifies MLP0 perturbations, including random directions. Amplification alone
does not show that the response reproduces information present in the natural donor. This assay
compares the induced MLP1 write change against the actual donor-minus-recipient MLP1 write at the noun.

All16 prompt pairs have equal token lengths, equal anchor indices, and exactly one changed token
before the shared anchor. Every prefix transfer is therefore position-aligned. In both directions:

- **Local MLP0:** install the donor MLP0 write only at the noun.
- **Prefix MLP0:** install donor MLP0 writes at every position through the noun.
- **Random local MLP0:** four signed permutations of the local write difference, matched in norm.
- **Donor MLP1 oracle:** directly install the donor MLP1 write at the noun, as a target-reconstruction control.

The residual bypass, original normalized embeddings, first-layer value bus, attention and later
computation remain recipient-native except for the explicitly replaced writes. The prefix and local
arms have different support and norms; this is a localization comparison, not a matched-budget contest.

## Internal write result

For target difference t and induced response p, report:

- signed projection `sum(t·p) / sum(||t||²)`;
- pooled cosine and relative L2 error;
- `write_R² = 1 - sum(||p-t||²) / sum(||t||²)`.

R² uses **no change** as the baseline. It is a pooled, energy-weighted squared-error reduction,
not a percentage of causal responsibility, semantic information, or whole-model understanding.

| Intervention | Directional rows | Signed projection | Cosine | Relative L2 error | Write R² |
|---|---:|---:|---:|---:|---:|
| Local MLP0 |32| .1771 | .4576 | .8919 | .2044 |
| Prefix MLP0 |32| .7618 | .9145 | .4128 | .8296 |
| Random local MLP0 |128| .0032 | .0092 | 1.0549 | −.1129 |
| Donor MLP1 oracle |32|1|1|0|1|

Prefix transfer reduces squared target error by **83%**, versus **20%** for local transfer.
Prefix write-R² is positive in all four context families: .6995, .8233, .8895 and .8931.
The local context-derived response has alignment absent from its norm-matched random controls.
Random perturbations can therefore amplify outputs (P3) without reproducing the natural MLP1 signal.

## What the prefix/local difference localizes

Both arms install the same MLP0 write at the anchor and preserve the same recipient embeddings.
Their difference is at earlier MLP0 positions. At this depth, that difference can reach the anchor
only through **attention1**; MLP1 itself is position-wise. This is an architecture-derived path
constraint, not discovery of a particular head or attention factor. The measured result is that
including these earlier writes substantially improves transfer of the native MLP1 difference.

This supports a distributed cross-position contribution through MLP0→attention1→MLP1 on this panel,
rather than a purely local echo or complete independence from MLP0. It does not exclude parallel
embedding/value-bus routes, assign a unique mediation fraction, or establish that 83% transfers on
other tasks. Attention remains recipient-conditioned and all weights stay native.

## Output fidelity is a separate result

Remaining donor-output discrepancy is `sum KL(donor || intervened) / sum KL(donor || recipient)`:

| Intervention | At the noun | Mean over continuation |
|---|---:|---:|
| Local MLP0 |1.0048|.9995|
| Prefix MLP0 |.6203|.6037|
| Random local MLP0 |1.0009|1.0001|
| Donor MLP1 oracle |.9796|1.0019|

A perfect local MLP1-write match barely recovers the complete donor-output difference. Prefix MLP0
transfer improves output fidelity more, but it also changes other positions and residual pathways.
**That improvement cannot all be credited to the measured noun-position MLP1 write.** This is why
an interpretable internal signal must be evaluated separately from downstream behavioral fidelity.
No target-answer correctness or task competence is measured here.

## Verification and limits

288 full native-model forwards; exact identity replay and exact oracle reconstruction. All224 saved
vector rows independently recomputed with NumPy: maximum relative metric error1.08e-15; overall
summary error6.67e-16. Stored output-KL ratio arithmetic checked; logits were not independently replayed
in this audit. Source and panel hashes bound. All16 prefix-alignment gates passed and are recorded.

The panel remains eight variants across four nouns, bidirectional and highly dependent, with one
continuation. No fresh data, fits, reserved validation, causal-share identification, or representative
corpus claim. Local low recovery cannot establish independence, and prefix high recovery need not imply
a small or reusable feature basis. This is an intervention using native donor information.

[Results](validation/mlp1_mediation.json) · [Audit](validation/mlp1_mediation_audit.json)
· [Runner](probe_mlp1_mediation.py) · [Independent vector audit](audit_mlp1_mediation.py)

The runner stores the224 small vector pairs in `/tmp/emlp_p4_vectors.pt`; run the audit afterward.
They are retained in the main research workspace, outside the compact share packet.

## Next bounded causal question

The prefix MLP0 intervention improves donor-output fidelity, but which path supplies that gain?
Cross prefix replacement on/off with **all-position MLP1 writes frozen/live**, always keeping the
same intact recipient baseline. This distinguishes the changed MLP1 response from the residual bypass
and later computation. Report donor-gap recovery and unrelated output changes separately. A positive
result would give a meaningful causal-supervision target for a featurizer; a negative would prevent
mistaking internal alignment for behaviorally useful computation. Keep this opened panel diagnostic.
