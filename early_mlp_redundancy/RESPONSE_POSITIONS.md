# EMLP-P6: the changed lexical position carries most of the recoverable response

Exploratory native-model localization, 2026-10-28. Closes the first synthetic-panel sequence;
further rank/feature fitting on this panel is not justified.

## Fixed intervention

Recreate P5's induced MLP1 write difference from the aligned-prefix MLP0 transfer. On an otherwise
unedited recipient, transplant that response at (a) the single changed lexical token, (b) every
other position, or (c) all positions. All positions include the shared continuation. These are
subsets of the **induced response**, not deletions of native MLP1 writes.

Four signed permutations at the changed position match that response vector's norm and provide
same-site controls. Neither the response nor its direction is fitted. The exact original model
still generates every induced write. P5 already showed that natural prefix directions, unlike
matched random-prefix directions, produce donor-output recovery.

## Results

Remaining discrepancy is pooled KL(donor || intervened) / KL(donor || recipient):

| Transplanted MLP1 response | At noun | Future-position mean | Future discrepancy recovered |
|---|---:|---:|---:|
| Changed lexical position only | .7127 | .7140 | 28.60% |
| All other positions | .9592 | .9643 | 3.57% |
| Full response | .6638 | .6606 | 33.94% |
| Norm-matched random at changed position | .9995 | .9904 | .96% |

At the noun, corresponding recovered discrepancies are28.73%,4.08%,33.62%,and.05%.
The full response exactly replays P5's response-only arm on every direction.

The changed position contains71.46% of pooled response-vector squared norm. That norm fraction
is a different measure from output recovery. Centered-logit interaction between changed-position
and other-position responses is13.08% of the full future response RMS, so do not allocate recovery
as unique additive position shares.

The source-position arm's future recovery varies substantially by textual contrast:33.60%
(dog/dogs),2.19%(she/he),33.66%(enjoyed/disliked),35.39%(yesterday/today). She/he remains weak;
the full response is not uniformly better there. No confidence or corpus-general claim follows.

## What this adds

Much of the useful induced MLP1 response is a **source-token response**, rather than a write at
the later noun where we initially read the internal signal. This reconciles two earlier findings:
MLP0 prefix transfer strongly aligns the later noun's MLP1 difference, yet directly replacing that
noun's MLP1 write alone gives little output recovery. A readable downstream signal is not necessarily
the position where the most useful causal write occurs.

This identifies a position in a finite transfer experiment. It does not identify a monosemantic
neuron, a low-dimensional manifold, or a standalone computational unit. Removing all but this
response is not evidence that the native MLP1 computation elsewhere is unnecessary.

## Important design limitation: only eight source contexts

The four noun anchors occur **after** the changed word. Their causal prefixes through that word
are identical within each direction. By causal attention, the source response cannot depend on
those later anchors. Thus the32 evaluated directional rows contain at most **eight distinct source
receiver/donor contexts**. The token-ID audit verifies this prefix identity.

Varying the noun tests downstream consumption, not independent rediscovery of the source feature.
An apparently stable source vector or rank≤8 fit would be largely built into the design. We will
not use such a fit as evidence of a reusable feature basis.

## Next research step and stopping rule

Stop enlarging or fitting this synthetic panel. The candidate causal-supervision target is now
concrete: the MLP1 write response at a changed token induced by an earlier MLP0 write intervention,
plus its output-distribution consequences.

Test that target on **varied prefixes before the changed token**, using multiple competent task
families and fixed donor/recipient alignment rules. Check native competence first if task labels
are used; output-distribution transfer alone needs no intended-answer label. Preserve existing
whole-task training exclusions and distinguish previously opened examples from new confirmation.
Use training/development data to establish feasibility before fitting a featurizer or spending an
untouched task holdout. A pass must show useful target recovery beyond norm and token-identity
baselines across source contexts, not merely four later noun continuations.

This connects the bounded redundancy problem back to diverse causal supervision for featurizers.
It does not authorize a new rank/seed/prompt sweep on these eight source contexts.

## Verification

320 native-model forwards. Baseline identity and pre-changed-word response errors exactly zero.
All32 full-response arms replay P5 exactly. Overall means/ratio arithmetic checked; independent
NumPy checks of28 scalar KL metrics across seven saved arms differ by at most6.94e-16. Source
prefix identities checked directly from token IDs. No new audit forwards, fitted model or documents.

[Results](validation/response_positions.json) · [Audit](validation/response_positions_audit.json)
· [Runner](probe_response_positions.py) · [Metric audit](audit_response_positions.py)

The runner writes samples to `/tmp/emlp_p6_logits.pt`; the audit accepts `--samples`. Sample logits
are retained in the main research workspace outside the compact packet. Dependencies and source
hashes are recorded. These opened, dependent synthetic cases remain exploratory.
