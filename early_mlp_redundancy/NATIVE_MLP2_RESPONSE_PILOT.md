# EMLP-P1: native MLP2 responds to these MLP1 edits by amplifying output change

Exploratory result, 2026-10-28. No fitted model, no corpus sampling, no semantic labels trained.

## Question and intervention

Does MLP2 generically damp changes in the native MLP1 write, or was earlier attenuation specific
to the C512 approximation's errors? Eight authored context pairs all end with the same token
` the`. Swap only the **final-position MLP1 write difference** from donor to recipient, in both
directions. Recipient embeddings, attention, persistent value bus, and other MLP1 writes stay native.
The native MLP1 baseline is retained, so adding the actual difference installs the donor write at
that position; it does not install the donor residual, token sequence, or entire model state.

For each of 16 directional edits, also test four fixed-seed signed permutations of that exact
1152-dimensional write change. These preserve its Euclidean norm, not its semantics or naturalness.
All edits use the full measured amplitude; there is no amplitude selection or fitting.

Three MLP2 conditions:

1. **Live:** allow the native MLP2 to respond to the edited upstream state.
2. **Frozen write:** replace MLP2's output with its recipient-baseline write. Attention2, RMSNorm
   before subsequent computations, and the rest of the suffix remain live. The unedited baseline
   matches the intact model exactly, so live-versus-frozen isolates the changed MLP2 write's effect.
3. **Omitted:** replace the whole MLP2 write, including bias, with zero. Score relative to the
   *omitted* recipient baseline. This changes the reference distribution and is secondary context.

Measure KL(recipient baseline || edited output) at the final token over all 50,304 logits.
These are changes in a distribution, not error against desired donor behavior or task correctness.

## Results

| Direction source | Directional samples | Mean KL, live | Mean KL, frozen write | Ratio of summed KL, live/frozen | Live KL lower than frozen |
|---|---:|---:|---:|---:|---:|
| Same-token context difference | 16 | .00023302 | .00010744 | 2.169 | 0/16 |
| Norm-matched signed permutations | 64 | .00010751 | .00005432 | 1.979 | 6/64 |

Individual context-edit ratios range from 1.437 to 3.369. Native MLP2's response increases
output sensitivity in every tested context edit. Random changes are usually amplified too.
The context/random ratios are descriptive; their difference is not an established selective effect.
Opposite directions from each pair and controls from one recipient are dependent observations.

Mean omission-background KL is .00027819 for context changes and .00017510 for random changes.
Those values must not be compared as if deletion merely switches off a response while preserving
the intact reference. The frozen-write condition is the cleaner test of that question.

## What changes in our interpretation

A generic MLP2 damping story does not fit this panel. This does not contradict the historical
C512 result: C512 creates a particular upstream approximation error, and earlier analysis used
different interventions, positions and aggregate metrics. The leading possibilities now include
error-direction-specific attenuation, operating-state dependence, and a distinction between changing
a module's baseline output and changing its response to a perturbation.

There is no demonstrated special repair signal here, and amplification need not be harmful:
these context-derived edits may carry information the network normally uses. No behavioral accuracy
target was defined, so the experiment cannot call amplification correct or incorrect.

## Limits and verification

- All sites are a final ` the` token; other token identities and subsequent positions are untested.
- Only eight hand-authored pairs; no claim of representative natural text or independent replication.
- Exact donor write access and frozen recipient write are oracle interventions, not learned features.
- Context differences and random permutations can be off-manifold in the recipient state.
- Raw edit norms span 237.55–1273.02; norm matches pass at relative error below 1e-6.
- 288 short full-model forwards; native frozen-write identity gives exactly zero logit discrepancy.
- Local state/write accounting differs by at most .00024414 in float32 absolute coordinates;
  do not describe that arithmetic check as bit-exact.
- Three additional forwards independently replay the first live context edit. NumPy KL differs
  from the stored value by 1.02e-15. Every stored summary mean and metric finiteness was checked.
- No new natural-document dataset, trained featurizer, held-out-task discovery or full historical
  compensation replay is claimed.

[Results](validation/native_mlp2_response_pilot.json) · [Audit](validation/native_mlp2_response_audit.json)
· [Run script](probe_mlp2_response.py) · [Independent metric audit](audit_response_pilot.py)

## Next information-gain test

Freeze this intervention and control scheme. Cross several final-token identities with several
background contexts, keeping each donor/recipient final token matched. Include earlier sites with
future outputs scored separately. Test whether amplification persists or changes with token/position;
do not fit or widen a featurizer yet. Only then use a document-disjoint natural panel and task-relevant
contrasts to connect response geometry to reusable computation. Keep the C512 error direction as a
separate future bridge, requiring the exact saved program and its input contract.
