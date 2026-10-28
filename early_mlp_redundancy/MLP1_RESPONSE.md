# EMLP-P3: MLP1 strongly amplifies the tested upstream changes

Exploratory native-model result, 2026-10-28. This applies EMLP-P2's intervention to MLP0→MLP1,
returning directly to the question of what MLP1 adds beyond earlier token information.

## Comparison held fixed

Use exactly P2's opened 4 context pairs × 4 location-noun anchors and shared continuation.
For each of 32 directional pairs, add donor-minus-recipient **MLP0** write at the noun position.
Retain recipient embeddings, value bus, other MLP0 writes, attention and residual state. Four
signed permutations per edit preserve its norm and reuse P2's permutation/sign seeds.

Freeze or release the **MLP1** response at the edited position and later positions, yielding
frozen_all, local_only, future_only and live. Frozen writes are the exact unedited recipient's
outputs. All unedited baselines match. All later layers, including MLP2, remain live.

P2 and P3 share prompts, directions of donor exchange, support, control construction and metrics.
They do **not** have matched perturbation norms across layers: each uses its own native source
write differences. Thus differences in their gain ratios are not a layer-importance ranking.
No fitting, behavior labels, new natural-text documents or fresh confirmation split.

## Results

Mean KL(intact recipient || edited output) for context-derived MLP0 changes:

| MLP1 response | Edited token | Future-position mean | Final position |
|---|---:|---:|---:|
| Frozen all | .000002789 | .000000248 | .000000165 |
| Local only | .000082206 | .000007461 | .000004000 |
| Future only | .000002789 | .000001742 | .000002023 |
| Live | .000082206 | .000009534 | .000005914 |

Live/frozen ratios are **29.48 at the edit, 38.42 over future positions, and 35.88 at the end**.
Every one of the32 context edits increases all three KL measures when MLP1 responds.
The large ratios start from small frozen-baseline effects; report the absolute KL values too.

Random controls also amplify: ratios **52.63 / 38.37 / 34.95**. All128 controls increase all three
measures. This is strong evidence against claiming semantic specificity from amplification alone.

Context-edit ratios by anchor:

| Anchor | Edited token | Future mean | Final position |
|---|---:|---:|---:|
| house | 43.44 | 63.34 | 61.86 |
| park | 34.80 | 33.25 | 18.08 |
| school | 13.06 | 29.89 | 31.16 |
| station | 44.26 | 32.76 | 35.64 |

At the edited position, live equals local_only and future_only equals frozen_all exactly, as
causality requires. Local response supplies much of the increased effect on future predictions,
with a noticeable additional future-position contribution. Centered-logit factorial interaction
RMS is only **.383%** of the total live-minus-frozen future response for context edits, and **.255%**
for controls. This is approximate local additivity on these finite edits, not a global linear model.

## What this says about redundancy

MLP1 is not functionally inert to these context-derived MLP0 perturbations. It changes their downstream
impact substantially, even though earlier work found strongly token-predictable outputs and overlapping
class information. Repeated or decodable information can be transformed into a different causal effect.

But “MLP1 uses an MLP0 perturbation” is weaker than “MLP1's natural contextual computation is mostly
inherited from MLP0.” This intervention changes one MLP0 write while leaving the rest of the recipient
context intact. The donor's actual MLP1 difference also reflects other positions, attention1 and
embedding paths. The present data do not measure how much of that whole natural difference is mediated
by this one source. Nor does a large KL change establish improved or impaired task behavior.

The random-control result currently favors a broad local gain mechanism over a specifically selected
semantic amplifier. It does not exclude structured response directions within that gain. The next
useful distinction is between inherited contextual signal and parallel contextual construction.

## Next bounded discriminator

On these same opened examples, measure the finite MLP1 write response to the local MLP0 edit and compare
it directly with the full donor-minus-recipient MLP1 write. Report signed projection, cosine and residual
error, not just norms. Include all-position MLP0-write transfer only where exact token alignment is valid,
and keep persistent embeddings/value bus fixed. A low local recovery would not establish independence
because other MLP0 positions can contribute. No feature fitting should precede this mediation check.

This produces a specific causal target for a later featurizer: the context-conditioned response mapping,
with a measured scope. It avoids treating activation overlap or compression as proof of a reusable unit.

## Verification and limits

704 model forwards; identity and pre-edit logit errors exactly zero. Source SHA256, all overall summary
means, per-position means, metric finiteness and all causal equalities checked. Independent NumPy replay
of56 positionwise KL values differs by at most8.29e-16. No additional audit forwards.

The authored sample has four location nouns, one shared continuation, dependent forward/reverse edits,
and norm-matched controls that may be off-manifold. No corpus generality, task competence, natural
mediation fraction, or semantic interpretation is claimed. Raw source-edit norms range1028.99–4395.37;
these are not output-logit changes. Frozen writes observe the recipient baseline and are oracle controls.

[Results](validation/mlp1_response.json) · [Audit](validation/mlp1_response_audit.json)
· [Runner](probe_mlp1_response.py) · [Metric audit](audit_mlp1_response.py)

The runner writes sampled audit logits to `/tmp/emlp_p3_audit_logits.pt`; run `audit_mlp1_response.py`
afterward. Sampled logits are retained in the main research workspace, outside the compact share packet.
