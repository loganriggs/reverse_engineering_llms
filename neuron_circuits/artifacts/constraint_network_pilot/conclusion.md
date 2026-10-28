# Constraint-network experiment: conclusion

## Outcome

This experiment produced a validated result and a falsification.

1. **Validated:** the signed margin-gradient kernel is an accurate causal local
   explanation of finite parameter-edit effects in the trained factual
   transformers.
2. **Falsified:** in these high-fidelity AdamW checkpoints, increasing arbitrary
   factual load does not increase the magnitude of blockwise constraint
   coupling or collateral damage.  Both decrease sharply.
3. **Refined:** the arbitrary-fact networks are broader in effective support but
   much weaker per edge.  They are diffuse, approximately sign-balanced, and
   closer to orthogonal—not absent.

The preregistered design is in
[`../constraint_network_preregistration.md`](../constraint_network_preregistration.md).
The full 32-fact primary result is in [`results.json`](results.json), and the
128-fact sensitivity result is in
[`../constraint_network_sensitivity_n128/results.json`](../constraint_network_sensitivity_n128/results.json).

## Primary causal validation

For each source fact and parameter group, the experiment applied a gradient
edit normalized to predict a fixed increase in that source's margin.  It then
recomputed the whole network and predicted all off-diagonal fact-margin changes
from the signed gradient Gram matrix.

At the preregistered one-logit scale, the two fully arbitrary models obtained:

| width | Pearson | origin-anchored R-squared | permuted-source Pearson |
|---:|---:|---:|---:|
| 128 | 1.000 | 1.000 | -0.003 |
| 256 | 1.000 | 1.000 | 0.034 |

The result remains essentially exact at the four-logit scale.  Thus the kernel
is not merely ranking infinitesimal effects: it predicts the signed finite
collateral after downstream recomputation over the tested range.  This
validates it as a local/finite causal model, not as a unique explanation of how
facts are globally stored.

## The load hypothesis reversed

The locked load comparison regressed out same attribute, answer, competitor,
randomization status, trait overlap, and ID-digit overlap.  Averaging the four
transformer blocks gave:

| width | rho=0 residual absolute cosine | rho=1 residual absolute cosine | rho=0 one-logit collateral | rho=1 one-logit collateral |
|---:|---:|---:|---:|---:|
| 128 | 0.0475 | 0.0108 | 0.0773 | 0.0115 |
| 256 | 0.0504 | 0.0071 | 0.0345 | 0.0070 |

The 128-fact sensitivity panel reproduced the direction and scale:

| width | rho=0 residual absolute cosine | rho=1 residual absolute cosine | rho=0 one-logit collateral | rho=1 one-logit collateral |
|---:|---:|---:|---:|---:|
| 128 | 0.0419 | 0.0108 | 0.0519 | 0.0116 |
| 256 | 0.0517 | 0.0069 | 0.0406 | 0.0073 |

Therefore arbitrary load alone did not recreate the strong tension web found
in the saturated one-layer bilinear memorizer.  The structured model had more
shared blockwise sensitivity because many facts actually used the same rule
computation.  The high-fidelity random models instead separated their local
sensitivities.

This does **not** mean the arbitrary networks are more localized to individual
neurons.  Mean blockwise effective support increased from about 31 to 59 facts
at width 128 and from 16 to 62 at width 256 on the 128-fact panel.  The edges
were nearly evenly split between positive and negative signs.  The organization
is consequently better described as **broad, weak, signed coupling** rather
than a taut web of large interactions.

## Independent training replication

The rho 0 versus 1 comparison was repeated with two fresh manifests and model
initializations.  All four new arbitrary models met the original factual
fidelity gate.  Across all three seeds and both widths (six paired cells), the
arbitrary/structured residual-coupling ratio ranged from 0.100 to 0.258 and the
collateral ratio from 0.093 to 0.224.  Conversely, the effective-support ratio
ranged from 1.87 to 3.83.  The one-logit finite-effect predictor passed its
locked threshold in all six arbitrary cells.

The reversal is therefore not a panel-selection artifact or a peculiarity of
the original random table.  Full individual results are in the
[`three-seed confirmation`](../constraint_network_replication_summary/summary.md).

## Layerwise result

Under full randomization, controlled residual cosine was largest in the shared
input/position group and first block, then fell to roughly 0.005--0.007 in later
blocks.  The output group retained moderate coupling, partly from shared class
roles.  In contrast, the structured models showed substantial coupling in
later blocks.

This is evidence that different layers can occupy different points in the
organization space, but the present kernel is a sensitivity map rather than a
percentage allocation of stored information.

## Relation to the neuron audit

The already-locked activation audit found median semantic held-out R-squared of
0.804/0.934 for rho=0 and 0.132/0.104 for rho=1 at widths 128/256.  Native axes
were close to random rotations under full randomization.  Combined with the
new causal result, the current models are best summarized as follows:

- structured facts use strongly shared, semantically predictable computation;
- arbitrary facts acquire stable fact-specific addresses;
- arbitrary-fact parameter sensitivities are broad but weak and nearly
  orthogonal at the block level;
- the signed kernel predicts their edit interactions extremely well even when
  individual native neurons lack simple semantic explanations.

That validates mechanism-level interpretability without validating
monosemanticity.  It does not yet validate simultaneous semantic patch transfer
and a strong KKT-like tension web inside the same layer.

## Revised hypothesis

The evidence rejects “more arbitrary facts necessarily create a stronger
constraint web.”  A better hypothesis is:

> Arbitrary facts create optimizer- and capacity-dependent constraint
> geometries.  With sufficient slack, AdamW transformers can make factual
> sensitivities diffuse but weak and approximately orthogonal.  Under genuine
> saturation, or under an SGD max-margin bias, those same constraints may
> tighten into the load-bearing KKT-like web observed in the bilinear model.

The next decisive experiment is therefore not another rho sweep at the same
capacity.  It is a matched-accuracy slack/saturation and optimizer factorial:
AdamW versus SGD, with fact load adjusted so that both optimizers achieve the
same recall.  The prediction is that coupling magnitude, support multipliers,
and edit collateral track slack and optimizer geometry more strongly than rho
itself.  Semantic activation patches should be evaluated independently on the
structured subset.

## Limits

- One data manifest and one initialization are available; the larger panel is
  a sampling sensitivity check, not an independent training replication.
- Width 64 was excluded from the primary matched-fit test because it did not
  learn the arbitrary facts.  It cannot reveal the geometry of a successful
  saturated memorizer.
- The transformer checkpoints used AdamW, so Frobenius-KKT attribution from
  the SGD bilinear experiment should not be expected without re-deriving the
  optimizer-appropriate geometry.
- Gradient kernels establish local causal sensitivity.  They do not prove a
  global max-margin optimum or identify a unique stored-fact decomposition.
