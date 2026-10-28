## H794 — Early second-source attention interaction causally affects final candidate composition

Previous turn was progress: H793 traced a large L2 attention interaction at the second value source, followed by growth at the query, while query-local L2 mixing was smaller. H794 tests those two early sites causally rather than inferring importance from their activation norms.

For every cell of the same4x4 L1 source-identity factorial, use one of four arms: no clamp; additive clamp of L2 attention output at the second source; additive clamp of L2 MLP output at the query; or both. A clamp replaces the module output with its stored factorial additive prediction z_a0+z_0b-z_00. Every single-source and identity cell is included, permitting independent replay checks and recomputation of the remaining output interaction. Attention value-bus output is preserved.

The joint arm uses the original reference additive query-MLP prediction, not one recomputed after the source clamp. It is therefore a specified compound clamp, not an ordering-independent decomposition or unique causal mediation calculation. Other positions and downstream paths remain live.

### Causal result

Remaining centered16-candidate interaction RMS relative to the unclamped baseline:

| Format | Donors | Second-source attention clamp | Query MLP clamp | Both clamps |
|---|---|---:|---:|---:|
| Vertical bars | Matching | .536 | 1.000 | .537 |
| Colon/newline | Matching | .534 | 1.000 | .534 |
| Vertical bars | Mixed | .891 | 1.000 | .892 |
| Colon/newline | Mixed | .929 | 1.000 | .929 |

The second-source attention intervention substantially changes the final non-additivity for matching donors. The early query MLP clamp has almost no effect on this readout, despite its nonzero traced activation interaction. The joint arm closely follows the source-only arm. This supports causal relevance of the tested source-attention interaction, while directly illustrating why activation-interaction magnitude alone is insufficient.

These RMS ratios describe the response to specific interventions; they are not uniquely attributed fractions of computation. Remaining interactions are substantial, especially for mixed donors. The experiment does not establish a complete or minimal route.

### Direction of its effect

The source-attention clamp increases average donor specificity rather than erasing it:

| Format | Donors | Baseline specificity | Source-clamped specificity | Mean change |
|---|---|---:|---:|---:|
| Vertical bars | Matching | .510 | .573 | +.0628 |
| Colon/newline | Matching | .417 | .476 | +.0594 |
| Vertical bars | Mixed | .273 | .299 | +.0261 |
| Colon/newline | Mixed | .227 | .242 | +.0143 |

For matching donors, item-bootstrap intervals for the change are [.0146,.1107] and [.0147,.1157]. Mixed-donor changes have intervals [.0139,.0372] and [.0008,.0279]. These are descriptive intervals over16 reused item clusters per format, with no multiplicity correction. Candidate-specificity definitions are unchanged from H792, and matching/mixed target-set sizes differ.

The interaction therefore modulates or counteracts part of the additive donor signal under these hybrid interventions. It should not be labeled a beneficial content-transmission channel or a harmful native mechanism from this result. Native copying is already competent; stronger counterfactual donor promotion is not necessarily better task performance.

Per-example donor-specificity interaction RMS also decreases: matching-donor values move from .344 to .211 in vertical bars and .367 to .192 in colon/newline. Mixed values move from .143 to .129 and .114 to .104. Query-only values remain essentially unchanged. This separates a genuine change in response structure from a mere change in average specificity.

### Verification

2,048 model forwards: all16 factorial cells,32 item-format contexts,4 arms. Native L1 replacements and additive clamps are exact at requested positions. No-clamp candidate responses and all single-source/identity cells replay the prior reference within9.54e-6. Identity logits replay within1.77e-5. Metrics finite; model/input/reference/runner hashes unchanged. Original reserved test untouched; natural-document pool remains26,112.

No-clamp references and zero-interaction single cells were checked, rather than assuming an additive clamp is behaviorally harmless there. This remains a reused competent-copying panel; independent replication of a newly selected mechanism is still required. The negligible query-clamp result is specific to L2, this site, intervention and readout; it does not establish that query MLPs or normalization are generally irrelevant.

### Next mechanistic discriminator

At layer2, immediately after the L1 source replacements, each source input depends only on its own replaced output. At the second source, its attention query can therefore depend on donorB while the first source's keys/values depend on donorA. With this model's unnormalized attention, the first-source-to-second-source attention edge is a concrete candidate for the observed cross-source interaction.

Next reconstruct that edge's factorial interaction from actual query/key/value factors, audit that its projected contribution explains the module interaction, and then intervene on its head contributions with full-module reference controls. This can identify a specific multiplicative computation rather than introducing a generic learned DAG merely because the architecture permits one. Any compact representation must preserve the remaining identity-specific finite responses, not just match mean activation geometry.
