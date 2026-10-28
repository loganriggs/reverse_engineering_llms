## H816 — Shared component is causally useful across groups; geometric private selection is imperfect

Freeze the H815 dictionary and gates. For each held-out own/input-swapped response, compute its group from the original raw response once, then retain both components, shared only, selected private only, neither, or shared plus the other group's private block. Recompute RMS and the exact MLP response against the same recipient baseline, then replay downstream. No refitting, gate recomputation after ablation, or held-out selection occurs.

Errors below compare with the uncompressed full response, divided by the RMS full-versus-zero tested MLP contribution. They are fidelity errors, not fractions of uniquely attributable causal effect.

| Components retained | Copy error | IOI error |
|---|---:|---:|
| Neither | 1.000 | 1.000 |
| Shared + selected private | .256 | .335 |
| Shared only | .575 | .390 |
| Selected private only | .758 | 1.003 |
| Shared + other private | .509 | .294 |

### Shared component carries useful information in both groups

Removing shared coordinates while keeping the selected private block worsens the copy error from .234 to .654 in group0 and .318 to 1.026 in group1. For IOI the changes are .599 to 1.021 and .327 to 1.003. Thus the same two stored shared directions contribute to useful reconstruction under both group gates in this fitted representation.

This establishes limited operational reuse, not discovery of two semantic features or a natural model taxonomy. Sharing was imposed via the PCA common block. The response family, baseline, and downstream readout determine its effect. Removing two shared versus three private coordinates is not dimension- or norm-matched, so their raw ablation sizes should not be interpreted as intrinsic importance rankings.

### Private blocks and gate choice

Private coordinates are useful for copying overall: shared-only error .575 improves to .256 with the selected private block. For IOI the improvement is smaller, .390 to .335. In IOI group0, adding the selected private block actually worsens error (.541 to .599), although that cell contains only two lexical items and ten correlated response variants.

The other-private control exposes a limitation of geometry-based selection. It worsens aggregate copying fidelity (.256 to .509), especially group0 (.234 to .570). But for copying group1 it improves fidelity (.318 to .201), and for aggregate IOI it improves fidelity (.335 to .294). Group-specific effects are therefore neither uniformly interchangeable nor uniformly selective.

These findings concern the fixed-token log-probability metric used here. A block that better reconstructs that scalar may be worse for the rest of the output distribution; the present results do not establish full-distribution superiority of the other block. Wrong-private replacement keeps active dimensionality but does not match projected norm.

### Nonadditivity and checks

Output factorial interaction (both minus shared-only minus private-only plus zero) has RMS .000764 for copy and .000175 for IOI. This documents a finite nonlinear interaction; it is not a canonical causal allocation between components.

960 model forwards completed. The frozen H815 both-component results replay exactly in log probability, as do the original uncompressed references. Shared-plus-private input decomposition error is at most 1.78e-15, basis orthogonality error 3.11e-15, paired source directions identical, train/holdout sets disjoint, hashes unchanged, and all outputs finite. Original reserved test untouched; natural-document pool remains 26,112. An initial shell launch lacked the interpreter and exited before running the experiment; the corrected command completed normally with no failed numerical run.

Three held-out lexical items per task and correlated dose/source-state/arm variants limit the evidence. Group counts are copy44/52 and IOI10/38, and the gates themselves were initialization-sensitive in H815. The shared component is useful, but neither group labels nor private memberships are established as reliable manifolds.

### Next discriminating test

Evaluate these same frozen component interventions against the full uncompressed output distribution using KL, alongside the scalar readouts. This will determine whether geometric gate failures are limited to one token metric or extend to distributional fidelity. Only then consider a label-free fidelity-aware gate objective. Do not select private blocks on held-out outcomes or promote shared/private geometry to hierarchy or computational DAG without additional evidence.
