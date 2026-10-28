## H745 — Wording-transfer errors come from omitted query and key directions

An exact algebraic diagnostic decomposes H744's frozen-subspace coefficient error. With projected query/key q,k and omitted donor residuals dq,dk, the full-minus-projected second-factor dot product is dq·k + q·dk + dq·dk. The model's first-factor gain and signed bus multiplier are retained. Norm restoration is measured separately. This performs no fitting and no additional model forwards; it does not establish downstream effects for the new rescue conditions.

| Competent fresh stratum | Frozen coefficient error | Full query + compact key | Compact query + full key | Captured query/key edit energy |
|---|---:|---:|---:|---:|
| New names, old narrative wording | 12.4% | 11.2% | 6.3% | 92.7% / 96.3% |
| New names, old list wording | 8.8% | 4.0% | 5.7% | 84.0% / 91.9% |
| Old names, new narrative wording | 11.4% | 5.8% | 12.6% | 57.2% / 84.5% |
| Old names, new list wording | 29.6% | 15.4% | 16.9% | 55.2% / 51.1% |

Errors are RMS coefficient deviation from the full-factor donor intervention, normalized by its change from native. Energy is mean squared edit length captured by the frozen rank-four subspace, separately for query and key. Rescues replace one compact state with its full donor state while retaining the other norm-preserved compact state. These are explanatory diagnostics on the already evaluated panel, not new held-out successes.

The new-list error is essentially unchanged without norm restoration: 29.66% raw versus29.61% norm-preserved. Norm restoration contributes only0.50% of the full edit's RMS scale, versus17.12% query-residual,15.28% key-residual,3.54% bilinear-interaction component norms. Component norms do not add as percentages because their signed effects can reinforce or cancel. Both omitted query and key directions matter. New narrative wording is more query-sensitive, although cancellation means restoring the key alone slightly worsens the error. Small omitted geometric energy can still matter causally, as the new-name narrative key result illustrates.

The new-name/new-wording cells remain diagnostic because native binding fails the H744 competence gate. Their coefficient errors are16.2% narrative and33.5% list; the same low list edit-energy capture persists. No selection or refitting is performed using these cells.

Interpretation: normalization is not the main explanation for reduced transfer. The frozen operation edit contains meaningful reusable structure but misses wording-dependent directions at both ends of the attention edge. The next discriminating test should separate a shared causal component from template-specific directions, using independently split wording families and matched nuisance edits. A union of template subspaces may describe the data better than one universal operation manifold; this is a hypothesis, not an established architecture.

Audits: exact bilinear expansion residual2.22e-16; H744 norm-preserved coefficient replay exactly zero; all input hashes unchanged; finite values. Zero new model forwards, zero reserved-test evaluations; natural document count26,112 unchanged.

Next: freeze a wording-family split before fitting; compare a shared edit subspace, a union of format/wording subspaces, and equal-rank nuisance-derived controls. Require held-out causal effect preservation, not only energy capture. Continue using verified components to investigate computation; no featurizer is declared ideal on reconstruction alone.
