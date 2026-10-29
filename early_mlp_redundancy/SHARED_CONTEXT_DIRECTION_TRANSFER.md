# M50: conditional direction transfer is measurable, but needs a group-prior control

The target is prediction of a native intervention response, not maximizing disruption. Before native evaluation, this stage audits same-token/same-group donor availability and local direction transfer with recipient norm supplied. It does not yet measure the native response-prediction target.

## Fixed inventory and oracle magnitude

Keep every M49 input/group assignment. Deduplicate exact(prefix,group) within domain/source split. A donor must have the same token and group but a distinct causal prefix within that stratum. Cycle SHA256-sorted prefixes, without consulting gates, norms, output effects or task correctness. Unpaired positions remain excluded and counted.

Predict the recipient vector by scaling its donor vector to the recipient norm. If the target is zero, predict zero; if donor is zero but target is not, predict zero and retain the failure. This gives the predictor oracle magnitude/presence information. It tests conditional direction transfer, not deployable prediction. The normalized squared vector error is summed error / summed target energy; a zero-vector predictor scores1.

| Cohort | Unique prefix/group units | Eligible units | Eligible occurrence positions | Eligible token types | Eligible groups |
|---|---:|---:|---:|---:|---:|
| task train | 917 | 405 | 422 | 33 | 28 |
| task evaluation | 235 | 86 | 95 | 12 | 12 |
| natural evaluation | 4096 | 1999 | 1999 | 285 | 21 |

## Local direction results

Pooled across both seeds and source objectives. Models have different target vectors and occupancies, so this is descriptive, not a matched-target architecture ranking. Zero-donor failures stay in the error.

| Cohort | Grouping | Component | Relative vector error | Missing donors / nonzero targets |
|---|---|---|---:|---:|
| task train | random | exclusive | 0.6586 | 51/221 |
| task train | random | shared | 0.2575 | 44/196 |
| task train | random | union | 0.2777 | 51/221 |
| task train | coactivation | exclusive | 0.4434 | 40/191 |
| task train | coactivation | shared | 0.1756 | 33/172 |
| task train | coactivation | union | 0.1961 | 40/191 |
| task evaluation | random | exclusive | 0.7055 | 12/54 |
| task evaluation | random | shared | 0.3822 | 14/48 |
| task evaluation | random | union | 0.3202 | 12/54 |
| task evaluation | coactivation | exclusive | 0.4937 | 9/44 |
| task evaluation | coactivation | shared | 0.1980 | 9/44 |
| task evaluation | coactivation | union | 0.1989 | 9/44 |
| natural evaluation | random | exclusive | 0.8436 | 338/975 |
| natural evaluation | random | shared | 0.5322 | 312/857 |
| natural evaluation | random | union | 0.5619 | 338/975 |
| natural evaluation | coactivation | exclusive | 0.8144 | 343/1141 |
| natural evaluation | coactivation | shared | 0.3772 | 316/1041 |
| natural evaluation | coactivation | union | 0.4277 | 343/1141 |

Shared vectors transfer direction better than exclusive vectors in all six pooled grouping/cohort comparisons. On natural data, learned shared error.377 is below random shared.532, while learned exclusive.814 and random exclusive.844 are closer. These contrasts are conditional on model-specific targets and supplied magnitude. Different native sensitivities may reverse a geometric comparison.

## Interpretation and next native test

This supports some directional regularity within token/group identity. It does not establish that donors carry useful contextual information beyond a group’s stable average direction. Low dimensionality, a dominant atom, or a consistent signed direction could produce the same result. Missing donors remain common, especially on natural text; they must not disappear from later native evaluation.

Before native causal evaluation, add a training-only group-mean direction baseline computed from natural documents0–31 with the existing frozen models. Compare recipient native response against donor prediction, group-mean prediction, zero response, and a fixed norm-matched scrambled-direction control. Normalize nonzero candidates to the same recipient norm; keep zero baselines/failures explicit. Score native-probability-weighted log-probability response error relative to the recipient intervention, rather than standalone output disruption. This tests direction prediction conditional on oracle norm; it does not claim deployability, semantic reuse, or new-task discovery.

Freeze the group-mean construction and scrambling before native outcomes. No new memberships, regression, rank, strength, threshold or seed selection. Restrict comparisons to the fixed eligible population and report actual target response energy and gate support. A positive result must beat the group-constant baseline to support context-specific reuse rather than just a stable group direction.

## Verification

Inventory construction checks every source prefix/group occurrence, exact donor bijection, stratum/token/group preservation and distinct-prefix eligibility.192native forwards capture5248unique prefix/group writes; all192input write-energy totals independently replay M49 (maximum relative discrepancy4.26e-16). Local analysis keeps all8models×3components×3cohorts and performs384fixed sampled NumPy decoder/normalization checks. No independent full encoder replay or native response transfer was performed in this stage.

[Inventory protocol](validation/shared_context_inventory_protocol.json) · [Inventory](validation/shared_context_inventory.json) · [Local directions](validation/shared_context_directions.json) · [Capture audit](validation/shared_context_capture_audit.json)
