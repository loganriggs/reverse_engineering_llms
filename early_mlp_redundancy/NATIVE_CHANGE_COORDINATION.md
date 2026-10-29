# M66: native task changes are distributed and net constructive across MLP0–2

Across all 1002 nonzero directed layer/site conditions, native channel contributions to the donor-minus-recipient change are **net constructive**, not net cancelling. This covers 70 training pairs and 14 familiar source-group evaluation pairs, both endpoints, MLP0–2 and source/final sites. There are also 504 identity conditions and six retained ineligible source alignments. Reciprocal endpoints are not independent observations. This is a description of the fixed panel, not a universal statement about all prompts.

## What is being measured

For each native bilinear channel j, form its product change `c_j = product_j(donor) - product_j(recipient)` and vector contribution `v_j = Down[:,j] * c_j`. These contributions sum to the MLP write difference. We concatenate selected positions when measuring norms.

- `S = ||sum_j v_j||²` is energy in the combined change.
- `C = sum_j ||v_j||²` is summed individual contribution energy.
- `S - C` is the sum of cross-channel inner-product terms. Positive means net constructive coordination; negative means net cancellation. A positive total does not rule out particular cancelling pairs or groups.
- Participation is `(sum_j e_j)² / sum_j e_j²`, where `e_j = ||v_j||²`. It describes spread in the native channel basis, not a count of semantic features or sufficient causal units.

These are *change* contributions, not intact writes. No channels were removed from the functioning model in this atlas, so it does not establish literal removal redundancy.

## Layer and site comparison

Ratios below pool raw energies over each split/site. Participation is the median over directed conditions. Linear-only error is squared error divided by native channel-sum energy.

| Split | Layer | Site | C/S | Median participation | Linear-only error | Expected half error | Expected doubled-half error |
|---|---:|---|---:|---:|---:|---:|---:|
| train | 0 | source | 0.741 | 210.8 | 1.496 | 0.435 | 0.741 |
| train | 0 | final | 0.785 | 491.5 | 0.094 | 0.446 | 0.785 |
| train | 1 | source | 0.600 | 285.2 | 0.368 | 0.400 | 0.600 |
| train | 1 | final | 0.759 | 563.4 | 0.031 | 0.440 | 0.759 |
| train | 2 | source | 0.654 | 121.0 | 0.348 | 0.414 | 0.654 |
| train | 2 | final | 0.789 | 589.2 | 0.037 | 0.447 | 0.789 |
| source_eval | 0 | source | 0.744 | 215.4 | 2.248 | 0.436 | 0.744 |
| source_eval | 0 | final | 0.878 | 545.8 | 0.110 | 0.469 | 0.878 |
| source_eval | 1 | source | 0.563 | 366.9 | 0.480 | 0.391 | 0.563 |
| source_eval | 1 | final | 0.798 | 609.8 | 0.025 | 0.450 | 0.798 |
| source_eval | 2 | source | 0.717 | 320.6 | 0.533 | 0.429 | 0.717 |
| source_eval | 2 | final | 0.847 | 516.6 | 0.036 | 0.462 | 0.847 |

MLP1 source changes have the strongest pooled constructive coordination in both splits (C/S about0.600 training and0.563 evaluation). MLP2 is also net constructive; this does not support a general “MLP1 redundancy versus MLP2 cancellation” distinction. Final-position changes spread their energy more broadly than source changes in all six layer/split comparisons, with median participation roughly491–610 channels versus121–367 at source positions. Panel composition and change magnitudes differ across sites.

## Theory makes a concrete prediction about redundancy

For N channels and a uniformly selected fixed-size subset of k retained change contributions, let p=k/N. The expected squared error from dropping the other contributions is

`E[error] = (1-p)² S + p(1-p)(N C - S)/(N-1)`.

At exactly half retained, doubling the retained change is unbiased over masks, but its expected squared error is `(N C - S)/(N-1)`. Unbiasedness does not mean lower error. Doubling beats raw half only if `C/S < (N+2)/(3N)`, approximately one third.

No nonzero condition in this atlas satisfies that inequality. Predicted raw-half errors are about0.391–0.469 after pooling; doubled-half errors about0.563–0.878. A fixed32-mask Monte Carlo agrees closely with these analytic expectations. Masks were shared across conditions, so their aggregate errors are not independent samples. The exact formula, rather than Monte Carlo agreement, establishes the local expectation.

Thus contributions are distributed and reinforce each other, but are not sufficiently aligned for simple doubled-half averaging to improve local fidelity. This is more precise than labeling distributed activity “redundant.” Whether downstream behavior follows this local prediction is the next causal test.

## Nonlinearity and cancellation depend on the decomposition

At the normalized MLP input boundary, with donor input `z+dz`, the exact bilinear change is

`Down[L(z)*R(dz) + L(dz)*R(z) + L(dz)*R(dz)]`.

The first two terms are the recipient-anchored linear change; the third is quadratic. This expansion does not linearize upstream normalization or attention. The donor-anchored linear change minus the same quadratic term also gives the full difference. Both identities were checked.

Dropping the quadratic term gives only about2.5–11.0% pooled squared error at final positions, but34.8–224.8% at source positions. Large lexical changes need substantially nonlinear corrections. The quadratic term can exceed the final change because it offsets part of the linear term. That is cancellation between algebraic expansion terms, not a contradiction of net constructive coordination across native channels. The magnitude of a Taylor term also depends on the chosen expansion anchor; it is not automatically a semantic operation.

## Implication for computational units

The current evidence favors testing continuous, context-dependent task-change coordinates over treating hard-selected decoder groups as established native units. It does not yet tell us how to group channels into reusable mechanisms. A useful group must predict native responses and transfer across tasks; simply matching the spectrum, participation or local write is insufficient. The atlas supplies exact native finite-change targets for that next representation step.

## Validation and scope

168 native forwards replayed all saved baseline final logits exactly. Available M57/M59 native write deltas replayed exactly. Three linear branches were checked to have no biases. All1506 eligible conditions passed an independent NumPy audit of channel sums, energies, participation, both finite expansions and every32-mask error vector. Twelve small synthetic cases exhaustively enumerate fixed-size subsets, including duplicate, orthogonal and cancelling constructions, and verify the formulas. Maximum scaled metric difference is2.43e-14; mask error difference4.86e-16. Actual float32-native versus float64-expanded residuals remain in the raw rows.

No new task was exposed, no featurizer fitted, and historical whole-task training exclusions remain intact. This atlas is closed. The fixed next protocol, `validation/native_change_subset_response_protocol.json`, tests raw-half and doubled-half change interventions in the native model; it will not be described as ordinary neuron deletion.
