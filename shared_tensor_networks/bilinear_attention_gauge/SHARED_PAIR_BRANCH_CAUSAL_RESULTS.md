# A semantic function gauge does not preserve physical QK-branch edits

The [dense-token canonical result](DENSE_CODED_SHARED_PAIR_RESULTS.md)
finds three function-level count variables and reusable quadratic products
in a rank-three attention reduction. This experiment tests whether that
same three-dimensional key space describes the **physical** computation
when the two QK score branches are edited independently.

For each of ten paired-token and ten dense-token trained models, construct
four exact first-layer query/output-by-key tables from weights: ordinary
scores, QK1 set to constant one, QK2 set to one, and both set to one.
The intervention changes a branch score *before* the two scores multiply;
it does not merely rotate the ordinary output. A separate Torch forward
implementation performs the same edits. Its five-logit outputs agree
with the table-plus-learned-quadratic reconstruction within `5.5e-12`
over checked sentences, establishing that these are actual model edits.

Fit a shared key projector either to the ordinary table alone or to all
four tables stacked after unit-Frobenius normalization per condition.
No sampled sentences or task labels enter the fit. On 10,000 independent
sentences per model, compare the reduced model against the **physically
edited** model, separately for each edit. Table error, five-logit MSE,
and logit-sign agreement are all retained in the JSON. Edited outputs
have no supplied ground-truth task labels, so sign agreement here means
agreement with the edited model, not task accuracy.

| Token code and fitted projector | Rank | Ordinary output MSE | QK1 edit output MSE | QK2 edit output MSE | Double edit output MSE |
|---|---:|---:|---:|---:|---:|
| Paired, ordinary only | 3 | .001 | .759 | .938 | .790 |
| Paired, all four | 6 | .005 | .006 | .004 | .017 |
| Paired, all four | 7 | <.001 | .001 | .001 | .001 |
| Dense, ordinary only | 3 | <.001 | .889 | .938 | .940 |
| Dense, all four | 6 | .155 | .206 | .142 | .105 |
| Dense, all four | 7 | .028 | .020 | .022 | .038 |
| Either code, all four | 8 | 0 | 0 | 0 | 0 |

These are ten-seed means of relative squared five-logit error. Ordinary
rank three agrees in sign with the edited model on only about .60–.69
of its output coordinates, depending on token code and edit, despite
perfect unedited five-output task accuracy. The rare held-out slice is
also badly missed: mean relative output MSE after single or double edits
is roughly .88–1.11 at ordinary rank three. Full-rank eight is exact by
construction; the intermediate-rank tradeoff is the substantive result.

The distinction is mathematical, not a defect of a particular gauge:
all orthogonal gauges of a fixed rank-three reduction have the same key
projector `P`. For an intervened token table `T_edit`, the unrepresented
part `T_edit(I-P)` and its Frobenius error are identical under every
rotation of that subspace. The sparse-key, condition-wise-product, and
joint-nullspace gauges therefore cannot make the ordinary three-space
branch-faithful. The intervention-aware *reduction metric* must admit
additional directions before a legibility prior chooses coordinates.

The dense code needs more dimensions than the paired code to cover the
branch counterfactuals well. Separate QK factors expose directions
hidden by their ordinary product; the table errors alone do not prove
which physical components matter downstream. The edits are large
constant-one interventions. A dose sweep below checks whether the rank
gap also appears near the original model. Before moving to the 546M model, keep ordinary
function fidelity, physical branch-response fidelity, and downstream
behavior as distinct scores.

## The mismatch persists for small interventions

Interpolate an edited QK score as `s(α)=(1-α)s+α·1` before multiplying
the branches. Evaluate `α=.01,.05,.1,.25,.5,1`; the table construction
and a separate physically edited Torch forward agree at checked doses.
For each projector, subtract its **own** unedited baseline. Compare that
predicted five-logit change with the actual physical-model change,
normalizing squared error by the actual change energy. This response
metric does not vanish merely because an intervention is tiny.

| Token code, projector, dose | QK1 response relative MSE | QK2 response relative MSE | Both response relative MSE |
|---|---:|---:|---:|
| Paired, ordinary rank 3, `.01` | .306 | .413 | .263 |
| Paired, all-four rank 7, `.01` | .001 | <.001 | .001 |
| Dense, ordinary rank 3, `.01` | .409 | .387 | .267 |
| Dense, all-four rank 7, `.01` | .016 | .020 | .019 |
| Dense, ordinary rank 3, `1` | .765 | .813 | .820 |

For a single edit the first-layer table is affine in `α`; the output's
quadratic readout makes the five-logit response
`Δy(α)=α J·ΔT+O(α²)`. Thus a nonzero relative-error limit near zero
measures a genuine missing *derivative direction*, not simply excessive
edit size. The table-response errors at `.01` tell the same story:
.28/.32 for the two single edits in paired coding and .38/.41 in dense
coding under ordinary rank three. Joint branch fitting at rank seven
reduces those to ≤.001 (paired) and about .01 (dense).

Reproduce with `shared_pair_branch_interventions.py` and
`shared_pair_branch_dose.py`. All per-seed, per-rank, per-condition, and
per-dose measurements are in their corresponding JSON files.
