# A downstream fidelity gate selects a much wider head subspace

The previous [new-family test](NEW_FAMILY_PATH_TRANSFER_RESULTS.md) found
that a training rule based on the immediate `c_proj` output did not buy
transfer to unseen edit families. We therefore selected the reduction
using **finite final-logit head-patch paths** instead of the local output.

For L9H7, the training set contained four cases from each of the ten
familiar designed edit families. We compared ranks 32, 64, 96, and 112;
at each rank, we tried a weight-only `c_proj` Gram cut, pooled-edit PCA,
selected-family floors .1 and .5, and all-type-balanced edit PCA. The
candidate must have **mean relative path MSE at most .10 for every
training edit family** and mean ordinary base-prompt forward KL at most
**.001**. The rule chooses the smallest passing rank, then the smallest
worst-family path MSE. No validation or novel-family example was used to
choose the candidate.

| Rank | Best worst-family training path MSE among tested measures | Did any candidate meet both gates? |
|---:|---:|---|
| 32 | .439 (pooled) | No |
| 64 | .147 (all-type balanced) | No |
| 96 | .052 (pooled) | **Yes** |
| 112 | .022 (pooled) | Yes |

The rule selected **rank-96 pooled-edit PCA**. On training, its worst
family path MSE was .0517 and mean forward KL .000422. The familiar
type validation panel, using 12 independent cases per type, also passed:
worst family path MSE .0671 (random-context substitutions), mean forward
KL .000391. Rank 96 retains 75% of the 128-dimensional head. Under this
particular stringent fidelity criterion and tested spectral family, the
result is a modest compression rather than evidence of a small canonical
circuit. It does not provide a lower bound for other projectors.

The previously frozen four-family panel already contains the selected
rank-96 pooled cut. Its mean new-family path MSE is .0682 and ordinary
base-prompt KL .00075. Three family means pass .10; modal verbs reach
.1036, just above the gate. Twelve modal cases make this a weak marginal
miss, not decisive evidence against transfer. A second independent set
of six unseen edit families was therefore constructed without changing
the selected cut.

| Second unseen family | Mean finite path MSE | Mean native head-path cosine with whole edit effect |
|---|---:|---:|
| Quantifier (`all`/`some`) | .057 | .779 |
| Pronoun (`he`/`she`) | .056 | .589 |
| Arithmetic operator (`+`/`*`) | .082 | .475 |
| Temporal order (`before`/`after`) | .077 | .656 |
| Polarity (`yes`/`no`) | .101 | .522 |
| Conditional operator (`if`/`unless`) | .073 | .587 |

Mean ordinary forward KL on these 72 base prompts is .000565. The
polarity family misses the .10 path criterion by only .0013. Across both
unseen panels, eight of ten family means pass and two miss marginally;
the cut has useful transfer at this relatively high rank, but the strict
per-family gate is not proved for unseen types. The native head paths
have positive alignment with the whole input-edit effect on all six
new family averages, so these paths are not merely large orthogonal
perturbations. That alignment is still not a named answer-level task test.

The distinction between measurements remains important: path MSE says
how well a *particular head patch* is reproduced, forward KL says how
much ordinary predictions change, and alignment of the native head path
with the whole input-edit effect says whether that patch plausibly
explains the edit. None alone establishes task accuracy or interpretable
coordinates. The next stage-two prior comparison should use a fixed
rank-96 function and exact rotations, while a separate stage-one effort
searches for better projectors under the same broad fidelity gates.

Reproduce the training selection with `flagship_train_finite_fidelity_gate.py`
and independent familiar validation with `flagship_gate_familiar_holdout.py`.
The second new-family test is `flagship_gate_second_family_stress.py`.
Saved measurements are in `full_width_results/train_finite_fidelity_gate.json`,
`full_width_results/gate_familiar_holdout.json`, and
`full_width_results/gate_second_family_stress.json`.
