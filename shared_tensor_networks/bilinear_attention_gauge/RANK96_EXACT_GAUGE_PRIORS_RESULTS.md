# Exact gauges of a fixed, downstream-gated head reduction

The [finite-path gate](FINITE_PATH_GATE_RESULTS.md) selected rank-96
pooled-edit PCA for L9H7. This experiment holds that reduced function
fixed and changes only its coordinates. It therefore separates the
stage-one question, *which head-state directions survive?*, from the
stage-two question, *which axes make the surviving function legible?*

Let $U\in\mathbb{R}^{128\times96}$ be the frozen orthonormal cut, $W$
the head's learned `c_proj` block, and $G=U^\top W^\top WU$. Define

$$
A(x)=xU G^{1/2},\qquad D=WU G^{-1/2}.
$$

Then $D^\top D=I$ and the reduced immediate output is
$A(x)D^\top=xUU^\top W^\top$. For **any orthogonal** 96×96 rotation
$R$, $(A(x)R)(DR)^\top=A(x)D^\top$. The output-isometric coordinates
make each decoder axis have unit immediate-output norm. The measured
relative reconstruction error for every gauge was about $10^{-15}$
on held-out native edit responses; the script also verifies the formula
against direct `c_proj` projection. Thus all prior stage-one final-logit
path and KL results are exactly shared by these gauges.

We fitted four coordinate priors using the old 800-edit **training**
panel, plus the output weights alone: output-weight varimax (sparse
decoder entries), balanced-edit activation varimax (sparse edit
coordinates, no edit-type labels), a fixed-code edit-type covariance
signature (uses the ten designed type labels), and pooled-edit PCA.
Identity and 32 random orthogonal rotations are controls. The known
types are evaluated on the independent 800-edit validation panel;
the ten whole-new-family types from two earlier panels are evaluated
without fitting on their 120 examples.

| Gauge | Decoder mean absolute entry ↓ | Familiar validation effective axes ↓ | Familiar type purity ↑ | Unseen-family effective axes ↓ | Unseen-family type purity ↑ |
|---|---:|---:|---:|---:|---:|
| Weight varimax | **.02228** | 27.13 | .198 | 27.57 | .270 |
| Edit-activation varimax | .02349 | **13.70** | **.419** | 22.16 | **.299** |
| Edit-type signature | .02345 | 17.62 | .283 | 24.99 | .273 |
| Pooled-edit PCA | .02352 | 13.86 | .272 | **18.77** | .279 |
| Identity | .02347 | 14.85 | .275 | 21.77 | .282 |
| Random mean (32) | .02350 | 33.37 | .199 | 33.40 | .272 |

Here **effective axes** is the mean inverse-Simpson participation of
each edit's squared output-isometric axis contributions; lower means
each edit uses fewer coordinates. **Type purity** is the energy-weighted
maximum share of an axis's variance attributable to one edit family
**after normalizing each family's total response energy to one**;
higher means more type-selective. This prevents a high-energy family
from winning merely by scale. These are
structural proxies, not scores of semantic correctness or task accuracy.
The exact decimal values depend on the chosen edit ensembles and output
metric. The type-signature gauge is explicitly supervised by the ten
designed family labels, though it uses no answer labels.

The priors disagree even though the reduced function is identical.
Weight varimax makes $D$ about 5% sparser by mean absolute entry than
random controls, but spreads familiar edit effects across roughly twice
as many axes as edit varimax. Edit varimax sharply concentrates familiar
edits and improves familiar type purity, yet PCA concentrates the
**unseen** edits more (18.77 effective axes versus 22.16). Edit-varimax's
unseen type purity improves only modestly over random (.299 versus .272),
so familiar-label separation is not evidence of novel-task discovery.

Independent refits reveal only moderate axis identity:

| Gauge | Train-half axis match | Train vs. independent validation axis match | Train vs. validation top-20-energy axis match |
|---|---:|---:|---:|
| Edit varimax | .383 | .443 | .618 |
| Edit-type signature | .346 | .370 | .510 |
| Pooled-edit PCA | .375 | .422 | .615 |
| Independent random rotations, mean | .269 | .269 | .281 |

Matching uses optimal one-to-one assignment of absolute axis cosines;
top-20 selects axes by training edit energy and matches each to any
validation axis. The learned gauges beat random axes, but edit varimax
does not materially stabilize individual axes beyond PCA. At rank 96,
it would be premature to name these axes as a reproducible ontology.

**Decision.** Exact functional gauge fixing can change sparsity and
apparent type structure substantially without changing any model
prediction. No tested single prior dominates across weights, familiar
edits, unseen edits, and axis stability. The next experiment should
evaluate *causal group* or reusable-subspace claims at fixed function:
freeze axes/groups using training edits, then test independent new
families and named behaviors with matched deletion dose. The stage-one
search for a smaller faithful cut remains separate.

Reproduce with `flagship_rank96_exact_gauge_priors.py` and
`flagship_rank96_gauge_stability.py`. Saved metrics and exact rotations
are in `full_width_results/rank96_exact_gauge_priors.json`,
`full_width_results/rank96_exact_gauge_rotations.npz`, and
`full_width_results/rank96_gauge_stability.json`.
