# Legible edit-response groups inside the faithful joint bracket cut

The [joint reduction](JOINT_HEAD_MOMENT_FRONTIER_RESULTS.md) produced
a rank-four layer-13/head-8 cut that preserves the tested bracket path
while improving natural-text KL. Here we hold that **exact four-dimensional
subspace fixed** and ask which orientation makes its response to input
edits compact and interpretable. Every rotation retains the same
rank-four projector; the maximum projector Frobenius difference is
`3.1e-15`. On four fresh prompts, rotating the basis changes any
full-vocabulary logit by at most `6.7e-6` in float32.

We fit six orientations inside the cut: its original coordinates,
raw edit-response PCA, row-normalized edit PCA, edit-response L1,
fourth-moment/varimax, and sparsity of the output-projection weights.
The first four edit-derived orientations use 80 original
parenthesis/square input edits and **no answer labels**. Sixteen random
orthogonal rotations provide matched controls. Within *each*
orientation, we select the first one or two axes by raw training edit
energy; the same selection rule applies to every method. None of the
24 fresh test expressions or their answer margins enters fitting or
axis selection.

We transplant each test edit's selected head-state response into the
base prompt and measure the finite `)` versus `]` margin effect. The
first error column asks how well a small group reproduces the **complete
joint rank-four cut**. The second asks how well it reproduces the
**native full-head path**; it includes the cut's own roughly 2% error.

| Orientation | One-axis error vs joint cut ↓ | Two-axis error vs joint cut ↓ | Two-axis error vs native head ↓ |
|---|---:|---:|---:|
| Original coordinates | 39.30% | 19.51% | 20.39% |
| Raw edit PCA | 5.84% | **0.78%** | **2.22%** |
| Normalized edit PCA | 6.22% | 0.79% | 2.24% |
| Edit L1 | **5.41%** | 0.81% | 2.26% |
| Varimax | 6.18% | 0.81% | 2.27% |
| Output-weight L1 | 11.95% | 5.44% | 6.49% |
| Sixteen random rotations | median 38.93%, best 4.15% | median 15.34%, best 2.82% | — |

Raw edit PCA's two-axis error against the joint cut has a 95% bootstrap
interval of **[0.27%, 1.23%]** when resampling the six token-identity
families. It beats all sixteen sampled random rotations on that metric;
the conservative empirical rank is 1/17, so the control panel is too
small to claim a strong significance threshold. All 24 rank-one and
rank-two PCA paths have the same sign as the rank-four path. The
edit-PCA top axis has absolute cosine **0.99997** between independently
fitted 80-edit train and validation panels.

The top raw-PCA axis carries **99.15% of the bracket edit-response
energy** inside the cut, yet its one-axis finite path still misses the
cut by 5.84%. A second, low-energy direction reduces that error to
0.78%. This is a concrete reason to test functional interventions in
addition to variance explained. The normalized-PCA, edit-L1, and
varimax top axes align with raw PCA at absolute cosine above 0.9998;
their selected two-dimensional subspaces also have principal cosines
above 0.9994. These priors find essentially the same bracket response
group here, so this result does not favor a unique sparsity objective.
Weight L1 produces a different orientation and a weaker two-axis
finite-path approximation.

We also captured **800 independent input edits**, 80 in each of ten
designed types. Ranking them by the fitted PCA axis response puts
bracket edits in all top 80 slots. But ranking by the **full 128-dimensional
head response norm** or even the full joint rank-four norm also gets
80/80. The group compresses and stabilizes a known bracket-related
path; it does **not** discover a new task label beyond an obvious
head-level response signal. The head and the bracket edit family were
chosen using prior knowledge, and the validation uses the same opening
symbol distinction under different contexts.

The next test should challenge this group with a different competent
relation or use task-blind head selection and whole-edit-type holdouts.
Within this selected head, however, it is now meaningful to compare
other human-legibility priors at a **fixed, causally faithful reduction**
instead of optimizing coordinates inside a cut that has already lost
the bracket path.

Reproduce with `flagship_joint_bracket_legibility.py` and
`analyze_joint_bracket_legibility.py`. Rotations, raw edit responses,
per-case finite paths, selectivity scores, and bootstrap summaries are
in `full_width_results/joint_bracket_legibility*`.
