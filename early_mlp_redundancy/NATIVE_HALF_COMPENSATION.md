# M41: native half compensation distinguishes averaging from deletion tolerance

The expanded objective asks for a conceptual explanation backed by theory and empirics. This test returns directly to native MLP0–2 channels. It is separate from the featurizer group-gate assay, which remains pending.

## Theory and discriminating prediction

Let channel writes be c_i, net write y=sum_i c_i, channel energy C=sum_i ||c_i||², and net energy S=||y||². The bias is excluded from these quantities and always retained once in the actual intervention. For a uniformly sampled half K of N channels, the fixed estimator y_hat=2 sum_{i in K} c_i is unbiased over masks. Its expected squared error is

    E||y_hat-y||² = (N C - S)/(N-1).

The expected error of unscaled half deletion is one quarter of this quantity plus S/4. Therefore compensation reduces expected local squared error iff

    C/S < (N+2)/(3N), approximately 1/3.

This criterion is stronger than net constructive interference (C/S<1). A layer can be distributed and net constructive while simple averaging compensation makes deletion worse. Duplicate aligned contributions allow exact compensation; orthogonal private contributions do not; coordinated cancelling components can make compensation especially damaging. Exhaustive enumeration of all20 half-masks for four six-channel toy constructions verified the expectation before native outcomes were inspected. This is an algebraic result under fixed contributions, not a theory of why training chose them.

Previously measured task-panel C/S values were approximately .229/.403/.520 for MLP0/1/2. They predict local improvement for MLP0 but deterioration for MLP1/2 in expectation over masks. No output-KL theorem follows: downstream sensitivity can change this ordering.

## Fixed experiment

Reuse both complementary half masks from D1, all160 original task endpoints, and32 previously used natural evaluation documents. Compare retained contribution at scale1 versus scale2, with all downstream computation live and biases unchanged. No scale fitting, search, selected channels, or additional task-holdout exposure. Each input has three layers × two masks × two scales. The two complementary masks are dependent; they do not provide independent uncertainty estimates.

## Results

The table averages output KL over inputs and the two masks. Local error is pooled squared product-write error divided by pooled native product-write energy, excluding bias. Task evaluation is the existing source-separated development split, not a pristine whole-task holdout.

| Cohort | Layer | Raw half KL | Doubled half KL | Raw local error | Doubled local error | Mask-expectation prediction |
|---|---|---:|---:|---:|---:|---:|
| task train | MLP0 | 0.04038 | 0.03124 | 0.3065 | 0.2260 | 0.2297 |
| task train | MLP1 | 0.03689 | 0.04842 | 0.3498 | 0.3993 | 0.4037 |
| task train | MLP2 | 0.02303 | 0.02538 | 0.3798 | 0.5190 | 0.5234 |
| task evaluation | MLP0 | 0.03215 | 0.02509 | 0.3058 | 0.2233 | 0.2260 |
| task evaluation | MLP1 | 0.03738 | 0.04266 | 0.3492 | 0.3970 | 0.3989 |
| task evaluation | MLP2 | 0.02350 | 0.02241 | 0.3763 | 0.5051 | 0.5051 |
| natural evaluation | MLP0 | 0.02885 | 0.02589 | 0.3054 | 0.2215 | 0.2213 |
| natural evaluation | MLP1 | 0.03282 | 0.03714 | 0.3488 | 0.3954 | 0.3995 |
| natural evaluation | MLP2 | 0.02264 | 0.02714 | 0.3723 | 0.4890 | 0.4881 |

MLP0 improves locally and in KL across all three cohorts. MLP1 deteriorates on both across all three. MLP2 deteriorates locally throughout; KL worsens on task training and natural text, but improves slightly on source-separated task evaluation. These aggregate patterns do not claim every example or family behaves identically.

## Conceptual update

The current working account is distributed partially aligned contributions, with differing amounts of common versus complementary information. MLP0 is more compatible with a simple averaging correction than MLP1/2 under this assay. MLP1 deletion tolerance is not adequately described as two identical interchangeable halves: doubling one surviving half does not repair its output. This rejects that simple sufficient model, not all learned substitutions or redundancies.

Failure of this correction also does not identify cancellation: MLP1/2 can fail it while remaining net constructive. MLP2’s mixed output ranking despite uniformly worse local error exposes a separate reader-sensitivity axis. Neither result proves active downstream compensation. The resemblance to the mask expectation is descriptive and partly uses measured native energies; it does not establish an optimizer-level explanation or a semantic feature ontology.

## Scope, audit, and next question

2,496forwards and2,304intervention rows. Independent audit checked complete inventories/summaries,12sampled NumPy decodes/output metrics, complementary-half local-error identity, and exact replay of all960 original task half-deletion rows. Maximum sampled decoder relative discrepancy2.49e-7; output metric discrepancy1.19e-8. No new model training.

Stop scalar/mask tuning. The next native substitution test should ask whether a training-fitted low-complexity mapping between halves transfers to other inputs better than a mean-only baseline; it must evaluate local replacement and downstream causal consequences separately. Such a mapping is an oracle diagnostic unless executable from retained activations, and decodability alone is insufficient. Keep the featurizer shared/exclusive intervention question explicit rather than treating architectural overlap as the answer.

[Protocol](validation/native_half_compensation_protocol.json) · [Theory check](validation/native_half_theory_check.json) · [Results](validation/native_half_compensation.json) · [Audit](validation/native_half_compensation_audit.json)
