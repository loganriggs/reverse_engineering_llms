## H897 — Output rank helps when discovery covers contextual variation

The previous turn made progress: H896 completed all ten frozen-unit natural evaluations and found limited, heterogeneous fidelity beyond removal. H897 separates input-axis variation from output fitting. The original A number axis and coordinate origin remain fixed. Only the affine approximation of its contextual cross-write C(w) changes.

Fit output PCA using task-family regular nouns: four pairs (teacher, worker, student, farmer) or eight pairs (adding artist, driver, doctor, player), in template A, templates A/B, or all A–F. Evaluate all fits/ranks on disjoint seven irregular noun pairs across six templates and task/copy families:168 contexts. These are existing reused diagnostic contexts, not new confirmation. The full six-template condition shares evaluation templates but never validation nouns; it tests lexical transfer with template coverage, not unseen-template generalization. More templates also increase sample count, so the factorial alone does not isolate a pure coverage effect. The equal16-endpoint comparison below partially addresses that confound.

Ranks0,1,2,4,7 and a seeded random rank7 control are all retained. Rank0 is a fitted mean, not removal. The denominator for every error is the exact component's energy, equivalent to the error of removing it. Reported numbers are sqrt(sum squared reconstruction errors / sum exact squared cross-write norms). Values below1 beat removal locally. They are not whole-model KL, downstream causal fidelity, or per-example error averages.

### Native local cross-write relative RMS error

|Discovery|Endpoints|Agreement rank1|Agreement rank7|Copy rank1|Copy rank7|
|---|---:|---:|---:|---:|---:|
|four_A|8|0.566|0.554|0.882|0.866|
|four_AB|16|0.570|0.476|0.835|0.808|
|four_ABCDEF|48|0.460|0.242|0.802|0.770|
|eight_A|16|0.554|0.538|0.875|0.867|
|eight_AB|32|0.559|0.463|0.827|0.795|
|eight_ABCDEF|96|0.449|0.227|0.793|0.741|

With eight nouns and only A, rank1→7 gives0.554→0.538, a small gain. With all six templates, the same rank increase gives0.449→0.227. Four nouns show the same pattern (A0.566→0.554; six templates0.460→0.242). Additional dimensions learned from narrow discovery do not contain the missing contextual variation; broader coverage makes them useful. Doubling noun pairs at fixed template coverage produces much smaller changes in this panel.

At equal16 discovery endpoints, four nouns across A/B yield rank7 error0.476, versus0.538 for eight nouns in A. However, rank1 is slightly worse with A/B (0.570 versus0.554). Coverage and capacity interact; broader data is not uniformly beneficial at rank1. This matched-count comparison changes lexical identity and template distribution together and is not a randomized pure-template estimate.

The six-template eight-noun agreement fit has mean-only error0.526 and random7 error0.524, versus learned rank7 error0.227. The result therefore requires learned output directions rather than merely seven arbitrary dimensions. The same best-covered condition still has copy error0.741 at rank7, limiting a universal-computation claim. Copy formats repeat across some A–F labels, so84 copy rows are not84 independent prompts.

### Competence and intervention-vector checks

The all-template agreement summary includes templates that failed the earlier full-vocabulary task gate. Restricting descriptively to the previously competent irregular B/C cells preserves the pattern: eight-noun A-only rank1/rank7 errors0.513/0.499, versus six-template0.426/0.223. Those28 endpoints provide competent-task geometry evidence but remain reused diagnostic data.

Using the opposite-number-centroid intervention displacement instead of the native number coordinate gives similar local errors: eight-noun A rank1/rank7 is0.551/0.535; six-template is0.452/0.231. This assesses the intervention write vector before downstream propagation. No live intervention was executed here, so improved behavioral fidelity is still unproven.

### Interpretation and next test

The strongest current explanation is that the fixed number direction participates in multiple context-dependent output directions, and a narrow discovery panel cannot learn them simply by adding rank. This is evidence against treating the original rank1 contextual write as a broadly sufficient unit. It does not prove a seven-dimensional manifold: output rank is not intrinsic input dimension, template coverage remains limited, and copying generalization is weak.

Next run live native replacement and number-coordinate interventions for the frozen output fits, comparing narrow/broad coverage at rank1 and7, plus the equal-count A-eight versus AB-four contrast. Use synthetic nouns/templates absent from these output fits, retain task competence failures separately, and compare with exact and removal controls. This will test whether local geometric gains survive downstream computation. Do not select a best fit using the previously evaluated384 natural documents or present reused panels as fresh confirmation. Broader shared-dictionary/BSF manifold and computation objectives remain active.

### Verification

All discovery/validation noun sets disjoint, fixed-unit/source/model hashes unchanged. Output basis orthonormality error2.51e−14; compiled affine encoders match explicit cross-term projections to relative1.06e−15. All local metrics finite. Zero full-model forwards; cached native activations and fixed model weights used. No natural or reserved test evaluation; document pool remains26,112. Fits, full rank curves, and per-example errors saved.
