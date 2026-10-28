

## H281 — More adapter supervision helps modestly, without resolving the remaining error

The fixed 64-coordinate linear adapter was fit with nested prefixes of 192, 384 and 768 discovery documents. Feature geometry, normalization and the base program are held exactly fixed. Those frozen components already used 768 documents and native supervision: this measures marginal adapter supervision, not total data exposure. All fits use the same next256 validation documents, five ridge choices, eight KL runs and 400-update budget. The 768 models are frozen H280 fits; new 192/384 fits use the same procedure. This is one nested sequence, not replicated subsampling. Equal updates imply different passes through the datasets.

A fresh128-document panel excludes2496 prior unique complete-document hashes. Full source-document and window hashes, window contents, uniqueness, exclusions and saved-file SHA256 were independently checked. It is not a model-training holdout. Evaluation asserts identical feature normalization across counts. Native-head replay relative error is4.30e-7. Three focused adapter/ridge/KL-gradient tests passed.

| Adapter documents | MSE fresh KL | KL-selected fresh KL | KL train | KL validation | Selected KL step |
|---|---:|---:|---:|---:|---:|
|192|0.056614|0.056614|0.043492|0.054608|0|
|384|0.055541|0.055201|0.028647|0.052553|200|
|768|0.054893|0.054665|0.034373|0.052036|300|

All MSE fits select ridge1. At192 the KL selection retains the MSE initialization (step0); at384/768 it selects MSE initialization, rate.3, penalty.1, steps200/300. Thus the192 KL result is not evidence of a successful KL update.

- mse_192_768: relative KL reduction 3.04%, paired document-bootstrap95% [0.97, 5.62]%.
- kl_192_768: relative KL reduction 3.44%, paired document-bootstrap95% [-0.27, 7.01]%.
- objective_gain: relative KL reduction 0.42%, paired document-bootstrap95% [-2.77, 3.42]%.

All four preregistered gates fail. Both learning gains are below5%; the KL interval also includes zero. The objective gain misses10%. The768 KL adapter recovers43.07% of original compiler KL error, with CE increase.05324 nats and native top1 agreement89.26%, missing repair thresholds80%/.02/98%. Thresholds were not relaxed. Bootstrap intervals condition on these fitted models and do not include training-sample or optimizer variability; exploratory adjacent-count contrasts are retained in JSON without multiplicity-adjusted claims.

**What this changes.** There is a small positive MSE learning effect on this fresh panel. A dominant data-bottleneck account is less compelling than after H280, but has not been ruled out: counts are limited, fixed bases already used the larger set, the optimizer is not certified converged, and one nested sequence cannot characterize sampling variance. KL training gains still generalize poorly. Neither the small gains nor their apparent flattening establish an asymptotic capacity ceiling. No claim about semantic units follows.

**Next information-gaining experiment.** Compare three matched-size added feature families on the same frozen base: (A) the existing64 linear coordinates, (B)64 quadratic combinations within that same read span, and (C)64 linear directions outside the span. Fix feature construction using discovery data only, use the same768/256 train/validation split, coefficient count, MSE/KL objective choices and update budget, and freeze selections before a new test. Include matched shuffled-feature controls as diagnostics. A B advantage supports missing interactions within current coordinates; a C advantage supports missing directions; similar training improvements without held-out gains points back toward estimation. If all fail, distinguish failure to fit from lack of improvement before drawing representational conclusions. Predeclare contrasts and practical thresholds in the next config before collecting outcomes; do not promote an adapter without measuring source-response transfer. This is a proposed discriminator, not an experiment already run.

Artifacts: bilin18_l17_readout_learning.json/_rows.npz/.log; per-count selection, frozen matrices and logs under bilin18_l17_readout_n192, _n384 and bilin18_l17_fixed_readout. Original weights/compiler/base unchanged. All jobs terminal. Broad research goal remains active.
