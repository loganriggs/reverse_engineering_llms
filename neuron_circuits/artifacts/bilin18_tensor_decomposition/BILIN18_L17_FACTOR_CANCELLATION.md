

## H325 — Retained and omitted native-factor contributions

Freeze the full H323 learned readout and H324 ranked/random subset refits. For each query, split the full fitted response P into retained S and omitted O=P−S. Compare S+O against the target, measure pooled output cosine(S,O), and measure how well the refit correction C=refit−S recovers O:1−sum||C−O||²/sum||O||². The latter compares to the full fitted predictor, not exact native truth. No new fitting or selection. Summary uses discovery documents896:1024; rows768:1024 also saved. This is reused diagnostic data.

|Subset|Target|Retained energy / target|Omitted energy / target|Component cosine|Frozen fidelity|Refit fidelity|Omitted recovery|
|---|---|---:|---:|---:|---:|---:|---:|
|ranked256|actual|0.102|0.889|0.003|9.83%|3.01%|-8.31%|
|ranked256|angle|0.099|0.896|0.004|9.62%|3.02%|-7.91%|
|ranked512|actual|0.179|0.807|0.009|17.34%|8.39%|-12.46%|
|ranked512|angle|0.174|0.817|0.008|16.89%|7.88%|-12.25%|
|ranked1024|actual|0.311|0.669|0.015|30.14%|14.13%|-27.14%|
|ranked1024|angle|0.305|0.680|0.013|29.41%|13.40%|-26.59%|
|ranked2048|actual|0.543|0.430|0.021|52.01%|33.50%|-51.71%|
|ranked2048|angle|0.537|0.445|0.016|50.65%|31.19%|-52.34%|
|random256|actual|0.053|0.937|0.007|5.12%|1.61%|-4.21%|
|random256|angle|0.054|0.940|0.007|5.22%|1.56%|-4.40%|
|random512|actual|0.107|0.879|0.011|10.43%|3.34%|-9.03%|
|random512|angle|0.108|0.883|0.010|10.40%|2.85%|-9.63%|
|random1024|actual|0.218|0.762|0.016|21.23%|9.52%|-17.66%|
|random1024|angle|0.219|0.764|0.016|20.96%|8.99%|-18.39%|
|random2048|actual|0.433|0.541|0.019|41.65%|23.90%|-38.32%|
|random2048|angle|0.436|0.545|0.017|41.03%|21.46%|-42.37%|

Locked gates: ranked1024_cancellation: FAIL, ranked1024_omitted_recovery50: FAIL. Cancellation gate requires cosine<=−0.1 for ranked1024 in both families; recovery gate requires>=50% omitted recovery for ranked1024 in both.

**Finding.** At ranked1024, retained/omitted cosine is+0.0148 actual/+0.0132 angle, inconsistent with strong pooled cancellation under this split. Omitted energy is0.669/0.680 times target energy. Keeping full-model output rows yields30.14%/29.41% fidelity; refitting drops this to14.13%/13.40%. At2048, inherited weights yield52.01%/50.65% versus33.50%/31.19% refitted. Thus H324 conflates removing useful factors with fitting poorer weights to the reduced feature set. This changes the next experiment: compare inherited-weight subsets and native Down subsets before concluding that all subset deficits are representational.

**Interpretation limits.** This diagnoses the fitted shared interaction predictor. It is not an ablation of the native hidden layer, semantic identification, or proof of irreducible information. Low recovery may reflect finite reference coverage, regularization or subset selection. A negative cosine shows cancellation but does not alone show cancellation is the main cause of subset failure. A refit can improve target fidelity while failing to reproduce the omitted fitted component because the full predictor itself has error.

**Audit.** Exact energy identity ||S+O||²=||S||²+||O||²+2S·O and independent replay of every H324 refit error passed. {'max_energy_identity_relative': 2.7301199097679716e-16, 'max_refit_error_replay_relative': 2.0880975144445446e-16}. Frozen full and subset models plus response hashes verified. Native weights unchanged.

**Next discriminator.** First compare inherited-weight subsets, native Down subsets and refit sample scaling; then test learned mixtures of native-factor features against individual-factor subsets. This distinguishes a low-dimensional distributed representation from a genuinely broad response map; count feature-computation storage as well as readout parameters. Confirm selected models on fresh documents before semantic interpretation. Broad goal active.
