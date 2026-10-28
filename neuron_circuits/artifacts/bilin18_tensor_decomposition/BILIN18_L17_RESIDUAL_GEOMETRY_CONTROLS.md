

## H254 — Generic readout geometry explains much, with extra native-error alignment

Previous turn H253 made progress: new oracle correction evidence and updated
records. H254 tests whether those local charts identify MLP-specific computational
organization or mostly reflect generic downstream sensitivity.

Freeze H253 compiled states and top256-probability rank16/64 local charts. Compare
actual native-minus-compiled residual, three independent isotropic Gaussian draws,
and three document-shifted residual controls. Each control is rescaled to the
native residual norm at EACH site. Shuffling preserves token position and uses a
fixed random nonzero cyclic document offset, not a uniform arbitrary permutation.
Compare scales .1 and1. Synthetic target distributions are head(state+scaled control
error), not native predictions. Measure exact KL(target || corrected distribution),
normalized by each family's own uncorrected KL. Corrections are oracle projections;
charts precede native-error computation. Wrong-context charts shift one document.
No true next-token labels enter this experiment. Same128 documents/512 sites reused.

| Scale | Error family | Local16 KL recovered | Wrong16 KL recovered | Local64 KL recovered |
|---|---|---:|---:|---:|
|.1|Native|81.4%|7.3%|90.2%|
|.1|Isotropic|66.9%|9.5%|81.7%|
|.1|Shuffled residual|73.1%|15.0%|85.3%|
|1|Native|78.8%|7.1%|88.7%|
|1|Isotropic|61.7%|9.2%|77.9%|
|1|Shuffled residual|67.4%|14.0%|81.3%|

Control summaries pool KL numerator/denominator across the three seeds before
forming recovery (ratio of sums), not arithmetic averages of seed recovery ratios.
The config's shorthand 'seed-averaged recovery' means this pooling; implementation
and reported values retain that explicit distinction. Bootstrap2000 resamples
whole documents jointly across families/seeds, conditional on these three draws.

At full scale native-minus-isotropic recovery is17.12 percentage points, paired
95% interval[14.14,19.86]; native-minus-shuffled11.45[8.19,14.45]. At .1 scale the
gaps are14.47[11.92,16.97] and8.32[5.86,10.51]. Native alignment advantage is
nonzero on this panel. The locked native-specialness gate nevertheless FAILS:
required >=10 points against both controls at BOTH scales; the small-scale
shuffled gap is8.32 points. Generic-sufficiency gate PASSES: isotropic local16
recovery >=60% and matched-minus-wrong >=40 points at both scales. The full-scale
isotropic recovery interval[58.40,64.92]% straddles60%; the gate was a point-estimate
criterion, not a confidence-bound criterion. Neither gate is an ontological proof.

**Scientific update.** Much of H253's local-chart advantage occurs even for isotropic
errors, supporting generic probability-weighted readout geometry. Native residuals
also align more strongly with these directions than equal-norm shuffled residuals,
supporting an additional context/error relationship. These contributions coexist;
the percentages are not an additive causal decomposition. Random/shuffled errors
may be off the natural manifold; the .1-scale result limits, but does not eliminate,
that concern. Local sufficiency does not identify MLP semantic features or predict
missing coefficients. Prior32 tests plus H253/H254 gives34-test inventory; four
directly relevant tests passed in this run, not the entire suite.

**Next information-gain experiment.** Freeze top64/256/1024 probability-support
variants at ranks16/64 and evaluate recovery by fixed compiled-entropy strata.
If gains collapse at high entropy or vary strongly with support, a sparse likely-
token readout account strengthens. Stable gains would support broader conditional
sensitivity structure. Then replicate the frozen comparison on distinct documents
before investing in coefficient prediction. Do not tune a winning rank/support on
this reused evaluation panel. No model weights or compiled program changed.
All H254 runs complete; research goal remains active.
