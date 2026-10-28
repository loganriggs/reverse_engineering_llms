

## H348 — Finite behavior versus local sensitivity

Reuse H346 head6 recipients and exact matched payloads. At intact final residual state s, compare KL(p(s) || p(s+αΔ)) for four paths: compiled response; full native source intervention (−payload + native MLP response); direct −payload; and remaining native MLP response minus compiled response. Compiler/full paths use α=0.25 and1, others1. Quarter strength interpolates the final-state delta; it does not rerun a quarter-strength source intervention. Compiler-only and other isolated paths are diagnostics, not native replacements.

|Path / scale|Actual mean KL|Moment mean KL|Donor mean KL|Moment/actual [95% interval]|Donor/actual [95% interval]|
|---|---:|---:|---:|---:|---:|
|compiled_0.25|8.50643e-05|0.000324983|0.000380549|3.820 [2.878, 5.125]|4.474 [3.298, 5.911]|
|compiled_1.0|0.00133398|0.00492942|0.00566851|3.695 [2.804, 4.887]|4.249 [3.200, 5.546]|
|full_0.25|0.000182543|0.000339095|0.000356921|1.858 [1.609, 2.154]|1.955 [1.639, 2.325]|
|full_1.0|0.00264848|0.00539064|0.00564879|2.035 [1.748, 2.390]|2.133 [1.759, 2.585]|
|direct_1.0|0.00248126|0.00553517|0.00563848|2.231 [1.986, 2.497]|2.272 [2.017, 2.546]|
|remaining_1.0|0.000425593|0.000899935|0.000904856|2.115 [1.767, 2.537]|2.126 [1.787, 2.525]|

Compiler finite/local Fisher ratios (actual, moment, donor) are 0.972, 0.938, 0.915 at full strength and 0.992, 0.990, 0.983 at quarter strength. Local prediction is half the Fisher energy of the full finite compiler delta, not just its input tangent. Finite output nonlinearities modestly reduce the predicted KL; they do not remove the large compiler-path control gap.

Locked gates: compiled_finite2: PASS, full_finite2: FAIL. Both control-family lower95% bounds must exceed2. The full-native ratio is smaller than the compiler ratio. Ratios alone do not quantify cancellation: direct and remaining paths also change under controls, and finite KL is nonadditive.

Uncertainty uses4000 joint recipient-document resamples and global realization resamples, conditional on frozen compiler, six draws and fixed donor assignments. Intervals are pointwise; donor dependence is not reconstructed. Documents are reused from H346. This tests output distribution change, not whether predictions become more correct or which semantic content changes.

Audit: cached H346 coefficient-output replay error 7.73e-07; payload norm error 6.48e-07. Per-site KL and top1 agreement saved; all computed KL values nonnegative within1e-10. Native weights unchanged.

Next: identify which token-probability changes carry the compiler-path sensitivity and whether those changes are explained by generic output confidence, RMS normalization, or specific aligned output directions. Avoid assigning semantic units from KL magnitude alone. Broad research goal remains active.
