

## H327 — Fresh native-factor confirmation

New128 documents exclude3904 complete-document hashes across all listed prior panels. Source documents and windows independently reconstructed and verified. This is experimental-panel freshness, not a model-training holdout or new corpus. Freeze H323 full native-factor learned and rotated-factor models, H324 ranked1024/2048 subsets and refits, and native Down weights. No fitting, penalty selection or subset selection on this panel. Evaluate all9 heads with actual deletion and new matched-angle controls. Target remains the missing residual cross response.

|Model|Actual fidelity [95%]|Angle fidelity [95%]|
|---|---:|---:|
|full_native|86.97% [85.52,88.20]|87.91% [86.24,89.46]|
|full_inherited|90.90% [90.62,91.16]|90.53% [90.26,90.77]|
|full_rotated|0.69% [0.64,0.75]|0.57% [0.50,0.64]|
|native1024|29.47% [29.11,29.86]|28.61% [28.14,29.05]|
|inherited1024|30.16% [29.84,30.51]|29.33% [29.05,29.62]|
|refit1024|14.25% [14.03,14.47]|13.40% [13.18,13.61]|
|native2048|52.02% [51.57,52.45]|51.11% [50.59,51.60]|
|inherited2048|52.01% [51.68,52.34]|50.72% [50.47,50.98]|
|refit2048|33.54% [33.25,33.82]|31.33% [31.07,31.60]|

|Paired contrast|Target|Gain (pp),95% interval|
|---|---|---:|
|native_alignment|actual|90.21 [89.91,90.49]|
|inherited1024_over_refit|actual|15.91 [15.67,16.16]|
|full_learned_over_native|actual|3.93 [2.65,5.38]|
|native_alignment|angle|89.96 [89.67,90.23]|
|inherited1024_over_refit|angle|15.93 [15.73,16.14]|
|full_learned_over_native|angle|2.62 [1.12,4.28]|

Locked gates: full85: PASS, alignment80: PASS, inherited_refit10: PASS. full85 requires>=85% full inherited fidelity in both families; alignment80 requires>=80pp native-factor over rotated gain with positive interval; inherited_refit10 requires>=10pp inherited1024 over refit1024 with positive interval. Bootstrap resamples128 documents, conditioning on frozen models and one new angle realization.

**Meaning and limitations.** This tests transfer of a shared response computation to new experimental documents. It does not identify semantic factors, prove sparse irreducibility, or validate intact replacement. Native Down is a known-weight witness omitting the compiler correction. The full fitted model uses roughly native-MLP storage; subsets additionally require shared geometry and the baseline program for complete predictions. One rotated control was tested.

**Audit.** Frozen models, geometry and corpus hashes passed. Direct subset feature calculation agrees with slicing full features to8.31e-14 relative error. Targets use native float32 cross accounting in256-site batches. Document provenance, exact predictions' errors, intervals and gates saved. Native weights unchanged.

**Next discriminator.** Investigate whether response-important information is low-dimensional in mixtures of native factors despite being spread across individual factors. Compare data-weighted output modes and factor subsets at explicit storage costs; then test downstream behavior and semantic selectivity separately. Broad research goal active.
