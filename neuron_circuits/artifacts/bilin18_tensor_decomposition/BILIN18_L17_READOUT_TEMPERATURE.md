

## H257 — Holding computation fixed exposes strong dependence on the readout metric

H256 was progress: frozen replication supported generic sensitivity plus additional
native error alignment. H257 isolates metric dependence experimentally. Keep all
compiled/native residual states, MLP writes, errors and vocabulary logits fixed.
Apply temperatures .5/1/2 AFTER softcap, logits/T, for both chart construction and
KL evaluation in a full3x3 cross. Original native model corresponds to T=1 only.
Charts retain top1024 tokens, ranks16/64, with error scales.1/1. Same128 H256
documents reused; all charts selected before native errors. Coefficients are oracle.
Exact full-vocabulary KL evaluates all settings, with each temperature's own
uncorrected baseline. Ratios are pooled across sites, bootstrapped by document.

| Fixed chart T | Scale | Evaluate T=.5 | Evaluate T=1 | Evaluate T=2 |
|---|---|---:|---:|---:|
|.5|.1|98.07%|80.50%|40.66%|
|1|.1|97.93%|83.30%|45.07%|
|2|.1|94.14%|82.33%|51.60%|
|.5|1|96.52%|77.87%|30.96%|
|1|1|96.73%|81.67%|36.22%|
|2|1|93.44%|80.76%|42.36%|

Table: exact KL error recovered by rank16 oracle correction. In EACH ROW the
correction vector and all internal computations are identical across evaluations.
For chartT=1, .5-minus2 recovery gap is52.86 percentage points at small scale,
paired document-bootstrap95%[48.61,57.15], and60.51 points at full scale
[56.19,65.20]. The locked >=15-point metric-dependence gate PASSES at both scales.
This comparison establishes a causal dependence of the reported recovery on the
chosen readout metric, not a causal effect of entropy alone on internal processing.

The chart-adaptation gate FAILS: at evaluationT=.5, rebuilding the chart atT=.5
changes recovery by−.20 points[−.59,.10] relative to chartT=1, rather than required
+2. At evaluationT=2 rebuilding improves6.15 points[5.07,7.34], satisfying that
half only. No requirement is retrospectively weakened. Rank64 and all3x3 outcomes
remain in JSON/rows; no winning temperature is selected.

**Truncation limitation.** Mean retained top1024 probability mass is99.996%,97.44%,
49.39% at chartT=.5/1/2; minima99.50%,68.29%,20.43%. Mean entropy .851/2.721/8.431
nats. The flatter chart is severely truncated, so its improvement or failure does
not characterize optimal full-Fisher charts. However this does not invalidate the
fixed-chartT=1 contrast: its correction vector is fixed and evaluation uses exact
full-vocabulary KL at every temperature. Entropy and distributional weighting
change together by design. Temperature does not change argmax token identities;
this test says nothing about improved top1 decisions or native model performance.

**Interpretation.** Compact behavioral interfaces are not intrinsic counts of MLP
features. The same missing internal computation looks nearly repaired or poorly
repaired under different output metrics. H254/H256 still support extra native
alignment beyond generic controls, but interpreting readout-derived directions as
fundamental units is unwarranted. Internal structure, an executable representation,
and downstream relevance must be investigated separately, then connected.

Six directly relevant tests pass, including exact temperature score gradients
against autograd at all3 temperatures and equality with the old native formula.
Full project suite not rerun. Plot generated and visually inspected. All H257 runs
complete; native weights and compact program unchanged.

**Next experiment: remove oracle coefficients.** Return to an internal-computation
question. Fit a predeclared affine residual predictor using only the1024 discovery
documents and normalized pre-MLP activations. Compare its full vector correction,
its projection into the originalT=1 local chart, constant residual prediction, and
oracle chart correction on evaluation documents. Use a fixed regularization grid
with discovery-only validation; report all candidates and freeze before test.
Count parameters/reads and explicitly retain costly chart construction. A positive
result would show missing relevant coefficients are predictable from available
activations; a failure would reject this predictor class, not all compact units.
This differs from earlier response-linear fits: target the executable program's
natural-state residual and score downstream behavior. Do not call the charted
version cheap or semantic. Broad research goal remains active.
