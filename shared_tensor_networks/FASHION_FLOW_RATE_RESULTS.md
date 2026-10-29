# Training-only initial-loss-descent calibration

Completed18new fits: corrected tied and free occurrence SGD, depths2/3/4, three seeds. Reuses the exact prior55k/5k split, width16, RMSNorm, float64 initialization and215minibatch order. Ordinary tied SGD baseline is reused from the audited prior run with result hashes. No clipping/momentum/decay. This is a one-epoch dynamics control, not a convergence claim; official test untouched.

For each arm choose lr=.01*E_ordinary/E_arm using only the initial first256training examples. E is the gradient-flow loss-decrease energy for the relevant metric. Freeze rates thereafter. This equalizes predicted infinitesimal loss decrease on the calibration probe, not output velocity, finite-step decrease, later dynamics, or parameter displacement. The next three disjoint256example training probes check calibration transfer but do not choose rates. No validation tuning.

| Depth | Ordinary tied, base .01 | Corrected, base .01 | Corrected, calibrated | Free, base .01 | Free, calibrated |
|---|---:|---:|---:|---:|---:|
| 2 | 76.46% | 73.70% | 77.50% | 78.38% | 78.64% |
| 3 | 76.70% | 67.02% | 77.42% | 76.52% | 76.20% |
| 4 | 75.02% | 60.58% | 76.88% | 75.06% | 74.56% |

Entries are median validation accuracy over3seeds at215steps. Corrected learning-rate multipliers are approximately1.97–1.98 atdepth2,3.78–3.85 atdepth3,7.46–7.58 atdepth4. Free multipliers vary0.60–1.26.

On independent training probes, calibrated corrected predicted descent divided by ordinary descent spans 0.965–1.010; exact matching holds only on the calibration probe.

On independent training probes, calibrated free predicted descent divided by ordinary descent spans 0.659–1.376; exact matching holds only on the calibration probe.

The earlier corrected-SGD slowdown is substantially removed by this scalar calibration. Atdepth4 its median improves from60.58% to76.88%, while calibrated free-tree median is74.56%. These small one-epoch samples do not establish that sharing wins; they establish that the fixed-base-rate comparison was highly sensitive to metric-induced scale. Differences can remain because the metric is anisotropic, calibration uses one probe, trajectories diverge, and these are finite SGD steps. Large initial disagreement energy alone did not predict a robust free-tree accuracy advantage under this control.

All18final models replay independently in NumPy for validation CE and accuracy. Calibration arithmetic, baseline result hashes and source snapshot are checked. Initial energies independently match the prior gradient control on the same probe; that prior control checked loss directional derivatives by finite differences. All fits finite, no failures removed. GenericCPU+pinnedcore0 runtime completed without crash; no assertion that earlier runtime problem is fixed.

Next: stop treating this one-epoch control as an accuracy tournament. Retain multiplicity-aware optimization in future comparisons; use multiple calibration probes or independently specified rate panels if drawing optimization conclusions. Return to fixed-degree reuse-family learning experiments and full-data seed/sample-size scaling. A parallel exact repeated-squaring certificate and scalar likelihood/flow analysis is in REUSE_COMPLEXITY_AND_FLATNESS.md.
