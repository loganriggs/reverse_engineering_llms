

## H328 — Distributed mixtures of native-factor outputs

Freeze H323 full learned map P=phi C. Learn output eigenvectors V from uncentered second moments of6144 reference predictions, either pooled or with each head/payload group normalized to unit trace. Approximate P by phi(C V_r)V_r.T. This is an executable linear mixture of all native-factor features, equivalent to projecting the fitted output into a reference-learned subspace. Compare nested random output directions. No new label fitting or evaluation selection. The frozen C inherited H323 calibration selection. Evaluate reused discovery896:1024; H327 fresh panel is not used here.

|Mixtures|Basis|Actual fidelity|Angle fidelity|Total executable scalars incl. geometry|
|---:|---|---:|---:|---:|
|16|pooled|1.36%|1.26%|10,894,466|
|32|pooled|3.23%|3.10%|10,986,626|
|64|pooled|7.49%|7.26%|11,170,946|
|128|pooled|16.02%|15.48%|11,539,586|
|256|pooled|31.36%|30.37%|12,276,866|
|512|pooled|55.65%|54.31%|13,751,426|
|1024|pooled|86.29%|85.34%|16,700,546|
|1152|pooled|90.69%|89.97%|17,437,826|
|16|equalgroup|1.45%|1.18%|10,894,466|
|32|equalgroup|3.49%|3.18%|10,986,626|
|64|equalgroup|7.74%|7.30%|11,170,946|
|128|equalgroup|16.59%|15.79%|11,539,586|
|256|equalgroup|32.17%|30.94%|12,276,866|
|512|equalgroup|56.52%|55.26%|13,751,426|
|1024|equalgroup|86.60%|85.68%|16,700,546|
|1152|equalgroup|90.69%|89.97%|17,437,826|
|16|random|1.22%|1.21%|10,894,466|
|32|random|2.48%|2.47%|10,986,626|
|64|random|5.03%|4.99%|11,170,946|
|128|random|10.05%|9.94%|11,539,586|
|256|random|20.08%|20.01%|12,276,866|
|512|random|40.23%|39.84%|13,751,426|
|1024|random|80.60%|79.96%|16,700,546|
|1152|random|90.69%|89.97%|17,437,826|

Locked gates: pooled256_80: FAIL, pooled256_gain10: PASS. pooled256_80 requires>=80% fidelity in both families; pooled256_gain10 requires>=10pp over random256 in both.

**Storage and computation.** All4608 bilinear factors must still be computed. Storage includes10,616,832 input-factor scalars,4608r mixture coefficients,1152r output basis entries, and185474 geometry/normalization scalars. At full width the factorized readout is redundant and should revert to the original dense readout. Baseline compiler storage is additional. Mixture width is not a count of bilinear neurons and cannot be compared to subset width as equal compute or storage.

**Interpretation.** This asks whether the successful shared factor map has a compact linear output bottleneck on the tested population. Failure does not rule out nonlinear, conditional, task-specific or downstream-weighted bottlenecks. Success would not by itself identify semantic units. Pooled and equal-group objectives may prioritize different outputs; their spectra are saved, with all per-document errors. One random basis is a control, not a universal null distribution.

**Audit.** All1152 modes reproduce H323 full prediction errors; maximum relative discrepancy8.89e-13. Native factor and reference hashes verified. Native weights unchanged.

**Next discriminator.** Use the rank curve to distinguish a small shared output interface from a broad one. If broad, inspect whether downstream-sensitive directions or context-conditioned output maps are smaller, without treating response-energy compression as behavioral equivalence. Broad research goal remains active.
