## H768 — Context sensitivity is highly heterogeneous; special error/context alignment is not established

Previous goal turn was progress: H767 produced a concrete local-versus-global discrepancy and a refreshed hypothesis set. H768 tests that discrepancy without categorical Monte Carlo sampling.

For each of the same 16 validation documents at lengths 32 and 120, take its adaptive write-PCA reconstruction error at budgets 8 and 64. Apply that identical error direction to every one of the 16 contexts of matching length. This yields four 16-by-16 curvature matrices, 1,024 context/direction pairs. Rows identify the context that produced the error; columns identify the context in which its local output sensitivity is measured. Central differences at 1% and 2% estimate directional logit derivatives, and exact full-vocabulary native probabilities give their variance. These are numerical local Fisher evaluations, not full finite-edit transfer tests.

A diagonal entry measures the original context/error pairing. Averaging a row measures that same error direction under the panel-average metric. This directly tests averaging over contexts, without the older metric's categorical sampling noise or training/validation context mismatch.

| Prefix length | Budget | Matched / other-context mean curvature | Ratio after destination sensitivity normalization | Panel-average prediction relative RMS error |
|---|---:|---:|---:|---:|
| 32 | 8 | 8.12 | 1.41 | 90.4% |
| 32 | 64 | 4.81 | 1.31 | 87.0% |
| 120 | 8 | 0.87 | 1.11 | 115.9% |
| 120 | 64 | 0.56 | 0.73 | 274.4% |

Even an average over these same contexts predicts their individual curvatures poorly. Therefore categorical sampling error in H764 cannot fully explain the local-versus-global failure. Increasing sampling precision alone will not eliminate the heterogeneity demonstrated here. This does not establish how much of H764's particular error came from each source.

However, the stronger hypothesis that reconstruction errors systematically align with their own context's special sensitive directions is not established. The raw matched/off-context ratio is large at length 32 but reverses at length 120. Destination sensitivity normalization divides each column by its mean over all 16 error directions; that reduces the apparent length-32 advantage substantially. All four normalized diagonal means lie within their descriptive 95% permutation reference intervals. These permutations are exploratory comparisons on a small reused panel, not confirmatory tests or proof of no coupling.

Across destinations, mean sensitivity to this fixed bank of directions varies by 284–351 fold within a length/budget. This is directional-bank sensitivity, not a claim that the full Fisher matrices differ only by a scalar. A multiplicative source-strength times destination-sensitivity approximation has matrix relative RMS residuals of 19–39%, so non-scalar structure remains. Its diagonal errors are 16–71%; it is fitted to the whole matrix, not a held-out predictor.

The best-supported next distinction is therefore **overall sensitivity versus direction-specific geometry**, rather than immediately introducing a different feature basis for every context. A scalar context weight is a simpler control that must be tested before attributing gains to richer conditional groups or manifolds. None of these diagnostics verifies a semantic manifold or reusable computation.

Numerical checks: 5,120 model forwards; native logit replay <=4.24e-5; diagonal curvatures reproduce H767 within 0.78% relative RMS. Two finite-difference steps agree within 0.08–0.56% curvature RMS across the crossed matrices. Inputs/model hashes unchanged and outputs finite. No original reserved-test evaluation; natural document pool remains 26,112. Only adaptive write-PCA directions are crossed here, so extension to other featurizers and causal swap directions remains untested.

Next experiment: on fresh training contexts, estimate sensitivity using a fixed bank of task-blind probe directions shared across all contexts. Test whether a scalar sensitivity estimate predicts held-out probe curvature and whether adding a small number of directional regimes materially improves it. Keep probes used to infer the scalar/regime separate from probes used to evaluate it, and split by document. Then compare scalar-weighted feature learning against context-dependent geometry with actual KL and diverse causal tasks. This sequence distinguishes a weighting problem from a representation problem before adding architectural complexity.
