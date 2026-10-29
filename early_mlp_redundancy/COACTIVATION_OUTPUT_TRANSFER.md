# M40: learned grouping improves ordinary outputs over random grouping

All 12 M38 candidates evaluated without fitting or selection on the unchanged 1,152-input inventory. This completes the ordinary-output counterpart to [M39 native responses](COACTIVATION_NATIVE_TRANSFER.md).

KL compares native versus reconstructed full output distributions, averaged over positions and then inputs. Lower is better. Joint evaluation below excludes both lexical sources and wrapper templates from fitting, but remains familiar development data.

| Seed | Source objective | Cohort | Scalar KL | Random-group KL | Learned-group KL | Learned/random |
|---|---|---|---:|---:|---:|---:|
| 71 | intact_KL | natural | 0.14898 | 0.50159 | 0.43620 | 0.8696 |
| 71 | intact_KL | expanded | 0.08499 | 0.13810 | 0.13105 | 0.9489 |
| 71 | intact_KL_plus_native_response | natural | 0.16362 | 0.52106 | 0.48441 | 0.9297 |
| 71 | intact_KL_plus_native_response | expanded | 0.09951 | 0.14774 | 0.14076 | 0.9528 |
| 72 | intact_KL | natural | 0.13198 | 0.47457 | 0.44444 | 0.9365 |
| 72 | intact_KL | expanded | 0.06961 | 0.16209 | 0.12213 | 0.7534 |
| 72 | intact_KL_plus_native_response | natural | 0.15415 | 0.50958 | 0.46407 | 0.9107 |
| 72 | intact_KL_plus_native_response | expanded | 0.09467 | 0.17330 | 0.12770 | 0.7369 |

Learned groups improve on random groups in all four natural-text comparisons (6.3–13.0%) and all four jointly excluded lexical/template comparisons (4.7–26.3%). They remain worse than the original scalar representation in all eight comparisons. Each natural cohort contains32documents; each joint task cohort56endpoints. These are descriptive paired comparisons with two seeds, not uncertainty estimates or new-task discovery.

Interpretation: training coactivation gives useful predictive grouping, consistent with M39, but forcing these groups loses substantial fidelity relative to independently selected atoms. The dictionaries and coordinate encoders were trained as scalar representations and frozen before grouping. Scalar uses32active atoms, while learned unions use approximately28; therefore neither an architecture-wide ranking nor a causal attribution of the gap solely to grouping is established. No additional optimization or active-budget sweep is authorized by these outcomes alone.

Validation:14,976forwards,13,824rows,12model hashes; complete inventory and summaries checked. All4,608 scalar-source rows exactly replay M37. Independent NumPy checks cover84decoder/output samples, with no support near-tie discrepancies; maximum decoder relative error1.94e-7 and metric discrepancy5.42e-8. Full results retain natural/original/expanded inputs, lexical/template splits, family summaries, native competence failures, CE, top1, local error and actual active counts.

Next mechanistic discriminator: separate removal of an active group gate (which preserves shared atoms used by other gates) from removal of its atom contribution (which removes those shared atoms too). Evaluate the exclusive and shared contributions with native-response controls. Shared support alone is built into the architecture and cannot establish native reuse. Substitutability of native channel groups on other contexts/tasks remains a separate, unresolved test of distributed redundancy.

[Protocol](validation/coactivation_output_protocol.json) · [Results](validation/coactivation_outputs.json) · [Audit](validation/coactivation_outputs_audit.json) · [Matched comparison](validation/coactivation_output_comparison.json)
