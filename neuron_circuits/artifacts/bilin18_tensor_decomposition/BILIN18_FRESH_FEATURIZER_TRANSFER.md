## H733 — Fresh cross-task tests expose substantial missing computation in every sparse candidate

We evaluated the frozen scalar Top-8 and both H730 learned-group variants at L11H3 and L13H0 on all 231 token-valid H732 prompt variants. Every dictionary seed and three degree-preserving rewires per learned grouping were retained. No representation was selected or fit on the fresh panel. Failed-competence strata remain in the results and are labeled diagnostic.

Two interventions were kept separate. Task transfer adds F(donor) minus F(receiver) to the native receiver head, leaving downstream computation live. Native reconstruction replaces the head by F(native) and measures KL from the original output distribution, plus change in intended-answer-set negative log probability. That latter task-target metric is not observed-corpus cross-entropy.

Mean retained exact-head task effect in competent strata (averaged over three dictionary seeds):

| Head and task / format | Scalar Top-8 | Learned correlation groups | Learned random-init groups |
|---|---:|---:|---:|
| L11H3 agreement A | 88.5% | 85.8% | 87.3% |
| L11H3 agreement B | 94.3% | 92.2% | 90.2% |
| L13H0 capital recall B | 65.1% | 51.7% | 49.5% |
| L13H0 entity binding A | 37.5% | 25.8% | 22.8% |
| L13H0 entity binding B | 35.1% | 21.7% | 27.6% |
| L13H0 sequence copying A | 41.0% | 27.1% | 27.4% |
| L13H0 sequence copying B | 43.6% | 31.1% | 27.0% |

Retention is the ratio of mean donor-directed margin changes, not accuracy. Ratios were suppressed wherever the absolute mean exact-head effect was below 0.25, a threshold fixed before this run. Absolute RMS effect errors were calculated everywhere. Scalar Top-8 has the smallest mean RMS error in all seven competent, non-negligible-effect cells above: 0.731/0.363 for agreement, 2.412 for capital recall, 0.612/0.484 for binding, and 0.703/0.769 for copying. Thus its advantage is not just a favorable aggregate ratio, although considerable individual-effect error remains.

The grouped candidates' advantages over degree-preserving rewires are small and inconsistent. For example, learned-correlation groups retain 21.7% of binding-B effects versus their rewires' 22.6%; learned-random groups retain 22.8% of binding-A effects versus rewires' 24.8%. Other cells favor inferred groups modestly. There is no general group-specific causal advantage on this panel.

Native output-distribution KL on the same competent task contexts:

| Head and task / format | Scalar Top-8 | Learned correlation groups | Learned random-init groups |
|---|---:|---:|---:|
| L11H3 agreement A | 0.00742 | 0.00947 | 0.00822 |
| L11H3 agreement B | 0.00595 | 0.00839 | 0.00766 |
| L13H0 capital recall B | 0.03569 | 0.06070 | 0.06931 |
| L13H0 entity binding A | 0.02149 | 0.03002 | 0.03017 |
| L13H0 entity binding B | 0.01583 | 0.02164 | 0.02197 |
| L13H0 sequence copying A | 0.00394 | 0.00527 | 0.00475 |
| L13H0 sequence copying B | 0.00404 | 0.00481 | 0.00532 |

These are task-context KL values; they should not be compared as if sampled from the same distribution as earlier 32-natural-context means. Scalar Top-8 again performs best among these candidates in the displayed cells, but none is an adequate complete account of the recall head's cross-task computation.

The near-zero-effect cross-task controls remain small in absolute terms: for competent tasks outside agreement, L11H3 effect RMS errors are approximately 0.003–0.014 across candidates; for L13H0 on agreement they are approximately 0.007–0.011. We do not inflate these into unstable retention ratios. Sentiment and capital Format A remain native-competence failures; their numerical results are archived but do not count as successful task transfer.

Interpretation: shared use of the architectural recall head does not establish shared feature-level computation, and compressing it into eight selected coordinates discards much of its binding/copying influence. The better scalar reconstruction seen earlier is useful but insufficient validation for semantic labels or manifold claims. The groups have not earned promotion to hierarchy or computational-DAG interpretation.

Audits: 25,872 full-model example forwards completed. All input/source hashes remained unchanged and task/KL values were finite. Exact-head task replay maximum error was 6.68e-6; identity native-reconstruction KL was exactly zero. Existing union and membership-inference correctness tests underlie these frozen implementations. No new natural corpus documents or original reserved-test evaluations were used. Small templated strata, repeated capital facts and finite seed/control counts limit generality; no significance claim is made from these means.

Next information-gaining test: distinguish a restrictive eight-coordinate budget from poor learned directions. Evaluate a predeclared write-PCA capacity curve on the same cross-task effects, with full-rank identity control, and separately intervene on the component omitted by scalar reconstruction. That determines whether missing computation is low-dimensional but missed by the featurizer or spread across a wider span. Only then should selective component-reuse/manifold claims proceed; a successful unit must account for the relevant effects rather than only the retained fraction. The separate sentiment format/content diagnostic remains outstanding.
