## H932 — Task-blind prototype rules transfer across templates unevenly

H931 found causal fidelity on templates included in prototype fitting. H932 refits each model using only template A, then separately using only B. In each direction, fit48 examples from the first four lexical pairs per domain, and evaluate96 examples from the other four pairs on both templates. The excluded template does not enter prototype or coefficient fitting. This is exclusion from the current fit, not an unseen research dataset: both templates and all names have been examined before.

Use1,4,8 prototypes, all three seeds, live gates and the same joint persistent-component edits at L8H3/L13H0/L14H4. Both native and opposite-endpoint donor-value conditions are evaluated. The held-out-template prediction cannot use its task/template coefficient means. The supplied-context baseline transfers the fitted template's absolute-position coefficients; prototype rules use end-relative distance. Uncovered cluster distances fall back to the fitted pooled-distance coefficient; globally missing distances use zero. All rules and fallback policy were fixed before outcomes.

### Excluded-template outcomes

Ranges cover all three seeds. Native KL compares predicted native-value output with the original model. Edited KL compares predicted donor-value output with the reference native-routing donor intervention. Effect error compares donor-minus-native answer-margin responses, normalized by reference RMS.

|Fit→evaluate|Domain/family|Prototypes|Native KL|Edited KL|Effect error ratio|
|---|---|---:|---:|---:|---:|
|A→B|agreement task|4|0.001–0.001|0.001–0.001|0.713–1.038|
|A→B|agreement task|8|0.001–0.001|0.001–0.001|0.984–1.012|
|A→B|agreement copy|4|0.009–0.015|0.008–0.014|0.268–0.296|
|A→B|agreement copy|8|0.004–0.005|0.004–0.005|0.251–0.357|
|A→B|language task|4|0.024–0.484|0.036–0.686|0.126–0.798|
|A→B|language task|8|0.023–0.024|0.036–0.038|0.126–0.133|
|A→B|language copy|4|0.097–0.160|0.043–0.077|0.554–0.711|
|A→B|language copy|8|0.048–0.103|0.031–0.071|0.387–0.667|
|A→B|binding task|4|0.072–0.072|0.067–0.067|0.113–0.113|
|A→B|binding task|8|0.072–0.072|0.067–0.067|0.113–0.113|
|A→B|binding copy|4|0.020–0.030|0.018–0.021|0.357–0.375|
|A→B|binding copy|8|0.008–0.010|0.004–0.005|0.097–0.103|
|B→A|agreement task|4|0.001–0.001|0.001–0.001|6.521–10.489|
|B→A|agreement task|8|0.001–0.001|0.001–0.001|4.905–7.599|
|B→A|agreement copy|4|0.017–0.026|0.017–0.023|0.198–0.414|
|B→A|agreement copy|8|0.020–0.021|0.018–0.020|0.372–0.385|
|B→A|language task|4|0.054–0.489|0.068–0.591|0.197–0.725|
|B→A|language task|8|0.048–0.054|0.062–0.068|0.186–0.197|
|B→A|language copy|4|0.031–0.046|0.063–0.093|0.338–0.429|
|B→A|language copy|8|0.044–0.093|0.108–0.140|0.604–0.804|
|B→A|binding task|4|0.009–0.013|0.010–0.037|0.150–0.427|
|B→A|binding task|8|0.009–0.009|0.010–0.010|0.150–0.150|
|B→A|binding copy|4|0.058–0.078|0.010–0.012|0.426–0.480|
|B→A|binding copy|8|0.007–0.008|0.002–0.002|0.110–0.139|

Eight-prototype binding task models have11.3% error A→B and15.0% B→A, versus6.3% and6.7% on their respective fitted-template lexical-transfer controls. Language task error is12.6–13.3% A→B and18.6–19.7% B→A. These results support transfer beyond exact template membership on this small panel; they do not establish broad semantic generalization.

Distribution fidelity and effect fidelity diverge. Binding A→B native KL is about0.072 despite11.3% effect error; its fitted-template native KL is about0.007. Thus a useful approximation to the donor response need not reproduce baseline predictions equally well. Both metrics must remain visible.

Copying is less uniform, especially language copying: eight-prototype A→B error38.7–66.7%, B→A60.4–80.4%. Agreement task remains a low-reference-effect condition at these readers, so relative errors can be large while absolute output discrepancies are small. All within-template controls, one-prototype baselines and seeds are retained in the raw results rather than selecting favorable settings.

### Interpretation and next test

This weakens an explanation based solely on memorizing the exact training template. It does not distinguish semantic context from shared final tokens, common syntax, lexical categories or similar query geometry. The source templates are close paraphrases and copying formats are shallow variants. No continuous manifold is established by discrete prototype success.

Next test sample stability independently of initialization: fit on disjoint lexical subsets while holding the prototype count and template protocol fixed, then compare assignments on a common evaluation panel and compare causal effect profiles. Do not choose a seed by downstream success. Separately test whether the native-distribution drift comes from coefficient prediction or from grouping distinct contexts under one prototype—for example, retain the same frozen gate while allowing a small discovery-fitted continuous correction within each group. That is a targeted featurizer extension, but it must beat the uncorrected and continuous-without-groups controls before being interpreted as meaningful manifold structure.

The stable objective remains to identify reusable units of computation, not merely a small surrogate that passes these prompts. Evidence now favors lexical carriers plus context-dependent routing for this partial pathway, with template transfer and remaining fidelity failures explicitly bounded.

### Verification

5,568 forwards across the two directions (2,784 each). Identity replay exactly zero, native replay within1e-4, fit/evaluation lexical indices disjoint. Fit templates are asserted; prototypes are refit only on their48 selected discovery examples. One-prototype models exactly equal their pooled-distance baseline. Source hashes unchanged; outcomes finite. Per-direction fits, all native/intervention metrics, live gate decisions and fallback counts are saved. No reserved test or added natural documents; pool remains26,112. Broad goal active.
