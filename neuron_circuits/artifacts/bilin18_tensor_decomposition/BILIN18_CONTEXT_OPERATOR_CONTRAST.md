## H912 — The context contrast is reproducible and high rank, not a small output subspace

The previous turn made progress: H911 found similar operators from disjoint country discovery sets but substantial residual differences. H912 subtracts the copying-context operator from the language-context operator within each discovery set. This tests whether whole-map similarity was only shared background processing.

All means remain frozen. At L13/14 compute ΔJ=J(task mean)−J(copy mean), compare the two discovery sets, and take each explicit1152×1152 operator's SVD. Evaluate rank1/2/4/8/16/32/64/128/256 actions on all96 reused new-country endpoint changes. No ranks or layers are selected and no new behavioral intervention is run. New-country C remains a failed-gate diagnostic; the action results below are geometric, not competent-task behavioral claims.

### Stability after subtracting the copying map

|Layer|Contrast operator cosine|Relative operator difference|Action cosine on task changes|Relative action difference on task changes|
|---|---:|---:|---:|---:|
|13|0.979|0.204|0.977|0.215|
|14|0.973|0.233|0.970|0.245|

The L13 contrast cosine0.979 exceeds the full-task-map cosine0.959 from H911; relative difference decreases29.2%→20.4%. L14 contrast cosine is0.973 with23.3% relative difference. The contrast norm is approximately as large as the task-map norm (ratios0.98–1.02), so it is not merely a tiny unstable subtraction. This supports reproducible context-dependent processing differences, with material residual variation still present.

### Spectrum and action approximation

|Layer|Discovery set|Rank for50% energy|Rank for90%|Rank for99%|
|---|---|---:|---:|---:|
|13|first|155|501|831|
|13|second|155|501|831|
|14|first|153|499|829|
|14|second|153|499|829|

Energy is squared Frobenius norm, not causal importance or downstream logit variance. The two discovery sets have similar broad spectra. Rank1 retains less than0.8% of matrix energy; rank4 only about2.5%.

For the first discovery set, task-input action relative RMS errors are:

|Rank|L13 action error|L14 action error|
|---|---:|---:|
|1|0.999|0.998|
|4|0.992|0.990|
|16|0.959|0.963|
|64|0.857|0.866|
|128|0.750|0.748|
|256|0.560|0.564|

Low global operator rank does not recover this contrast on the sampled lexical changes either: rank4 errors~0.99, rank64~0.86, rank256~0.56. Therefore this case does not currently support a small shared output subspace. This does not rule out a data-weighted or downstream-effect approximation; global SVD optimizes a different objective, and even these sampled action metrics are not behavioral fidelity.

A single context displacement can still induce a high-rank change in a bilinear map: ΔJ=J(task mean−copy mean), since J is linear in its evaluation point. Thus a low-dimensional conditioning variable and a high-dimensional operated-on space are compatible. This is an algebraic interpretation, not evidence that the model has already been shown to use a unique semantic context coordinate.

### Crucial remaining confound and next experiment

The present language and copying prompts differ in syntax, sequence length, query words and final tokens (language questions often end in “is,” copying in “word” or “item”). The contrast is therefore not an isolated causal effect of task identity. High reproducibility could reflect consistent surface differences.

Next construct a matched prompt skeleton such as “Country: Italy
Language:” versus “Country: Italy
Country:”, both ending in the same punctuation token. Check actual token lengths and competence before interpreting it. Freeze lexical discovery/transfer splits, retain eligibility and competence failures, and compare context contrasts under the matched surface conditions. This more directly tests whether the reusable operation tracks the requested relation rather than final-token or formatting differences. Only then consider low-dimensional conditioning or shared-dictionary interpretations. Binding and broader manifold validation remain open.

### Verification

No new full-model forwards. All source/model/frozen-mean hashes unchanged. ΔJ=J(mean difference) holds to relative2.51e−15; complete SVD reconstructs operators to relative5.82e−13. All action errors finite. Spectra and truncated factors saved; all tested ranks retained. No natural or reserved-test evaluation; document pool remains26,112. Broad goal active.
