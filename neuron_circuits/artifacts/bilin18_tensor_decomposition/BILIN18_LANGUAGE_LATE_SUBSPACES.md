## H907 — Influential language sites do not yield a transferable four-dimensional source-change subspace

The previous turn made progress: H906 localized stronger language MLP donor effects at L13–14 rather than L12. H907 repeats pooled-versus-paired subspace tests at all three layers, with L12 retained as reference. L13/14 are screening-derived choices, and evaluation pairs are reused; this is follow-up localization analysis, not independent confirmation.

At each layer/template, four task discovery pairs fit centered activation PCA and uncentered paired-difference SVD. Ranks1/2/4 plus seeded random rank4 are evaluated on four disjoint country pairs in task/copy contexts. Full and projected source changes are inserted only at the final-query MLP, with exact bilinear quadratic terms and live downstream computation. Projected normalized inputs are not renormalized; no norm-matched selectivity claim is made.

### Task transfer at the influential sites

The error ratio below is RMSE versus full donor-intervention margin effect divided by the RMSE from making no change. Zero is exact recovery; one means the approximation is no better than no change by this metric. It is not a fraction of explained causal computation.

|Layer|Subspace|Held-out change energy captured|Effect-error ratio to no change|Full-edited to approximate-edited KL|
|---|---|---:|---:|---:|
|13|pooled_1|0.021|1.008|0.04019|
|13|pooled_2|0.054|0.922|0.03078|
|13|pooled_4|0.126|0.802|0.02518|
|13|paired_1|0.024|1.003|0.03904|
|13|paired_2|0.077|0.861|0.02721|
|13|paired_4|0.115|0.832|0.02586|
|13|random_4|0.003|0.998|0.03721|
|14|pooled_1|0.021|0.985|0.02468|
|14|pooled_2|0.051|0.905|0.02387|
|14|pooled_4|0.131|0.857|0.02134|
|14|paired_1|0.024|0.979|0.02402|
|14|paired_2|0.082|0.821|0.02236|
|14|paired_4|0.120|0.807|0.02145|
|14|random_4|0.003|1.001|0.02394|

At L13, pooled/paired rank4 capture12.6%/11.5% of held-out input-change energy and retain effect-error ratios0.802/0.832. Their mean margin changes are0.209/0.171 versus the full substitution's1.009. At L14, rank4 capture13.1%/12.0%, with error ratios0.857/0.807 and mean changes0.124/0.140 versus full0.790. Rank1 is near no change at both sites. Random rank4 is also near no change, so the learned directions contain some structure but fall far short of recovering the full effect.

Increasing paired rank2 to rank4 yields only a small L14 margin-error improvement (0.821→0.807); pooled rank4 beats paired rank4 at L13 and the reverse holds at L14. There is no consistent large paired-discovery advantage. All ranks and failures are retained, rather than choosing a favorable criterion or layer.

L12 still has weak full target influence (RMS0.120); its lower normalized error ratios do not make it a better site for identifying the language computation. The strong sites' full RMS effects remain1.103/0.950 and reproduce the localization screen exactly.

### Copying and interpretation

At L13–14, task-trained rank4 bases capture only about6% of copying changes; effect-error ratios remain0.825–0.861. No successful shared copying mechanism emerges. These results do not disprove structured language computation or a higher-dimensional lexical representation. Four discovery country contrasts provide a very limited span, and new countries may use distinct directions.

The distinction now worth testing is a shared operation versus a shared low-dimensional activation subspace. An operation can process many country-specific directions without those directions collapsing onto one to four axes. Repeatedly increasing rank on four contrasts cannot distinguish that possibility from insufficient lexical coverage.

Next preserve each recipient's full paired source change and test whether discovery-only context means can approximate the bilinear partner state that processes it. Compare own task-template means, pooled task means, and matched copying means, with exact/native-linearized/quadratic controls. This tests reusable context-dependent processing without first throwing away most of the lexical change. It is an operator-reuse diagnostic, not a learned low-dimensional feature claim or a parameter-compression result. Broader manifold, shared-dictionary, and binding-pathway work remains incomplete.

### Verification

960 full-model forwards across three layers. Identity patches are exact. Bases orthonormal to6.00e−15; cached activation replay relative error5.11e−7. Full donor-intervention margins reproduce H906 exactly at every tested layer. All source/model/frozen-fit hashes unchanged and KL finite. The exact finite-expansion formula was independently audited in H905; no new formula approximation is introduced. No natural or reserved-test evaluation; document pool remains26,112. Goal active.
