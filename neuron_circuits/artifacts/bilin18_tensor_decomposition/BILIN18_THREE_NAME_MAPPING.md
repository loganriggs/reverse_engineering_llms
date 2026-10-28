## H545 — Three-name mappings expose partial competence and weak association selection

The preceding goal turn made progress by rejecting a fixed query suppressor and finding little correct-value preference in two-name mappings. This experiment uses eight name triples, all six permutations, three cyclic demonstration orders and all three queries: 432 newly constructed prompts. Every key occupies each entry position across order rotations. Keep all four readers fixed and report every condition without filtering successful examples.

Score all three candidate names and the complete vocabulary. For nonfixed queries, also compare the correct target with the other nonquery candidate. Suppressing the query alone cannot distinguish those two. The strong competence gate requires >=75% of complete three-query blocks correct and >=50% intended full-vocabulary top-1 within a permutation kind.

| Permutation kind | Prompts | Three-way accuracy | Full-vocabulary top-1 | All-three-query block accuracy | Nonquery-rival accuracy | Strong gate |
|---|---:|---:|---:|---:|---:|---|
|identity|72|73.6%|73.6%|37.5%|—|fail|
|transposition|216|44.0%|44.0%|1.4%|63.9%|fail|
|cycle|144|61.8%|61.8%|6.2%|61.8%|fail|

No kind passes the strong gate. These results show partial ability, not total inability: cycle accuracy is 61.8%, above a fixed nonquery-choice baseline of 50% on this balanced design, but complete mapping execution is unreliable. The model never chooses the query itself on the 144 cycle prompts. Identity mappings produce 73.6% individually correct outputs but only 37.5% completely correct three-query blocks.

A useful failure occurs in transpositions, where two names swap and one is fixed. Accuracy is only 5.6% on the fixed-point queries, versus 63.2% on the nonfixed queries. This is consistent with a coarse nonidentity/query-suppression tendency rather than reliable query-specific association selection; it does not identify the full algorithm. A secondary most-recent-nonquery baseline has 50% cycle accuracy and matches only 36.8% of cycle model choices, so that single recency rule is not a sufficient account either.

### Causal source neutralizations

For each query and demonstrated value, replace one source-to-head persistent-bus contribution with the mean of the three name-value vectors. This removes some identity-specific evidence along that path while preserving shared name content, but also introduces equal candidate-name components. All other paths and downstream computation remain active. These are local sufficiency/neutralization probes, not whole-circuit necessity measurements.

On cycle prompts, neutralizing the repeated query changes the correct-versus-other-nonquery margin by only +0.0029 for L14H4 and +0.0045 for L8H3 on average. Neutralizing the correct value changes that margin by −0.7195 and −0.4782. Thus these value paths carry evidence that distinguishes candidate identities, while the query paths mostly act on the query identity itself. This does not show that the heads select which value is correct.

Indeed the correct value has the strongest identity-copying routing coefficient among the three values only 34.0% of cycle cases for L14H4 and 45.1% for L8H3. Uniform ranking would give one third; these are descriptive rates without population significance claims. The small L8H3 preference is not a complete lookup mechanism, and raw coefficient comparisons are only a proxy because token transformations differ across identities.

In transposition fixed-point cases, the query coefficient is suppressive in 100% of cases for L14H4 and 88.9% for L8H3. Neutralizing that path improves the correct-answer margin by +0.610 and +0.171, respectively. The same neutralization harms nonfixed cases on average (−0.151 and −0.038). This identifies a local contribution that can help easy cases while hurting fixed points. It does not establish that it accounts for the entire error.

### Direction update

The conditional identity-reader primitive remains useful, but strong claims about a general arbitrary-mapping algorithm are unsupported. Broad value copying plus context-dependent query correction remains plausible; selective association information may be elsewhere or insufficiently learned. Continuing to interpret poor arbitrary-task performance as successful binding would be misleading.

Next: test a more natural three-person narrative with an explicitly irrelevant person, while keeping three-candidate/full-vocabulary scoring and frozen readers. Vary distractor placement so a simple recent-name strategy can fail. Native competence and all failures must be reported, not used to silently choose a passing subset.

Audits: 432 native evaluations, 6,912 geometry records and 7,344 interventions, including self controls. Self correct/nonquery margins reproduce exactly; source token/position assertions pass; all outputs finite; manifest/source hashes unchanged. No new FineWeb documents; tracked total 26,112. Known names are recombined in new prompts, not claimed absent from model training. Broad goal remains active.
