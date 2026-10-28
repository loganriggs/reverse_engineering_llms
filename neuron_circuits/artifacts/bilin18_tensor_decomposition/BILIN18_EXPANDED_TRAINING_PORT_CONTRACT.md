## H1040 — Added task cohorts supply different causal stream targets

Previous goal turn was progress: H1039 screened20candidate families and produced20eligible training pairs across five additions. This turn completed the fixed eight-corner donor/recipient boundary factorial on every eligible pair in both directions,320suffix evaluations, without fitting or inspecting reserved task results.

The ports are residual pre-attention11, initial embedding recurrence input, and cached first-layer values. Full donor transplant is the reference. Answer margins compare the two continuation groups; full-output KL is also measured. Shapley allocation averages each port's marginal effect over all insertion orders, then projects onto the native donor-minus-recipient answer effect. These signed allocations sum to one but are not fractions of overall model understanding and can exceed100% or be negative.

| Training cohort | Residual aligned allocation | Embedding allocation | Cached-value allocation | Residual-only relative donor KL | Cache-only relative donor KL |
|---|---:|---:|---:|---:|---:|
| Antonym |68.9%|0.2%|30.9%|0.0975|0.7004|
| Country-language |23.9%|0.4%|75.7%|0.5368|0.0270|
| Number successor |90.2%|0.04%|9.7%|0.0075|0.8122|
| Past tense |91.8%|0.05%|8.1%|0.0424|0.9558|
| Noun plural |117.3%|0.3%|-17.5%|0.0978|1.0020|

KL values are normalized by full-donor versus no-edit output divergence;0is exact and1equals no edit. They are not the same metric as margin allocations. Residual+cache transplant leaves relative donor KL below0.00016for every cohort while embedding remains native. This supports stream-specific supervision with joint evaluation; it does not justify deleting the embedding port globally.

The noun-plural cache allocation opposes the aggregate answer effect. This is cancellation in a small panel, not evidence of a negative semantic feature. The native model already makes some plural errors; all four eligible-cohort pairs remain included. Fidelity targets reproduce native behavior, including errors, whereas supervised correctness targets desired answers. Do not silently conflate them. Single-port additivity error is approximately25.5%antonym and26.5%past tense, so a purely additive explanation is incomplete under these finite hybrid interventions.

Semantic scope: these pairs usually vary lexical content under a fixed task format. 'Today I walk. Yesterday I' versus the corresponding play prompt measures verb-identity effects in a past-tense setting; it does not isolate changing tense. Likewise noun-plural contrasts change cat versus dog while both contexts demand plurality. Number sequences change several aligned positions together. The bank supplies within-task counterfactual targets, not a proven causal factorization into grammatical operations. Equal token counts enable positional transplantation but do not prove semantic position equivalence. Future operation claims need content-matched task/operator contrasts or transfer controls.

Featurizer deliverable: the expansion contains residual-dependent sequence/morphology contexts and cache-dependent factual recall, avoiding an exclusively residual training objective. Preserve all three inputs for matched comparisons and report stream-conditioned plus joint causal fidelity. Shared groups must demonstrate transfer, not merely coexist with these coarse port effects. No feature/manifold discovery is established by this experiment.

Verification: source captures, eligibility, and screen hashes unchanged; zero reserved pairs selected and zero fitting rows. Native and full-donor boundary logits replay exactly. Shapley efficiency error3.56e-15. Independently recomputed Shapley for all40directed pairs using explicit permutations, and all eight saved logit corner margins/KLs for one direction; maximum discrepancy4.89e-15. No new natural documents or fresh prompts were acquired.

Stopping criterion for training-port localization is met. Next repair the reserved task prompt/alignment panels while preserving family assignments and retaining all failed attempts. Then construct the expanded joint training comparison with both state-reconstruction and context-preserving edit objectives. Record contrast variable explicitly as lexical identity, sequence offset, or operation change; require an operation-isolating control before attributing a learned group to an operation. Do not launch further head-localization detours for these cohorts.
