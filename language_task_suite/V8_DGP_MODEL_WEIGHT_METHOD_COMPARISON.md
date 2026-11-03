# V8 DGP variation: learned source copy and what weight methods recover

The research objective is to build a realistic, variable sampled language distribution with known, controllable rules, then compare whether weight-based mechanistic-interpretability methods recover the rules and structure that models learn across DGP settings. Task capability is a behavioral eligibility gate for interpreting a rule, not the primary objective. This V8 experiment is a **controlled pilot**, not a realism pass or a complete tensor canonicalization. The V8 generator has known world facts, source-attributed updates, aliases, state queries, and optional delayed questions. Syntax entropy varies fact-bearing clauses while keeping latent worlds and answer sequences paired. `extra_source_recap_rate` controls how often the source rule is queried; source delay controls whether a different-key report intervenes. All models use ordinary next-token CE, not rule labels for training.

## DGP form and rule checks

On two independent 64×100-word FineWeb comparison panels, high-entropy V8 reference documents have 225/224 word types, top-ten share .329/.326, and first-40 repetition .255/.232. FineWeb has 2358/2435 types, .207/.217 top-ten share, and .160/.189 repetition. The intentionally task-rich setting (`extra_source_recap_rate=.5`) raises source-answer sites from about .7 to roughly 3 per eligible document; its form still has only 214–225 word types. Low syntax entropy repeats substantially more (.365/.374 on reference panels). Delaying half the source questions changes task dependencies but little in these first-window form statistics. [Aggregate form panels](results/v8_dgp_factors_fineweb_skip38000_n64.json) and [second panel](results/v8_dgp_factors_fineweb_skip39000_n64.json) contain the full measurements. These are descriptive, separately eligible windows, not matched natural documents. V8 remains visibly synthetic.

The earlier [V8 semantic audit](results/document_world_v8_discourse_audit_300x2.json) passes 1,417 visible-text answer checks and upstream edits. The [task-rich semantic audit](results/document_world_v8_source50_semantic_audit_300x2.json) passes all 2,865 sites and matched upstream edits on two additional 300-document panels; its source-site counts are 929/945. The saved form panels show source sites rising from about .7 to about 3 per eligible document. In every paired syntax training run, the train and test samples have identical latent worlds and `(family, answer)` sequences across syntax settings.

## Behavioral gate: exposure changes the learned rule

All models are depth-two bilinear attention/MLP blocks, width 64, four heads, RMSNorm, 8,000 train documents, 1,000 held-out documents, 3,000 full-next-token steps. The paired syntax settings share world seeds, initialization seed, parameter count, vocabulary, and context width. Results are one seed at reference source prevalence and two seeds at task-rich prevalence.

| Training condition | Held-out source accuracy | Native-style immediate source-edit response | Delayed source-edit response |
| --- | ---: | ---: | ---: |
| Reference prevalence, syntax 0 | .326 | .161 | .035 |
| Reference prevalence, syntax 1 | .392 | .176 | .024 |
| Task-rich, immediate, syntax 1, seed 34801 | .959 | .958 | .033 |
| Task-rich, immediate, syntax 1, seed 34802 | .976 | .929 | .014 |
| Task-rich, 50% delayed, syntax 1, seed 34801 | .557 aggregate | .545 | .128 |

Edit responses are changes in probability assigned to the new correct answer after a supported upstream source edit; the oracle response is 1. Immediate-source nuisance edits are near zero in the competent task-rich models. The high-syntax immediate models transfer to low-syntax immediate questions at .963/.970 accuracy across seeds. The low-syntax model from seed 34801 does not transfer to high-syntax questions (.365 accuracy). On delayed questions, the high-syntax immediate models fall to .226/.230 accuracy, and their causal responses are only .033/.014. The 50%-delay model improves delayed response to .128 but still does not implement reliable keyed attribution. Thus the behaviorally eligible rule for the following method tests is **recent-source retrieval in immediate questions**, not general source binding. Detailed fresh-seed panels live alongside each checkpoint.

## Local bond bases: causal fidelity and ordinary language loss differ

At the second attention block's pre-`W_O` bond, compare weight-only right singular vectors of `W_O`, unlabeled activation PCA, a weight/data hybrid, a source-edit-supervised basis, physical coordinates, and random directions. Unlabeled PCA is fitted on documents with **no scored source-question site**; source reports remain in their prose. The basis fit excludes the `summary` genre, while tests use fresh summary documents. The model itself trained on all genres. Lower-rank projections are applied at every position in that bond; full rank replays the native model exactly. These are local bases, not a canonical decomposition of the whole function.

| Seed | Basis, rank | Source-edit response, syntax 1 | Token NLL, syntax 1 |
| --- | --- | ---: | ---: |
| 34801 | Native | .927 | .765 |
| 34801 | Weight-only, 16 / 32 | .754 / .921 | .906 / .773 |
| 34801 | Unlabeled no-source PCA, 16 / 32 | .022 / .731 | 1.176 / .828 |
| 34801 | Supervised edit basis, 16 / 32 | .920 / .924 | 2.349 / 1.443 |
| 34802 | Native | .950 | .762 |
| 34802 | Weight-only, 16 / 32 | .328 / .935 | 1.090 / .767 |
| 34802 | Unlabeled no-source PCA, 16 / 32 | .046 / .064 | 1.334 / .840 |
| 34802 | Supervised edit basis, 16 / 32 | .936 / .939 | 3.194 / 1.992 |

The weight-only basis consistently beats the tested global activation-energy basis on both causal response and token NLL at rank 32, including the low-syntax transfer panels. Rank-16 retention varies substantially by model seed. The supervised basis retains the narrow edit but is poor for ordinary next-token prediction. A simple task-blind product of QK and OV Frobenius norms ranks the causal head first in seed 34801 but ranks a noncausal head first in seed 34802. Global weight magnitude is therefore not a reliable head-discovery rule in this small replication. The [two-seed frontier figure](results/v8_bond_basis_frontier_two_seeds.png) shows the tradeoff; the [first](results/v8_discourse_source50_paired_depth2_seed34801/v8_syntax1_weight_bases_seed351001.json) and [second](results/v8_discourse_source50_paired_depth2_seed34802/v8_syntax1_weight_bases_seed351101.json) basis runs retain every method/rank and full-rank check.

## Sparse recovery and a causal source edge

Starting from each high-syntax dense checkpoint, iterative L1 and global pruning leave exactly 19,661 of 49,152 attention projection weights nonzero (40% live). Embedding, unembedding, MLP, and normalization weights remain dense and may adapt during fine-tuning. Immediately after the 60%-zero prune, held-out source accuracy falls to .315/.464. A 200-step **masked, no-L1** recovery restores it to .952/.967, versus dense .959/.976, with token NLL .766/.770 versus dense .766/.772. Fresh supported immediate-source edits remain .917/.932 versus dense .958/.929; nuisance effects are small. Delayed binding remains absent. Sparse compression does not by itself make a circuit readable.

Head ablations make the retained computation more specific. In seed 34801, zeroing second-attention head 1 drops source-edit response by .946 dense and .906 sparse; seed 34802 gives .913/.910. The important first-attention head differs across seeds (head 1 versus head 3), and the sparse checkpoint preserves the within-seed dependence. Using the sampled event plan only to select the *specific report-source token*, zeroing its single attention coefficient into the answer query in second-attention head 1 drops response by .955/.911 (dense/sparse seed 34801) and .921/.917 (seed 34802). On the same cases, ablation of a different person's token or the query self-edge changes response by at most about .02. The evaluator parses report forms, including possessive names, and checks that each selected source token belongs to the specified event. This is a strong causal edge result for the learned immediate-copy behavior, but it was **selected with ground-truth task labels**. It does not show that a task-blind weight method discovered the edge, or that the edge is a complete unique circuit.

## What this changes next

V8 is useful for testing how syntax variation and rule prevalence alter learned algorithms, but it cannot be the final “realistic language” benchmark. The large lexical/coherence gap requires a probabilistic document planner and semantic realization with more entity types, discourse purposes, and interacting tasks. The next weight-method test should ask whether a task-blind method can nominate the causal source head/edge across model seeds and syntax settings, then validate those nominations on held-out edits. Stronger claims about tensor canonicalization require a gauge-aware, function-level object, not only local `W_O` SVD. Keep reference-form evaluation separate from deliberately task-rich competence panels, and gate every rule before interpreting it.
