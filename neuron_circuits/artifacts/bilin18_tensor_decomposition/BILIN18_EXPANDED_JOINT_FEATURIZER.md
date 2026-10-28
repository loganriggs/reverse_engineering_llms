## H989 — Expanded context data does not reliably resolve joint-featurizer transfer

Previous turn was progress: H988 produced a prompt-disjoint expanded-context bank and explicit discovery-only training eligibility. This turn uses that data in actual joint-featurizer fits, closing the promised integration step rather than leaving it as a task-bank artifact.

Twenty-four evaluated models: eight frozen H984 baselines plus sixteen new fits. Cross two architectures (disjoint blocks/fixed balanced shared memberships), two objectives (reconstruction/pair-change), two seeds and three regimes. Frozen small6 uses original48training slots/41unique prompts. Expanded6 uses144eligible slots/132unique prompts from the original six families. Expanded7 adds eight antonym slots for152slots/140unique prompts. Country-language stays excluded fromMLP12. Neither copying nor spatial inversion enters fitting or scaling.

Architecture and optimization fixed:128atoms,32groups, block four-atom groups/Top8 versus shared eight-atom groups/Top4 with union decoding, Adam0.001,600full-batch steps. Expanded objectives average reconstruction and normalized pair-change error equally across families, preventing unequal cohort counts from reweighting task families. On the balanced original data this equals the prior reduction. Full-batch expansion means more examples and compute per step; this is step-matched, not FLOP-matched. Training-derived centers/scales differ, and expanded6 changes context distribution as well as count. Do not interpret it as a randomized pure sample-size intervention.

All models evaluated on the SAME H988 contexts plus old whole-task holdouts. Main table averages pair-difference squared-error ratios over50 native-passing, non-discovery original-family context cohorts and both seeds (100rows). Holdout column averages four native-passing copying/spatial cohorts and both seeds (8rows), including their old discovery split because these tasks never train. One equals predicting zero activation change. This is activation-space fidelity, not downstream behavioral fidelity. Failed native cohorts and discovery scores remain in all2,880raw rows.

| Architecture/objective | Training regime | New-context delta error | Whole-task holdout delta error |
|---|---|---:|---:|
|block / reconstruction|frozen_small6|1.187|1.046|
|block / reconstruction|expanded6|1.275|1.127|
|block / reconstruction|expanded7|1.155|1.136|
|block / paired_changes|frozen_small6|1.046|1.042|
|block / paired_changes|expanded6|1.109|1.055|
|block / paired_changes|expanded7|0.942|1.089|
|shared_fixed / reconstruction|frozen_small6|1.743|1.155|
|shared_fixed / reconstruction|expanded6|1.448|1.250|
|shared_fixed / reconstruction|expanded7|1.452|1.180|
|shared_fixed / paired_changes|frozen_small6|1.191|1.051|
|shared_fixed / paired_changes|expanded6|1.111|1.120|
|shared_fixed / paired_changes|expanded7|1.287|1.174|

There is no consistent coverage gain. Shared reconstruction improves with expanded6, but block reconstruction and block pair-change do not. Adding antonyms improves block pair-change on the original-family aggregate (1.109→0.942) while worsening shared pair-change (1.111→1.287). Both are two-seed descriptive comparisons, not established transfer mechanisms. All whole-task means remain above1. No new-context antonym validation cohort passes native competence, so no competent antonym-generalization claim can be made. Its effect on other families can still be measured.

The data expansion has genuinely reached the featurizer, but these representations still do not reliably preserve unseen-task changes. This weakens the simple explanation that more contexts alone would fix the tiny-data pilot. It does not rule out scale, better task balancing, optimized memberships, better causal objectives, site choice or a different representation architecture. No memberships are learned in this comparison; fixed overlap remains a control for the earlier membership-collapse problem.

Audits: sixteen600-step fits completed, eight frozen models replayed on original inputs within1e-5; all24 models and new validation predictions saved. Exact union support and finite predictions checked. Source/model hashes unchanged. Independent CPU recomputation of all2,880 delta-error rows agrees within1.20e-6; training-only centers and excluded holdouts checked; new training/validation exact prompt separation retained. No full-model forward examples, no reserved-test access. Natural-document pool26,112.

Next high-information diagnostic: separate representation coverage from encoder/gating failure before adding more losses or contexts. Compare train-only PCA baselines and teacher-informed projection into each frozen decoder span, including its actual active support. If oracle reconstruction is strong but encoded changes fail, improve coordinate/gate inference; if even oracle coverage fails, revise dictionary capacity/site/objective. Oracles must remain explicitly teacher-informed diagnostics, never called deployable featurizers. Then carry justified candidates into finite downstream validation and selective interventions. The desired bootstrap discovery/retraining loop remains open; H961 context-bridge predictor evaluation remains pending.
