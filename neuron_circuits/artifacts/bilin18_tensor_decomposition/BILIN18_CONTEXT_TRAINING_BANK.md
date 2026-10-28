## H988 — Context expansion yields184 eligible training slots and172 unique prompts

Previous turn was progress: H987 causally checked new task sites and assembled the seed dataset, exposing limited unique coverage and old capital-recall validation overlap. This turn expands contexts for eight non-holdout families and checks exact train/validation prompt separation before model execution. No fitting occurs yet.

Fixed design: agreement, sentiment, negation, entity binding, pronoun gender, capital recall, antonym and country-language; eight source pairs/family crossed with eight context templates. Contexts0–3 are training wording candidates,4–7 validation wording; sources0–3 discovery,4–7 lexical transfer. Cross them to obtain discovery/lexical/structural/joint splits. Each family/context/source-half is a four-pair competence cohort. Semantic groups for agreement/sentiment/negation remain inherited; others exact single-token answers.

Context changes are task-specific: intervening locations for agreement, reviewed subject for sentiment/negation, referent-location and reporting-verb wording for pronouns, setting/object/transfer-verb changes for binding, and question paraphrases for factual/antonym tasks. These are controlled template variations, not a representative natural-language distribution. Historical lexical items are reused. Capital pairs were rebuilt from16 distinct countries so training and lexical sources do not share endpoints. Other families can share answer labels and word parts across splits. Sentiment/negation intentionally share some training prompts.

512declared pairs;504 token-valid pairs/1008 endpoint slots. All18 last-token MLP writes and native logits saved. Unchanged competence threshold per four-pair cohort: at least75% both-endpoint binary correctness and50% intended full-vocabulary top1, with allfour pairs valid. Training eligibility uses ONLY discovery-cohort gates; no individual correctness filtering or validation-based training selection. All validation cohorts are saved, with failed native cohorts explicitly diagnostic.

| Family | Eligible training slots | Passing contexts: discovery / lexical / structural / joint (out of4 each) |
|---|---:|---|
|agreement|24|3 / 3 / 4 / 3|
|antonym|8|1 / 0 / 0 / 0|
|capital_recall|8|1 / 0 / 3 / 0|
|country_language|32|4 / 3 / 3 / 3|
|entity_binding|24|3 / 3 / 4 / 4|
|negation|24|3 / 3 / 3 / 3|
|pronoun_gender|32|4 / 4 / 1 / 1|
|sentiment|32|4 / 4 / 4 / 3|

Total184 training endpoint slots,172unique prompts, across23 passing discovery cohorts. No exact training/validation prompt overlap. For the commonMLP12 comparison, excluding country-language leaves152training slots/140unique prompts across seven families. Country-language's32states remain available forMLP13 or explicitly layer-conditioned work; H987 showed weakMLP12 causal effects, so do not silently count these as equally useful common-site targets. The dataset is still small and templated, but substantially expands the earlier48slot/41unique-prompt training set.

Coverage limits are substantive: antonym and capital recall retain one training context each and no lexical validation contexts. Their competence depends on wording and/or lexical items; growing nominal prompt counts does not guarantee usable supervision. Sentiment, country-language, entity binding and agreement transfer more broadly across this panel. Pronoun-gender training contexts allpass, but onlyone validation wording passes in each source half. These gates are behavioral observations, not proof of causal interchangeability of every captured state.

Audits:1,009/1,009 full-model forwards, including one single-example replay. Native logits agree within2.48e-5; activation replay absolute0.02344 and relative5.92e-7, under a declared relative1e-5 bound. Source and manifest hashes unchanged, finite outputs. All128 cohort gates independently recomputed; eligibility count and zero cross-split prompt overlap independently verified. Eight token-boundary pair exclusions retained. No original whole-task holdouts enter this bank, no reserved-test access, no featurizer fitting. Natural-document pool26,112.

Saved concrete training ingredient: bilin18_context_training_bank_training_eligibility.json specifies indices, family counts, discovery-only selection and all validation indices; captures contain complete all-layer writes. This is ready for a direct expanded-versus-small-data comparison, not another plan-only task expansion.

Next: fit fixed-budget joint featurizers on the expandedMLP12 cohort while retaining earlier frozen models as baselines evaluated on the SAME new validation examples. Separate six-original-family coverage gains from adding antonyms, if feasible, and retain both seeds and reconstruction/pair-change objectives. Use prompt-disjoint validation and retain native-failing diagnostics. Country-language stays separate atMLP13. Then compare downstream finite-edit fidelity and whole-task transfer; only stable/selective unlabeled structure should initiate new-task discovery. Broad bootstrap objective remains unfulfilled; H961 context-bridge predictor evaluation remains pending.
