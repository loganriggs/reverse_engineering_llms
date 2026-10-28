## H986 — Ten additional task candidates yield two clean training families

Previous turn was progress: H985 completed4,368 finite-edit model evaluations, establishing modest familiar-task fidelity and weak whole-task transfer. This turn broadens supervision candidates rather than continuing the48-state loss sweep. No featurizer fitting occurs.

Fixed candidate bank: antonyms, animal young names, plural forms, country-language recall, continent recall, profession tools, month successor, number words, comparative forms and object functions. Each has eight source pairs and two wordings, with A firstfour discovery, A lastfour lexical, B firstfour structural and B lastfour joint. Continent and object-function families are additional whole-task holdouts; original copying/spatial/role-binding/tense holdouts remain excluded. Month vocabulary necessarily overlaps source splits; other new families have distinct queried items across source slots, not necessarily distinct outputs or word parts.

160declared pairs;144 token-valid pairs/288 endpoint examples. Sixteen pair exclusions are retained, chiefly multi-token animal-young completions, plus plural goose/geese and one profession-tool pair per format. Strict one-token continuation evaluation limits task coverage; exclusions are not inability evidence. All18 last-token MLP writes are captured for every valid endpoint, enabling causal screening and later joint training without another native capture pass.

Unchanged gate: at leastfour distinct pairs per cohort, at least75% both-endpoint binary correctness and50% intended full-vocabulary top1, no individually correct endpoint filtering.

| Candidate | Native-gate passing cohorts | Training interpretation |
|---|---|---|
| Country-language | discovery, lexical, structural, joint | New candidate across both wordings |
| Antonym | discovery, lexical | New candidate, alternate wording fails full-vocabulary gate |
| Continent | discovery, lexical | Diagnostic only; holdout and semantic-label issue |
| Object function | discovery | Diagnostic only; holdout and multiple plausible completions |
| Animal young, plural, profession tool, month successor, number word, comparative | none | Retain failure/exclusion records; no training promotion |

Country-language binary correctness is100% in allfour cohorts, full-vocabulary rates87.5%,100%,87.5%,75%. Antonym original-wording cohorts are100% binary and75% full-vocabulary; alternate wording remains100% binary but0% full-vocabulary. Several other candidates similarly pass binary contrasts but fail open completion. This is why adding arbitrary labels would inflate the usable task count.

Semantic audit: the continent labels use under-specified America for Brazil/Peru, where the intended continent can require South America; this undermines the strict one-token target. Tool/function prompts admit alternatives, and some function labels are not natural completions of both templates. These defects were identified during the run, were not patched post-outcome, and exclude those families from training eligibility even where native gates pass. They should be redesigned as new panels if pursued, never silently relabeled as validated tasks.

Saved eligibility manifest selects only eight discovery pairs/16states from antonym and country-language. Non-discovery examples are reserved for development validation; causal qualification is still pending. Combined with the earlier six-family48-state bank this would yield eight families/64states, still much too small to claim that training coverage is adequate. This is a concrete training ingredient, not completed retraining or discovery of unlabeled tasks.

Audits: all40 cohort gates and every endpoint score independently replay exactly from saved logits. Zero exact prompt overlap with H962; this is only against that checked panel, not all historical prompts or pretraining. Final run289forwards including one repeat; initial run also289forwards, giving578 total. Initial absolute activation-replay threshold1e-4 failed; its log is retained. Same stimuli rerun with relative activation-norm check: absolute discrepancy0.01831, relative5.39e-7, native logit replay2.81e-5. This is float32 batch-versus-single replay, not exact tensor equality. Source and manifest hashes unchanged within the final run, finite scores, no reserved-test access. Natural-document pool26,112.

Next bounded deliverable: validate native MLP donor-write effects for these two new families and expand contextual coverage for eligible anchors, feeding a merged joint-featurizer dataset with explicit source/task splits. Preserve fresh contexts for validation and all declared whole-task holdouts. Broader task exploration remains useful when it supplies such datasets or causal targets; avoid substituting another long single-task mechanism study. The bootstrap loop (unlabeled groups → new task hypotheses → fresh validation → retraining) remains outstanding. H961 context-bridge predictor evaluation remains pending.
