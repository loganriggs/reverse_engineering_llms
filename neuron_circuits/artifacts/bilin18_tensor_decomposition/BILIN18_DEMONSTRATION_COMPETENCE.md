## H676 — Competent demonstrations enable a sharper mechanism test

The previous no-demonstration held-out transformation subset had0% exact next-token accuracy, limiting claims about successful computation. H676 fixes two demonstration formats before inference and screens six task families with matched copying controls. Four examples precede each query. Queries comprise four prior held-out and four prior fresh-panel inputs, eight per family. Demonstration/query inputs are disjoint. Formats are arrow pairs or Input/Answer lines with a task instruction. Vocabulary/items are reused; prompt sequences are new, not independent fresh-data validation.

Predeclared rule: a family/format qualifies only if both its transformation and copying control achieve at least6/8 full-vocabulary top1 correctness. All192 prompts and all12 cells are retained. No interventions or template revisions were selected after observing results.

| Family | Format | Task correct /8 | Copy correct /8 | Qualified |
|---|---|---:|---:|---|
|plural|arrow|7|7|True|
|plural|labeled|6|8|True|
|antonym|arrow|5|7|False|
|antonym|labeled|1|8|False|
|capital|arrow|4|8|False|
|capital|labeled|2|8|False|
|successor|arrow|7|8|True|
|successor|labeled|5|8|False|
|addition|arrow|2|7|False|
|addition|labeled|2|8|False|
|agreement|arrow|8|8|True|
|agreement|labeled|8|8|True|

Five cells qualify: plural in both formats, successor in arrow format, and agreement in both formats. Agreement copying always asks for The, so its perfect copy score is a weak control for item-sensitive copying. Task agreement has balanced is/are answers. Plural and month results supply successful single-token transformations for the previously investigated mechanism. Antonyms, capitals and addition remain below the fixed threshold and are not silently discarded.

This is evidence for competence under these demonstrations, not proof that the earlier mechanism explains it. Format sensitivity is substantial: antonym correctness is5/8 with arrows versus1/8 with labels; capitals4/8 versus2/8. Demonstrations alter context length, attention opportunities and task specification together. The present screen does not attribute the improvement solely to any one of those factors.

### Refreshed hypothesis set and discriminating experiments

These are qualitative priorities, not posterior probabilities. Multiple explanations can coexist.

| # | Hypothesis | Current assessment | Discriminating observation |
|---|---|---|---|
|1|A signed gate controls generic input support.|Strong local evidence on earlier prompts.|Native sign predicts full finite input-write removal on competent prompts.|
|2|The same gate controls whether to copy or transform.|Plausible, incomplete.|Matched task/copy gate changes predict distinct performance effects.|
|3|Carried content is the main value source across formats.|Supported earlier, transfer untested.|Local/bus decomposition stays carried-dominated in demonstration prompts.|
|4|Layer14 projection supplies suppressive alignment.|Supported as a composition, source attribution incomplete.|Matched readout mapping and scrambled-coordinate controls across layers.|
|5|Head4 is a dedicated successor operator.|Disfavored by broad content effects.|Selective successor dependence beyond matched copying and pluralization would revive it.|
|6|Opposing heads cancel broadly during native computation.|Disfavored as a general account.|Measure coefficient-weighted cancellation on new prompts, not bare directions.|
|7|Input support also supports semantically associated answers.|Supported by plural/capital removal effects.|Task-answer damage persists after controlling input-copy competition.|
|8|Transformations rely on additional heads beyond head4.|Likely but not localized.|Whole-input-write removal has distinct task impact beyond head4 removal.|
|9|Demonstration answers provide a competing retrieval route.|New high-priority alternative.|Remove demonstration-answer writes separately from query-input writes.|
|10|Example relation, rather than answer retrieval, specifies the operation.|Plausible.|Change example relations while preserving output-token counts.|
|11|Recent demonstrations dominate task selection.|Untested.|Reverse fixed example order with the same query and compare gates/behavior.|
|12|Output formatting controls much of apparent competence.|Supported as a sensitivity, mechanism unknown.|Transfer across arrow/labeled formats at matched competence and analyze punctuation contributions.|
|13|The tested late gate is a downstream correlate of earlier task selection.|Plausible.|Gate swaps fail while earlier-state interventions transfer behavior.|
|14|Later layers compensate for an altered input write.|Supported in local derivative magnitudes.|Compare small versus full removal and direct versus full responses.|
|15|Distributed redundant heads preserve correctness under removal.|Plausible.|Joint removal produces effects absent from isolated removals.|
|16|A compact task-independent value basis survives context changes.|Plausible but unproven.|Fit only discovery inputs, predict held-out intervention effects across tasks/formats.|
|17|Different task contexts require different effective bases.|Compatible with wording sensitivity.|Shared-basis predictions systematically lose to matched-context bases at equal capacity.|
|18|Behavior-equivalent compression hides distinct causal responses.|Open, central to canonical-form question.|Equal-KL models diverge under the same held-out interventions.|
|19|Observed transfer reflects narrow vocabulary/domain reuse.|Live limitation.|Freeze rules and replicate on independently specified new identities and relations.|
|20|Some apparent task behavior is a shortcut rather than the intended relation.|Live alternative.|Counterbalance answer frequencies, example overlap and relation-consistent distractors.|

Highest immediate information value: test hypotheses1–3 against9 and13 on the competent panel, with failed cells retained as secondary comparisons. Predeclare interventions before outcomes: remove query-input writes from head4 versus all heads, and remove demonstration-answer writes from head4 as a distinct source control. Measure native signed coefficients and full downstream effects; report task/copy CE and exact accuracy separately from KL. Query-input spans and all demonstration-answer spans must be determined from tokenizer offsets, with multi-token agreement/addition inputs handled explicitly. Do not select individual correct examples as the sole analysis set; use every item in each prequalified cell, plus all unqualified cells descriptively.

Audits:192 whole-model forwards; all four tracked hashes unchanged; finite logits; single-token targets and disjoint demonstration/query inputs asserted. Manifest/config written before model loading and inference. Captures retain native logits and layer8/14 QK projections, persistent bus and final residual for subsequent analysis. No new FineWeb documents (26,112 unchanged). Previous turn supplied a requested research explanation but no new experimental state; this turn resumes concrete experimental progress. Broad goal remains active and incomplete.
