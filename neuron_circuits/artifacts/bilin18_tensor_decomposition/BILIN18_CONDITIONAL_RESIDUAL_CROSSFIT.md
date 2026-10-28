## H1052 — Conditional residual supervision fits but transfers mainly to pronouns

Previous goal turn was progress: H1051 tested conditional frozen residual models across8nonfactual whole-family exclusions and found limited learned benefit. This turn trained the residual-only objective directly, completing24fits under identical exact-content access.

Each method has147456learnable unit-normalized residual decoder parameters, fixed training-only PCA128encoder from its H1048fold, fixed scalar/block/shared support rule, and at most16active residual coordinates per token. Donor embedding/cache values come from the validated H1050token-content program. Native recipient residual and suffix remain;2304exact content values/token and retained pretrained machinery are still charged. All methods use9training families,16visits each/144steps, Adam0.001, final checkpoint only. All remaining pairs covered; each evaluated family remains excluded from both encoder and decoder fitting.

Objective: actual bidirectional donor-output KL under the conditional content program, plus the same small direction-displacement penalty. No gate STE, no full-state reconstruction loss, no evaluation-dependent selection. Every one of24fits lowers full equal-family training KL.

| Method | Training KL initial→final | Held-out relative KL initial→final | Prior all-stream-trained conditional KL |
|---|---:|---:|---:|
| Scalar |0.1910→0.0984|0.5766→0.5689|0.5724|
| Block |0.4292→0.0676|0.6458→0.6308|0.6410|
| Shared |0.5210→0.0611|0.6802→0.6654|0.6817|

Better-matched training helps modestly in aggregate, especially grouped decoders relative to their earlier training objective. Yet the improvements concentrate in pronoun gender: scalar0.2308→0.1742, block0.4381→0.3356, shared0.4808→0.3566 (initial baselines in the current run). Excluding pronouns, seven-family macro barely changes: scalar0.62597→0.62535, block0.67548→0.67294, shared0.70873→0.70949. Training fit remains a poor proxy for broad task transfer.

Final scalar errors remain0.819agreement,0.609antonym,0.053binding,0.760number successor,0.723past tense,0.737plural,0.174pronoun,0.677sentiment. Grouped methods do not beat scalar on any of these eight folds. Macro donor-answer top1is20.96%scalar and14.32%each grouped method. These results concern fixed random groups, one seed, small competence-screened panels, and a frozen encoder; they do not rule out learned groups/manifolds generally.

Verification:14752suffix calls including6912student training passes,24completed fits, all finite; source/program/geometry hashes unchanged; exact content matches native inputs; max16active; no exact heldout prompt or heldout family in fitted statistics. Independently recomputed48saved logit KL samples, error2.45e-14. A strict1e-8cross-run initial-KL replay check failed: maximum difference1.37e-6from H1051. Initial normalized decoder values match exactly; the remaining cross-run numerical cause was not isolated. This is recorded rather than called exact replay, and is much smaller than the reported objective differences. Native/content and current-run metric checks remain valid. No parameter tuning followed that discrepancy.

Stop the decoder-only sequence as promised. Actual optimization now works, objective mismatch has been reduced, and extra training/causal labels alone have not produced broad cross-family reusable directions. Dominant remaining hypotheses include limited frozen-encoder coverage, random group structure, lexical-specific representations, context-dependent geometry, and the fact that whole-family exclusion may withhold a computational primitive rather than merely a new composition.

Next research design should distinguish that last possibility: a frozen crossed content×context benchmark with held-out combinations, where each content item and each context format occurs separately in training. Preserve existing whole-task exclusions and report the new compositional test separately; success on it must not replace the broader goal or be called unseen-primitive transfer. Include matched native competence and zero/context-only controls, retain all attempts, and do not repurpose old heldout prompts into training without explicit provenance. Before fitting, inventory which factors truly vary and whether the native model performs the intended combination. This is a new representation hypothesis after stopping the unsuccessful decoder-only family, not another optimizer-duration or rank sweep.
