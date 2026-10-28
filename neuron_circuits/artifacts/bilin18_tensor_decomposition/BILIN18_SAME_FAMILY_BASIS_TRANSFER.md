## H1038 — Residual bases transfer within familiar tasks, conditional on native other streams

Previous goal turn was progress: H1037 showed excellent training-change coverage but poor excluded-task transfer. This turn completed the bounded frozen same-family evaluation, then added a necessary native-assistance control after observing the initial two-arm results. No basis, threshold, rank, or gate was tuned.

Panel: all19 historical lexical-development pairs/38endpoints across the five training families, exactly prompt-disjoint from discovery. Native both-endpoint binary accuracy100% for every family. Native top-answer accuracy is87.5%agreement,100%capital,100%binding,75%pronoun,75%sentiment. Capital has only3equal-length pairs, so remains diagnostic; the other families have4pairs each. These are previously opened development examples, not fresh task validation.

Both frozen rank128 residual bases use the same training center. Whole reconstruction compresses the residual and keeps other streams native. Paired edits preserve recipient residual context and transfer the projected residual difference, with exact donor information in the embedding and cached-value ports. The added mean-only decoder produces zero residual change in paired edits, retaining the same exact donor other ports. Its whole reconstruction is a mean-residual ablation, not a matched-rank candidate.

| Family | Activation-basis paired relative KL | Difference-basis paired relative KL | Zero residual change, donor other ports |
|---|---:|---:|---:|
| Agreement |0.0121|0.0133|0.9660|
| Capital recall |0.0107|0.0131|0.0211|
| Entity binding |0.0259|0.0248|0.0981|
| Pronoun gender |0.0106|0.0036|0.6753|
| Sentiment |0.0568|0.0510|0.8573|

Lower is better; normalization is against no edit at any port, not the added zero-residual-change control. The residual representations supply substantial causal information for agreement, pronouns, and sentiment, since donor other ports alone perform poorly there. Capital is largely explained by the other ports; its tiny candidate error must not be read as strong residual explanatory coverage. Binding shows an intermediate incremental contribution. This agrees with H1030's stream-level task differences and avoids mistaking native assistance for a learned explanation.

Five-family macro paired error0.0232activation versus0.0212difference, compared with0.4787/0.5006on excluded copying/spatial tasks inH1037. These panels differ in tasks, strength of effects, examples, and native port reliance, so this is a descriptive contrast rather than a controlled effect size of task coverage. Within-family causal transfer is nonetheless demonstrated on three residual-dependent families.

Whole-state reconstruction differs sharply: activation-basis whole KL0.0322, macro top187.5%; difference-basis whole KL0.7774, top120%. A representation useful for local causal changes need not encode enough background state to replace the whole residual. Mean held-out difference energy byfamily is59.4%activation and61.4%difference, again not a model-understanding score. Paired differences and whole-state representation should remain distinct featurizer objectives.

Verification: native saved logits replay exactly across all38endpoints; no fitted evaluation rows or exact prompt overlap; source hashes unchanged. Initial two-arm evaluation used190suffix calls; final three-arm rerun used266,456total. Independent saved-logit KL replay covers one endpoint per arm, three samples, with error below1e-10. The control was added after the initial results to resolve an interpretation confound; it was not prespecified, and no candidate was selected or trained from its outcome.

Decision: stop this basis-diagnostic series. There is enough same-family success to justify broadening supervised task coverage before another architecture search. Return to the requested loop: screen a diverse inventory of candidate tasks for native competence and aligned counterfactual pairs; freeze whole-family train/development/evaluation assignments before feature fitting; train residual task-change dictionaries plus context-preserving controls; compare scalar/block/shared support at matched capacity and intervention access. Include tasks beyond agreement/capital/binding/sentiment/pronouns, while preserving copying and spatial inversion as historical excluded diagnostics rather than silently folding them into training. New candidate families should span lexical relations, morphology, temporal relations, numerical relations, and symbolic transformations. Discovery remains unproven until an unlabeled component yields a new independently validated task.
