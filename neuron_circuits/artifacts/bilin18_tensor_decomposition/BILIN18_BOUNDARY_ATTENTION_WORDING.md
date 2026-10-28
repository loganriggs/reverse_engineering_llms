## H660 — Wording mismatch explains most mean-prediction error except month succession

H659 suggested attention prediction was the larger conditional limitation of original-wording mean edits. H660 distinguishes wording transfer from residual item variation. Fit new means from only the four discovery items under the second wording, using the same family/form groups, copy-to-task changes and receiver-native baselines. Evaluate the same four held-out items; no held-out activation or output enters either fit.

Cross original-wording, new-wording and exact boundary states with original-wording, new-wording and exact attention writes: nine conditions. Add native/native, native/exact and exact/native as replay references, totaling12. The configuration's inherited H659 prose mentions other-task/live-additive conditions; an explicit design clarification records that those were not rerun here. The condition records and tables below describe the actual experiment.

Predicted boundary and attention writes replace the native copying values plus discovery means at block9 and attention10–17; MLPs compute naturally. Exact controls use donor task values. Native attention reference conditions recompute from evolving states. All96 transfer prompts remain. This is a retrospective diagnostic with independent discovery fitting, not a new blind panel.

**KL(task||edited), nats:** O=original-wording mean; N=new-wording mean; E=exact donor value. Lower is closer to the natural task distribution.

| Family | O boundary/O attention | O/N | N/O | N/N | N/E | E/N |
|---|---:|---:|---:|---:|---:|---:|
|addition|0.04057|0.01524|0.02394|0.00795|0.00390|0.00167|
|agreement|0.05855|0.01470|0.03491|0.00529|0.00225|0.00171|
|antonym|0.10863|0.04180|0.04822|0.01121|0.00765|0.00978|
|capital|0.06047|0.03424|0.03188|0.01752|0.00336|0.01314|
|plural|0.06744|0.02761|0.02666|0.01084|0.00595|0.00737|
|successor|0.09690|0.10245|0.09048|0.08653|0.01289|0.08167|

Matching both components to the evaluation wording reduces KL sharply in addition, agreement, antonyms, capitals and plurals, to0.0053–0.0175. Antonyms improve from0.1086 to0.0112, reversing the previous weak cross-wording behavior without adding item-conditioned coefficients. Thus their cross-wording failure was not evidence that a within-wording mean lacked useful structure.

Month succession differs: O/O0.0969 becomes N/N0.0865, only a modest improvement. With exact boundary state, new-wording attention still leaves KL0.0817. With new-wording boundary and exact attention, KL falls to0.0129. This isolates residual attention variation as the stronger conditional issue for this family under the four-item discovery design. It does not prove intrinsic item dependence rather than limited discovery coverage: the original split uses January–April for discovery and May–August for transfer. That ordered split and small sample must be considered before claiming a fundamental month-specific representation.

The attention-priority conclusion from H659 therefore needs qualification. Much of its attention error is removable by matching wording, and after wording matching the relative importance of the two components varies. Addition/agreement have more remaining error with predicted boundary plus exact attention than the reverse; capitals and succession have the opposite pattern. Plural and antonym residuals are closer. These are conditional KL comparisons with nonlinear interactions, not additive causal shares.

**Target loss remains separate:**

| Family | O/O target CE | N/N target CE | N/E target CE | E/N target CE |
|---|---:|---:|---:|---:|
|addition|3.747|3.681|3.694|3.731|
|agreement|8.110|8.224|8.237|8.189|
|antonym|5.304|5.592|5.463|5.735|
|capital|5.569|4.933|4.993|4.924|
|plural|5.695|5.224|5.257|5.170|
|successor|2.777|2.627|2.333|2.509|

Low output KL is fidelity to a possibly incorrect task-prompt distribution, not evidence of task execution. Native candidate and exact correctness limitations from prior held-out panels still apply; all runs retain the original model and task/form labels. No feature dictionary, unique canonical form or compressed end-to-end model has been established. The practical description is a wording-conditioned mean transformation with uneven residual variation across operations.

Next distinguish where the remaining attention error acts from whether the discovery items cover it. First use single-layer exact-attention repairs and leave-one-layer-predicted controls with the new-wording mean baseline, retaining all six families. That can show whether a small subset of attention writes dominates the remaining output error or whether errors are distributed. For succession, follow with balanced or leave-item-out discovery splits to test coverage; do not equate the ordered four-item split's failure with irreducible complexity. Avoid adding a large predictor before these discriminating tests.

Audits:1152 whole-model forwards for12×96 prompts;3840 cached attention-stage evaluations across original discovery, new discovery and receiver baselines. All previous original-mean prediction arrays reproduce exactly; all four exact/native H658 reference arrays reproduce exactly. Runtime hashes unchanged and metrics finite. No held-out fitting or exclusions. Metadata clarification added after execution solely to disambiguate inherited prose; numerical results and executed source are unchanged. FineWeb remains26,112 unique documents. Previous goal turn was progress. Broad research goal active and incomplete.
