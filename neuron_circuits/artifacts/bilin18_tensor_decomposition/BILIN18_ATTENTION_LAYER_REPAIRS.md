## H661 — Layer14 concentrates the unresolved month-succession attention error

H660 found that matching wording largely removes prediction error in five families, while month succession retains substantial attention error. H661 localizes that error without fitting new parameters. Keep the second-wording discovery means fixed. For every family and held-out prompt, either retain predicted boundary state or use the exact donor boundary. Under each boundary condition evaluate18 attention configurations: all eight predicted, all exact, eight single exact repairs, and eight conditions leaving only one layer predicted. MLPs compute naturally. All96 held-out prompts are retained.

Each repaired layer overwrites its final-position attention output with the natural task prompt's value. Unrepaired layers use native copying attention plus the fixed second-wording discovery mean, rather than recomputing attention from edited states. This deliberately isolates attention-write approximation errors. Repairs use held-out donor information and are diagnostic, not predictions. Layers10–17 are tested exhaustively; the best layer in the tables is selected descriptively after evaluation, not prospectively validated.

**With predicted boundary state, KL(task||edited):**

| Family | All attention predicted | Best single exact layer | KL after repair | Worst sole predicted layer | KL with only that layer predicted | All attention exact |
|---|---:|---:|---:|---:|---:|---:|
|addition|0.00795|14|0.00486|14|0.00658|0.00390|
|agreement|0.00529|11|0.00390|11|0.00327|0.00225|
|antonym|0.01121|13|0.00885|15|0.00838|0.00765|
|capital|0.01752|14|0.00860|14|0.00743|0.00336|
|plural|0.01084|13|0.00790|13|0.00706|0.00595|
|successor|0.08653|14|0.02195|14|0.04142|0.01289|

**With exact boundary state, isolating attention prediction:**

| Family | All predicted | Best single exact layer | KL after repair | Worst sole predicted layer | Its KL |
|---|---:|---:|---:|---:|---:|
|addition|0.00167|14|0.00032|14|0.00091|
|agreement|0.00171|11|0.00105|11|0.00032|
|antonym|0.00978|13|0.00475|13|0.00168|
|capital|0.01314|14|0.00390|14|0.00423|
|plural|0.00737|14|0.00355|14|0.00179|
|successor|0.08167|14|0.01404|14|0.03351|

**Month-succession detail:**

| Layer | Single exact repair, predicted boundary | Sole predicted layer, predicted boundary | Single exact repair, exact boundary | Sole predicted layer, exact boundary |
|---|---:|---:|---:|---:|
|10|0.08311|0.01152|0.07705|0.00028|
|11|0.08081|0.01559|0.07726|0.00129|
|12|0.08627|0.01348|0.08201|0.00003|
|13|0.08126|0.01565|0.07866|0.00067|
|14|0.02195|0.04142|0.01404|0.03351|
|15|0.07651|0.01179|0.07070|0.00052|
|16|0.06114|0.01166|0.05194|0.00475|
|17|0.08474|0.01269|0.07968|0.00008|

Layer14 is the strongest month-succession repair under both boundary conditions and the most damaging layer to leave predicted. With predicted boundary, repairing only14 reduces KL0.08653→0.02195; all exact attention gives0.01289. With exact boundary, layer14 alone reduces0.08167→0.01404, while leaving only14 predicted gives0.03351. Layer16 also has a notable single-repair effect. A single repair does not recover the full task output, and these improvements must not be interpreted as additive causal shares.

The pattern is partially shared across tasks: layer14 is also the strongest single repair for addition and capitals under both boundary choices, and for plurals with exact boundary. Agreement instead favors11 and antonyms13; plural ranking changes with boundary choice. This does not establish a universal layer14 semantic unit. It identifies a promising location for the residual error of this particular conditional mean approximation.

Some exact repairs worsen KL or a leave-one-predicted condition beats all-exact attention with predicted boundary, reflecting compensation between boundary and attention errors. Therefore geometric residual magnitude, individual repair benefit and necessity need not agree. Both single-repair and leave-one controls are useful; neither alone defines a minimal circuit.

Native task competence limitations remain: output KL targets a possibly incorrect natural task distribution. The current audit localizes approximation error, not successful task computation. No model weights, mean coefficients, amplitudes or layer subsets were optimized on these outputs. The original model and native receiver baselines remain required.

Next test whether layer14 residual variation is largely discovery-coverage error. Compare the original first-four-item mean with a fixed balanced four-item scheme and a leave-one-item-out seven-item mean, always excluding the evaluated item and retaining all forms of that item together. Keep the other layers and boundary predictions fixed so the change is attributable to layer14 prediction. The month task's January–April discovery versus May–August evaluation is ordered and small; this control should precede a claim of irreducible month-specific features. Report the new schemes as retrospective cross-validation, not blind replication. If error persists, layer14 head/write decomposition becomes a better-targeted next mechanistic experiment.

Audits:3456 whole-model forwards (2 boundary conditions×18 attention configurations×96 prompts),768 cached native attention-stage evaluations. All four all-predicted/all-exact baseline arrays reproduce H660 logits identically. Source, mean, exact-reference and runner hashes unchanged; KL metrics finite. All six families and all cases retained. FineWeb remains26,112 unique documents. Previous goal turn was progress. Broad research goal active and incomplete.
