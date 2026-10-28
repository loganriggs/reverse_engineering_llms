## H663 — Layer14 head4 concentrates month-succession mean-prediction error

H662 showed that balanced-index and seven-item discovery means only modestly reduce layer14 error. H663 decomposes layer14 into its nine projected head writes (head indices0–8). Capture each head's final-position vector immediately before the attention output projection and multiply by that head's corresponding projection columns. The projection has no bias. These writes sum to the native attention output within floating-point tolerance.

For each family/form, fit each head's original four-discovery-item mean copy→task shift under the second wording. For each held-out item, define the head residual as true task-head write minus native copying-head write minus that head's discovery mean. Add selected exact residuals to the existing layer14 mean prediction. Other attention layers stay predicted. Evaluate all predicted, all nine heads repaired, nine single-head repairs, and nine leave-one-head-predicted conditions, under both predicted and exact boundary states. All96 held-out cases remain, for40 forward conditions.

Repairs use true held-out donor information and localize approximation error; they are not fitted predictors. The head basis is defined by architecture, not a newly discovered unique semantic feature basis. Single-head rankings are descriptive and need independent validation.

**Predicted boundary, KL(task||edited):**

| Family | No head repair | Best single head | KL after repair | Worst sole unrepaired head | KL with only it unrepaired | All9 heads repaired |
|---|---:|---:|---:|---:|---:|---:|
|addition|0.00795|7|0.00553|7|0.00702|0.00486|
|agreement|0.00529|4|0.00516|4|0.00516|0.00505|
|antonym|0.01121|4|0.00985|4|0.00984|0.00885|
|capital|0.01752|4|0.00998|4|0.01497|0.00860|
|plural|0.01084|7|0.00985|7|0.00940|0.00874|
|successor|0.08653|4|0.03208|4|0.07175|0.02195|

**Exact boundary:**

| Family | No repair | Best single head | KL after repair | Worst sole unrepaired head | Its KL | All9 repaired |
|---|---:|---:|---:|---:|---:|---:|
|addition|0.00167|7|0.00053|7|0.00117|0.00032|
|agreement|0.00171|4|0.00160|4|0.00147|0.00138|
|antonym|0.00978|4|0.00756|4|0.00720|0.00536|
|capital|0.01314|4|0.00581|4|0.00998|0.00390|
|plural|0.00737|4|0.00522|4|0.00511|0.00355|
|successor|0.08167|4|0.02185|4|0.06942|0.01404|

Month succession is strongly concentrated in head4 under both boundary choices. With predicted boundary, a single head4 repair reduces KL0.08653→0.03208; all nine repairs give0.02195. Leaving only head4 unrepaired gives0.07175. With exact boundary, single-head4 repair reduces0.08167→0.02185, while leaving only4 unrepaired gives0.06942. Other heads contribute and interactions remain; these improvements are not additive causal shares or proof that head4 alone computes succession.

Head4 also ranks highest for capitals, antonyms and agreement within layer14. Addition instead favors head7. Plural ranking depends on boundary state (head7 predicted boundary, head4 exact boundary). Agreement's total layer14 repair benefit is small because earlier layer-level tests favored layer11. A shared head-level error pattern does not make head4 a month-specific semantic unit; it may perform a broader contextual routing or answer-format function.

### Refreshed twenty-hypothesis tournament

These are qualitative priorities, not posterior probabilities, and several can coexist.

| # | Hypothesis | Evidence and next discriminator |
|---|---|---|
|1|Head4 routes input-token information differently across items|Leading testable account of concentrated residual; partition write contributions by input versus instruction tokens.|
|2|Head4's instruction-token routing changes with item identity|Also plausible; the same partition can distinguish it from input-token routing.|
|3|Query-dependent coefficients are the main source of variation|Test coefficient swaps with token values held fixed, using exact reconstruction.|
|4|Projected value content rather than routing explains residuals|Compare value and coefficient contributions at aligned input positions.|
|5|A shared instruction-conditioned mean plus small content-dependent correction suffices|Strong within-wording means for many tasks; fit a low-dimensional head4 residual only after identifying its source.|
|6|Wording transfer remains the main global limitation|H660 strongly supports five families; require new independent phrasings for generality.|
|7|Month succession fails because discovery coverage is narrow|H662 weakens coverage-only explanation but does not rule out larger representative data.|
|8|Head4 reflects calendar-item structure rather than succession computation|Possible despite low task competence; compare copying and other month operations on identical months.|
|9|Head4 serves broad factual/relational output processing|Capital and antonym patterns support exploring shared function; compare position-level writes across families.|
|10|Output formatting dominates head4 repairs|Open because exact task output is often incorrect; compare input-token and suffix-token contributions.|
|11|Head7 carries a distinct numerical component|Addition repair ranking motivates tests, not a label; compare arithmetic with number-copy controls.|
|12|Boundary and attention errors compensate|Supported by differing rankings and nonmonotonic repairs; retain both boundary controls.|
|13|A large geometric residual can be functionally inert|Prior geometry/KL differences support; assess projected writes against finite output effects.|
|14|A small residual aligned with sensitive directions causes most KL|Open; compare output-relevant directions with raw head residual norms.|
|15|Remaining error is distributed within a head rather than across heads|Head4 concentration permits testing low-rank residuals inside its128-dimensional output channel.|
|16|Several heads have redundant or opposing effects|Leave-one and single-head asymmetries suggest checking combinations before declaring minimality.|
|17|Exact attention transplants hide complex upstream computation|Certainly a limitation of these controls; predictors must avoid donor lookup and report full dependencies.|
|18|Mean output-write prediction is useful but not an intrinsic feature system|Consistent with wording/task conditioning; test independent datasets and transformations.|
|19|Output-KL equivalence differs materially from loss equivalence|H657–659 show concrete ranking reversals; retain distinct objectives.|
|20|A dataset-relative canonical description may require conditional units rather than fixed global vectors|Still plausible, not established; compare conditional head-write descriptions with fixed bases on prospective data.|

Next reconstruct head4 as a sum of token-position contributions, with explicit tokenizer alignment to the input, instruction and answer suffix. Verify that those contributions reproduce the native head write before patching them. Then repair role-specific residuals to distinguish content routing from instruction/format routing. Only after that should query/key versus value swaps be interpreted, because unequal prompt lengths make arbitrary positional swaps misleading. Retain all six families, task-competence caveats and independent fidelity/loss metrics.

No claim of successful task execution follows from repairing a natural task distribution. This remains a donor-informed audit of a conditional approximation; the original model and native receiver baseline are retained. The broad question of fundamental computational units remains open.

Audits:3840 whole-model forwards (2 boundary settings×20 head configurations×96 cases),1152 cached attention-stage evaluations. Projected head sums match native layer14 outputs with maximum relative error2.66e-7. Sum of head residuals matches the existing exact-minus-predicted layer14 residual within1.32e-5 relative error. Unrepaired baseline logits reproduce exactly; all-nine-head repairs reproduce exact-layer14 reference logits within8.11e-6. All source/runner/mean hashes unchanged; KL finite. All cases retained. FineWeb remains26,112 unique documents. Previous goal turn was progress. Broad research goal active and incomplete.
