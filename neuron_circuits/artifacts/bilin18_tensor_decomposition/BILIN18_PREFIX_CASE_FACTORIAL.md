## H607 — Prefix and capitalization interact strongly before MLP11

The earlier prefix edit prepended "For the record," and lowercased the first character of the original sentence. This factorial separates those changes in the two copying formats. Hold names and answers fixed; cross prefix present/absent with initial uppercase/lowercase. Use eight name pairs and both endpoints in each format: 128 native prompts. All eight wording/prefix/case competence cells have both-endpoint candidate correctness and full-vocabulary intended-answer top1 of 1.000. This removes task incompetence as an explanation for the measured local changes.

Let z00 be the native normalized MLP11 input, z01 lowercase-only, z10 prefix-only retaining the initial uppercase character, and z11 both changes. Define:

- Case component C=z01-z00.
- Prefix component P=z10-z00.
- Interaction I=z11-z10-z01+z00.
- Total change D=z11-z00=C+P+I.

The interaction measures non-additivity of upstream context formation. It is not the quadratic-in-e term of MLP11's bilinear response expansion. The local response map T(dc,e) is linear in dc for fixed incoming e, so its three component responses sum exactly to the full response. Downstream behavioral effects need not add, and component norms are not causal percentage shares.

For each original uppercase/unprefixed recipient, inject equal-norm number/task/random Attention5 signals with both signs. Capture 192 incoming responses. Run full/unchanged, individual components, additive C+P, and matched cross-format component replacements using the same name/endpoint: 1,920 interventions. No fitting or selection occurs. The prefix-only strings retain an uppercase word after a comma; case changes tokenization/orthography and is not a clean semantic factor. Nevertheless, all cases preserve copying competence.

**The capitalization confound is material, but does not by itself explain the earlier prefix results.** Mean context-component norms relative to the total are:

| Wording | Case / total norm | Prefix / total norm | Interaction / total norm |
|---|---:|---:|---:|
|Old: He sent|0.6996|0.8123|1.0544|
|New: A letter was delivered|1.0560|0.9962|1.2319|

The interaction is as large as or larger than the total because the vectors partly cancel. Ignoring it gives substantial error. At the new wording, additive C+P has relative output KL 1.326–1.421: worse than leaving the entire context-switch contribution out. Prefix-only is closer (0.401–0.943) but still incomplete. Case-only is substantially worse (2.282–2.427). Thus the full effect cannot be explained by summing two independent edit effects.

| Wording | Signal | Case-only relative KL | Prefix-only relative KL | Additive C+P relative KL | Interaction-only relative KL | Full cross-format replacement relative KL |
|---|---|---:|---:|---:|---:|---:|
|0|number|1.2026|0.5666|1.2291|1.6342|0.4995|
|0|task|0.8955|0.4550|0.8202|1.3638|0.3609|
|0|random|0.9950|0.4805|0.9893|1.5021|0.6181|
|1|number|2.2823|0.4014|1.4200|2.1867|0.9178|
|1|task|2.4272|0.5074|1.4213|1.3541|1.0430|
|1|random|2.2936|0.9434|1.3264|1.6268|1.1605|

Cross-format geometry also depends on component identity. Mean old/new cosine similarity is about 0.697 for the total change, but only 0.487 for case, 0.408 for prefix, and 0.559 for interaction. The total can appear more stable than its constituent changes because of cancellation. Replacing one component with its counterpart from the other format produces substantial errors; the replacement is conditional on retaining the recipient's other two components. These are not additive attributions of the total transfer failure.

All relative KL/margin/CE errors are normalized against full-context-switch versus unchanged-context discrepancy. They measure fidelity to this conditional local intervention, not total task error. Full reconstruction is a reference, not a claim that the prefix should alter the copied answer. The actual incoming e remains observed. Raw internal, answer-margin and intended-token CE metrics are stored separately.

Interpretation: two simple textual factors interact non-additively in the representation reaching MLP11 even though the task answer remains unchanged and the model is fully competent. This gives a more concrete mechanism behind part of the earlier format dependence: context effects are jointly formed, not necessarily sums of reusable edit directions. It neither establishes universal feature interactions nor rules out a different coordinate system that exposes simpler structure. Prefix content, added length/positions, orthography and tokenization remain entangled within these controlled factors.

Next locate where this interaction emerges and which computations carry it. Trace the four-condition difference through attention and MLP writes across layers; distinguish a large upstream interaction from downstream amplification or cancellation. Then intervene on candidate writes to test whether reducing that interaction changes the later context-switch response. Length and verb-construction controls remain useful, but locating this now-verified interaction has higher immediate information value than treating the combined edit as a single feature.

Audits: all eight competence cells perfect; 128 native prompts, 192 perturbed captures, 1,920 interventions; exact unchanged-response logit replay; component response sum error <=7.67e-7 relative; runner and manifest unchanged; all metrics finite. No new natural-text documents; FineWeb remains 26,112. Broad goal remains active and incomplete.
