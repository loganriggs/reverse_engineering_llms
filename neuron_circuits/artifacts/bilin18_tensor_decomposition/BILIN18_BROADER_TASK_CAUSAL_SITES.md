## H987 — New task anchors need different sites; merged dataset exposes duplicate coverage

Previous turn was progress: H986 screened ten additional task families and saved native outputs/all-layer MLP captures. This turn causally qualifies the two clean candidates and builds a concrete joint-featurizer seed dataset. No new fitting or context expansion yet.

Discovery-only scan of all18 last-token MLP writes for antonym and country-language, bidirectional donor replacement, identity and paired norm-matched Gaussian edits. Select maximum mean donor-directed answer-margin change on four discovery pairs/family. Freeze that choice, then evaluate all validation cohorts at both selected layer and fixed shared site MLP12. Same final token is required; all32 pairs satisfy it. Same-answer discovery controls are unavailable because these families use distinct answers; missing controls are not synthesized from validation.

| Family/cohort | MLP12 donor-directed margin | Selected-site margin | Selected site |
|---|---:|---:|---:|
| Antonym discovery |1.203|2.233|8|
| Antonym lexical |0.849|1.718|8|
| Country-language discovery |0.035|1.339|13|
| Country-language lexical |0.039|0.339|13|
| Country-language structural |0.026|0.734|13|
| Country-language joint |0.003|0.208|13|

Antonym alternate-wording cohorts fail the native full-vocabulary gate and remain diagnostics; their selected-site effects are0.539/0.710, not claims of competent semantic generalization. Country-language passes all native cohorts but its MLP12 effects are comparable to or below random-control RMS (~0.05–0.07). Antonym MLP12 discovery donor RMS1.230 versus paired random RMS0.481/0.525; lexical donor RMS0.925 versus0.459/0.344. Donor effects are not additive fractions of model computation or proof that a direction is uniquely semantic.

Consequence: country-language is useful for a layer13 causal benchmark, but weak as an MLP12 anchor. Adding more labels at one shared layer can fail simply because those tasks use different sites. Antonym is a clearer addition to the current common-site benchmark. Preserve this distinction before interpreting a joint-model failure as a lack of reusable structure.

Built bilin18_joint_task_dataset.pt plus manifest using a saved builder script.320 endpoint slots/160pairs across the previous eight evaluated families and two new families; all18 layer writes retained, with MLP12 extracted as the backward-compatible common site.64 discovery training slots across eight non-holdout families; copying/spatial remain excluded, as do role-binding/tense and new continent/object-function holdouts. No centering or fitting occurs in the merge.

The merge's initial unique-prompt assertion failed usefully:64 slots correspond to57 unique training prompts. Existing positive sentiment/negation examples overlap, and neighboring capital-recall pairs reuse countries. Earlier claims of48 unique training indices were true but did not imply48 unique prompts. Keep task-balanced slots while reporting actual coverage. Two capital-recall lexical-validation endpoints also exactly overlap training prompts. Manifest explicitly flags these endpoints and marks affected pairs ineligible for clean-validation reporting. Previous H983–H985 development scores are not recomputed here; their source-slot splits must not be reinterpreted as strictly prompt-disjoint. Broad task-holdout claims are unaffected by these capital overlaps.

Audits:1,552/1,552 full-model forward examples. Identity logits agree with H986 captures within3.00e-5; random norm relative error2.26e-7. Frozen source/selection hashes unchanged. Independent summary replay exact and discovery-only selected layers reproduced. All-layer merged tensor shape320x18x1152, MLP12 extraction checked exactly,64training slots/57unique prompts and two validation overlaps independently counted. Cached native captures are reused, not recaptured in this experiment. No featurizer training or reserved-test access; natural-document pool26,112.

Next: expand contexts and examples for site-appropriate anchors, with prompt-disjoint splits checked before evaluation. Keep common-site and task-specific-site questions distinct: first add antonyms to MLP12 coverage, retain country-language for a layer13 or explicitly layer-conditioned comparison. Do not silently pool coordinates from different layers as a single representation. The merged dataset is an integration artifact, not adequate training coverage or a successful bootstrap loop. Unlabeled task discovery and subsequent retraining remain outstanding; H961 context-bridge predictor evaluation remains pending.
