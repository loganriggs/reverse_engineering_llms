## H1026 — Both distractor identity and contrast wording modulate routing

Previous turn was progress: H1025 completed exact query/key/rotary recombination and identified the wording-versus-competition confound. This experiment tests that confound without fitting a gate.

Panel preregistered before model evaluation: original four discovery country pairs, two endpoints; sixteen one-token distractors (countries Canada/Brazil/Mexico/India; cities London/Paris/Boston/Berlin; people Alice/Peter/David/Mary; abstract nouns music/science/poetry/history); three templates plus eight baseline controls. Total392 prompts. Within each template all distractor replacements preserve source and destination positions and token count. Every paired target-country example differs only at the source token. These constructed combinations are a new diagnostic panel on familiar discovery targets, not a sealed evaluation set or expansion of the natural-text corpus.

Templates: 'Unlike X, the main language spoken in COUNTRY is'; 'The main language spoken in COUNTRY, rather than X, is'; 'The guide mentions X. The main language spoken in COUNTRY is'. Contrast is not perfectly matched to neutral wording in length/positions, so across-template differences are effects of the whole template, not isolated contrast semantics.

Baseline signed head13.0 final-query/source-key coefficient mean is −0.673823. Positive shifts below mean reduced magnitude. Each cell averages four distractor words and eight target endpoints; repeated targets are not independent observations.

| Template | Countries | Cities | People | Abstract nouns |
|---|---:|---:|---:|---:|
| Unlike X |+0.0972|+0.0776|+0.0650|+0.0674|
| Rather than X |+0.1008|+0.0802|+0.0552|+0.0904|
| Guide mentions X |+0.0729|+0.0531|+0.0241|+0.0217|

Country distractors have larger mean shifts than other classes in all three templates, but the category separation is incomplete. In the neutral mention template, city word means span coefficients −0.656 to−0.586 and country means −0.640 to−0.565, overlapping each other and parts of the non-geographic ranges. In the 'rather than' template, abstract-noun shifts nearly match country shifts. Thus neither a category-only gate nor a wording-only gate accounts for these observations. Geographic associations and discourse context are both plausible contributors; this panel does not identify their internal features.

Competence must constrain interpretation. Both-endpoint binary accuracy is100% in every class/template except country/rather-than (93.75%). Full-vocabulary intended-answer top1 rates:

| Template | Countries | Cities | People | Abstract nouns |
|---|---:|---:|---:|---:|
| Unlike X |71.9%|87.5%|87.5%|87.5%|
| Rather than X |40.6%|81.3%|87.5%|84.4%|
| Guide mentions X |59.4%|84.4%|93.8%|93.8%|

Country/rather-than falls below H1024's50%top1 competence threshold and must remain a diagnostic failure condition, not a positive generalizing task demonstration. No examples were excluded or tuned after observing these results. Binary margins alone would hide the substantial full-vocabulary failures. Score attenuation and answer errors covary, but this observational panel does not establish that this one edge causes those errors.

Interpretation for representation research: a useful computational representation likely needs a continuous context gate that combines lexical associations and surrounding structure. The data do not warrant supervising a single binary 'competitor country present' feature. Native query values still come from the full prefix; this is not an executable query predictor or evidence for a discovered manifold. The fixed lexical value map remains a separate candidate reusable operator.

Limits: only four target pairs, four words per class, no frequency match; abstract nouns differ in capitalization; some unlike/rather-than non-country combinations are semantically awkward. Cities have country and language associations. Across-template positions differ. These are explicit reasons to avoid a clean taxonomy or universal mechanism claim.

Audits:392prefix and392suffix sequences; fixed positions within template; all endpoint pairs differ only at target source; independent reduction of all summary cells from saved outputs; baseline coefficients replay H1025 within1.11e-16; all finite; zero fits and zero reservedBSFtest rows. Saved prompts, all model outputs, two constituent QK scores, coefficients and per-word ranges. Natural-document pool26112unchanged. Frozen lexical program unchanged, no process remains live.

Next: finite edge intervention on the neutral-mention panel. Substitute the baseline routing coefficient only for the target source's cached head13.0 contribution to the final token, retain live routing elsewhere, and test whether this restores target-answer margins or merely changes an incidental score. Include equal-size control writes and distinguish target improvement from recovering the whole-model baseline. Freeze the assay before selecting examples; retain low-competence conditions explicitly. Afterwards test a distinct task sharing this operator to return to the multitask featurizer/reuse question rather than extending a single-task gate indefinitely. Broad goal remains active.
