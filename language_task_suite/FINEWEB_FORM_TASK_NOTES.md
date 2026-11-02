# Natural-form clues for the next joint toy language

This is a **form-stratified spot check**, not a prevalence estimate. I decoded the first 72 tokens of selected rows from the frozen 480-window FineWeb panel used by the surface audits, choosing early rows containing quotes, questions, digits, or parentheses. The panel and tokenizer paths are in `audit_fineweb_surfaces.py`. Because windows can start mid-document, even apparently simple spans may omit the evidence needed to infer their true task.

| Panel rows | Observed form | Plausible computational demand to test in a controlled DGP |
|---|---|---|
| 4 | A craft-progress update gives completed and remaining quantities, followed by a named speaker's quoted phrase | Numeric state and attribution: distinguish a current total from a quoted claim and update the remainder |
| 7, 38 | Headings, table/list-like fragments, dates, and institution bullet points | Section/list scope: determine which statement belongs to which heading or numbered item |
| 14, 20, 22 | Interview or comment-thread questions embedded with answers and quoted replies | Speaker/source binding: attribute a claim to the right person and keep an embedded question distinct from its answer |
| 16, 23 | Quoted speech, self-correction, and a parenthetical speaker label | Quote and parenthesis scope, source attribution, and correction of an earlier claim |
| 35 | A parenthetical alternate-language name for the same entity | Alias resolution: reuse the entity while preserving surface-form and language context |

The current v5 joint stream has verb/object rule competition, object-to-place retrieval, and main-subject counting. It does **not** model source attribution, aliases, corrections, section scope, or quoted/parenthetical structure. Its document-form marginal frequencies and lexical repetition remain far from the panel. Counting a question mark or digit is an inadequate realism test: the *relationship* between text spans matters.

A next version should sample a small **document graph** before lexicalization: entities and aliases, events and numeric states, claims with source and time, and headings/list membership. Multiple surface templates should express the same graph, with incidental details and distractors. Useful exact-oracle prediction sites and causal tests include:

1. **Source-attributed claim:** Given a quoted or paraphrased claim, predict which speaker made it. Swap speaker labels while keeping claim content fixed; changing an unrelated speaker should have no effect.
2. **Correction/state update:** Predict the current value or location after a later correction. Swap only the corrected field; an earlier stale value must not control the answer.
3. **Alias binding:** Predict the same entity under an alternate name, with multiple aliases and distractors. Swap an alias mapping without changing the surrounding topic.
4. **Numeric remainder:** Predict a remaining amount after observed additions/subtractions. Edit one amount and compare exact answer changes; keep units and irrelevant numbers as controls.
5. **Nested span scope:** Predict quote/parenthesis closure or which clause a modifier belongs to. Hold out nesting depths and surface templates, not just random sequences.

These should be interleaved with narrative text and each other rather than selected by a single genre header. For each task, measure ordinary next-token fit, whole-family task transfer, and supported counterfactual effects separately. Before declaring the language realistic, annotate a new stratified natural panel for the actual presence and distance of these dependencies, then evaluate human readability and surface statistics on data **not used** to tune the generator. The current frozen panel can guide hypotheses but is already a development set.
