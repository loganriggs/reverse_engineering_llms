## H559 — Frozen subspaces retain substantial copying error on fresh lexical examples

Freeze H558's family/global PCA bases and all ranks. Test new name pairs William/Rachel, Patrick/Diana, Samuel/Julia and Dennis/Linda in the same two copying formats and binding template. Proposed color pairs are amber/beige, coral/ivory, crimson/scarlet and teal/navy. Before model evaluation, exclude pairs with unequal token lengths or invalid single-token continuations. Only coral/ivory survives among colors; all four name pairs survive. The resulting 28 constructed prompts produce 616 interventions with no refitting.

Names meet the existing competence gate: both copying formats are 8/8 candidate and full-vocabulary correct; binding is 8/8 candidate, 7/8 full-vocabulary. Color cells are 2/2 correct but have only one pair per format, below the four-pair gate. Their results are descriptive only. No outcome-based filtering or replacement of excluded pairs occurred. “Fresh” refers to these constructed prompts, not guaranteed absence from model training or all previous unrelated vocabulary use.

| Family, +1 amplitude | Family mean: response error / KL ratio | Family rank 7: error / KL ratio | Global rank 32: error / KL ratio |
|---|---|---|---|
|binding|21.8% / 0.035|20.4% / 0.034|19.4% / 0.031|
|color_prose|59.1% / 0.537|58.3% / 0.490|55.1% / 0.415|
|color_record|66.6% / 0.303|63.7% / 0.283|59.6% / 0.323|
|name_prose|50.7% / 0.572|47.8% / 0.458|45.4% / 0.409|
|name_record|69.0% / 1.121|66.6% / 1.006|63.4% / 0.883|

Binding retains the earlier pattern: a mean context gives 22% internal response error but only 3.5% of zero-response KL; 32 global directions give 19% error and 3.1% of baseline KL. Name copying is harder: global rank 32 leaves 45% prose and 63% record response error, with 41% and 88% of baseline KL. For record-style names, family mean KL is 1.12 times the zero-response baseline: adding an approximate response can be worse behaviorally than omitting it. Family rank 7 remains approximately tied with that baseline (1.006 ratio).

These results weaken a simple context-PCA canonical form for this local mechanism. They do not disprove a compact data-based representation: fitting only four lexical pairs may inadequately cover response-relevant directions, and PCA optimizes the context's variance rather than the response or downstream distribution. A KL-oriented approximation and an internally faithful approximation are different research targets; neither has been established here. The exact recipient displacement and quadratic term were retained, so even a successful context compression would not be a complete replacement model.

Next H560 should distinguish poor metric choice from insufficient coverage: compare context-variance and response-weighted subspaces on identical discovery data with fixed rank budgets, then test prospectively on new pairs. Avoid selecting a favorable rank from these evaluation results. The fresh color panel must be expanded with token-valid pairs chosen before inference before making color-transfer claims. Physical word-position and downstream-layer controls remain open branches; a local compression result would not resolve them.

Audits: exact replay max margin error 1.15e−5 and max KL 1.31e−11; source/basis unchanged; all finite; no refitting. H558–H559 complete 1,496 interventions and add 28 constructed prompts. No new FineWeb documents; total 26,112. Broad goal active and incomplete.
