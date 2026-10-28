## H567 — First query factor carries most contextual gate correction

Reuse all 192 edited H566 prompts. At each edited destination position, hold the recipient source key fixed and independently choose base or edited context for each of the two query projections. Cross four cases: base/base, edited/base, base/edited and edited/edited. Recompute the bilinear coefficient and replace only that source's persistent-bus contribution at the final destination. Later computation and all other edges remain live. Two fixed heads give 1,536 interventions.

For base factors a,b and edited factors u,v, the coefficient change is lambda*((u−a)*b + a*(v−b) + (u−a)*(v−b)). These are exact algebraic terms, not independently identified semantic features. Cancellation can make individual absolute terms larger than the total change. This diagnostic holds the edited position fixed, so it isolates query-content changes relative to the phase-only prediction.

| Task/head | Neither query updated: coefficient MAE | First updated | Second updated | First-update live-margin MAE |
|---|---:|---:|---:|---:|
|binding/L8H3|0.01172|0.00500|0.00979|0.00470|
|binding/L14H4|0.25178|0.01748|0.23786|0.02099|
|copying/L8H3|0.02314|0.00730|0.02418|0.00209|
|copying/L14H4|0.02646|0.00238|0.02688|0.00171|

At L14H4, first-factor-only updates remove about 93% of binding coefficient error and 91% of copying error. Updating only the second factor removes little. L8H3 is less clean: first-only updates remove about 57% of binding and 68% of copying error, leaving a material second-factor contribution. The interaction term is small on average in these cases but is retained exactly in the accounting.

This supports asymmetric factor roles for the selected reader/edits, not a universal first-factor semantic label. It is not a complete query swap: source-specific coefficient interventions need not reproduce all effects of changing a query throughout attention. Original competence and all edit cases are retained.

Audits: native coefficient replay within 1.12e−7; exact algebraic decomposition within 1.39e−17; self-margin replay <=1.82e−5; all finite and source unchanged. No new prompts or corpus documents. Next H568 fits one mean raw query change per task/phrase/head on the first four lexical pairs, freezes it, and predicts gates on a separate name panel before edited-prompt evaluation.
