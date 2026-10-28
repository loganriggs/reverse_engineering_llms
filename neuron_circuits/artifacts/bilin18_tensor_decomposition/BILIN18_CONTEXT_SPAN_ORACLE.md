## H562 — Even a response oracle cannot reconstruct copying within the discovery span

Evaluate the 40 excluded H558 prompts and 40 fresh H561 prompts at +1/16 amplitude. Use every one of the 39 centered training-context directions. First project the recipient context normally. Then, separately for each recipient displacement, solve a float64 least-squares system using the exact target linear response to choose oracle coordinates in the same span. Keep the mean-context response and quadratic term exact. This measures the minimum attainable local response error under these constraints. The oracle is not a usable predictor: it is given the target response. No downstream intervention or KL is evaluated here.

| Fresh family | Full-span projection response error | Oracle response error |
|---|---:|---:|
|binding|17.9%|16.0%|
|color_prose|56.8%|46.7%|
|color_record|65.3%|54.2%|
|name_prose|39.7%|31.6%|
|name_record|56.5%|44.9%|

Even the oracle leaves 32–54% mean response error on fresh copying and 16% on binding. It improves the ordinary full-span projection but does not approach exact reconstruction. The old held-out panel gives a similar result, with copying oracle errors 31–45%. These values are relative vector-norm errors, not squared explained-variance fractions.

Therefore reweighting or learning better coefficients confined to this fixed discovery span cannot eliminate the remaining local vector error for these directions. Important response components lie outside the span of its induced outputs. This is a limitation of the sampled representation, not proof that the underlying mechanism is intrinsically high-dimensional or that all compact models must fail. A different span, more diverse contexts, token-conditioned structure, or a behavioral rather than internal objective may behave differently.

Audits: all 80 oracle solutions no worse than ordinary projection; maximum relative normal-equation residual 6.14e−16; finite results; source unchanged. This is 80 algebraic response diagnostics, not 80 causal interventions.

Next H563 should expand discovery-context coverage while keeping evaluation lexical pairs and rank budgets fixed. Compare coverage curves and oracle residuals to separate insufficient sampling from dimension requirements. Choose token-valid donor examples before inference, exclude evaluation vocabulary from discovery, and use an additional prospective test panel if making new generalization claims. Retain a task/behavior distinction and physical-position/downstream-layer questions; context compression remains only one branch of the feature/computation investigation.

H560–H562 completed 1,760 causal interventions plus 80 algebraic diagnostics and added 40 constructed prompts. No new FineWeb documents; tracked total 26,112. Broad goal active and incomplete.
