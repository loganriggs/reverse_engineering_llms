# Preregistration: interaction order of the suffix-mask response surface

Each site has exact finite-effect fidelity and analytic objective values for all
84 three-layer masks. This experiment asks whether that combinatorial surface
has a short additive or pairwise description.

For each site independently, fit:

- additive features: nine layer-inclusion bits;
- pairwise features: the nine bits plus all 36 pair products.

Use five deterministic outer folds over masks. Ridge strength is chosen inside
each outer training set by three-fold validation from `[1e-4, 1e-2, 1, 100]`.
Report held-out R2 and the regret incurred when choosing the predicted-best mask
within each held-out fold. Repeat for exact finite fidelity and the analytic
full-Gram objective.

An additive grammar passes at median R2 >=0.70 and median regret <=0.05 for
finite fidelity on both corpora. A pairwise grammar passes at R2 >=0.80 and
regret <=0.03. Pairwise necessity additionally requires an R2 gain >=0.20 on
both corpora. Pairwise failure is declared if median R2 <0.50 or 90th-percentile
regret >0.10 anywhere.

These are local descriptive fits: each site supplies its own coefficients.
Success would identify interaction order, not yet a cross-context router.
Failure can reflect finite-scale nonlinearities or rare discontinuous program
changes, not necessarily literal third-order neural operations.
