## H972 — Earlier adjective-position MLPs create substantial local factorial interaction

Previous goal turn was progress: H971 confirmed that removing transported-value interaction reproduces attention5's full interaction-removal effect. This turn traces adjective and following-period representations through blocks0–5 and separates attention5 current values from its layer0 bypass values.

All96 locked states captured at each source position: block input, embedding skip, normalized attention input, attention output, normalized MLP input, MLP output, block output. The adjective alignment is its last subtoken; all competent familiar adjectives are single tokens. Failed-cohort multi-subtoken diagnostics cannot be interpreted as the sum of every adjective token's contribution. Period ID13 verified throughout. Final output logits replay exactly.

Exact bilinear decomposition uses x=c0+a*ca+n*cn+a*n*ci after MLP input normalization. Output interaction splits into inherited B(c0,ci)+B(ci,c0) and local B(ca,cn)+B(cn,ca). Below are signed local projections onto the total MLP output interaction on the sole native-competent cohort. These are algebraic shares, not fractions of task computation or causal necessity.

| Layer | Adjective local share | Period local share |
|---|---|---|
|0|95.62%|92.89%|
|1|78.46%|28.60%|
|2|48.60%|26.13%|
|3|58.42%|14.73%|
|4|13.74%|1.46%|
|5|5.90%|1.65%|

Unlike the mostly inherited interaction at the final token in late layers, earlier adjective-position MLPs have substantial local cross-product terms. Adjective MLP2 and MLP3 interaction norms average4358 and8379, respectively; local shares are not merely a ratio of negligible coefficients. Norm magnitudes themselves are not semantic-importance measures, and learned residual scales change their propagation.

The period increasingly receives or transforms existing interaction. Its attention4 interaction norm averages2639 and MLP4 interaction norm6179, with only1.46% of MLP4's interaction projection from local cross-products. This is consistent with a possible route from earlier adjective computation to a period representation, but no direct source-to-period mediation has yet been tested.

Attention5's layer0 value bypass has exactly zero factorial interaction at both aligned source positions on the competent cohort. Current-layer values carry the entire value interaction there; the value mixture independently replays exactly. This follows from contextual current values versus the context-free original token-value bypass, but does not imply bypass values have no effect on other attention output components. Both initial source-position residuals also have exactly zero factorial interaction.

Independent audits:96/96 whole-model forwards; native logits and value mixing replay exactly; residual recurrence relative error3.93e-8; bilinear interaction discrepancy relative to all output coefficients5.95e-7. Separate direct evaluations of actual double-precision MLP modules recover local and inherited components at allsix layers, both positions, allfour competent quartets:288 local evaluations, maximum relative error2.70e-8 against saved components. Aggregate shares independently checked; all outputs finite and inputs unchanged. No training or reserved-test access; natural-document pool26,112.

Limits: native trajectory algebra on a small, mostly familiar panel, with normalization included in inherited inputs and alternate cohorts failing competence. The adjective main factor captures lexical polarity differences and the not factor includes positional/context changes. Local interaction does not by itself establish semantic negation, a reusable operator, or a meaningful unsupervised manifold. H969's negligible MLP0 effect was at the final token and does not settle the effect at the adjective position.

Next: finite source-position removals of local versus inherited MLP interaction through layers0–4 at adjective and period, preserving downstream recomputation and including norm-matched controls. Measure both final behavior and whether attention5's transported value interaction changes, separating upstream effect from mere algebraic size. H961 frozen context-bridge predictor evaluation remains pending.
