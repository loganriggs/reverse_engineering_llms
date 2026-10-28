## H970 — Attention5 primarily transports composed interaction from adjective and period positions

Previous goal turn was progress: H969 finite interaction removals identified attention5 as the strongest earlier final-token site. This turn reconstructs attention5 by source token and head, then factorizes each source contribution into routing weights and projected mixed values. All96 locked states retained; only familiar original-format quartets are used for interpretation/ranking because other cohorts fail native competence.

Architecture check: this checkpoint uses CausalBilinearSelfAttention with two normalized/rotated QK pairs. Its routing weight is the product of their dot products divided by head_dim squared, without row normalization or softmax. These weights can be signed and are not probabilities. Values mix current-layer projections with the native layer0 value bus before head output projection. An initial assertion expecting the single-QK variant failed before any forward evaluation; the implementation was corrected against the checkpoint config and actual model source. No results from the incorrect variant were produced.

Each source/head contribution is P*V, where P is a scalar routing weight and V the mixed value projected into residual space. With four-state coefficients indexed by constant, adjective a, negation n and interaction an, the exact interaction is:

P0*V_an + P_an*V0 + P_a*V_n + P_n*V_a.

Call these transported value interaction, routing interaction, and cross terms. This is an algebraic decomposition on native states, not a causal mediation result. In particular V_an already exists before attention5 aggregation, but could have been computed at any earlier layer or introduced through input normalization.

Competent-cohort signed projection onto the total attention5 interaction:
- Value interaction at common aligned token slots:96.81%.
- Routing interaction:3.24% common slots plus0.030% noncommon slots.
- Cross terms:-0.112% common slots plus0.030% noncommon slots.

Source-role shares: adjective50.06%, suffix48.41%, prefix1.48%, explicit not token0.060%. The period immediately after the adjective supplies44.83% by itself; later sentiment-label tokens supply the remaining suffix contribution. A tiny direct not-token share does not mean not is unimportant: it can alter representations at later positions. Head4 supplies37.55%, head7 supplies32.33%, head6 supplies13.57%; these are native-trajectory projection shares, not intervention effects or attention probabilities. Indices zero-based.

Token alignment uses character offsets and prefix/negation/adjective/suffix role plus within-role token index. Missing slots receive zero P and V; factor allocation at such slots depends on that convention. Report common versus noncommon separately. On the competent cohort all adjective positions are single tokens, and noncommon slots contribute only0.060% of total signed projection. Thus the main transported-value conclusion comes from common slots, not the missing-token padding rule. All aligned and unaligned source contributions are saved for independent replay.

Audit:96/96 whole-model forward examples; native logits replay exactly. Reconstructed final-token attention outputs differ by at most3.06e-7 relative; full interaction differs from the prior float32 capture by6.55e-6 relative. Product factorial identity holds to1.32e-15. Independent NumPy contraction over original unaligned source positions reproduces the saved interaction within3.44e-15, and summary aggregation matches. Inputs unchanged, outputs finite. No training or reserved-test access; natural-document pool26,112.

Interpretation: attention5 appears primarily to gather already-composed content, prominently from the adjective and following period. Together with MLP12's inherited-input result, this moves the search for original composition upstream and to earlier token positions. Neither attention5 nor MLP12 has yet been shown to implement a reusable negation operator.

Next: finite selective removals of transported-value versus routing/cross interaction at attention5, with role/head removals and norm-matched controls. Then trace validated source-position interaction through earlier blocks rather than continuing to inspect the final token alone. H961 frozen context-bridge predictor evaluation remains pending.
