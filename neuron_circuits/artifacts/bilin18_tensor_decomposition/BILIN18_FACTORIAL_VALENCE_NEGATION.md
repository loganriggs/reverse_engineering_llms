## H966 — Frozen sentiment direction primarily carries composed valence on a competent factorial panel

Previous goal turn was progress: H965 established separate MLP12 directions with sentiment/negation overlap, then identified shared answer groups as a confound. H966 tests8 adjective pairs x3 prompt/output contexts x4 states: positive/negative adjective crossed with absent/present not. H965 sentiment and negation directions remain frozen. All12 directed edges per quartet tested with full-write, identity, each frozen projection and its complement. Coefficients still use true donor deltas; this is span adequacy, not encoder transfer.

Native gate: at least4 quartets, >=75% quartets with allfour binary answers correct, >=50% full-vocabulary intended-group top1. Only original review format A with the four familiar adjective pairs passes (100% allfour binary,75% full-vocabulary). New-to-H965 adjectives in A get50% allfour correct and50% full-vocabulary, failing the quartet gate. Failures are negated negative adjectives: not abysmal and not dreadful still have slightly negative task margins. B (meal, Overall sentiment) and C (performance, The review is) both fail in both adjective cohorts. All96 states and failures retained. Twelve prompts deliberately overlap H962/H965 source data; the four newly introduced not-negative states within the competent cohort supply the previously missing factorial corner. This is not a wholly unseen-context generalization result. Binary reversal for not-negative expressions is an operational expectation; natural language can permit neutral or attenuated interpretations.

For each quartet let a=+1/-1 be lexical adjective polarity, n=+1/-1 indicate plain/not. Exact four-state decomposition is mean + a*C_a + n*C_n + a*n*C_an. Under ideal binary reversal, a*n corresponds to resulting valence; C_n instead tracks a polarity-independent effect of not. The decomposition is descriptive across prompt interventions and does not locate where the interaction is computed.

On the competent familiar-A cohort:
- Mean fraction of centered full-write squared norm: lexical adjective13.2%, not-presence18.4%, interaction68.4%.
- Frozen sentiment coordinate pooled squared factor coefficients: adjective8.0%, not-presence1.7%, interaction90.4%.
- Frozen negation coordinate: adjective4.4%, not-presence9.4%, interaction86.2%.
- Native answer-margin mean coefficients: adjective0.273, not-presence0.128, interaction1.138.

These fractions are descriptive squared-coordinate/squared-write decompositions, not percentages of causal task computation. The frozen direction mostly carries composed valence in this setting, rather than a context-independent not marker.

| Edge type, competent cohort | Full-write margin-change RMS | Sentiment-projection RMS | Sentiment-projection KL ratio | Negation-projection KL ratio |
|---|---|---|---|---|
| Change adjective only |0.839|0.750|0.056|0.079|
| Toggle not only |0.847|0.731|0.080|0.097|
| Change both (same ideal valence) |0.174|0.251|0.580|0.558|

KL ratio is summed KL(full-write donor patch||candidate) over summed KL(full-write donor patch||native). Zero is exact full-write fidelity; one is no edit. Crossed both-factor changes keep a*n fixed, exposing lexical/not-specific effects and incomplete cancellation: their fidelity is much worse, and projected margin RMS can exceed the full-write RMS. Hence a single composed-valence direction is useful but not a complete four-state causal representation. In failed cohorts, projections still sometimes track actual model patch effects, but cannot establish correct semantic transfer. All diagnostics retained in JSON.

Audits:1,824/1,824 whole-model forward examples; frozen inputs/directions/manifest unchanged; identity logit max difference2.10e-5; exact factorial write reconstruction; independently recomputed NumPy factors and aggregate KL ratios; outputs finite. No training, whole-task-holdout examples, or reserved-test evaluations. Natural-document pool26,112.

Interpretation: evidence now favors a shared composed-sentiment result direction over a standalone reusable negation operator at the output of MLP12. It remains unresolved whether that block computes the composition or converts already-composed upstream input into an output direction. Native format fragility sharply limits universality claims.

Next: capture pre-MLP12 inputs on the locked quartet panel. Decompose the bilinear MLP's factorial output interaction into inherited input interaction and products of input adjective/negation components, including the model's actual normalization and bias. Independently replay exact outputs before interpreting the decomposition. Keep the complete failed cohorts as diagnostics. H961 frozen context-bridge predictor evaluation remains pending.
