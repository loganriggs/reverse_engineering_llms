# Preregistration: cross-seed causal invariance of grammar computations

The within-model assays found that semantic activation edits increasingly resemble
a generic answer-output edit.  This assay asks whether there is nevertheless a
reusable rule computation across independently trained models.

The three pure-rule bilinear models implement the same input grammar, but have
different initializations, data seeds, and random answer-label permutations.  For
each ordered source/target pair, fit one affine ridge map per layer from
post-product MLP activations on shared, unlabeled anchor prompts.  Anchor prompts
use training template combinations and are unrelated to the counterfactual edit
panel.  Freeze the map and report its reconstruction R2 on new prompts in held-out
template 6.

For each direct rule/value transition, estimate a mean semantic activation edit in
the source model, map its *difference vector* into the target coordinate system,
and add it to unseen target-model recipients in template 6.  Compare its complete
softmax-Fisher logit effect and target accuracy against:

1. the target model's own semantic edit (ceiling);
2. a norm-matched mapped edit for the wrong grammar transition;
3. a mapped generic source answer-class edit; and
4. the unpatched target model.

The alignment uses no answer labels, counterfactual pairs, or intervention
outcomes.  A causal null result is adjudicative only at layers whose held-out
activation R2 passes the frozen 0.50 gate in every ordered model pair.  Passing
would show a linearly alignable causal equivalence class, not identical neurons.
Failure with a valid alignment would falsify this particular global-linear
invariance hypothesis; it would not exclude nonlinear or context-conditional
correspondence.

The exact hypotheses and thresholds are stored in
`configs/grammar_cross_seed_semantic_invariance.json` before running the assay.
