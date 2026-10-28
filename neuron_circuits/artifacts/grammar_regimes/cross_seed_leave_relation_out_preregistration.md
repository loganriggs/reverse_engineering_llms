# Preregistration: leave-one-relation-out cross-seed red team

The initial cross-seed assay found strong causal transfer after a global linear
alignment, but the alignment anchors included all tested relations.  This could
let the map learn a separate correspondence for every rule.  Its generic-answer
control was also deliberately difficult because source and target models use
different answer permutations.

For every ordered model pair and tested direct relation, this red team refits the
activation map using only anchor prompts from the other seven relations.  The map
is evaluated on new prompts from the omitted relation in unseen template 6.  The
omitted relation's semantic edit is then mapped and causally patched exactly as
before.  A layer is adjudicative only if every model-pair/relation alignment has
held-out R2 at least 0.30.

Controls are the target model's own semantic edit, a norm-matched wrong-rule edit,
the mapped generic source-answer edit, and—correcting the interpretation of the
first assay—the target model's own generic answer edit.  The source-answer control
tests label identity across incompatible codebooks; the target-answer control
tests whether the final causal effect is a generic output interface.

Passing establishes global-linear causal correspondence that generalizes to an
unseen rule family.  It does not prove identical algorithms or axes.  A failed
transfer with valid state alignment falsifies this global-linear version of
cross-model abstraction; failed alignment leaves nonlinear correspondence open.
