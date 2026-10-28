## H954 — Complete factorial number panel passes native competence

Seven fixed noun pairs × two templates × four subject/distractor number combinations =56 sentences,14 quartets. Templates: “The [subject] with the [guard/guards] near the window” and “The [subject] that the [guard/guards] noticed near the window.” The invariant embedded past-tense verb repairs H952's token-alignment limitation without choosing replacements from model outputs. Every quartet yields all12 directed swaps, labeled subject-only, distractor-only, or both. All168 swaps are at the same preMLP4 relay token/position. No exclusions.

Native competence: all four binary answers correct in every quartet for both templates. Intended full-vocabulary top1 is53.6% prepositional and100% relative clause. Both pass the predeclared >=4quartets, >=75% all-four-correct, >=50% full-vocabulary gate. Thus relative clauses now have a valid test, and failures of frozen featurizers can be assessed on demonstrated teacher capability.

280 whole-model forwards (native,identity,and exact relay swaps),336 local MLP endpoint evaluations. Checkpoint and manifest hashes unchanged, identity replay exact, finite effects. Independent audit confirms all12 directed swaps per quartet, correct factor labels and relay alignment, and donor reconstruction relative norm error9.62e-9. An initial absolute1e-4 check failed at1.2207e-4; the scaled check is below float32 epsilon and downstream margin replay independently validates the intervention. No training or reserved-test access; natural document pool26,112.

Scope: this is a prospective factorial follow-up, not a wholly new lexical sample. Some prepositional endpoints intentionally repeat H952. Full subject/distractor factors are separately manipulated here for the first time in this branch. The fixed relay may mediate only part of the model's complete task computation.

Next: H955 evaluates every frozen H949 fit and top-eight subset against these exact causal effects, separately for factor/template and isolated MLP versus joint residual path, retaining full-distribution KL.
