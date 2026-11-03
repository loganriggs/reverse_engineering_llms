# Document-purpose pilot: semantic pass, form rejection

The research target is a variable sampled language distribution with known,
controllable rules on which to evaluate weight-based interpretability. Model
capability is an **eligibility gate** for a rule: first verify the rule was
learned behaviorally, then ask whether a weight method recovers it. A more
capable toy model is not itself the outcome.

## Prototype and checks

`document_world_purpose_pilot.py` reuses the V8 sampled event world but renders
paired worlds as a report, correspondence, or a register. Each purpose has
different introductory and update clauses and layout. A parser reads only the
visible prefix, reconstructs the current facts and attributions, and checks
each prediction site. The audit additionally makes supported source and event
value edits and checks the edited document from visible text.

Across two 64-world panels, **2,022 original sites passed**, along with 128
value edits and 128 source edits. The edits produced 213 changed and 3,831
unchanged answer-site comparisons, all matching the visible parser. These are
semantic checks; they do not show that a model has learned the rules.

## Paired form screen

All generated arms were cropped to the first 100 words of the same eligible
sampled worlds. The FineWeb and previous V10 aggregates are the two stored
independent sample-10BT panels, at offsets 40,000 and 41,000. "Shared
trigrams" is the fraction of trigram occurrences whose trigram appears in
more than one document in that 64-document panel.

| Arm | Shared trigrams, panel 1 / 2 | Word types, panel 1 / 2 |
|---|---:|---:|
| FineWeb | .007 / .010 | 2,532 / 2,258 |
| Prior V10, fact acts + typed lexicon | .368 / .392 | 476 / 471 |
| Report, typed lexicon | .515 / .543 | 328 / 327 |
| Correspondence, typed lexicon | .680 / .684 | 333 / 331 |
| Register, typed lexicon | .442 / .453 | 324 / 324 |

The unlexicalized purpose arms have shared-trigram fractions .765–.876. The
register has only 22–23 sentence-start types, versus 217–267 in FineWeb.
Different wrappers and formatting did **not** solve the repeated core-clause
problem. Correspondence is especially repetitive. The pilot should **not**
be used for model training or weight-method claims as an improvement over
V10. This is a useful failure: genre labels alone do not provide document
level variation when every act still comes from a tiny clause inventory.

## Next generator unit

Build a sampled proposition/discourse graph before choosing surface wording.
Vary the number and relations of propositions, rhetorical moves, participant
roles, ordering constraints, and clause composition. Realize the same semantic
graph in more than one genre while preserving an independent visible-prefix
checker and controlled causal edits. Keep the V8/V10 generator as a stable
semantic baseline. A grammar or corpus-derived realizations can add local
syntax, but every semantic slot needs a known interpretation; arbitrary
natural-sentence insertion would break the ground-truth claim.

Use the same independent form panels and whole-task/composition holdouts. A
candidate advances to model training only if semantic checks pass and the
structural metrics improve materially over V10 without relying on merely
randomizing names. Then train a modest 1–4-layer grid across DGP settings,
measure each rule's behavioral and causal response, and evaluate
canonicalization/sparsity only on eligible learned rules. Report reconstruction,
prediction, causal fidelity, and discovery separately.

The reference metrics are descriptive, not a claim that toy language must
match all of FineWeb. FineWeb contains many unrelated web genres; the large
gap still diagnoses highly reused scaffold text.

Reproduce with `audit_document_purpose_pilot.py` using the two saved FineWeb
aggregate JSONs as references. Results are in
`results/document_purpose_pilot_seed94121_n64.json` and
`results/document_purpose_pilot_seed94122_n64.json`.
