# Predicting real web text: what the prefix actually supplies

This is a design reading of the 64 independent FineWeb sample-10BT document
previews already audited in [the inventory](FINEWEB_PREDICTIVE_RULE_INVENTORY.md)
and its fixed random 24-document opportunity screen. I re-read eight
contrasting previews (offsets 23006, 23010, 23014, 23019, 23033, 23038,
23043, 23062) to ask what a next-text predictor can legitimately infer.
The underlying text is not copied into this repository. These are selected
examples, not prevalence estimates or a claim about the author's hidden
thought process.

The useful object is a **conditional distribution**, not one rule per
document. A document kind sets a broad prior; its topic and entities set
more specific priors; a discourse plan sets which facts are likely to be
mentioned; grammar and local word associations constrain the next few
tokens. Some later tokens are implied by an earlier fact, others are merely
more likely, and some introduce genuinely new information. A toy DGP can
represent this as sampled latent document type, world/entity graph,
discourse plan, sentence realization, and local lexical noise. The exact
oracle is the marginal over *possible latent states given the visible
prefix*, rather than access to the generator's private state.

| Real preview and target | Predictive structure | Constraint from visible prefix | Toy rule and discriminating edit |
|---|---|---|---|
| 23006, report about a dog attack: later pronoun for a witness | A woman was introduced with a role in the event; grammar and referent identity both matter | `she` is strongly favored, but a writer could repeat `the witness` | Sample people/roles; vary referent and wording independently; separate local gender cue from distant identity. |
| 23019, two cruise records with the same room number | Ship and date provide record scope; room number alone is ambiguous | A later category is recoverable only if the query identifies which record | Sample repeated keys under distinct scopes; swap ship while holding room fixed; balance first/last order. |
| 23033, baseball report: later final score after a 4–2 lead and three runs | Event update plus team-role binding | Arithmetic is possible, but the final score is already in the headline, so copying also works | Train leaked/unleaked headlines, change the scoring event, and ask about either team. |
| 23014, walkthrough moving through floors | Goal and repeated procedure give sequence and page-genre cues | Third floor is already named as the goal; a later mention need not require counting | Separate goal leakage from step transition, and swap intermediate steps. |
| 23038, daylight-saving reference | Parallel start/end schema, opposite earlier/later direction | The row type narrows the sign of the clock change | Sample two contrasting rows with dates and sign; swap row heading without changing topic words. |
| 23043, product specification list | Bullet schema and field type strongly predict units and format | A width field suggests a measurement; its exact number is a newly supplied fact | Sample product fields, values, and prose rendering; test unit/type versus exact-value retrieval separately. |
| 23062, dog first named in a narrative | Entity continuity across puppy/dog/name | The first name is new information; a later mention of it can use identity | Sample alias introduction then later reference; edit name, not just nearby dog words. |
| 23010, car-light failure forum post | Symptom and mechanical knowledge narrow plausible causes | Bulb and ballast are alternatives; no single correct diagnosis follows | Sample a fault graph and calibrated answer distribution, including uncertainty rather than a forced label. |

The fixed random 24-document screen found clear entity reuse in 19,
source attribution in 10, a repeated page/list/field schema in 7, and an
explicit state change in 6. Ordered procedures appeared in 2, while a
scoped-record candidate and semantically related number appeared in 1
each. This is only a short-preview, one-reader opportunity count. It does
suggest a frequency hierarchy for the **reference-rate** toy stream:
entity reuse, attribution, page form, and local syntax should be routine;
clean arithmetic and conflicting-record queries should be harder,
task-rich oversamples with separately reported rates. Zipf-like word
frequencies would be a useful marginal check, but they cannot supply these
conditional relations by themselves.

The next annotation pass should randomly select *prediction sites*, not
just documents or regex cues. For each site record: target span, minimal
earlier evidence, plausible alternate continuations, whether the target
is deterministic/probabilistic/new, the required context length, and a
counterfactual that is still grammatical. Two readers should adjudicate a
subset. Then estimate both the fraction of sites and the fraction of
next-token loss attributable to each mechanism using full/local/edited
prefix scoring, with the caveat that a model's score difference is a
sensitivity estimate rather than ground-truth causal structure. Split
independent documents from the development panel for validation.

For synthetic training, build the *joint distribution* before training:
sample one world with an entity graph, events, sources, and document
intent; render it through multiple forms (prose, list, quote, short
table); and let local bigram, skip, induction, and bracket tendencies
interact with those same facts. Require exact visible-prefix replay and
supported edits. Keep a natural-reference rate and a task-rich rate
separate. A model can be studied on any task it passes, but task-specific
accuracy, edit fidelity, and held-out combinations must be reported
separately from overall token loss. The [V6 transfer result](DOCUMENT_MIXTURE_V6_TRANSFER.md)
shows why: nearly identical token loss hides a sharp failure on held-out
record assignments and score combinations.
