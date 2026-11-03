# Predictive rules suggested by real web documents

The toy language should be designed by asking what information in a real
prefix changes the distribution of a later word, not by matching a Zipf
curve in isolation. This inventory is an exploratory pass over **64
document rows**, offsets 23000–23063 of the public [FineWeb sample-10BT
split](https://huggingface.co/datasets/HuggingFaceFW/fineweb). The rows had
64 distinct URLs and hosts. Only their first 1,200 characters were read
for this pass; the committed [audit](results/fineweb_predictive_inventory_skip23000_n64.json)
contains hashes and aggregate counts, no copied source prose. The
[reproduction script](audit_fineweb_predictive_inventory.py) can re-fetch
the same offsets. The earlier 480-window cache contains adjacent windows
from some pages, so it should not be used to estimate independent-document
task incidence.

These documents include articles, forum questions, product listings,
walkthroughs, press releases, biographies, reference tables, and short
scraped or boilerplate pages. In the first 1,200 characters, 61/64 have a
line break, 51/64 a digit, 26/64 a quote mark, 16/64 a question mark, and
8/64 a pipe. These are **surface cues**, not counts of actual linguistic
dependencies. Among the 49 previews containing 100 alphabetic words,
their first 4,900 words contain 1,993 types and 1,336 singletons; the top
ten words account for 21.45%. A few clauses repeated in a small uniform
lexicon cannot reproduce this document mixture.

## Concrete prediction sites

The offsets below identify source documents in the audit. The described
continuations are sometimes strongly constrained and sometimes only
probability shifts; the real corpus does not provide a synthetic oracle.

| Offset | Visible evidence and predictive relation | Candidate sampled task |
|---|---|---|
| 23033, sports report | A team trails 4–2, then scores three runs; the later 5–4 score is a state update, not an isolated number bigram. | Keep a game state, sample scoring events, and emit the updated score in prose and a header. Edit the earlier score or run count. |
| 23019, cruise reviews | Two records use room number 6085 but refer to different ships and dates. A room-only lookup is ambiguous; the enclosing record determines the field. | Repeated keys in scoped records, followed by a natural question or descriptive sentence. Swap record scope while preserving the key. |
| 23062, pet narrative | A puppy is bought, later called a dog and named Madison. Identity must survive noun changes, a proper name, and intervening sentences. | Sample an entity graph with aliases and descriptions; probe a later mention after distractors and alias swaps. |
| 23006, attack report | The injured man, dog, witness, spokesman, and pilot recur under roles and pronouns. The speaker of a claim and the subject of an event differ. | Distinguish entity identity, event roles, and source attribution rather than one generic copy rule. |
| 23014, game walkthrough | Instructions move from the first floor to the second and third, repeating a procedure with a changing location/state. | Sample ordered steps and state transitions; hold out new step order or count. |
| 23038, daylight-saving reference | The start/end entries pair dates with clocks moving later/earlier. Parallel rows share a schema but have contrasting values. | Two-field table plus prose, with row-scoped retrieval and contrastive sign edits. |
| 23043, product listing | Bullet-like specification lines include dimensions, units, material, and warranty. The page template predicts formatting; each field predicts a different value type. | Sample a product schema and render it as bullets, prose, or a table. Probe field-value swaps and units. |
| 23063, press release | The same agent and firm recur in a seminar account, quote attribution, award, and call-to-action. | Link a source, event, and later reference across several discourse functions. |
| 23010, repair forum | A failed car light motivates possible bulb/ballast causes. The next reply is uncertain, but the failure constrains which explanation is plausible. | Sample a causal fault graph and a probabilistic advice response; do not fabricate one uniquely correct answer. |
| 23045, doctrinal passage | Feature, mark, and beam form nested repeated descriptions with large counts. | Test compositional hierarchy and repeated expansion separately from memorized phrase repetition. |

The contrast across these examples suggests a **hierarchical generative
model** as the next design target: choose a document kind and intent, then
a topic and world/entity state, then a discourse plan, then sentence or
record structures and words. Local syntax, collocations, copying,
coreference, arithmetic, and world knowledge jointly affect the next-token
distribution. A natural continuation is usually stochastic. We should
plant exact *conditional distributions* and supported counterfactuals in
the toy, not pretend every natural sentence has a deterministic rule.

## Fixed-sample opportunity screen

To check whether the examples above are merely memorable, a second pass
used a fixed random draw of 24 of the 64 independent documents
(`random.Random(23012)`). One reader inspected the first 1,200 characters
of **every** selected document, recorded its main page form, and marked
only clear positive opportunities under [explicit flag
definitions](results/fineweb_manual_opportunities_skip23000_seed23012_n24.json).
The counts are:

| Visible opportunity | Documents / 24 |
|---|---:|
| Particular entity reused across clauses | 19 |
| Proposition or quotation attributed to a source | 10 |
| Repeated list, page, or field schema | 7 |
| Explicit before/after state change | 6 |
| Ordered multistep procedure | 2 |
| Candidate parallel records with separate scope | 1 |
| Semantically related numbers | 1 |
| Failure plus alternative causal hypotheses | 1 |

These are **not population prevalence estimates**. The preview may miss
later dependencies, one annotator can disagree with another, and a reused
entity does not prove that an LM needs a long-range identity circuit to
predict any particular token. The scoped-record and numeric examples that
motivated the inventory were not typical of this small fixed sample. This
changes the sampling plan: ordinary entity reuse, source attribution, and
page/genre structure should be common in a reference-rate stream;
arithmetic and record-collision tasks can be deliberately oversampled in
a separate task-rich training/evaluation regime, with the reference rate
reported separately. The next annotation pass should mark **target spans,
minimal evidence spans, alternate candidates, and ambiguity**, then have
a second reader adjudicate a subset before treating any rate as a target.

Several concrete targets show why a single notion of “the rule” is too
coarse:

| Offset and target | What the prefix supplies | Predictability judgment |
|---|---|---|
| 23014: next floor in the walkthrough | The goal is the third floor, and prior steps visit first then second. | Strong, visible sequence constraint; a good exact toy oracle. |
| 23006: pronoun for the witness | A young woman was introduced and acted on the dog. | Entity/role evidence raises `she`, but wording remains a writer choice. |
| 23037: surname on a later attribution | An official is introduced with full name before a quotation. | Long-context identity helps; the quote wording is not determined. |
| 23043: next specification field | A product page has bullet and key-value conventions. | Strong format/type prior; particular order and value are uncertain. |
| 23010: bulb or ballast | The car light failed after previously working. | Several causal explanations remain possible; test probability calibration. |
| 23059: sentence length in months | Headline says roughly eight years. | `97 months` is compatible, but not recoverable exactly from the rounded headline. |

Thus the toy should distinguish **visible deterministic implications**,
**context-sensitive probability shifts**, and **new exogenous facts**. A
next-token model legitimately spends capacity on all three; only the
first two support a claim that the prefix contains a recoverable
computation. For naturally occurring text, a full-prefix versus truncated
or counterfactual-prefix score can test predictive sensitivity, but a
single model's score is not ground-truth causal evidence.

## Target-site red team

An eight-site, hand-inspected [target ledger](results/fineweb_prediction_sites_skip23000_n8.json)
records a hash, exact character offset, candidate dependency, competing
shortcut, and proposed edit for each target. The [script](audit_fineweb_prediction_sites.py)
verifies every offset against the local source preview without committing
source passages. The sites were chosen for conceptual diversity, **not** to
estimate frequencies or model accuracy.

Two attractive examples are less decisive than they first appear. In the
sports report, the score later explained by a 4–2 starting score and three
runs is already given in the headline. In the walkthrough, the third-floor
goal is named before the first/second/third progression. A model could get
both targets right by copying an earlier explicit answer. The toy should
include matched **leaked** and **unleaked** versions, plus edits that make
copying and state computation disagree.

Other targets isolate different levels of constraint. A witness pronoun is
strongly biased by a nearby gendered noun, but its exact wording is a writer
choice. A repeated surname after a quotation can use an earlier full-name
introduction, though a local attribution pattern may suffice. A product
listing predicts that a case-measurement field has a dimensional unit, but
not its exact value. A dog receives a proper name for the first time; that
name is a new fact, while *later* uses of it can test identity tracking. A
room number occurs under two ships with different category values, so a
key-only lookup is ambiguous; record scope matters only at a later query
site that actually asks for a category. These distinctions should control
which sampled sites count as task evidence, which are nuisance surface loss,
and which are deliberate confound tests.

As a controlled first implementation, [document_world_v3_multisite.py](document_world_v3_multisite.py)
uses one visible event history for alias identity, source attribution, and
three final-state predictions in the same sampled document. The
[3,000-document audit](results/document_world_v3_multisite_audit_n3000.json)
checks all 15,000 answer sites against an independent visible-prefix parser,
genre-preserving answers, and relevant event edits. It passes those semantic
checks. It fails the natural-form gate badly: in 49 documents × the first
100 alphabetic tokens, the toy has 84 word types and zero singletons,
versus 1,993 types and 1,336 singletons in the analogous FineWeb screen.
The toy also ends every document with the same five questions. It is a
control for semantic binding, **not** yet a realistic replacement stream or
an exact full next-token oracle. The next DGP iteration should make those
dependencies appear inside variable-length prose, records, and dialogue,
with stochastic continuations and interaction with the original local rules.

The [interleaved structure pilot](document_world_v4_interleaved.py) removes
the fixed five-question ending: it samples one to eight declarative answer
sites at different points in the visible event history. Its
[audit](results/document_world_v4_interleaved_audit_n3000.json) checks 6,995
sites across 3,000 documents and 2,671 paired upstream edits. The median
site falls at 65% of document length. This establishes that the parser and
edits still work with variable placement. It **does not** improve natural
form: the same 49 × 100 alphabetic-token screen has only 82 types, no
singletons, and a top-ten share of 39.4%. Source-attribution sites occur
in 1,401/3,000 documents, roughly the 10/24 document-level opportunity
rate of the small FineWeb screen, but the two rates are not directly
comparable: one is a planted target and the other a broad human flag.
Do not promote this pilot to the main training stream on that basis.

An [exploratory prefix-sensitivity check](FINEWEB_PREFIX_SENSITIVITY.md)
scores the eight natural target sites with GPT-2 and the trained 18-layer
bilinear model. Both are strongly sensitive to earlier mentions at a later
surname site, while a witness pronoun depends mainly on nearby gender
evidence. The apparent long-context benefit at a second record's category
is contaminated by priming from the first record; the sports score is
contaminated by its headline. The daylight-saving direction site does not
show successful contrastive reasoning in either model. These are model
diagnostics on selected examples, not natural task rates or proof of the
source writer's rule. They sharpen the next sampled controls: separate
name binding from local grammar, priming from scoped lookup, and copying
from state computation.

## What to build and measure

The next version should integrate the original competing bigram,
skip-context, induction, and bracket rules with shared latent structures
for **entity identity/aliases, sources, and event states**, then add
scoped records as a targeted harder family. Each structure should appear
through multiple outputs
(ordinary prose, a short table/list, a quote or question) in the same
sampled document. Add optional distractors, variable document and clause
lengths, and a class-appropriate long-tail lexicon. This makes a later
retrieval or update depend on information in text the model actually saw.

For every task family, record the number of valid prediction sites, the
entropy of the exact next-token oracle, collisions with other rules, and
the effect of a supported edit. For natural text, manually annotate a
stratified document panel with the evidence span, candidate prediction,
ambiguity, and genre; a regex match alone is insufficient. Compare the
natural and toy distributions of genre, length, formatting, entity reuse,
distance between evidence and target, task incidence, and word-frequency
head and tail. Reserve a disjoint FineWeb document panel for the final form
screen. Matching marginals does not establish that the model learns the
same computations.

Only after the revised DGP passes exact-prefix replay, grammatical review,
task-site coverage, and a held-out natural-form screen should we train
1–4-layer bilinear transformers. Interpretability comparisons then need
separate results for sampled-text loss, task-specific prediction,
counterfactual fidelity, and discovery of task structure withheld from
featurizer training. The current V3 short-form pilot passed oracle replay
but failed the word-frequency gate, so it is a control, not the successor
stream.
