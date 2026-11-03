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

## What to build and measure

The next version should integrate the original competing bigram,
skip-context, induction, and bracket rules with three shared latent
structures: **entity identity/aliases**, **scoped records**, and **event
state updates**. Each structure should appear through multiple outputs
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
