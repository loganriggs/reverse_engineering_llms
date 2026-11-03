# Corpus-informed opening moves for the next sampled language DGP

The target is a sampled, controllable language distribution that supports
weight-based mechanistic interpretation of rules a bilinear model has
actually learned. More V8 clause variants failed three times. The next
generator must sample **what a document is doing** before realizing its
sentences.

I read the first 500 whitespace-normalized characters of 32 distinct
FineWeb sample-10BT documents, offsets 42,000–42,031, and manually coded the
dominant page form and visible opening moves. The
[audit](audit_fineweb_opening_moves_42000.py) stores only SHA-256 hashes,
offsets, labels, and counts in the [result JSON](results/fineweb_opening_moves_42000_42031.json),
so no source prose is committed. These are one reader's exploratory labels
on one contiguous panel, **not** estimates of web-wide form probabilities
or verified prediction sites.

| Dominant opening form | Documents / 32 | Typical moves observed |
|---|---:|---|
| Report/news | 11 | headline or dateline → change or lead claim → attribution, evidence, or comparison |
| Personal/opinion | 8 | question or scene → background, position, example, or reflection |
| Reference/explainer | 4 | question, title, or count → definition, enumeration, qualification |
| Quoted feature/interview | 3 | context or byline → attribution → quotation |
| Procedure/recipe | 2 | item list or title → ordered steps and measurements |
| Collection/roundup | 2 | short item → link or second item, often changing scope |
| Commercial/review | 2 | product specification or review → details, examples, seller information |

Even this small panel has several different opening *programs*. For
example, one document lists quantities before giving ordered actions;
another defines a condition and enumerates symptoms; a roundup changes
its topic between linked items; a report introduces an event, then a
speaker's claim about it. The same next-token problem can thus require
counting, temporal order, entity and source tracking, scope, discourse
state, or only a local convention. The manually coded moves are visible
text functions, not proof that every corresponding dependency matters to
prediction.

The result changes the DGP interface. A document should sample:

1. A **page form** and genre-conditioned discourse state. The seven forms
   above are initial coverage targets. Their measured counts are too noisy
   to use as fixed priors; expose form weights as DGP hyperparameters.
2. A **proposition graph** with typed entities, attributes, events,
   sources, quantities, ordered steps, and scoped records. It includes
   exogenous facts, derived facts, and references to earlier facts.
3. A **move plan** that selects which graph nodes to introduce, attribute,
   compare, revise, quote, list, revisit, or ask about. Moves have
   preconditions: a revisited source needs an earlier claim; a comparison
   needs two relevant values; an ordered step has a predecessor.
4. A **semantic surface grammar** whose syntactic and lexical choices are
   conditioned on the move and its arguments. The induced PCFG is a useful
   source of syntactic statistics, but its earlier samples had poor
   selectional relations; raw grammar sampling should not be relabeled as
   a realistic semantic renderer. Expand the typed lexicon substantially
   and vary sentence structure, clause packing, and layout together.

For scored sites, store the proposition dependency, candidate set, and
whether the visible prefix determines one answer or merely changes a
distribution. A visible-only checker must reconstruct answers and reject
contradictory assertions. Supported edits should change the relevant
prediction and leave unrelated sites stable. Compare held-out form and
rule combinations. Audit word frequencies, first-100-word trigram reuse,
sentence starts/lengths, layout, and coherence on development panels and
an untouched natural-text holdout.

Once that generator works, train a small 1–4-layer bilinear grid across
document-form and rule-coupling settings. Test each rule behaviorally and
causally first. Compare weight-only canonicalization, data-aware
reductions, sparse weight objectives, and intervention-aware objectives
only on learned rules, reporting function fidelity and recovery of the
underlying rule graph separately.

The immediate implementation target is a typed proposition graph plus
form-conditioned move planner with explicit preconditions. It should
produce auditable plans before any model training or surface-realism claim.
