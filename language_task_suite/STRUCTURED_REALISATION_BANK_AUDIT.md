# Corpus-derived realizations: useful parts, missing document distribution

The research target is a **sampled, variable language distribution with known
rules**, used to test whether weight-based interpretability methods recover
the rules a trained bilinear model actually uses. Model capability is a
rule-specific behavioral gate, not the reason to make the distribution.
The [linked project-report DGP](PROJECT_REPORT_DGP_V1_AUDIT.md) has sound
local semantics but an 89.7–95.0% repeated-cross-document trigram share on
100-word panels, against 0.73% on the fresh FineWeb comparison. This screen
asks whether existing grounded text datasets can expand its realization
space without losing its causal trace.

## DART

[DART](https://aclanthology.org/2021.naacl-main.37/) pairs sets of triples
with text; its [official release](https://github.com/Yale-LILY/dart) combines
human and automatically derived partitions. The local V1.1.1 training JSON
has 30,526 triple-set records and multiple annotations. The
[screen](audit_dart_slot_realisations.py) accepts an annotation only when
every non-marker subject/object string from a multi-triple record appears
exactly once and without overlapping another. It then replaces those
strings with slots and counts distinct forms for each predicate signature.

| Partition | Annotations | Exact-once multi-triple annotations | Distinct slotted forms | Signatures with ≥10 forms |
| --- | ---: | ---: | ---: | ---: |
| WebNLG | 18,102 | 3,664 | 3,604 | 82 |
| E2E | 33,482 | 13,618 | 12,546 | 100 |
| WikiTableQuestions LILY | 4,700 | 1,044 | 1,027 | 1 |

The [aggregate results](results/dart_v111_slot_realisations_audit.json) show
real surface variety, especially in the restaurant-centered E2E partition.
They do **not** certify semantic fidelity. Manual inspection found references
with claims absent from the triples and awkward clauses; replacing a slot can
also break agreement or coreference. A quick mosaic of independently selected
WebNLG sentences had 39.6% repeated-cross-document trigrams and no coherent
document plan. Thus DART is a candidate sentence-form bank, not a ready DGP.

## ToTTo

[ToTTo](https://github.com/google-research-datasets/totto) supplies highlighted
Wikipedia table cells and human-corrected one-sentence descriptions. The
[streaming screen](audit_totto_document_realisations.py) read 120,761 local
training records, counted exact highlighted-cell string mentions, and grouped
eligible sentences by page and section. It found 59,093 records in which all
highlighted cell strings occur in the final sentence, 40,448 page-section
groups with at least one eligible sentence, and 1,102 such groups with at
least four eligible sentences. Exact mentions are a permissive screen, not
proof of meaning or safe slot substitution.

For a form diagnostic, the script assembled 64 distinct-page-section,
100-word samples from different highlighted rows. The
[aggregate result](results/totto_document_realisation_audit.json) has 1,289
word types, 33.7% top-ten-word share, and 33.5% repeated-cross-document
trigram share. These are much less repetitive than the report DGP, but far
from the independent FineWeb panel. Manual inspection of three selected
groups found lists of sports seasons or film credits: facts remain on topic,
yet there is little discourse progression, and the order is arbitrary.
ToTTo is a stronger source of table-grounded *sentences* than of documents.

## Decision and next test

Keep both datasets as optional realization resources, with separate
faithfulness checks. Do not concatenate their sentences into the main
training distribution or claim their exact-string matches define a semantic
oracle. The next generator should sample a document kind, persistent
entities/events or records, a variable discourse sequence, and several
typed ways to realize each event. It should make local language statistics
and longer-range rules compete within the same stream. DGP controls should
change rule prevalence, ambiguity, entity reuse, dependency distance,
lexical tail, and surface variation independently where possible; hold out
entire rule combinations and surface forms for evaluation.

Before training, audit fresh generated documents for form *and* task-bearing
span repetition against independent natural-text panels, verify sampled
rules from visible prefixes, and test paired edits of the controlling fact.
After training, gate each claimed rule on held-out prediction plus the
relevant edit response. Compare weight-only reductions, sparse gauges, and
full-path methods only on rules that pass that gate; score whether their
identified components predict causal effects across seeds, DGP settings,
and held-out combinations. Keep reconstruction, behavioral task accuracy,
causal fidelity, and discovery of previously unlabeled rules as separate
outcomes. A known data rule is not a unique ground-truth neural circuit.

The local DART and ToTTo archives are not committed. Only scripts and
aggregate statistics are retained here.
