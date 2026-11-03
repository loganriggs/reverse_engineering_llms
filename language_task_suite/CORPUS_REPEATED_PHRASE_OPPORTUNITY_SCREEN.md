# Repeated phrases in natural documents: a denser copy-rule candidate

The [delimiter method comparison](CORPUS_DELIMITER_FULL_PATH_COMPARISON.md)
has reached its stopping point. To broaden the language rules tested in
realistic variable text, this screen asks whether repeated multiword phrases
could support a document-local copying intervention. It is an **opportunity
screen**, not yet a generator or a trained-model result.

The [audit](audit_corpus_repeated_phrases.py) inspected the first 8,000
characters of 500 independent FineWeb `sample-10BT` documents at offsets
52,000–52,499. A simple capitalization regex selected exact two- to
four-word repeated phrases separated by 20–300 words, excluding common
sentence leads and phrases that were immediately followed by another
capitalized word. It found **213/500** eligible documents and **672**
eligible consecutive-repeat pairs. Requiring the first word to come from
the existing small curated person-name bank leaves only **7/500**
documents and 12 pairs. This large sensitivity to the extractor is why
yield alone cannot justify an entity-binding task.

A deterministic 24-pair sample was read at both occurrences. A single
manual reviewer marked 13 as *apparently* referring to one specific
person, organization, place, event, or project; seven were generic surface
repetitions (for example, a product category or listing field), three were
partial names or aliases, and one was unclear. These labels are short-snippet
judgments, not verified identities. The [aggregate audit JSON](results/corpus_repeated_phrase_opportunities_skip52000_n500.json)
and [hash-linked assessments](results/corpus_repeated_phrase_manual_assessment_seed72991_n24.json)
contain no source passages or phrase strings.

**Interpretation:** broad exact phrase recurrence is a plausible *surface
reuse* task but an unreliable *entity identity* label. A future sampled
transformation could consistently substitute a same-type alias at every
occurrence, so the later surface becomes a known copying target. It must
first pass three checks: (1) the alias preserves grammatical/entity type
well enough that text still resembles natural prose, (2) a paired earlier
alias edit changes the later answer while an unrelated edit does not, and
(3) the model learns the rule on whole-source-document holdouts. Until
those checks pass, do not use broad repeated phrases to claim that weights
encode entity binding. The strict full-name transform remains a sparser,
higher-precision person-reference control.

This corpus-backed route complements the harder main objective: a sampled
document/world generator with known interacting bigram, skip, induction,
reference, and discourse rules. Natural source text supplies form but no
complete symbolic next-token oracle. The synthetic generator supplies
oracle access but still fails the independent natural-form screen. Both
limitations should remain explicit in any later weight-method comparison.
