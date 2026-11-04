# Natural attribution frames are plentiful as syntax, scarce as safe slots

This screen tested a possible shortcut to a more natural sampled language:
extract whole reporting-sentence shells from FineWeb, replace the speaker and
clausal claim, then use those shells to realize a sampled document world. It
does **not** establish that such substitutions preserve meaning or naturalness.

The [reproducible audit](audit_corpus_attribution_syntax.py) parsed two
source-disjoint, local 2,000-document FineWeb preview panels with spaCy. It
accepted a sentence when its root was a reporting verb with one overt nominal
subject and one finite clausal complement; it also rejected obvious named
entities, numbers, and demonstratives outside the candidate slots. Only
aggregate statistics, hashes, and source index ranges were saved.

| Panel | Accepted sentences | Documents with an accepted sentence | Distinct parser signatures | Distinct surface shells | Bare speaker–verb–claim shells | Shells with quote marks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 130k–132k | 202 | 150 | 100 | 118 | 105 | 34 |
| 140k–142k | 163 | 127 | 82 | 101 | 79 | 25 |

The ten most common surface shells account for 45.5% and 42.9% of accepted
sentences. Those numbers overstate useful diversity. A manual substitution
probe replaced the slots with *Alice* and *that the map was blue*. Long shells
often retained source-specific context: a COVID-era supply-chain preface, a
reference to a particular show or investigation, an unrelated second clause,
or quotation marks whose placement no longer fit the new claim. The parser
also accepted some grammatically broken source fragments. About half the
accepted shells contained a newline in the preview segment, which can create
spurious surface differences. A syntactic slot screen cannot certify that
meaning-bearing adjuncts are detachable.

**Decision:** Do not use the accepted shells directly as a training renderer.
This is a stronger reason than sample size: a large bank could still preserve
context-dependent claims or make false-looking prose. Retain this screen as a
negative control and a source of *structural* frequencies, not source text or
validated sentence programs. The earlier
[dependency opportunity audit](SCORED_CLAUSE_COMPOSITION_AND_RELATIVE_RESULT.md)
found roughly 1,210–1,227 distinct natural root-child signatures in 512
documents, whereas the best scored-coordination pilot had 185–212. Natural
sentences often combine coordination, adverbial, relative, and complement
clauses, so a new generator needs compositional structure at those sites.

The next implementation should sample a **typed semantic derivation** and a
**syntactic derivation jointly**. A document-level world and discourse planner
chooses an entity, a state change, a source claim, or a comparison. A
probabilistic attribute grammar then realizes that meaning through clause
operators such as complement, coordination, subordination, and reference.
Each operator must specify how the child meanings compose; syntax choices
cannot carry unrelated corpus facts. Production probabilities may be fitted to
source-disjoint natural structural counts, but grammar output needs its own
semantic replay from visible text and independent form audit.

The first vertical slice is one report-like document with introduction,
state change, and attribution in a single 256-token context. It should
expose at least two separately testable next-token dependencies and their
combination, not append a fixed question sentence for each. Compare it with
the existing V4 and real-text controls on source-disjoint 41-word rule
windows and whole documents, including repeated trigrams, sentence starts,
dependency signatures, and held-out LM score. Require enough joint sites
for whole-combination holdouts before training. Model training then uses
ordinary full-token cross-entropy; rule-specific competence and causal edits
gate which weights are interpreted. Exact full-document prefix probabilities
are valuable on a bounded fixture but are not a prerequisite for this pilot.

Audit outputs: [130k panel](results/corpus_attribution_syntax_130000_n2000.json)
and [140k panel](results/corpus_attribution_syntax_140000_n2000.json).
