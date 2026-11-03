# Proposition-plan surface V1: causal fixture, form gate failed

The new document planner now has a text realization and a separate
visible-only parser. It samples seven page forms and typed proposition
graphs; the renderer expresses introductions, state changes, source
attributions, comparisons, record counts, ordered steps, scoped records,
and later revisits or questions. Every scored site has a candidate set
and a dependency on prior visible text. The parser receives only emitted
tokens, never the sampled plan.

This V1 grammar is deliberately explicit. It is a **semantic integration
fixture**, not yet a realistic toy language or a model-training stream.

## Semantic and causal checks

The [audit](audit_document_proposition_surface.py) sampled 200 plans for
each of seven forms, rendered each in canonical and typed-lexical wording,
and parsed every complete document and scored prefix: 2,800 documents and
10,214 scored sites passed. The scored families include 3,912 current/initial
facts, 962 updated facts, 402 sources, 2,194 comparison outcomes, 348
record counts, 2,146 step numbers, 98 later step actions, and 152 scoped
record values. These are counts over **both** lexical arms, not independent
semantic worlds.

Targeted paired checks passed 656 update-value flips, 336 source flips,
870 comparison flips, 304 count flips, 88 step-action flips, and 136
scoped-record flips, again counting both lexical arms. Removing an earlier
completed step shifted the next step-number target in 186 valid canonical
prefixes. The parser rejected 454 deliberately false comparison assertions.
These checks establish internal rule integrity and causal target support;
they do **not** establish that a neural model learns any rule.

One potential shortcut was caught before training. With independently
sampled initial attributes, comparisons are mostly “different.” A
`shared_attribute_rate` DGP parameter now controls cross-entity overlap.
Across two 1,000-plan panels, the majority-class comparison baseline was
about .84–.86 at rate 0, .69–.71 at rate .35, .50–.53 at rate .6, and
.88 at rate 1. The default .6 supplies a roughly balanced development
setting; other rates should be deliberate distribution shifts. These
numbers are synthetic setting diagnostics, not estimated natural rates.

Typed lexicalization initially created category errors, such as a
lexically substituted vehicle receiving a liquid-volume measurement.
Counts now describe external records—copies, reports, entries—so they
remain meaningful under the existing broad object substitution. The
separate parser checks their allowed units and values. A future richer
lexicon still needs stronger semantic subclasses.

## Form gate

On two independent 64-document first-100-word synthetic panels, with
canonical and typed arms paired by proposition plan:

| Seed | Arm | Word types | Cross-document trigram reuse | Sentence-start types |
|---|---|---:|---:|---:|
| 94121 | Canonical | 154 | .765 | 44 |
| 94121 | Typed lexical | 368 | .455 | 83 |
| 94122 | Canonical | 153 | .770 | 44 |
| 94122 | Typed lexical | 371 | .474 | 81 |

For scale, previous V10 acts+typed-lexical controls had .368/.392 reused
trigrams on their two development panels, and the sampled FineWeb
reference panels had .007/.010. These are not paired across the new and
old worlds. Manual inspection also finds awkward record/count wording and
overly regular sentence starts. Seven different move distributions are
not enough when each move still has only a couple of fixed clauses and
the typed vocabulary is small.

**Decision:** do not train bilinear models on V1 as the improved language
distribution. Keep it as a semantic and causal test fixture. The next
substantive change is a corpus-informed *semantic* surface grammar with
many compatible clause structures, a larger type-checked lexical bank,
and more varied paragraph/layout realization. Re-run visible-prefix,
contradiction, targeted-edit, task-base-rate, and natural-form checks on
that version. Once it clears those gates, train 1–4-layer bilinear models
across form and rule-coupling settings, behaviorally select learned rules,
then compare weight-only/data-aware canonicalization, sparse objectives,
and intervention-oriented decompositions on those rules.

Code: [planner](document_proposition_plan.py),
[surface grammar and parser](document_proposition_surface.py),
[plan audit](audit_document_proposition_plan.py), and
[surface audit](audit_document_proposition_surface.py). Full aggregates:
[plan](results/document_proposition_plan_audit_2800.json) and
[surface](results/document_proposition_surface_v1_audit.json).
