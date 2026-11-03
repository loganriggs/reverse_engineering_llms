# Introduction paragraphs and a lexical ceiling on repeated scaffolds

The aim remains a sampled language distribution whose rule strengths and
combinations can be varied, then used to evaluate weight-based methods on
rules that trained bilinear models demonstrably learned. Two form experiments
now rule out adding hand-written templates to V8 as the main route.

## Composed introductions

`document_world_intro_paragraphs.py` samples five grammatical roles for each
object's initial owner, color, place, and alias, shuffles objects, and packs
one to three descriptions into a sentence. The visible-prefix transducer
recovers the initial facts without reading the latent world. On 300 worlds,
it passed 2,592 scored sites in canonical and typed-lexical forms, 2,700
direct initial-fact queries, and 300 targeted initial-fact edits.

Two paired 64-document first-100-word panels gave:

| Seed | Arm | Types | Reused trigrams | Start types |
|---|---|---:|---:|---:|
| 94121 | Prior V10 acts + typed lexicon | 466 | .362 | 104 |
| 94121 | Composed introductions | 461 | .401 | 105 |
| 94121 | Introductions + acts, no asides | 419 | .374 | 97 |
| 94122 | Prior V10 acts + typed lexicon | 471 | .385 | 108 |
| 94122 | Composed introductions | 453 | .397 | 103 |
| 94122 | Introductions + acts, no asides | 413 | .372 | 98 |

Paragraph packing increased mean segment length modestly, but did not
materially improve the distribution. It exchanged old fixed frames for new
ones and still provided little variation in document-level rhetorical
structure. No model was trained on this pilot.

## What lexical expansion alone can and cannot fix

`audit_v10_frame_trigrams.py` masks a broad set of variable words in the
prior best V10 stream: every typed lexicon value, alias, topic, incidental
noun/adjective, and incidental event word. A trigram containing **none** of
these words is a conservative fixed-frame trigram. If only slot words change,
its occurrences remain unchanged, assuming one-word substitutions and the
same 100-word crop. The two independent 64-document panels show:

| Seed | All repeated-trigram share | Fixed-frame-only share of all trigrams |
|---|---:|---:|
| 94121 | .366 | .123 |
| 94122 | .390 | .133 |

Thus even an arbitrarily large one-word slot vocabulary cannot take the
current V10 text below **12–13% repeated trigrams** on these sampled panels
while the scaffold remains fixed. The corresponding FineWeb panels have
only .7–1.0% total cross-document trigram reuse. This does **not** prove that
the eventual toy must match FineWeb's number exactly: the vocabularies and
document mixtures differ. It does prove that lexical diversity alone cannot
close the observed gap in this generator. The variable-touching trigrams
also contribute substantial reuse, so a richer lexicon remains necessary.

The next implementation should change both levels together: a much broader
typed lexicon and a **corpus-informed discourse/syntax model** that samples
which propositions are communicated, grouped, revisited, attributed, or
omitted before rendering. Each semantic statement must have an explicit
interpretation and visible-prefix check. This is a change in generator unit,
not another set of fixed opener or aside templates. Evaluate form on
development panels and a disjoint holdout; preserve rule-specific causal
edits and whole-task/composition splits. Only then train the 1–4-layer
bilinear grid and compare canonicalization, sparsity, intervention-aware
objectives, and other decompositions on behaviorally eligible rules.

The [introduction audit](audit_intro_paragraphs.py) and
[results](results/intro_paragraphs_semantic_form_300x2.json), and the
[frame audit](audit_v10_frame_trigrams.py) and
[results](results/v10_frame_trigram_ceiling_2x64.json), reproduce these
figures. These are descriptive form and semantic checks, not proof of
natural-language realism or model competence.
