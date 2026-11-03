# Structural language gap: stop extending the V8 sentence list

The primary objective is a realistic, variable **sampled** language DGP with known, tunable rules, followed by behavioral gates and comparison of weight-based interpretability methods. V8–V12 are useful controls and parser fixtures, but their form audits repeatedly reject them as that DGP. This audit asks which level of structure is missing, using two fresh independent FineWeb sample-10BT panels (offsets 40000 and 41000) and paired V8–V10 generated worlds. Each panel has 64 eligible first-100-word document windows. Only aggregate statistics are saved.

| Offset | Panel | Word types | Top-ten word share | First-40 repeated-word fraction | Sentence-start types | Cross-document trigram reuse |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 40000 | FineWeb | 2,532 | .206 | .157 | 267 | .007 |
| 40000 | V8 | 220 | .325 | .241 | 37 | .714 |
| 40000 | V9 lexical | 469 | .308 | .240 | 109 | .386 |
| 40000 | V10 acts + lexical | 476 | .309 | .240 | 104 | .368 |
| 41000 | FineWeb | 2,258 | .214 | .162 | 217 | .010 |
| 41000 | V8 | 229 | .327 | .245 | 36 | .713 |
| 41000 | V9 lexical | 476 | .315 | .246 | 103 | .407 |
| 41000 | V10 acts + lexical | 471 | .313 | .246 | 106 | .392 |

“Cross-document trigram reuse” is the share of three-word occurrences in the 100-word windows whose exact lowercased trigram appears in **more than one** of the 64 documents. This is a descriptive finite-panel statistic, not a population estimate or semantic similarity measure. The generated arms share latent worlds and eligibility; FineWeb is separately sampled. At this scale, V10 plus lexical variation still reuses exact three-word strings roughly 40–55 times as often as the natural panel. The first 40 words barely change when extra true statements are placed after events. [Offset 40000 results](results/v8_v10_structure_fineweb_skip40000_n64.json); [offset 41000 results](results/v8_v10_structure_fineweb_skip41000_n64.json).

The gap is structural as well as lexical. The top five sentence-start words cover .233/.240 of natural segments versus .415/.367 for V10 on the two panels. A simple punctuation/newline segmenter estimates mean segment length 13.9/15.2 words in FineWeb and 9.2/9.2 in V10. Colons appear in 37.5%/26.6% of natural windows but 87.5%/82.8% of V10 act windows. These segment and punctuation figures are crude—web headings, abbreviations, and markup can distort them—but they reinforce the exact-ngram result. Typed nouns increase the number of distinct first words without changing the dominance of a few opening frames.

This changes the next implementation. The core unit should be a sampled **document purpose and proposition graph**, not a fixed inventory of sentences. Sample a document kind (for example report, forum exchange, listing, procedure, reference entry, narrative), then entities, events, claims, and what the author wants to communicate. A discourse planner should choose moves such as introduce, elaborate, attribute, contrast, correct, quote, revisit, and ask, with genre-dependent transition probabilities. A semantic renderer should combine several propositions into variable-length clauses and paragraphs using coordination, subordinate clauses, referring expressions, and layout. The existing event world, visible-state parser, typed lexicon, and supported edits can supply the truth conditions; the V8 sentence forms remain baseline controls.

The implementation should be judged on three separate axes before training a new model grid:

1. **Rule integrity:** a visible-text checker reconstructs scored answers; supported upstream edits change the right answers; changing omitted latent facts leaves text and answers unchanged. Ambiguous targets are scored as distributions, not as the sampled hidden choice.
2. **Form:** two development panels and a later untouched holdout compare lexical tail, sentence openings/lengths, exact cross-document trigram reuse, layout, and within-document repetition. These are continuous diagnostics, not one pass/fail claim of “realism.” Inspect actual samples for semantic coherence.
3. **Model eligibility:** train 1–4-layer bilinear pilots on documented generator settings, test each rule and held-out combination behaviorally, then compare weight-only, data-aware, sparse, and intervention-optimized methods only where behavior supports that rule. Distinguish function fidelity from human-readable rule recovery.

The 24-document manual FineWeb opportunity panel already distinguishes news, opinion, promotion, organization pages, search aggregations, products, reviews, forum text, instructions, and reference material; it is too small to set reliable mixture weights. The next generator should start with a few **genuinely different** document purposes and then widen the mixture, keeping task prevalence separately tunable. The previous corpus-sentence bank and induced PCFG experiments remain negative or partial controls: a sentence bank makes the continuation nearly deterministic, while the induced PCFG improves word counts without reliable semantic relations. The new renderer needs compositional syntax **conditioned on** propositions.
