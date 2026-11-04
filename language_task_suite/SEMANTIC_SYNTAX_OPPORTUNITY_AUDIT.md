# The sampled proposition renderer lacks clause composition

The current semantic proposition renderer has passed visible-answer and counterfactual checks, but its first-100-word windows remain much more repetitive than FineWeb. This audit asks whether its **sentence structure**, not just vocabulary, is the binding constraint. It also screens whether natural sentences offer a plentiful bank of simple, truth-preserving subject–predicate templates. The answer to the first question is yes; to the second, probably not.

The [aggregate-only audit script](audit_semantic_syntax_opportunities.py) parses two local, source-disjoint 512-document FineWeb development panels (offsets 130k and 140k) and separately samples 512 threaded proposition-surface V3 documents per panel. It also applies a [reversible coordination operator](document_proposition_coordination.py) to the *same* V3 documents: adjacent unscored assertions can become one sentence joined by “, and”. Each text is cropped at its first 100 word matches. spaCy `en_core_web_sm` 3.8.0 produces dependency labels. A sentence is counted if the parser finds 5–60 word matches. A dependency signature records the root part of speech and root-child dependency labels, without storing words or source passages. Signatures are crude structural proxies, not linguistic equivalence classes.

| First-100-word panel | Sentences | Sentences per 100 input words | Distinct dependency signatures | Top-ten signature share | Relative-clause sentences | Coordinated sentences | Negated sentences |
|---|---:|---:|---:|---:|---:|---:|---:|
| FineWeb 130k | 2,584 | 5.24 | 1,227 | .179 | 457 | 1,220 | 234 |
| Proposition V3, paired seed | 4,722 | 10.35 | 123 | .614 | 3 | 359 | 0 |
| V3 with coordination, same plans | 4,046 | 8.82 | 195 | .545 | 4 | 944 | 0 |
| FineWeb 140k | 2,588 | 5.21 | 1,210 | .171 | 483 | 1,261 | 202 |
| Proposition V3, paired seed | 4,673 | 10.19 | 113 | .603 | 2 | 339 | 0 |
| V3 with coordination, same plans | 3,949 | 8.57 | 177 | .531 | 2 | 958 | 0 |

The synthetic documents are shorter even before the crop: mean input words per document are 89.1/89.6 versus 96.4/97.1 for FineWeb, and 300/277 synthetic documents versus 78/61 FineWeb documents fall below 100 words. Normalizing sentence count by actual input words still leaves the baseline generator at roughly **twice** the natural sentence frequency. FineWeb contains many relative, coordinated, and negative clauses that the renderer nearly or entirely omits. The signatures have different total sentence sample sizes, but the natural panel has *fewer* sentences and roughly ten times as many distinct signatures. This is not plausibly repaired by adding a few nouns or alternate opening phrases.

Coordination is a useful controlled structural intervention: at a .75 join rate it combines 681/744 sentence pairs. Both panels passed full visible-token expansion back to V3 for all 512 documents, independent visible parsing of all documents, and 1,295/1,276 marked-site answer checks. It moves sentence frequency and coordination toward FineWeb but leaves nearly all the structural gap. It adds neither relative clauses nor true negation. It also does **not** make the actual rule sites look natural:

| 41-word window around one randomly selected rule site per document | Natural random window | V3 task window | Coordinated V3 task window |
|---|---:|---:|---:|
| 130k panel: repeated cross-document trigram share, 430 windows | .023 | .645 | .642 |
| 140k panel: repeated cross-document trigram share, 431 windows | .022 | .644 | .642 |
| 130k panel: word types | 5,255 | 404 | 404 |
| 140k panel: word types | 5,282 | 398 | 398 |

The natural windows are random positions, not task-matched. Still, this shows why an aggregate document-form improvement is insufficient for the interpretability benchmark: the mechanism-bearing local wording remains almost unchanged. Do **not** train a new bilinear model on this coordination-only arm as a purportedly realistic DGP.

The rule-carrier opportunity screen is deliberately stricter. It requires a direct nominal subject, a simple main predicate, no negation or embedded/coordinate clause, at most 18 spaCy tokens, and a complete declarative sentence. Under these filters, the two FineWeb panels have only **11/6 simple adjective predications** and **49/33 simple transitive events**, versus 91/71 and 548/580 in the synthetic panels. Broader parser matches are more common—natural panels have 115/110 adjective predications and 630/573 transitive events—but manual spot checks found additional qualifications, attached clauses, partial web headings, and parser errors. Even the strict matches are *opportunities*, not a safe sentence bank: substituting an entity or attribute can change agreement, reference, or the truth conditions of surrounding modifiers. Therefore direct slot replacement in natural sentences is not established as a realistic, meaning-preserving renderer.

The generator should change its unit of composition *at the rule sites*, not merely join background assertions. A typed clause representation can retain the existing proposition graph but add operators for (1) coordinating compatible propositions **including scored clauses**, (2) relative clauses that assert an already-visible fact about the same entity, and (3) genuine negative propositions whose truth can be checked against the world state. Those operations should be sampled with DGP parameters and rendered through a grammar. A separate visible-text parser must reconstruct each asserted fact and the scored answers; the latent plan alone is not a correctness oracle. Attach task sites to propositions rather than fixed sentence positions. Re-run the same full-document *and task-window* form screens, using independent natural panels for confirmation. Only after behaviorally learned rules and held-out combinations are established should weight-only, data-aware, sparse, and intervention-optimized interpretations be compared.

This audit is a **design gate**, not proof that a new generator is realistic or that the parser's categories are gold labels. The FineWeb ranges have already been used for other development comparisons, and the corpus mix differs from the synthetic forms. The exact aggregate counts and parser versions are in [130k results](results/semantic_syntax_opportunities_fineweb130000_n512.json) and [140k results](results/semantic_syntax_opportunities_fineweb140000_n512.json). No raw source text is committed.
