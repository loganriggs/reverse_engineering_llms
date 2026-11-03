# Coordinating the same propositions does not fix the language gap

The [V13 packed-introduction pilot](document_world_v13_packed_intro.py) asks a narrow structural question: does combining several true object descriptions into coordinated clauses improve form over V10's separate object introductions? It retains V10's sampled event world, source attributions, typed lexical mapping, discourse acts, and answer sites. Three objects' owner, color, place, and alias facts can be expressed in four clause orders, grouped into sentences with semicolons or “, and”. A visible-text parser recovers every introductory fact, compiles it to the canonical V10 representation, and checks later answers without reading the sampled world or clause plan. The introduced preposition is checked against the surface place's type.

The [paired audit](audit_document_world_v13_packed_intro.py) uses two independent 64-world panels. All four arms share worlds and the same first-100-word eligibility; each arm has 100 or more words in all 64 draws per panel. The V10 control and V13 packing rates 0, .75, and 1 are compared. Each panel has 576 packed introductory facts parsed and checked against the sampled worlds (1,152 total). Every answer site passed the visible-prefix checker, including 383 source-attribution sites in each arm across both panels; 128 supported upstream edits changed their intended first answer and all edited sites still checked. This validates the experiment's truth conditions, not model learning.

| First 100 words, 64 worlds | V10 control | Pack 0 | Pack .75 | Pack 1 |
|---|---:|---:|---:|---:|
| Word types, seed 53701 / 53702 | 467 / 482 | 439 / 442 | 437 / 446 | 428 / 442 |
| Top-ten word share | .306 / .309 | .336 / .335 | .341 / .336 | .343 / .337 |
| Cross-document repeated-trigram share | .398 / .393 | .407 / .397 | .411 / .394 | .409 / .392 |
| Sentence-start types | 105 / 101 | 106 / 110 | 106 / 106 | 103 / 101 |

In 41-word windows at a source-attribution site, repeated-trigram share is .339/.380 for V10 and .349/.340 for full packing; there is no clear local improvement. A separate matched-length FineWeb rule-window screen found roughly 1% repeated-trigram share, although those natural windows are not source-task matched.

The result is negative: changing how the *same three fixed object propositions* are packed does not diversify the document's content or its rule-bearing contexts. It raises the common-word head and reduces word types. This is not a candidate main training DGP, and no new model was trained on it. The parser/plan code is useful as a controlled test of proposition composition, but more connector and clause variants are not the next research step. A stronger DGP must vary the proposition graph itself: which entities and relations matter, which claims are asserted or revised, how many acts appear, and how those acts connect to later predictions. That change should be checked on both full documents and task windows, with whole-combination holdouts and behavioral gates before weight-based method comparisons.

The [aggregate results](results/v13_packed_intro_n64.json) include generated examples and form statistics, no natural source passages.
