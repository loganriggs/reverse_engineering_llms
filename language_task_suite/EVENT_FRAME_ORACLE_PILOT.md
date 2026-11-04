# Variable event frames with an exact visible-prefix oracle

The existing sampled V14 stream has useful known rules and a learned local action circuit, but fresh FineWeb screens show that its fixed sentence frames and short-word head remain very unlike natural text. The [event-frame fixture](event_frame_oracle.py) isolates one missing technical capability: **the same semantic event can be expressed through different word orders and lengths while the next-token oracle marginalizes a still-hidden realization and document genre**. This is a component for a larger generator, not a replacement DGP.

Each document samples one of two latent genres and eight sentences. A sentence may state or update an object's attribute, or describe an action selected by the latest visible attribute of that object. There are three fact and three action surfaces. For example, the same `book → rare` fact can appear as `The book is rare.`, `A rare book was noted.`, or `As for the book, it is rare.` A later action can be `Frank sees the book and inspects it.`, `The book is inspected by Frank.`, or `After examining the book, Frank inspects it.` The grammar uses six attributes/actions from V14, 16 objects and eight people from the frozen V17 typed lexicon, and a count-smoothed frequency exponent of 0.5. These are illustrative choices, not corpus-calibrated task rates.

The oracle enumerates compatible sentence completions under each genre, replays completed facts from visible text, and sums the two genre likelihoods for every prefix. It never conditions on the sampler's private genre draw. The [audit](audit_event_frame_oracle.py) independently compares the product of next-token probabilities with the directly summed complete-document probability, checks support and normalization for every sampled token, and changes an earlier attribute across every one of the **3 fact × 3 action** frame combinations. The supported edit swaps the appropriate active or passive action probability while preserving the other next-token probabilities. Generated examples were manually reviewed; an initial ungrammatical passive/tense realization was corrected before the final audit.

Two independent 100-document panels passed all 200 paired-event checks: fixed and variable arms have identical sampled genre and semantic event sequences, differing only in wording. Both panels contain 328/329 later action sites. The worst direct-versus-chain probability relative error is below `8e-15`. In the variable arm, 107/114 action sites have more than one next-token possibility because the prefix `The book is` can start either a fact or a passive action. Those sites must be judged by **oracle distribution/KL**, not a planted deterministic target label. The fixed arm has 328/329 deterministic action sites.

| 100 paired eight-sentence documents | Variable frames, seed 42001 / 42002 | Fixed frames, same events |
|---|---:|---:|
| Distinct sentence-start words | 12 / 12 | 9 / 9 |
| Top-ten word share over full documents | .566 / .560 | .653 / .657 |
| Action sites | 328 / 329 | 328 / 329 |

The [fresh FineWeb form screen](compare_event_frame_oracle_fineweb.py) scanned 128 unselected `sample-10BT` documents at offsets 440,000–440,127; 80 had at least 40 words in all three paired arms. On exactly the first 40 alphabetic words in each eligible document:

| 80 matched windows | FineWeb | Variable frames | Fixed frames |
|---|---:|---:|---:|
| Distinct word types | 1,583 | 55 | 41 |
| Top-ten word share | .194 | .558 | .663 |
| Within-window repeated-word fraction | .154 | .510 | .601 |
| Distinct segment-start words | 141 | 12 | 9 |

Variable realization improves the controlled form comparison, but **both generated arms fail the realism gate by a large margin**. The problem is not only inflection or frame order: the grammar has a tiny ontology, only two discourse acts, no document structure, and recurrent stock phrases. Its examples are grammatical after the correction but still stilted. No bilinear model was trained on this fixture as a purported realistic language model; the V14 learned-action checkpoint remains the available mechanism control.

The next integration should apply this candidate-summing prefix method to a broader proposition/document planner: vary act order and sentence form, use the typed lexical tail, and ensure each completed utterance has a unique visible semantic parse. Audit the full-token oracle, supported edits, task-family exposure, and fresh FineWeb full-document and task-window form separately. Only after that should 1–4-layer bilinear models be compared behaviorally on whole-form and whole-task-combination holdouts, followed by matched-fidelity canonicalization and sparsity tests on rules they learned.

Aggregates: [seed 42001](results/event_frame_oracle_seed42001_n100.json), [seed 42002](results/event_frame_oracle_seed42002_n100.json), and [fresh FineWeb form screen](results/event_frame_oracle_fineweb_start440000_n128.json). The reports include only generated prose examples; no FineWeb source prose is published.
