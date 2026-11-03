# Hidden-genre joint-language pilot

This pilot tests a specific idea from probabilistic language modeling: a document has an unobserved style/genre, so each new word updates the reader's belief about it. Genre changes the distribution of **later** sentence types, not merely the opening label. It is a structural benchmark, not a realistic web-text simulator. The natural-language gap remains a hard failure.

[`genre_document_v3.py`](genre_document_v3.py) extends the [joint V2 source/task stream](JOINT_DOCUMENT_V1_V2_RESULTS.md). The hidden genre is court, encyclopedia, procedure, or math. All four can generate profiles, narrative events, claims, source questions, numbered sections, numbered steps, and addition problems, but their probabilities differ. A court document favors competing claims and source questions; an encyclopedia document favors profiles and sections; a procedure favors `Step 1 : ...`; a math document favors `Problem : 7 + 5 = ? Answer : 12 .`. The same document still contains planted bigram, induction, topic/place, claim, bracket, and section rules from V2. Its 519-token vocabulary and finite sentence templates are deliberately modest.

The generator samples a fixed hidden genre per document, but its exact visible-prefix next-token oracle marginalizes over genre:

\[
p(w_t\mid w_{<t})=\sum_g p(g\mid w_{<t})p(w_t\mid w_{<t},g),\qquad
p(g\mid w_{\le t})\propto p(g\mid w_{<t})p(w_t\mid w_{<t},g).
\]

Every genre gives positive support to every sentence form. The weak title cue can therefore be revised by later text. A title-topic edit is supported and changes the oracle's next-sentence distribution by mean total variation 0.1525 over 200 fresh documents. This is an actual probability shift, not a deterministic genre label.

The [task-rich](results/genre_document_v3_task_rich_audit_n200.json) and [source-reference-rate](results/genre_document_v3_reference_audit_n200.json) audits each replayed 200 × 256 prefixes, including independently recomputed prefix oracles at selected positions, normalization, sampled-token support, arithmetic answers, step numbers, and source-answer parsing. In task-rich documents, 701 source sites all have multiple speakers for the same item; the last-claim-speaker baseline gets 268/701 (38.2%). One hundred earlier-attribute edits were checked to preserve support and change the answer. The oracle's most likely genre matches the sampled hidden genre on 52% after the four-token title, 80% after 32 tokens, 89% after 64, 95% after 128, and 99% after 255. These are **oracle** identification rates, not evidence that the model has a genre circuit.

The realism check is negative: in 49 documents × the first 100 alphabetic words, task-rich V3 has 368 types, 63 singletons, and a 35.8% top-ten share, versus 1,993 types, 1,336 singletons, and a 21.4% top-ten share in the [FineWeb screen](results/fineweb_predictive_inventory_skip23000_n64.json). The title is often a generic place even for math; sentence frames repeat; there are no newline, quote, citation, or capitalization processes. The four hidden-genre priors are uniform in **both** regimes. “Reference” only lowers V2 source-question exposure; it is not a corpus-calibrated reference distribution. The 24-document [FineWeb opportunity panel](results/fineweb_manual_opportunities_skip23000_seed23012_n24.json) is too small and has different page forms, so it cannot justify these genre rates.

## Bilinear learning pilot

[`train_genre_document_v3.py`](train_genre_document_v3.py) trained the original width-64, four-head bilinear attention/MLP models with RMSNorm for 1,000 full-next-token CE updates at depths one through four, on 8,000 independently sampled 256-token task-rich streams. Two disjoint 1,000-stream panels use task-rich and source-reference rates. No genre or task labels were supplied to training. Exact train/evaluation document overlap is zero, but all task families and operand pairs occur in training. There is one seed per depth.

| Blocks | Task-rich token NLL | Genre sentence-start oracle KL | Addition answer accuracy | Step number accuracy | Bracket closer accuracy | Source answer accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.619 | 0.444 | 45.6% | 94.2% | 99.5% | 16.4% |
| 2 | 1.446 | 0.263 | 91.3% | 97.5% | 99.9% | 44.0% |
| 3 | 1.365 | 0.227 | 98.4% | 96.4% | 100% | 42.2% |
| 4 | 1.349 | 0.180 | 98.3% | 98.2% | 100% | 43.3% |

The [full model JSON](results/genre_document_v3_bilinear_depth1_4_1000.json) reports exact-oracle KL at every scored site, both panels, parameter counts, and training settings. “Addition” here is a tiny **seen operand-pair table** (0–9 + 0–9), not evidence of an extrapolating arithmetic algorithm. Step numbering and bracket closure are strong competent probes. Genre sentence starts improve but still have substantial KL. Source binding remains unlearned: depth-four accuracy is only modestly above the 38.2% shortcut, and no V3 source edit fidelity is claimed.

The supported [title-topic edit evaluator](eval_genre_document_v3_title_edit.py) tests model response to a genre cue before the first body sentence. On 500 fresh titles, the oracle's mean next-token total variation is 0.154. Model variation is [0.057](results/genre_document_v3_depth1_title_edit_n500.json), [0.073](results/genre_document_v3_depth2_title_edit_n500.json), [0.160](results/genre_document_v3_depth3_title_edit_n500.json), and [0.041](results/genre_document_v3_depth4_title_edit_n500.json) for one through four blocks. Magnitude alone is misleading: even depth three has mean full-vocabulary effect L1 error 0.257 against an oracle effect L1 of about 0.307. This is partial cue sensitivity, not a faithful genre computation. One training seed prevents a depth-ranking claim.

## Theory and next tests

[Compound PCFGs](https://aclanthology.org/P19-1228.pdf) make syntactic rule probabilities depend on a sentence-level latent variable. [Adaptor grammars](https://homepages.inf.ed.ac.uk/sgwater/papers/nips07-adaptor.pdf) let earlier structures alter later production probabilities. [EntityNLM](https://aclanthology.org/D17-1195/) explicitly models the introduction and reuse of entities. These are inspirations for separate components of a document-level probability program; V3 implements only a tiny fixed-genre filter and visible V2 state. Recent [PCFG learnability theory](https://proceedings.mlr.press/v306/parley26a.html) suggests varying ambiguity and cross-scale correlations rather than reading in-distribution loss as recovered structure. [COGS](https://aclanthology.org/2020.emnlp-main.731/) and [SLOG](https://aclanthology.org/2023.emnlp-main.194.pdf) motivate whole-combination and structural holdouts.

Next, build a genuinely broader document program: multiple sentence shapes per communicative function, entity salience/aliases and exogenous first mentions, a heavy-tailed expanding lexicon, newline/page layout, and genre-conditioned but overlapping syntax. Measure it against an independent natural-text panel before promoting it to the main benchmark. Keep task-rich and corpus-reference distributions separate. For interpretation-method tests, use the currently competent bracket/number/local-rule controls, then ask whether a sparse or canonical basis preserves **oracle edits** and held-out task combinations. Do not use V3 source binding or addition extrapolation as a claimed learned circuit.
