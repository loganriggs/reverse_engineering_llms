# Corpus-induced grammar and visible-state bridge

This pass follows the probabilistic-grammar proposal while keeping two requirements separate: language-like surface form and an exact, visible-prefix predictive law. The results do not yet satisfy both at once.

## Corpus grammar scale and surface form

I ran the released [Jumelet–Zuidema grammar-induction pipeline](https://github.com/clclab/pcfg-lm) on local prefixes of its parsed treebank, with 4–8 GB Java heaps. The 20,000-line input slice has SHA-256 `503cf4c44ba4212435c83e8023d467506bc7979e81147492bce0fa88659171b8`; the 50,000-line slice has SHA-256 `c3b6b5331cbc4e69c78652ff91134d2549d3469dec385f43fd095089de365492`. The source treebank and derived grammars remain outside this repository. `berkeley_pcfg_sampler.py` safely reads the exported grammar/lexicon for form-only sampling; `audit_induced_pcfg_form.py` saves only aggregate comparisons with the held-out FineWeb 25000–25047 panel. Sampling used no explicit-bound rejection in these panels, but there is no claim here that this forms an oracle-scored training corpus.

| Grammar input lines / split–merge cycles | Rules after lexical expansion | Word types / 4,300 words | Top-ten word share | Top-ten adjacent-pair share |
| --- | ---: | ---: | ---: | ---: |
| FineWeb comparison panel | — | 1,804 | 19.9% | 3.23% |
| 1,000 / 1 | 7,011 | 1,178 | 23.1% | 2.97% |
| 5,000 / 1 | 18,792 | 1,386 | 23.7% | 3.06% |
| 5,000 / 3 | 77,309 | 1,402 | 24.0% | 3.02% |
| 20,000 / 1 | 46,414 | 1,456 | 23.2% | 2.52% |
| 20,000 / 3 | 190,391 | 1,543 | 22.3% | 2.69% |
| 50,000 / 3 | 276,247 | 1,484 | 22.8% | 2.95% |

The larger grammar improves lexical diversity up to 20,000 input lines, yet the sampled sentences still frequently have broken syntax or implausible word relations. A matched CPU GPT-2 form diagnostic uses the same whitespace-token normalization for natural and generated text. Its token NLL is 4.144 nats on the FineWeb panel, 6.976 on the 20,000/1 grammar, 6.779 on 20,000/3, and 6.607 on 50,000/3; the 50,000/3 per-UTF-8-byte loss is 1.491 versus FineWeb's 0.854. This is a large gap despite improving word/pair counts. GPT-2 NLL is not the source oracle or a proof of semantic naturalness: the V7 hand grammar scores 4.685 nats/token and **0.818 nats/byte**, below the natural panel's byte score, despite only 277 reference-regime word types. Simple repeated frames are easy to predict. V5 retrieved sentences score 5.013 despite near-matched local counts, while V6 n-gram text scores 6.465. These diagnostics disagree for understandable reasons; passing one is insufficient.

Across ten synthetic seeds for the 20,000/3 grammar on that fixed FineWeb panel, mean word types are 1,522 ± 22 SD and mean top-ten word share is 22.9% ± 0.45 percentage points. The observed single-seed improvement is not merely a lucky sample, but these SDs do not quantify uncertainty in the FineWeb population or grammar induction.

At 50,000/3, ten-seed mean word types **fall** to 1,474 ± 13 SD, while mean top-ten share remains 22.9% ± 0.52 points. The larger training slice modestly improves GPT-2 NLL by 0.173 nats/token but does not improve these lexical screens. This does not rule out improvement with the released full 500,000-treebank grammar; it does rule out treating mere expansion from 20,000 to 50,000 as a solved realism step. I am stopping the grammar-only scale sweep here and moving the next main experiment to semantic/discourse conditioning and richer realization, with the full grammar retained as a later comparator.

`audit_induced_pcfg_lm_gap.py` reproduces the normalized GPT-2 screen without saving source or generated prose. The form figures come from one 43-document held-out panel and one synthetic seed per grammar, so they are diagnostic rather than population estimates.

## Exact-prefix bridge and its failure modes

The paper's EarleyX parser matches an independently enumerated ambiguous toy grammar: token surprisal for `a c .` differs by at most 1.7e-16 nats from exact enumeration (`audit_earleyx_bridge.py`). On 100 sentences sampled from the 5,000/1 grammar, an initial conversion scored only 64 finitely. All 36 failures began at apostrophe-bearing tokens: the paper converter maps `'` to `<apostrophe>` in its lexical rules. Applying the same token mapping to parser input made all 100 sentences finite. This is a tokenization contract, not a grammar-learning failure.

Pruning rules below 1e-7 changed some sampled token surprisals by as much as 0.189 nats on a ten-sentence comparison. We therefore used zero pruning for the support/runtime audits so the sampler and parser use the same expanded rules. With apostrophe normalization, EarleyX scored all 341 tokens of 30 sentences from 5,000/1 in 6.27 seconds; all 482 tokens of 30 sentences from 20,000/1 in 66.33 seconds; and all 23 tokens of three sentences from 20,000/3 in 16.60 seconds, on CPU. These are sampled-target scores, not a full-vocabulary distribution at every document position. The growth in cost makes an exact all-token oracle for a large document grammar a real engineering constraint. `audit_induced_pcfg_prefix.py` records grammar hashes, finite support, and runtime; the raw samples and parser logs stay temporary.

## Visible document state

`stateful_genre_grammar_v8.py` and `genre_document_v8.py` extend the small exact grammar fixture so already-visible profiles and source claims can influence later background mentions. Independent name draws are explicitly worded as guesses; they do not modify document facts. A supported swap of an earlier profile from Alice to Bob flips the later prediction after “The pilot in the report was” from Alice to Bob. A supported swap of which person described a rare badge does the same after “The badge was described as rare by”. The exact joint document oracle agrees with the direct grammar calculation in all four counterfactual cases. In 30 task-rich and 30 reference 256-token streams, sampled support, independent prefix replay, visible-state replay, and retained bracket/number/addition/source checks pass.

This is an inference and causal-consistency fixture, not a realistic renderer: its reference panel has only 347 types per 4,300 words and a 40.2% top-ten share. It also does not introduce new facts through prose; facts still enter via the old task grammar.

`audit_pcfg_visible_entity.py` tests a narrow bridge to an *induced* grammar. The name in a visible V4 profile changes the Berkeley PCFG's proper-noun lexical distribution, and each variant is converted into a matching EarleyX grammar. Across five sampled sentence contexts, changing the visible entity from Alice to Bob changes the candidate name probability by a factor of 304 (5.717 nats), with both names in support. This ratio is a consequence of the chosen lexical mixture, not evidence that a trained model learns entity binding. It does show that corpus-derived syntax, document-state-dependent lexical rules, and exact candidate scoring can be connected without a hidden-state label leak.

## Next gate

The next renderer should use a document planner with typed events/entities and a grammar that conditions *relations*, not just proper nouns. It must either make task/background branches observably distinguishable or marginalize both when their sentence prefixes overlap. For visible document history `D`, a fixed state/genre hypothesis `h`, task branch `T`, grammar branch `G`, and sentence prefix `u`, define `Z_h(u) = (1-p_bg) P_T(u|h,D) + p_bg P_G(u|h,D)`. Then the exact target is `P(w|D,u) = [sum_h P(h|D) Z_h(uw)] / [sum_h P(h|D) Z_h(u)]`. The posterior over `h` must itself be updated from visible text. A sampled derivation or sampled hidden fact is not the target.

The current PCFG is induced from the paper's book-like parsed treebank, not FineWeb. Before model training, test larger grammars and richer semantic realization against independent FineWeb lexical, layout, coherence, and context-benefit screens; verify exact target probabilities on a tractable subset. Keep whole-task and held-out-combination splits. Only then compare bilinear depth, oracle regret, causal edits, sparse circuits, and canonical decompositions on genuinely learned tasks. The existing V7/V8 models are not trained and should not be presented as successful language models.
