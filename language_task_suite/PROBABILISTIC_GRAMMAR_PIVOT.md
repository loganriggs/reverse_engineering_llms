# Probabilistic grammar pivot: controls and first oracle fixture

The current genre-document generator is too template-heavy to serve as a realistic language pretraining distribution. This pass tested two corpus-background shortcuts and implemented a small ambiguity-aware grammar oracle. The grammar is a correctness fixture; it is **not** the target generator.

## Why this pivot

[Jumelet and Zuidema (2023)](https://aclanthology.org/2023.findings-emnlp.288/) induce a large state-split PCFG from a parsed corpus, generate language-like data, and evaluate models against its known distribution. Their [released code](https://github.com/clclab/pcfg-lm) contains the Berkeley grammar-induction pipeline and EarleyX causal-prefix evaluation. It is a strong reference for our syntax and oracle architecture, but the paper also reports degraded selectional preference relative to real text and says the causal prefix evaluation was limited to about 1,100 token probabilities because EarleyX did not exploit their coarse-to-fine grammar. Their code specifies a 16 GB heap for induction and 32 GB for causal-prefix evaluation. None of that yet provides a document world state or a tractable exact joint oracle for source binding, copying, arithmetic, and grammar at once.

The released corpus-generation code also filters sampled sentences by length. If we adopt a similar rejection rule, the oracle must condition on acceptance; using the unconditioned PCFG prefix law would score a different distribution. The first integrated system should avoid rejection or account for it explicitly.

I cloned the paper code outside this repository and ran a *small local induction smoke test*: first 1,000 lines of its included parsed treebank, which the Berkeley trainer read as 1,700 trees, one split/merge cycle, 4 GB Java heap. It produced 1,228 grammar-text rules and 3,183 lexicon rows. This verifies that the induction path runs on this machine. The treebank, induced grammar, and lexicon stay outside this repository; this is not a calibrated corpus or an exact-oracle result.

## Corpus-background controls

V5 inserts whole sentences from a frozen local FineWeb token cache into the exact V4 task document. The trie gives an exact oracle for that **finite empirical sentence bank**. V6 samples from an interpolated trigram/bigram/common-unigram law over the same bank. Both retain V4's planted task logic and exact genre marginal. The source text/cache is local only; committed outputs contain aggregate statistics and hashes.

The independent FineWeb form panel has 43 eligible documents, with the first 100 alphabetic words per document. The figures below are diagnostic, not a representative population estimate.

| Distribution, reference regime | Word types / 4,300 words | Top-ten word share | Adjacent pair types | Top-ten adjacent-pair share |
| --- | ---: | ---: | ---: | ---: |
| FineWeb panel | 1,804 | 19.9% | 3,022 | 3.23% |
| V5 sentence bank | 1,711 | 21.8% | 3,028 | 4.49% |
| V6 n-gram sampler | 1,473 | 23.9% | 3,126 | 3.62% |

V5 gets close on these local statistics but its next-token distribution becomes effectively a lookup: in 100 reference documents, 99.9% of background sites after ten words have only one possible next token. V6 retains entropy (mean 3.74 nats after ten words, only 0.76% deterministic sites) but generated continuations lose syntax and semantic coherence. All sampled-support, replay, and retained task-oracle checks passed; that validates the **specified** distributions, not their realism. No new bilinear model was trained on either control.

Files: `corpus_backed_document_v5.py`, `corpus_markov_document_v6.py`, `audit_corpus_backed_document_v5.py`, `audit_document_form_gap.py`, and the matching `results/corpus_*_audit_n100.json` plus `results/genre_v3_v4_v5_v6_vs_fineweb_25000_form_gap.json`.

## Exact small-grammar fixture

`weighted_cfg_oracle.py` implements prefix filtering for **acyclic** weighted CFGs. It keeps a weighted distribution over pending symbol stacks and sums all derivations consistent with the visible prefix; the next-token law is normalized frontier mass. It intentionally does not claim to scale to the paper's recursive, million-rule grammar. `small_genre_grammar_v7.py` supplies a hand-built, genre-conditioned sentence grammar; `genre_document_v7.py` mixes it into the V4 document stream without changing the planted task oracle.

An independent exhaustive enumerator checks all 21 prefixes of a tiny ambiguous grammar. The surface sequence `a c . <eos>` has two derivations with total probability 0.245, which the prefix oracle recovers. In 30 task-rich and 30 reference 256-token documents, sampled probabilities normalized, independent prefix replay passed, and semantic replay plus bracket, step, section, addition, and source-answer checks passed. The word-token boundary is shared by grammar and model data; GPT-2/BPE compatibility is not established.

The form screen rejects V7 as a natural-language generator: reference documents have 277 word types and 45.0% top-ten word share, compared with FineWeb's 1,804 and 19.9%. Its long-background sites have mean next-token entropy ~1.01 nats and 26.7% deterministic probability support. That is better than V5's retrieval degeneracy but is still shallow, repetitive text. Its value is a tested oracle interface and ambiguous-parse marginalization.

Files: `weighted_cfg_oracle.py`, `small_genre_grammar_v7.py`, `genre_document_v7.py`, `audit_genre_document_v7.py`, `results/genre_document_v7_*_audit_n30.json`, and `results/genre_v7_vs_fineweb_25000_form_gap.json`.

## Next experiment and gates

1. Induce a corpus grammar on a larger, separately selected parsed sample; audit generated syntax, word frequency, sentence lengths, n-grams, and within-document repetition against held-out FineWeb. Do not use a sampled-sentence bank as the final oracle.
2. Connect grammar derivations to typed semantic slots and a bounded document planner/world state. The visible text, not sampled hidden state, determines evaluation. Preserve existing bigram, skip-rule, bracket, induction, arithmetic, source, and section mechanisms; make discourse acts and surface realizations probabilistic and cross them in the **same** sampled documents.
3. Implement an exact prefix oracle on a small state/grammar slice first. Verify against exhaustive enumeration and posterior normalization; measure how ambiguity and world-state size affect runtime. A parser that only scores the realized hidden derivation fails this gate.
4. Measure whether longer context provides predictive information at each task site, and run support-preserving counterfactuals. Keep whole-task and held-out-combination splits. Train pilot 1–4-layer bilinear models on generator variants with documented semantic rules and measured form properties; an exact full next-token oracle is useful where tractable, but is not a prerequisite. Establish which rules each model learned behaviorally before interpreting their weights. Compare canonicalization and sparsity at matched functional fidelity and report realism, competence, and causal recovery separately. The current study design is in `DGP_TO_WEIGHT_INTERPRETABILITY_PROTOCOL.md`.

The paper's state-split PCFG is the syntax reference, not proof that one grammar alone solves document-level mechanistic ground truth. V5/V6 are negative controls; V7 is an inference fixture. The main research question remains whether a learned bilinear model recovers reusable computations under a rich, known language distribution.
