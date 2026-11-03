# A corpus-derived syntax prior: raw PCFG fails, phrase substitution is viable

The current rule-known document generators have a conspicuous small-template form gap. To test a different source of surface variability, I used the public treebank released with [Jumelet and Zuidema's *Transparency at the Source*](https://aclanthology.org/2023.findings-emnlp.288/). Their paper induces a **state-split** PCFG from parsed BookCorpus sentences; this pilot is much smaller and does **not** reproduce their grammar or exact prefix oracle. The external treebank was read locally from [`clclab/pcfg-lm`](https://github.com/clclab/pcfg-lm) at commit `78c44cbf185217c4c62fabecba81bb151be3a584`; no source text or grammar archive is committed here.

Two sampled surface priors were compared:

* **Raw count PCFG:** count each parse-tree production and sample productions independently. This keeps explicit probabilities and derivation rules, but loses lexical and agreement dependencies between productions.
* **One-phrase tree substitution:** sample a complete source parse and replace one 2–10-token NP, PP, ADVP, or ADJP subtree with a training-corpus subtree of the same category. NPs also match a rough singular/plural head-POS signature. The replacement is sampled, not copied from the evaluation slice. This is a limited tree-substitution grammar, not a semantic generator.

The [raw-PCFG form audit](results/raw_pcfg_treebank_20k_n256.json) and [substitution form audit](results/tree_substitution_20k_n256.json) used 20,000 training parses and 3,000 disjoint reference parses. Five independently drawn sentences were concatenated into pseudo-documents and screened at 55 words. In that pilot, repeated-across-document trigram share was .291 for resampled reference text, .159 for raw PCFG, and .211 for tree substitution. **Lower repetition did not mean better English:** raw-PCFG examples were often syntactically broken or semantically incoherent. Tree substitution produced no exact training-sentence copies among 5,000 sampled sentences, yet some phrase swaps still damaged agreement or meaning. These pseudo-documents do not model discourse.

A stronger local-form screen uses one disjoint treebank slice as an n-gram reference and another as independent natural candidates. The [20k-fit comparison](results/syntax_priors_heldout_ngrams_20k_n1000.json) and [100k-fit comparison](results/syntax_priors_heldout_ngrams_100k_n1000.json) report the fraction of candidate n-gram **occurrences** also seen in 3,000 reference sentences. At 100k fit trees:

| Candidate source | Seen bigrams | Seen trigrams | Seen 4-grams |
|---|---:|---:|---:|
| Independent treebank | .462 | .126 | .0245 |
| Raw count PCFG | .261 | .0189 | .0007 |
| One-phrase substitution | .442 | .108 | .0176 |

The 20k fit panel has the same ordering: independent/tree-substitution/raw trigram shares .140/.100/.017. This supports the interpretation that a complete-source-tree backbone retains much more natural local structure than independently sampling unsplit productions. It does **not** show that substitution matches FineWeb, BookCorpus document structure, or the paper's state-split PCFG; it also does not supply a semantic task oracle.

Decision: keep phrase substitution as a **syntax carrier candidate** for a future typed world/meaning grammar. Do not train bilinear models on this arm as a substitute for the target multitask DGP. The next integration must make rule-bearing claims part of the sampled derivation and independently check their meaning, while testing whole-document and task-window form against held-out natural text. If that cannot be done, use the released state-split PCFG as a separate syntax-only baseline rather than claiming it solves semantic interpretability.

The reproducible local scripts are [raw PCFG audit](audit_raw_pcfg_treebank_pilot.py), [tree-substitution audit](audit_tree_substitution_pilot.py), and [held-out n-gram comparison](compare_syntax_priors_ngrams.py). They save aggregates only. The treebank path is an explicit argument, so the external data does not enter the repository.
