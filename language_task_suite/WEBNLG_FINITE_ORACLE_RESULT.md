# Exact WebNLG text oracle is a fixture, not a realistic training DGP

The [finite-support sampler and prefix oracle](webnlg_finite_oracle.py) use only original `good` English lexicalizations from enriched WebNLG v1.6. The explicit law samples a category with probability proportional to its number of entries raised to a configurable exponent, then an entry uniformly within that category and one of its good lexicalizations uniformly. Exponent 0 balances categories; exponent 1 weights categories by entry count. The emitted sequence is the original human text followed by GPT-2 EOS. A weighted token trie gives the **exact marginal next-token distribution** over this law, including duplicate texts. A direct finite-sum check on 16 sampled prefixes matched the trie to absolute error below `1e-10` in both settings, and root mass is 1. This is an exact oracle for a finite corpus distribution, not for arbitrary new semantic graphs.

The support has 17,979 lexicalization instances and 372,592 GPT-2 token-trie nodes. The two 512-draw audits show why this should not be promoted as the main language DGP:

| Statistic | Balanced categories | Entry-frequency categories |
| --- | ---: | ---: |
| Median support after 8 GPT-2 tokens | 2 | 2 |
| One next-token branch after 8 tokens | .783 | .782 |
| Median support after 16 tokens | 1 | 1 |
| One branch after 16 tokens | .926 | .936 |
| Generated 20-word cross-document repeated-trigram share | .412 | .376 |
| Size-matched natural FineWeb 20-word share | .010 | .011 |
| Generated / natural top-ten word share | .331 / .203 | .334 / .200 |

Only 219/512 and 216/512 sampled lexicalizations reached 20 words, so the form panels compare those eligible generated crops with equal-size random natural crops from an existing development preview. They are not topic- or task-matched, and the numbers should be read as a diagnostic screen. The gap is nevertheless too large for this finite-support distribution to satisfy the requested realism. Once a model recognizes a document prefix, it can finish a memorized sentence without learning the RDF relation. Training a bilinear model to low oracle regret on this law would therefore **not** establish recovery of reusable language rules; no such model was trained.

Keep the code as an oracle-correctness and memorization control. The next main-DGP implementation must sample *new* connected semantic graphs and varied realizations with independent relation checks, while using corpus annotations as a prior rather than a finite lookup table. Preserve whole-entity and whole-graph-combination holdouts; evaluate actual rule behavior before weight analysis. The original WebNLG annotations remain locally available for a reviewed subset, but their `good` label and entity-map coverage alone are not sufficient truth checks, as the [prior audit](ENRICHED_WEBNLG_SEMANTIC_FORM_SCREEN.md) found.

Aggregates: [balanced](results/webnlg_finite_oracle_balanced_n512.json) and [entry-frequency](results/webnlg_finite_oracle_empirical_n512.json). No source prose is committed.
