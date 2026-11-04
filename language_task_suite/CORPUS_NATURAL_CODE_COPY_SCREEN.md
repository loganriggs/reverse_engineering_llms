# Natural repeated-code copy: opportunity screen

The two learned rules in the [ordinary-order transfer test](CORPUS_RULE_REFERENCE_TRANSFER_RESULT.md) are list continuation and delimiter closure. This screen looks for a computationally different third target: a later occurrence of the same three-letter uppercase code in unedited FineWeb prose. It measures source opportunities only; it does not establish that a model predicts by copying.

The [auditor](audit_corpus_natural_code_copy.py) scans source documents uniformly, without selecting for a task. Its narrow detector finds an exact three-letter uppercase string repeated 20–160 words later, excluding obvious three-letter English words. Both occurrences must be a single identical GPT-2 BPE token and fit in the same ordinary, nonoverlapping 256-token chunk; no text is edited. The rule-site label is simply “repeat the earlier code.” All other tokens retain ordinary next-token targets, but have no exact symbolic oracle.

| FineWeb source offsets | Sources | Documents with a local code-copy site | Local sites | Distinct codes | Top-20 code share | Mean gap, words |
|---:|---:|---:|---:|---:|---:|---:|
| 373k–378k | 5,000 | 357 | 747 | 208 | .340 | 51.0 |
| 378k–383k | 5,000 | 322 | 697 | 194 | .377 | 52.1 |

This is substantially denser than the list rule's roughly 50–60 local documents per 5,000 sources, and more diverse than a tiny synthetic code vocabulary. It still selects a special subset: the first panel has 767 documents with a repeated-code candidate, but only 357 survive the one-token and ordinary-context filters. These rates belong to this narrow detector, not to all natural-language copying.

A local-only review of 24 sampled sites found many genuine document references (`API`, `SQL`, `GDP`, `NBA`, etc.), but also code repetitions that are readily predictable from topic or broad world knowledge. A few scraped documents appeared to concatenate unrelated passages. The top 20 codes account for 34–38% of sites, so a lexical-prior model could do well without reading the prior occurrence. Equality of the source and answer is therefore insufficient to call this an induction task. The earlier [acronym-substitution screen](CORPUS_UPPERCASE_COPY_DGP_SCREEN.md) found that replacing such codes with arbitrary familiar acronyms often distorts the meaning; do not use that substitution as a realism claim.

**Decision:** keep naturally repeated codes as a candidate third rule family in the unedited reference stream. Before using it as a weight-interpretation target, train or evaluate models on held-out whole source documents, then apply a *diagnostic-only* consistent code edit and measure whether the answer distribution moves toward the edited prior code. Include a context-masked and code-frequency baseline, hold out source-code combinations, and inspect any selected examples where unrelated scraped passages meet the regex. A model that merely predicts frequent acronyms or local topics has not passed the causal copy gate. The edited arm can test mechanism even if its factual semantics are alternate-world rather than authentic FineWeb.

The [first](results/corpus_natural_code_copy_start373000_n5000.json) and [second](results/corpus_natural_code_copy_start378000_n5000.json) aggregate JSONs contain no source prose. Review excerpts remain local.
