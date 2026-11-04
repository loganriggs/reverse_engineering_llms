# Numbered-list continuation as a corpus-grounded rule

Two prior sampled-generator changes, V15 alternate color frames and V16 a broader proposition bank, preserved visible rules but increased repeated language. This experiment instead locates a simple rule **already embedded in varied FineWeb prose**: line-initial numbered items `1.`, `2.`, `3.`. A plan records a maximal local run of consecutive one-digit list markers. Its counterfactual shifts every marker in that run by one, so the predicted third marker changes from `3` to `4`. The contents of recipe instructions, survey questions, product specifications, and other list items remain source prose. If a prose parenthetical lies nearby, its `(` and `)` can independently become `[` and `]`, giving four edit arms.

The renderer changes only selected marker digits and, when requested, one bracket pair. A separate visible-prefix checker derives the next list number from the last two line-initial markers. The audit verifies exact edit locality, both answer changes, all four BPE target alignments, and that the first list marker, bracket opener, and both target sites fit within a 320-token context. In every retained four-arm case, the BPE offsets agree through the list target and only tokens containing edited characters change. It does **not** certify that a page's other references to list numbers stay semantically consistent, or that the entire human article has a symbolic next-token oracle.

| FineWeb `sample-10BT` offset panel | 95,000–104,999 | 105,000–114,999 |
| --- | ---: | ---: |
| Documents screened | 10,000 | 10,000 |
| Strict `1,2,3` local runs | 136 | 125 |
| Two-arm list edit, BPE/context aligned | 128 | 121 |
| List plus nearby prose parenthetical | 57 | 42 |
| Complete list × bracket four-arm panel | 34 | 21 |
| Also strict token-local four-arm edits | 34 | 21 |

At the 256-token context of the existing delimiter models, the corresponding list-only counts are 113/114 and strict joint four-arm counts are 29/18. Thus the same architecture can be used for an initial comparison, with a smaller joint test panel.

The rule is uncommon, about 1.2–1.3% of documents at this strict context criterion, but present in distinct kinds of source pages. The 95k panel's 136 selected first-100-word passages have top-ten word share .2135 and repeated-across-document trigram share .0257; the 105k panel gives .2117 and .0255. At 41-word windows centered on the list target, these are .2100/.0117 and .2111/.0074. A separate FineWeb first-100-word reference was about .209/.016, but it used a different offset and was not list-conditioned, so these are **coarse form screens**, not proof of distributional equivalence. The source and shifted forms have identical word-only statistics because the audit's word regex excludes digits; the edit validator, not those statistics, establishes the number change.

This is a viable candidate for a third behavioral rule and a modest heldout list × bracket combination panel. The next decisive step is to build local token caches from a larger disjoint training offset range, with exact/near-duplicate filtering against the heldout ranges; train small one- and two-block bilinear models on ordinary next-token loss plus list-site weighting, mixing bracket-only examples without joint training examples; then test paired list and bracket behavior on fresh four-arm documents. Only if the list rule is learned should weight-based canonicalization or sparsity be asked to recover it. The data are sparse enough that an initial failure would call for more training examples or changed task prevalence, not a claim about circuit absence.

Reproduce with `corpus_numbered_list_dgp.py` and `audit_corpus_numbered_list.py`. The aggregate reports are `results/corpus_numbered_list_skip95000_n10000.json` and `results/corpus_numbered_list_skip105000_n10000.json`, with parallel `_ctx256.json` reports; no source prose is committed.
