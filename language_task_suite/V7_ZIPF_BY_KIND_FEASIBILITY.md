# V7 word-head feasibility by document kind

The [reproducible audit](audit_v7_zipf_by_kind.py) samples 500 documents of
each V7 kind with a fixed seed, lowercases alphabetic words using the same
regex as the [word-rank audit](V14_SAMPLED_EXAMPLES_AND_ZIPF.md), and compares
the share of `the` with the frozen 480 × 100-word FineWeb reference. The
[aggregate counts](results/v7_zipf_by_kind_n500.json) contain no natural
source text.

| Corpus / V7 kind | Mean words per document | `the` in first 100 words | `the` in full documents | Full-document top-ten share |
|---|---:|---:|---:|---:|
| FineWeb reference | 100 | 4.86% | — | 21.41% |
| Narrative | 152 | 19.97% | 19.60% | 35.94% |
| Event report | 126 | 14.97% | 15.10% | 33.63% |
| Record listing | 21 | — | 14.29% | 83.58% |
| Score report | 23 | — | 21.74% | 78.10% |
| Cross report | 170 | 14.51% | 16.01% | 33.55% |

The first-100-word comparison conditions on documents long enough to have
100 words: all narrative and cross reports and 433/500 event reports. Record
and score reports are too short for that window. Within those three long
kinds, no nonnegative reweighting can bring the `the` share down to the
FineWeb reference: the least concentrated kind is still 14.51% in the
matched-length window. Across complete documents, the minimum of the five
estimated kind-level shares is 14.29%. These are sample estimates, and the
FineWeb panel is independent rather than paired, but the gap is large.

This narrows the next DGP intervention. Changing genre frequencies alone
cannot solve the common-word head; the within-kind clause renderer needs
more variable sentence structures. A lexical Zipf prior over nouns would
add rare types but leave these fixed articles and repeated frames mostly
untouched. Any renderer change should retain the visible-prefix answer
checks and causal edits, then rerun the matched lexical and form screens
before training a new model. The result is a lexical feasibility finding,
not evidence about which neural circuit learns a rule.
