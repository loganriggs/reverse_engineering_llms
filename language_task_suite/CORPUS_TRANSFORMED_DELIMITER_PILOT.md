# Natural-parenthetical delimiter rule: denser controlled language task

This is a second **corpus-transformed control** alongside the sparse
[full-name reuse pilot](CORPUS_TRANSFORMED_NAME_REUSE_PILOT.md). Sample a
natural FineWeb passage, select an already-present prose parenthetical,
and sample whether its matching pair is `(...)` or `[...]`. The enclosed
words stay fixed. At the closing position, the visible opener determines
the correct closer. This is one known sampled rule inside empirical
natural language, not a symbolic oracle for the whole source text.

The [local audit](results/corpus_transformed_delimiters_v1_audit.json)
used the frozen 480-window, 513-GPT-2-token panel. Eligible pairs are
nonnested, contain 3–30 alphabetic words and at most 220 characters, and
exclude line breaks and code-like marks. It found **218 eligible windows**
with **447 eligible pairs**. For one selected pair in each window, all
218 paired opener/closer edits passed the string-level localized-change
check. **69** windows also supported a preceding, unrelated-pair edit
that left the selected closer unchanged. The enclosed span was 3–25
words, mean 7.0. With the **entire passage** tokenized, **183/218**
round-versus-square pairs have a stable visible-prefix tokenization and
the closer begins the next GPT-2 token in *both* styles. Only **93/218**
have a pure one-character closer token in both styles: GPT-2 often merges
the closer with following punctuation. A token-level evaluation should
use the 183 aligned pairs and retain the complete next-token target, or
score the full changed token sequence.

On a separate [500-document stream](results/corpus_transformed_delimiters_full_docs_skip43000_n500.json)
at FineWeb offsets 43,000–43,499, examining each document's first 8,000
characters, **177 documents** have at least one eligible pair and there
are **413 pairs** total. This is much denser than the strict full-name
reuse rule's 4/500 eligible documents on the same source panel.

## Realism and sampling-rate tradeoff

Word-level form is unchanged by the punctuation edit: among 218 eligible
local windows, first-100-word type count is 5,398 in both the source and
forced-square arms, and cross-window repeated-trigram share is 3.46% in
both. The elevated repeated-trigram figure is partly because this frozen
row cache can contain overlapping windows from the same source document;
it should not be compared directly to independent-document FineWeb
panels. The text remains natural source prose, but square brackets can be
less idiomatic than round parentheses in some contexts. A manual sample
included ordinary asides, citations, interface prompts and numeric notes;
it does not guarantee every eligible span is good editorial prose.

The `square_rate` DGP knob separates an enriched experiment from a
natural-form setting. The frozen panel originally has **20/480** windows
with a square-bracket pair. Resampling one eligible pair per window gives:

| `square_rate` | Selected square replacements | Windows with a square pair |
|---:|---:|---:|
| 0 | 0 | 20 |
| .01 | 5 | 24 |
| .04 | 16 | 35 |
| .10 | 21 | 40 |
| .50 | 100 | 116 |

The rates are a development sweep, not a corpus-calibrated prior. A
balanced .50 setting makes opener dependence easy to test but strongly
changes punctuation frequency. A rare setting is more natural in form
yet gives fewer positive square examples; train/test transfer between
settings is informative. A trivial always-round predictor gets 50%
accuracy at the balanced **selected** sites, so behavioral evaluation
must use paired opener edits and the eligible-site subset rather than
aggregate punctuation accuracy.

## What this permits and what it does not

This is a short-range structural rule. A small bilinear model that learns
it would not thereby learn entity binding, source attribution, syntax,
or the broader artificial-language rules. Nor would a sparse weight
pattern prove it implements a stack. Train/test splits must group whole
source documents and near duplicates, not individual windows. The
behavioral gate is held-out closing-token accuracy and a selective
probability shift under a supported opener edit; changing an unrelated
earlier pair should leave the selected close unchanged. Only a model
passing that gate is eligible for weight-only canonicalization, sparse
factor, or causal circuit comparisons on this rule.

**Decision:** promote this transform to the next *model pilot* as a dense,
natural-text control, while retaining the synthetic DGP as the route to
fully known world rules. Train across at least a balanced and low-square
setting; treat capability solely as the rule-specific eligibility gate.

Reproduce with:

```bash
python language_task_suite/check_corpus_transformed_delimiters.py
python language_task_suite/audit_corpus_transformed_delimiters.py \
  --output language_task_suite/results/corpus_transformed_delimiters_v1_audit.json
python language_task_suite/audit_corpus_delimiters_full_docs.py \
  --skip 43000 --n 500 \
  --output language_task_suite/results/corpus_transformed_delimiters_full_docs_skip43000_n500.json
```
