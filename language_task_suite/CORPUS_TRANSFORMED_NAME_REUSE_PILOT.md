# Natural-document transformation pilot: controlled name reuse

The fully synthetic document grammars still have narrow, repeated prose.
This pilot tests a complementary way to put a **known sampled rule inside
realistic variable language**: sample a FineWeb passage, find a repeated
person's full name, draw a compatible replacement first name that is absent
from the passage, and replace every selected occurrence consistently. The
later mention is an induction target. The replacement is random given the
source document, so a model must use the earlier visible occurrence rather
than memorize the original name–surname pairing.

This is **not** a symbolic oracle for FineWeb's entire next-token
distribution. It gives an exact transformation rule and target site on an
empirical natural-text background. It complements, rather than replaces,
the goal of a sampled artificial language with known generative structure.

## Identity correctness before form

The first loose implementation replaced repeated first names. Manual review
caught a bad case: two different people named Henry appeared in one passage.
Other first-name-only cases also mix people with the same name. The strict
policy now requires each occurrence to be part of the *same exact full
name*, with no standalone use of that first name in the selected window.
It rejects a middle-name position preceded by a different capitalized name.
The two name pools (38 and 43 choices) are a small, manually screened subset
of the existing V17 name bank cross-checked against NLTK's name lists;
they are only a surface-compatibility prior, not an assertion about a
person's identity.

On the frozen local panel of 480 FineWeb windows of 513 GPT-2 tokens, the
[strict audit](results/corpus_transformed_name_reuse_v1_audit.json) found
**8 eligible windows** and 17 replaced source-name occurrences. Every
transformation reversed exactly to the source text. Eight paired changes
to the randomly sampled name changed the correct later answer; four
changes to a different visible name left that answer unchanged. The scored
mentions were 42–148 words after the preceding matching mention, mean
99.6 words. The compatible replacement pool had 34–42 names, giving
about **2.7%** mean success for a predictor that knows the source passage
and name class but ignores the sampled earlier replacement. All eight
strict examples were manually checked for apparent same-person reference;
this is a small review, not a guarantee for unseen passages.

The [looser diagnostic](results/corpus_transformed_name_reuse_loose_diagnostic.json)
admits 42/480 windows but is unsuitable as a ground-truth task without
more reliable entity linking. Source windows can overlap and are not
independent documents.

A separate [full-document yield screen](results/corpus_transformed_name_reuse_full_docs_skip43000_n500.json)
looked at the first 8,000 characters of 500 streaming FineWeb documents at
offsets 43,000–43,499: **4** were strict-policy eligible; **29** were
loose-policy eligible. The four strict full-document matches were manually
reviewed and appeared to refer to the same person. Thus the current
high-precision rule is very sparse. Training a small transformer on it
would require much more source data or a broader, validated entity
extractor. We should not use the loose policy merely to raise yield.

## Form and limits

On the eight strict eligible local windows, the first-100-word crop has
469 word types before and after substitution and 0.26% cross-document
reused trigrams in both arms. A 100-word crop around the target has
455/456 types and 0.38% reused trigrams. These comparisons are unsurprising:
almost all text is unchanged natural prose, and eight windows provide a
weak form estimate. They do show that the controlled rule can live in a
language-like carrier without adding a repeated `Answer:` frame.

There are material limitations. Replacing a real person's first name makes
the passage counterfactual and can clash with facts outside the document.
The source distribution is empirical and unknown, so only the planted
reuse rule has a known oracle. A model might learn local name–surname
priors, so later weight claims must require **held-out source documents**,
new replacement draws, high target accuracy, and a selective response to
an earlier-name edit with little response to an unrelated-name edit.
Exact duplicate and near-duplicate source documents must be grouped before
splitting. The 480-window row cache is not suitable for that split.

**Decision:** retain the strict transform as a naturalistic induction
control, but not yet as the main training DGP. The next corpus-backed
version needs denser, high-precision sampled rules (for example,
structural delimiter transformations) and document-level source splits.
In parallel, the fully artificial generator still needs a better semantic
discourse grammar. Evaluate rule competence before weight-based
canonicalization or sparsity claims.

Reproduce the local audit with:

```bash
python language_task_suite/audit_corpus_transformed_name_reuse.py \
  --identity-policy strict_full_name \
  --output language_task_suite/results/corpus_transformed_name_reuse_v1_audit.json
```

The full-document audit uses `audit_corpus_name_reuse_full_docs.py` with
`--skip 43000 --n 500` and stores only aggregate counts and hashes.
