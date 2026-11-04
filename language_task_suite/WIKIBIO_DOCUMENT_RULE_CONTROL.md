# Corpus-grounded biography reuse: viable control, incomplete DGP

The semantic/syntax-tree DGP has exact rules but repetitive prose. This
pilot instead samples whole, real biography paragraphs from
[WikiBio](https://github.com/DavidGrangier/wikipedia-biography-dataset),
whose first-paragraph text is paired with an infobox. This is a different
language genre from broad FineWeb and a test of a **semi-parametric** DGP:
empirical human text plus a precisely specified, editable relation. The
infobox is not treated as a complete semantic oracle. The original WikiBio
train shard used here is the `thoughtworks/wiki_bio` Parquet revision
`c5f91f5d6a61a2cd1ce50bc33a7a7ac468030335`, SHA-256
`95ddb5303aff5f5c8462bf44de91cd8590bedee3310f91473177c3df9e8f1bca`.
The 165 MB shard remains in the local Hugging Face cache; no source
biographies are committed.

The [feasibility screen](audit_wikibio_document_rules.py) read its first
20,000 records. It found 4,225 paragraphs where the infobox name begins
the text and its surname appears again within 160 words. Literal visible
birth-year, nationality, and occupation mentions occurred in 3,149, 579,
and 463 of these, respectively. Those are **string mentions**, not verified
semantic tasks; family members, firms, aliases, or unrelated people can
repeat the same surname. Manual inspection caught such false positives.

The [strict subject-reuse DGP](wikibio_subject_reuse_dgp.py) therefore
requires exactly two surname occurrences: one in the opening full name and
one at the start of a later sentence followed by a subject-compatible verb.
An alternate-world edit replaces both occurrences with a different sampled
surname, and a visible-text parser must recover the new answer from the
opening name and later sentence without reading the infobox. The
[pair audit](audit_wikibio_subject_reuse_pairs.py) found 628 strict candidates
in the same 20,000 records. Of these, 627 passed the edit and parser checks;
625/627 have the entire target surname inside a 256-token GPT-2 context,
and 594/627 have the entire edited and original paragraphs inside that
context. The median target starts at token 59 and spans three BPE tokens.
This is enough for a training control after drawing independent source
shards and checking source/near-duplicate splits; it is not yet a trained
model or a claim of learned copying.

| First 40 words, 512 paragraphs | Top-ten word share | Cross-document repeated-trigram share | Word types |
| --- | ---: | ---: | ---: |
| Unselected WikiBio | .286 | .239 | 4,538 |
| Strict selected originals | .284 | .280 | 4,391 |
| Edited counterparts | .284 | .286 | 4,257 |
| Broad FineWeb preview | .206 | .027 | 5,840 |

These panels are length-matched but not genre-matched. The selected
biographies are somewhat more repetitive than unselected WikiBio, and
WikiBio itself differs sharply from FineWeb. The old WikiBio release is
lowercased and tokenized, including `-lrb-`/`-rrb-` markers, so its surface
form is not directly suitable as a realistic general-language distribution.
The paired edit leaves ordinary assertions in place while changing the
subject's name; that defines a controlled alternate world, not a true
Wikipedia article. The [SynthBio study](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/9fc3d7152ba9336a670e36d0ed79bc43-Paper-round2.pdf)
also documents the infobox/prose faithfulness problem, reinforcing why
literal field matches cannot license a full semantic oracle.

**Decision:** Keep this as a multi-sentence, corpus-grounded *rule control*
that can broaden genre and lexical coverage in later model comparisons.
Do not substitute it for the main goal of a varied, multi-rule sampled
language DGP. The next expansion needs clean original text plus several
independently checked semantic relations per document, and must test
whole-rule-combination holdouts and learned behavior before weight
interpretation. A useful next source is a document corpus with structured
facts and less destructive preprocessing; its provenance, semantic
alignment, and heldout form must be audited before training.

Aggregates: [field feasibility](results/wikibio_document_rules_trainfirst20k_n512.json)
and [paired edits/form](results/wikibio_subject_reuse_pairs_trainfirst20k_n512.json).
