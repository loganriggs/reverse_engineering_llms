# Random next-word sites in real web text

The earlier [prediction-mechanism map](FINEWEB_PREDICTION_MECHANISM_MAP.md)
used eight selected, memorable examples. This audit asks what happens when
we choose target words **before** reading their predictive relation. The
[selection script](audit_fineweb_random_prediction_sites.py) samples 20
distinct documents from the 37 eligible ≥900-character previews among
FineWeb sample-10BT offsets 23000–23063, excluding the eight illustrative
documents. Within each, it picks one word uniformly among word starts at
characters 300–899, using seed 24011. The [site ledger](results/fineweb_random_prediction_sites_seed24011_n20.json)
pins source hashes and character offsets, target words, and a single
reader's paraphrased judgment; it does not include source passages.

The primary judgment is **15/20 probability shifts**, **2/20 visible
phrase/word reuse**, and **3/20 newly introduced named facts**. None of
these mechanically selected targets had a clear exact answer computed from
an earlier state or a clean long-range scoped lookup. That is a feature of
this small, biased screen, not evidence that those tasks are absent from
FineWeb. Examples include `fiscal` after spending (local collocation),
`Material` at the start of a course-content section (page plan), `turkey`
in repeated wordplay (visible reuse), and `Hupmobile` as a new car make
(exact identity unavailable before introduction). The 20 sites span
article, promotional, testimonial, markup, forum, travel, legal, and
reference forms; their mechanisms cannot be represented by one generic
copy rule.

The [scorer](score_fineweb_random_sites.py) evaluated each observed word
under its full prefix and the final 100/40 whitespace-delimited words for
[GPT-2](results/fineweb_random_prediction_sites_gpt2_seed24011_n20.json)
and the trained [18-layer bilinear model](results/fineweb_random_prediction_sites_bilin18_seed24011_n20.json).
Full-prefix minus last-40-word observed-target log probability has median
**+0.027 nats for GPT-2** and **−0.023 for bilin18**; neither model has a
site with a >1-nat gain from the full prefix. The means are −0.101/−0.043.
Full versus last 100 words is especially weak evidence because 11/20
prefixes are shorter than 100 whitespace words, so those conditions are
identical. The longest prefix here is only 156 words. These scores test
model sensitivity to truncation, not the writer's causal rule, and the
selected word can itself be awkwardly tokenized. Larger and longer
disjoint samples are needed for prevalence or context-length claims.

This changes the next DGP implementation target. A natural-reference
stream needs a substantial *background* process for document genre,
collocations, quoted register, markup/list formatting, and class-specific
rare names. Exact first mentions should be sampled as new facts; later
mentions can test retrieval. Semantic tasks should be embedded at
auditable sites inside that background, with task-rich oversampling kept
separate from reference-rate sampling. The original bigram, skip,
induction, and bracket rules can occupy the local/back-reference part of
the same conditional distribution. For every synthetic site, record
whether its oracle depends on visible local words, an earlier entity or
event, a page plan, or an unobserved sampled choice. This prevents
interpreting a low overall next-token loss as evidence of the harder
mechanism, exactly as the [V6 transfer test](DOCUMENT_MIXTURE_V6_TRANSFER.md)
warns.

The next natural annotation should sample longer target-bearing spans
from independent documents, include multiple candidate words per site,
and have a second reader adjudicate at least a subset. A corpus-rate
estimate would need document and token sampling weights; this 20-site
pilot supplies neither.
