# Catalog retrieval in a versioned multi-genre language

This experiment adds a fifth, structured-record genre to the four-genre exact-oracle language. Its purpose is to test whether a model can bind an object key to one of two fields before using that model as a ground-truth target for canonicalization or sparse-circuit recovery. The present models do **not** pass that competence gate. Their answer distributions respond to record values but almost not at all to the queried object. No recovered Catalog circuit is claimed.

## Data-generating process

[`multi_genre_v2_language.py`](multi_genre_v2_language.py) samples Reports, Guide, FAQ, Policy, and Catalog streams. Catalog has unique object keys, unique owner values, unique status values, up to eight records, two record orders, and three question forms. One generated example is:

> Catalog:
>
> The badge belongs to Eve and is rare.
>
> The cup is old and belongs to Frank.
>
> For the badge, what is the status? The answer is rare.
>
> For the badge, who is the owner? The answer is Eve.

The generator supplies the exact next-token distribution from the visible prefix. A first query over two records has a two-way object choice; later queries can have more candidates. The audit replayed every prefix in 2,000 length-128 streams, independently checked 2,849 completed records and 1,523 completed queries, and verified four supported paired-edit families on another 500 streams. The edit families alter the query key, its relevant record value, its other field, or an unrelated record's same-field value. The first two must change the exact answer; the latter two must leave it unchanged.

The genre is motivated by a small inspected FineWeb panel containing structured records and field-value bindings, but **is not a natural-language realism match**. On 480 roughly length-matched windows, FineWeb has 8,719 distinct alphabetic words versus 167 in this mixture; mean sentence length is 17.0 versus 9.7 words. Colons occur at 0.54 versus 3.58 per 100 words, and every synthetic window carries a header colon. These are coarse surface checks using different tokenizers, not an estimate of task prevalence or semantic similarity. Full counts and examples are in [`multi_genre_v2_audit.json`](multi_genre_v2_audit.json).

## Training and capability controls

Models use the existing attention-only BilinearGPT shell with width 64, four heads, RMSNorm, batch 64, ordinary next-token CE, and 3,000 AdamW steps. The five-genre sweep has depths 1–4 and 127-token contexts. Independent evaluation uses two 1,024-stream panels, never the development panel used during training. All answer numbers below are exact-oracle KL in nats; lower is better. A query-key-blind Bayes control sees the field, all record values, and prior asked object-field pairs, but not the current queried object. This is a stronger control than a fixed uniform guess, because prior queries constrain which keys remain eligible.

| Training condition | Depth | Owner answer KL | Status answer KL | First owner query KL | First status query KL |
|---|---:|---:|---:|---:|---:|
| Five genres, bilinear | 1 | 2.044 | 2.401 | 1.442 | 2.428 |
| Five genres, bilinear | 2 | 1.368 | 1.328 | 1.253 | 1.166 |
| Five genres, bilinear | 3 | 1.319 | 1.321 | 1.212 | 1.168 |
| Five genres, bilinear | 4 | 1.356 | 1.332 | 1.229 | 1.194 |
| Five-genre query-blind Bayes control | — | ≈1.19 | ≈1.20 | ≈1.12 | ≈1.08 |
| Catalog only, two records, bilinear | 3 | 0.527 | 0.523 | 0.773 | 0.747 |
| Catalog only, two records, softmax control | 3 | 0.546 | 0.539 | 0.789 | 0.761 |
| Catalog only, two records, bilinear, first-answer CE ×10 | 3 | 1.716 | 0.795 | 0.813 | 0.756 |
| Two-record query-blind Bayes control | — | ≈0.47 | ≈0.47 | 0.693 | 0.693 |

The separate Catalog-only run with the ordinary record rate also failed to cross the query-blind control: development owner/status answer KL was 1.332/1.313. The two-record variant sets the record-addition probability to zero after two records, shortens context to 63 tokens, and keeps the same generated grammar. Its ordinary and softmax models have similar failure. The softmax control has fewer active parameters and is not a parameter-matched architecture comparison. The ×10 run adds CE on the first Catalog answer in each training stream; it is an objective ablation, not a tuned optimum.

## Causal test on first queries

The paired-edit test holds all other visible prefix tokens fixed. Total variation (TV) between model answer distributions should be **1** when the correct answer changes and **0** for an invariant edit. On the same 414 first-query pairs per edit kind in the two-record panel:

| Model | Query key TV (oracle 1) | Relevant value TV (oracle 1) | Other field TV (oracle 0) | Irrelevant record TV (oracle 0) |
|---|---:|---:|---:|---:|
| Bilinear, ordinary CE | 0.0047 | 0.586 | 0.011 | 0.557 |
| Softmax, ordinary CE | 0.0062 | 0.580 | 0.032 | 0.570 |
| Bilinear, first-answer CE ×10 | 0.0021 | 0.588 | 0.014 | 0.568 |

The model reacts similarly to a relevant and irrelevant same-field value edit while ignoring the key. That pattern is consistent with pooling field values without object binding. The measured behavior does not establish an internal pooling mechanism; it rules out competent keyed retrieval by these checkpoints. Low average token CE and plausible-looking completed answers would conceal this failure. First-query weighting does not resolve it and worsens aggregate owner KL.

## Decision and reproducibility

Stop this local Catalog-only loss and architecture sweep. These checkpoints are unsuitable targets for claims about recovering a keyed-attention circuit. Continue the broader task-diversity loop with an explicit competence gate: demonstrate first-query oracle KL below the query-blind Bayes control **and** large, selective key-edit response with low irrelevant-record response on fresh object/value and surface-form combinations. Only then compare weight-only and data-aware canonicalization, sparsity objectives, or featurizers on this behavior. Existing competent verb/object and FAQ tasks remain available for those methods meanwhile. Whole task-family featurizer holdouts and true new-task discovery remain separate from this model-training test.

Reproduce the generator audit with `audit_multi_genre_v2.py`; train with `train_multi_genre.py --language v2` and the saved `results.json` configurations; score with `eval_multi_genre_v2_fresh.py`, `eval_multi_genre_v2_key_blind.py`, and `eval_multi_genre_v2_edits.py`. The saved JSONs identify seeds, lengths, record rates, and model paths. The two-record edit files use seed 70021 and 2,048 generated streams; their `first_query_scores` contain the 414-pair comparison. The fresh KL files use seeds 9901 and 10910. The versioned generator and trained checkpoints are retained so the negative result stays reproducible.
