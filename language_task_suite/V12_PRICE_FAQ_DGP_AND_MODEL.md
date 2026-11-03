# V12: item facts, ordinary mentions, and cost questions in one stream

The [V12 generator](surface_mixture_v12_price_faq.py) extends V10's
jointly sampled narrative/source/section/bracket stream with item–price facts.
For example, one sampled 256-token stream contains these completed clauses,
with other clauses between them:

> The laptop costs 15 dollars.
>
> Afterward, Sam stores the temporary laptop outside the school after the exhibition because Priya had asked about it earlier.
>
> How much does the laptop cost? Answer: 15 dollars.

The middle ordinary narrative object slot receives a named `priced_item`
mixture component when a completed fact has not yet appeared in prose. The
later question can select any previously priced item, favoring those already
mentioned, and its answer is the item's visible price. Six products and six
prices are sampled without replacement within a stream, so a product's usual
price cannot solve the task. The generator also retains V10's source-linked
prose and question rules, ordinary competing verb/object proposals, and
bracket spans. All choices are made through one exact visible-prefix
`next_distribution`; the generator does not secretly schedule a three-clause
template.

The [audit](audit_v12_price_faq.py) samples two independent 100×256-token
panels per regime. It checks normalization, every sampled token's positive
oracle probability, exact replay of every recorded next-token distribution,
the named object-component sum, exact answers, and support-preserving earlier
fact-price edits. An edit changes the fact's number to one unused elsewhere in
the visible history and checks support at **every** later prefix before the
answer. Query-key edits to another still-answerable item receive the same
support check. Both edits change the exact answer. A separate 100×256 smoke
with the older `How many` count questions enabled found 79 count answers and
38 cost answers with exact replay at every sampled token, so the shared `How`
start correctly forks between the two forms.

| Per 100 streams | Reference 99611/99612 | Enriched 99601/99602 |
|---|---:|---:|
| Any completed price fact | 65 / 67 | 94 / 88 |
| Priced item participates in ordinary object selection | 20 / 31 | 35 / 43 |
| Complete price answer | 4 / 4 | 18 / 27 |
| Fact, ordinary object component, and price answer in one stream | 3 / 4 | 14 / 25 |
| Price answer and bracket close in one stream | 2 / 2 | 7 / 11 |
| Audited priced-object component sites | 24 / 39 | 42 / 52 |
| Support-preserving earlier price edits | 4 / 4 | 26 / 40 |
| Support-preserving queried-item edits | 4 / 4 | 24 / 34 |

The [reference audit](results/v12_price_faq_reference_candidate_n100_l256_seed99611.json)
and [enriched audit](results/v12_price_faq_diagnostic_n100_l256_seed99601.json)
store all counts. These rates are design choices, **not estimates of natural
language prevalence**. The low reference answer density motivates training
on the explicitly enriched stream and evaluating on untouched reference-rate
streams. Training on an enriched process changes the task distribution.

## FineWeb check: broader relation exists, exact template remains synthetic

A [new form screen](compare_v12_fineweb_fresh.py) streamed FineWeb
`sample-10BT` rows 15,000–15,063, disjoint from the earlier cached 7,000 and
11,000 offsets. At the prespecified 100-word cutoff, 59/64 document–toy
pairs were eligible. Question marks were present in 16.9% of FineWeb versus
15.3% of V12 crops, but colons were 39.0% versus 52.5%, parentheses 33.9%
versus 13.6%, and repeated-word fraction in the first 40 words 17.8% versus
26.1%. The [aggregate record](results/v12_price_faq_fineweb_fresh_skip15000_n64.json)
contains hashes of the selected documents, never their prose. Tokenizers and
document lengths differ; these numbers are diagnostic form checks.

A separate [manual review](review_v12_fineweb_dependencies.py) selected 16
of those 64 documents without using text cues. In the first 160 words,
7 showed a visible field–value relation and 8 a numeric binding. Examples
included a product pack with a template count and print resolution, and a
listing with a price and mileage. **None** contained V12's exact
item-price → intervening ordinary mention → later cost-question sequence in
the crop. The [coded review](results/v12_price_faq_fineweb_dependency_review_n16.json)
does not establish the sequence is absent from the corpus; it does show why
V12 should be treated as a causally auditable synthetic stress test, rather
than a realistic model of document frequencies or prose.

## Model gate

The [resumable trainer](train_v12_price_faq.py) uses a two-layer, four-head,
width-64 bilinear transformer with RMSNorm and ordinary next-token CE on the
enriched joint stream. A CPU interrupted-versus-uninterrupted four-step check
matched final weights exactly. The [fresh evaluator](eval_v12_price_faq_model.py)
will score both enriched and reference-rate streams, including exact price
answer CE, earlier-price edits, and query-key edits. Its smoke test passed
support checks; the untrained model appropriately failed the causal task.
The prospective circuit gate is price-answer CE below 0.8 nats and mean
new-target probability response above 0.5 for **both** earlier-price and
query-key edits on each of two untouched reference-rate panels, with at
least 30 edited answers per panel. No circuit or sparsification
interpretation should be attempted until the trained model passes this gate;
if a 512-stream panel is too sparse, increase the panel size rather than
relaxing the threshold. A six-price product-only guess has CE `log(6)=1.79`
nats.
