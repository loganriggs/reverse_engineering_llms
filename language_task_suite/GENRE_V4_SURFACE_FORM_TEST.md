# Does paraphrasing the task grammar close the natural-form gap?

V3's fixed sentence frames dominate its word frequencies. V4 is a controlled
surface-form intervention, not a claim to reproduce web text. It splits some
existing sentence probabilities between two expressions of the same canonical
event or fact: `According to X, ...` / `Filed by X: ...`, `Who described ...` /
`Which source described ...`, `At ...` / `In ...`, `After ...` / `Following ...`,
and `X is a role` / `X works as a role`. A deterministic transducer maps any
supported partial V4 prefix to a canonical V3 prefix. The original hidden
genre Bayesian filter then gives the exact conditional next-token law.
The added words and forms are deliberately small; the experiment tests how
far simple paraphrases move the surface statistics while preserving the
planted semantic tasks.

The [V4 generator](genre_document_v4.py) and [audit](audit_genre_document_v4.py)
checked 100 × 256 sampled tokens per regime. All sampled tokens had positive
probability, 180 selected prefixes per regime replayed from scratch, and
complete documents parsed to valid canonical semantic state. The task-rich
audit checked 340 exact source answers, 271 bracket closers, 259 section
numbers, 246 step numbers, and 249 arithmetic answers. The reference audit
also passed these checks, with 117 source answers. Each new form appeared in
both regimes. These checks establish internal generator consistency, not
naturalness or model learning.

The [form comparison](results/genre_v3_v4_vs_fineweb_25000_form_gap.json)
uses a disjoint 48-document FineWeb sample-10BT development panel at offsets
25000–25047. It includes the 43 documents with at least 100 alphabetic words
in their first 1,200 characters and matches them to 43 generated streams per
regime, taking the first 100 alphabetic words of each. For adjacent-word
pairs, two words must be separated **only by whitespace**, so skipping a
number or punctuation cannot create an artificial collocation. Source text
and URLs remain outside the repository; the result contains aggregate counts
and hashes.

| First 43 × 100 words | FineWeb | V3 task-rich | V4 task-rich | V3 reference | V4 reference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Distinct words | 1,804 | 361 | 353 | 370 | 366 |
| Singletons | 1,175 | 55 | 46 | 39 | 48 |
| Top-ten word share | 19.9% | 35.5% | 34.1% | 35.5% | 34.3% |
| Distinct adjacent-word pairs | 3,022 | 967 | 977 | 1,002 | 1,010 |
| Top-ten adjacent-pair share | 3.23% | 15.82% | 13.49% | 15.22% | 14.24% |

Thus simple paraphrases lower repeated-pair concentration somewhat, but
leave a roughly fourfold gap to this FineWeb panel and do not close the
lexical-tail gap. A [second generated seed panel](results/genre_v3_v4_vs_fineweb_25000_form_gap_seed68001.json)
shows the same pattern: task-rich top-ten adjacent-pair share falls from
16.55% to 15.26%, reference from 15.52% to 13.93%; distinct-word counts
move in both directions but stay below 370 versus FineWeb's 1,804.
This supports a stronger structural correction: more distinct
communicative functions, ordinary background prose, entity references,
optional clauses, and page/list layout. Merely adding surface synonyms to
the same few semantic templates is insufficient. This is a convenience
corpus panel and a first-100-word comparison, not a population estimate or
a judgment that every natural sentence needs an exact semantic task.

V4 still preserves V3's task-rich/reference distinction, but neither regime
is corpus calibrated. The same 0–9 addition table is used; no extrapolating
arithmetic claim follows. Exact source-answer generation does not imply that
a bilinear model has learned source binding.

## Matched two-layer bilinear pilot

The [V3/V4 trainer](train_genre_document_v3.py) used the same width-64,
four-head, two-block bilinear architecture and seed as the V3 pilot:
8,000 task-rich 256-token streams, 1,000 full-next-token CE updates, and two
disjoint 1,000-stream evaluation panels. The V4 vocabulary has 526 tokens
versus V3's 519, so the model has 214,784 versus 213,888 parameters. There
are no task/genre labels or whole-task training holdouts. One seed is not
enough to attribute small performance differences to the paraphrases.

| Task-rich measure | V3 depth 2 | V4 depth 2 |
| --- | ---: | ---: |
| Token NLL | 1.446 | 1.446 |
| Genre sentence-start oracle KL | 0.263 | 0.254 |
| Source-answer full-vocabulary accuracy | 44.0% | 46.1% |
| Bracket closer accuracy | 99.9% | 100% |
| Step number accuracy | 97.5% | 98.1% |
| Seen-table addition answer accuracy | 91.3% | 98.2% |

The [full V4 training result](results/genre_document_v4_bilinear_depth2_1000.json)
also gives reference-rate and per-genre scores; the checkpoint is alongside
it. V4 does not provide evidence that paraphrases teach source binding.
On 200 fresh [supported source-attribute edits](results/genre_document_v4_depth2_source_edit_n200.json),
the oracle switches answers completely (TV 1). The model's mean full-vocab
TV is only **0.0080** and its old-versus-new two-answer effect is **0.00085**
versus the oracle's 1. Its original/edited full-vocabulary answer accuracy
is 51.5%/39.0%. In a matched 200-site bracket-opener control, model TV is
0.998 and both original and edited closer accuracy are 100%. The model is
capable of following a causal edit, but has not learned this source relation.

A separate [query-paraphrase invariance check](results/genre_document_v4_depth2_query_paraphrase_n200.json)
changes `Who described ...` to `Which source described ...` or vice versa
without changing the oracle answer. Model mean output TV is 0.028, with
answer probabilities 0.312/0.311. This is modest surface sensitivity, but
low answer probability means it cannot be interpreted as successful semantic
paraphrase invariance.

**Decision:** keep V4 as a controlled surface-form baseline, not the main
language benchmark. The next generator must change the distribution of
document structures and utterance functions, and pass independent natural
form screens before we spend effort canonicalizing or sparsifying its
unlearned source task. Learned bracket/number/local tasks remain valid
competent controls for those interpretation methods.
