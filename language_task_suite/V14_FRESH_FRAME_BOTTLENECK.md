# V14's form gap is structural, not mainly sentence length

The active goal is a variable, realistic-enough **sampled** language distribution with known rules, so trained bilinear models can test whether weight methods recover the rules they actually learn. V14 has an exact visible-prefix oracle and several rule families in one stream, but its surface remains repetitive. A fresh FineWeb read included news articles, forum threads, listings, product descriptions, personal accounts, and explanations. That qualitative sample motivated a targeted check of document and sentence openings, not a genre-frequency estimate.

The [audit](audit_v14_frame_vs_fresh_fineweb.py) takes 128 first-100-word FineWeb windows from `sample-10BT` beginning at offset 94,500, outside the earlier V14 FineWeb panels, and compares two independent 128-window V14 draws under the trained reference-rate settings. It applies the same word and simple sentence-boundary heuristics to all arms. No natural source passages enter the [aggregate JSON](results/v14_frame_vs_fineweb_skip94500_n128.json).

| First 100 words, 128 windows | FineWeb | V14 seed 52301 | V14 seed 52302 |
|---|---:|---:|---:|
| Mean sentence segments per window | 7.82 | 8.09 | 7.66 |
| Mean words per segment | 12.79 | 12.36 | 13.06 |
| Distinct first-segment starts | **105** | 29 | 28 |
| Distinct sentence starts, all segments | **528** | 59 | 60 |
| Top-five sentence-start share | .190 | .608 | .643 |
| Distinct word types | **4,047** | 326 | 332 |
| Top-ten word share | .209 | .365 | .369 |
| Cross-document repeated-trigram occurrence share | **.016** | .634 | .632 |
| Repeated-word fraction in first 40 words | .164 | .279 | .265 |

The approximate segment count and average length match accidentally well. V14 repeatedly starts clauses with `In`, `The`, `During`, `At`, and `When`; those five starts dominate about 61–64% of its segments, compared with 19% for the top five in the fresh FineWeb panel. Its vocabulary tail and cross-document phrase variety are much smaller. This reproduces the earlier Zipf and `the`-attribution findings on a new source range, while adding direct evidence that the **frame inventory and start distribution** are the central form problem. The simple regex treats headings, fragments, abbreviations, and web markup as sentence starts; these are real parts of web text but not a syntactic parser's sentence labels. The panels are not topic-matched, so this is a form diagnostic, not a calibrated likelihood or proof of semantic similarity.

Both V14 seeds still sample the intended mechanism families in their 256-token streams: about 1,400 mixed-verb sites, 1,200 mixed-object sites, 200 attribute-object sites, 160–215 bracket closers, 35–39 source answers, and 66–69 conditional-action sites across 128 streams. Some other families are rare, notably only 4–5 price answers. Increasing linguistic variation must preserve and independently audit these rates and whole-stream combinations; a prettier generator that loses rule opportunities is not progress for mechanistic benchmarking.

**Next generator contract:** represent a completed event or fact independently of its wording. Sample a document-level genre/act state, then realize the same event through several genuinely different clause lengths and information orders, with varied determiners and a broad nonuniform lexicon. The exact next-token oracle must marginalize any still-hidden realization or genre choice from the visible prefix; it cannot condition on the generator's private sampled branch. Local bigram/skip proposals, induction, brackets, and stateful binding must still compete or interact at audited sites in one stream. First implement and verify two alternate clause frames over one event type, including prefix normalization, support, a visible-answer edit, and task-site coverage; then expand the act/genre mixture. Compare full documents and task-bearing windows on new disjoint FineWeb ranges, and train bilinear models only after generator semantics pass. Keep whole-task and whole-combination holdouts for method evaluation.
