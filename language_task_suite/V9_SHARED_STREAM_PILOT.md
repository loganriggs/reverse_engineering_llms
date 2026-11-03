# V9 shared-stream pilot: source facts affect prose; brackets share the stream

`surface_mixture_v9_source_bridge.py` is a **partial** successor to v8. It keeps v8's sampled narrative, source, and section clauses. A completed source claim such as `As Bob said, "the badge was missing"` now makes `badge` a candidate object when Bob appears as the subject of an ordinary narrative sentence. The candidate competes with the existing adjective-bigram, frame-cue, object-induction, and event-skip proposals. Source speakers also receive additional mass at narrative subject slots. The exact visible-prefix oracle computes this mixture directly; no hidden sampled rule label is needed to know its probabilities.

The same narrative sentence may now continue with `(` or `[` followed by one to three adjective/object pairs. Its correct closer is `)` or `]`, determined by the opener. The body distribution is identical for both openers, so the current adjective/object pair alone cannot identify the closer. Brackets are in the continuous sampled stream and ordinary narrative history; they are not a separate answer-only task. This pilot still does **not** connect section facts to prose, nor does it reproduce the original symbolic generator's exact proposal-weight collision scheme.

The [audit](audit_v9_source_bridge.py) replayed every sampled prefix and checked normalization, sampled-token support, source-mixture arithmetic against the independent v8 baseline, and opener-swap counterfactuals. Two independent 100×256-token panels per setting gave:

| Setting / seed | Source-influenced prose object sites | Sites sharing an output with an older object proposal | Source-answer sites | Bracket-closer sites | Streams containing source answer, source-influenced object, and bracket |
|---|---:|---:|---:|---:|---:|
| Diagnostic / 96001 | 75 | 10 | 234 | 195 | 23/100 |
| Diagnostic / 96002 | 58 | 16 | 258 | 205 | 10/100 |
| Provisional lower-rate / 97001 | 30 | 6 | 18 | 139 | 0/100 |
| Provisional lower-rate / 97002 | 25 | 5 | 17 | 165 | 2/100 |

Every source-influenced object site has multiple active *mechanisms*; the second numeric column specifically counts when the new source component and at least one older component also support the **same output token**. All 704 bracket-closer sites across the four panels passed the opener-swap test: swapping `(` with `[` (or vice versa) swaps closer probability and leaves the body-token probabilities unchanged. This verifies the intended long-range dependency in the generator, not in a trained model.

The diagnostic setting deliberately uses high source/query and bracket rates; it is a conditional training candidate, **not** a naturalistic reference. The lower-rate setting is likewise only a provisional reference candidate, not calibrated to FineWeb. Its 0–2/100 triple-overlap rate is too weak to test whether a model learns reusable computation across source binding, ordinary prose, and bracket closure. The next DGP change should bundle related events within a document: a source claim should raise the probability of a later narrative about that same speaker/object, with unrelated intervening clauses and a later source question. This makes joint coverage a designed property that can be audited at an unchanged reference distribution, rather than a byproduct of oversampling. A section detail should likewise affect some ordinary narrative choice. Then run disjoint FineWeb form/dependency comparisons with human review before describing the stream as realistic.

Raw results: [`v9_source_bridge_n100_l256_seed96001.json`](results/v9_source_bridge_n100_l256_seed96001.json) and [`v9_source_bridge_reference_candidate_n100_l256_seed97001.json`](results/v9_source_bridge_reference_candidate_n100_l256_seed97001.json). This pilot has **not** been used to train or interpret a bilinear model; v3 sparse-rule graphs remain the trained control.
