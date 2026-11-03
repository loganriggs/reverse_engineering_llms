# V10: visible-history links between claims, prose, and questions

V9 put source facts, prose, and bracket spans in one sampled stream, but at the provisional lower rates it almost never produced the same *event* across those forms. V10 adds a visible-history link without a hidden episode schedule. A completed claim raises the probability of later prose with its speaker and object. Once matching prose appears, a later source question is more likely to ask about that object. Other clauses may intervene; no fixed three-clause template is inserted. The same bracket and older verb/object mixtures remain available. All boosts are explicit probability mixtures in `surface_mixture_v10_linked_events.py`.

The [audit](audit_v10_linked_events.py) compared V9 and V10 on the same two 100×256-token seeds and the same provisional lower-rate parameters. “Complete” means an ordered claim → matching completed narrative → later source question about the claimed object, rather than mere presence of three labels in a stream.

| Metric per 100 streams | V9 seed 98001 | V10 seed 98001 | V9 seed 98002 | V10 seed 98002 |
|---|---:|---:|---:|---:|
| Streams with any claim | 68 | 68 | 71 | 71 |
| Streams with matching prose after a claim | 1 | 58 | 6 | 60 |
| Streams with complete ordered event | 0 | 19 | 0 | 16 |
| Streams with complete event and a bracket span | 0 | 9 | 0 | 8 |
| Source-influenced narrative object sites | 25 | 160 | 36 | 190 |

V10 produced 27 and 30 complete question events (some streams contain more than one). Of these, 10/27 and 16/30 had at least one intervening clause between claim and matching prose; 21/27 and 23/30 had one between prose and question; 7/27 and 11/30 had intervening clauses on both sides. Every sampled token had positive oracle probability, and the sampler's recorded next-token distribution replayed exactly. At all 113/129 pending-claim object sites with the matching subject, the claimed object received more than the configured .65 linked-component mass. At 32/32 later query-key sites per seed, the linked object received at least its configured .65 query-key mass.

V10 also exposes `object_rule_components(prefix)`: the effective mass and output distribution for adjective-bigram, induction, frame cue, event skip, source fact, and linked claim at each ordinary object site. Independently summing those named components reproduced the exact oracle to `<1e-10` at all 1,190 and 1,185 audited object sites. This keeps collision arithmetic inspectable rather than inferring a rule solely from the sampled label.

This is a DGP improvement, not a realism or model-learning result. The masses `.65/.65/.30/.65` are research choices and have not been calibrated to FineWeb. Even with linking, only 16–19% of 256-token streams contain a complete event; a conditional training setting may still be useful, but should be evaluated under this unchanged provisional setting. The bracket spans are still synthetic and section facts still do not influence ordinary prose. We need a disjoint FineWeb dependency audit with manual cue review before calling the stream realistic.

A [surface-form check](results/v10_fineweb_form_n192_l256_seed99101.json) on an **already used** 192-window FineWeb panel is negative: V10 question-mark windows are 65.6% versus 23.4% in FineWeb; colon windows 87.0% versus 43.2%; and repeated-word fraction over the first 40 words 27.0% versus 16.4%. Parenthesis windows are 30.2% versus 49.0%. The streams use different tokenizers, so these are diagnostic form mismatches rather than calibrated natural-language task rates. The prior manually reviewed FineWeb parentheses were often clarifications, statistics, dates, citations, glosses, or geographic notes—unlike V10's adjective/object bracket body. Do not treat the bracket task as naturalistic solely because a parenthesis character is present.

The bilinear LM training runner now accepts `--generator v10` and records bracket-close KL alongside the existing rule families. The first [two-layer capability and circuit screen](V10_BILINEAR_BRACKET_CIRCUIT.md) found competent bracket closure but no learned source binding after 1,000 steps. This gates subsequent interpretability work: analyze the bracket circuit now; adjust capacity/training only if source binding itself is the target. Raw DGP audit: [`results/v10_linked_events_reference_candidate_n100_l256_seed98001.json`](results/v10_linked_events_reference_candidate_n100_l256_seed98001.json).
