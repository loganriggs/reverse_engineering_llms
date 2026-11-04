# Bracket-mediated source events in the sampled language stream

The existing [V10 linked-event generator](V10_LINKED_EVENT_DGP.md) already makes a source claim affect later prose and a source question. The [bracket-mediated extension](surface_mixture_v10_bundled.py) tests a narrower compositional dependency: when prose reuses the claim's speaker and object, a bracket span becomes more likely; after at least one other completed clause, a question becomes more likely to ask about that **bracketed** object. All probabilities are functions of the visible prefix. There is no hidden event schedule, inserted answer-only block, or separate bracket task. The original V10 remains the matched control.

The [oracle audit](audit_v10_bundled.py) generated two independent panels of 100 streams × 256 toy tokens for both the provisional lower-rate and task-rich diagnostic settings. V10 and its extension shared every base source, question, bracket, and claim-link parameter. Every sampled token had positive oracle support and every recorded next-token distribution replayed from its prefix. All 1,337 bracket-opener-swap checks in the extension changed the correct closer while preserving the body distribution. At 88/113 linked bracket-opportunity sites in the lower-rate panels and 209/188 in the diagnostic panels, the extension's bracket mass met or exceeded V10's on the **same prefix**. The corresponding later query-object boosts passed 19/16 and 65/62 same-prefix checks.

| Setting / seed | Generator | Streams sampling source-linked prose object | Streams completing a linked bracket | Streams with source answer and linked bracket | Streams answering a later question about bracketed linked object |
|---|---|---:|---:|---:|---:|
| Lower-rate / 98001 | Existing V10 | 59 | 4 | 2 | 2 |
| Lower-rate / 98001 | Bracket bundle | 59 | 37 | 20 | 13 |
| Lower-rate / 98002 | Existing V10 | 61 | 8 | 6 | 2 |
| Lower-rate / 98002 | Bracket bundle | 61 | 42 | 13 | 9 |
| Diagnostic / 98101 | Existing V10 | 89 | 22 | 21 | 12 |
| Diagnostic / 98101 | Bracket bundle | 89 | 73 | 68 | 47 |
| Diagnostic / 98102 | Existing V10 | 92 | 29 | 27 | 16 |
| Diagnostic / 98102 | Bracket bundle | 92 | 69 | 64 | 45 |

The coupling produces enough within-stream overlap for a **conditional training experiment** without selecting only successful samples. It does not yet make the language realistic. A fresh, unselected 192-document FineWeb `sample-10BT` panel (offsets 430,000–430,191) gives the following form screen against 192 generated streams at provisional lower base rates:

| First 256 tokens, with lexical metrics on first 40 words | FineWeb | Existing V10 | Bracket bundle |
|---|---:|---:|---:|
| Windows containing a parenthesis | 40.6% | 30.2% | 44.8% |
| Windows containing a question mark | 24.5% | 49.0% | 51.6% |
| Windows containing a colon | 38.5% | 85.4% | 85.4% |
| Distinct word types across matched windows | 2,963 | 314 | 315 |
| Top-ten word fraction | 20.6% | 38.3% | 38.1% |
| Within-window repeated-word fraction | 17.2% | 27.1% | 27.1% |

The bracket rate happens to move parenthesis prevalence closer to this FineWeb panel, but the question/colon excess and tiny lexical tail remain large. Toy tokens and GPT-2 BPE tokens are not equal text lengths; the matched 40-word lexical comparison reduces that confound but does not prove semantic realism. In particular, a bracket body made of sampled adjective/object pairs remains unlike most manually reviewed human parentheticals. The lower-rate setting is **not accepted as a realistic reference**. The extension is an explicit task-rich DGP knob for training and circuit tests; the realism track still needs richer document forms, vocabulary, and calibrated dependencies.

Next, train 1–4-layer bilinear students on this *sampled stream* only as a conditional control, with whole-stream holdouts and oracle KL by rule family. Interpret source/bracket interaction weights only for models that pass both rule-specific behavioral and causal gates. In parallel, improve the renderer against fresh FineWeb form and lexical panels; do not use extra event overlap as a substitute for realistic language. Report generator event coverage, model prediction, intervention fidelity, and weight-method faithfulness separately.

Aggregates: [lower-rate event audit](results/v10_bracket_bundle_reference_n100_l256_seed98001.json), [task-rich event audit](results/v10_bracket_bundle_diagnostic_n100_l256_seed98101.json), [fresh FineWeb form screen](results/v10_bracket_bundle_fineweb_start430000_n192.json). No FineWeb source prose is published.
