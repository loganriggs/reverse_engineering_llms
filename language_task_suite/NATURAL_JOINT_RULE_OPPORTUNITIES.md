# Natural joint rule opportunities: a prevalence control for the DGP

The sampled proposition generator currently puts several scored semantic moves into almost every short document. That is useful for training but is not a measured property of real text. This [aggregate-only audit](audit_natural_joint_rule_opportunities.py) asks a narrower, reproducible question: how often does the existing corpus-rule detector find multiple *mechanical opportunities* inside one natural FineWeb preview? It samples no synthetic prose and does not treat a regex/structural candidate as a verified semantic rule or as evidence of model competence.

The audit used the first at most 2,500 characters of each of two local 2,000-row FineWeb preview panels, whose median visible lengths are 301 and 303 alphabetic words. The five eligible families are numbered-list continuation, paired bracket, repeated three-letter nonce alias, repeated literal code, and a local spelling-style cue. The nonce candidate detector is largely a subset of the repeated-code detector; those are not independent tasks. For a separately checked transformed replay, all eligible list, bracket, literal-copy, and style edits were applied together; nonce edits were off because they require a separate code pool.

| Opportunity in one preview | 130k panel / 2,000 | 140k panel / 2,000 |
|---|---:|---:|
| At least one family | 876 | 856 |
| At least two families | 161 (8.1%) | 192 (9.6%) |
| At least three families | 20 | 13 |
| Bracket | 616 | 586 |
| Literal copy | 249 | 276 |
| Natural style cue | 139 | 147 |
| Numbered list | 23 | 27 |
| Nonce alias | 32 | 26 |
| Bracket + literal copy | 85 | 93 |

An additional two previously inspected 64-row panels at offsets 450k and 451k had 3 and 6 documents with at least two families. These are small checks, not independent population estimates. The complete eligibility histograms, pairwise overlaps, jointly applied site counts, and median source-to-target distances are in [130k](results/natural_joint_rule_opportunities_130000_n2000.json), [140k](results/natural_joint_rule_opportunities_140000_n2000.json), [450k](results/natural_joint_rule_opportunities_450000_n64.json), and [451k](results/natural_joint_rule_opportunities_451000_n64.json). The source previews remain local and are not committed.

Selecting multi-opportunity **natural** documents did not create an obvious local-form penalty on these panels. For each selected source, the audit picked a distinct non-multi-opportunity source matched to within one visible word in preview length. On first-100-word windows, multi/control top-ten word share was .213/.212 at 130k and .210/.217 at 140k; repeated cross-document trigram share was .020/.022 and .022/.023. This does not establish unbiased selection: topic, genre, length beyond the preview, and task-site language may differ. It only says this particular enrichment did not obviously distort those two aggregate form measures.

**DGP consequence:** keep two explicit prevalence regimes. A natural-rate arm samples all documents and accepts that many contain no detected rule sites; a task-enriched arm samples distinct source documents with multiple eligible families or increases rule-event rates in the synthetic semantic planner. Do not call the enriched arm natural-rate, do not duplicate one source as if it were independent training documents, and hold out entire sources and rule combinations. Report source acceptance rate, eligible versus applied sites, and task-window form alongside whole-document form. More tasks in every training example may improve sample efficiency but changes the data law; that is an experimental knob to measure, not a realism claim.

This audit informs task prevalence and overlap only. It does **not** solve the separate main renderer problem: generating semantically grounded, variable language from sampled document state. That requires corpus-calibrated compositional syntax and a verifier/oracle that respects what is visible to the model. The natural carrier remains a useful form and training control while that rule-based DGP is built.
