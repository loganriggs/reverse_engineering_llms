## H999 — Residual-inspired probes qualify two candidate tasks

Screened64pairs/128states across clause completion, list markers, noun expectation and within-word continuation. All pairs share the final token; all tokenizations passed. There were129forwards including replay. Four source/format cohorts per family each contain four pairs; splits are source-slot splits, not guarantees of vocabulary independence.

Noun expectation and word fragments pass allfournative competence cohorts. Noun both-endpoint correctness is1,.75,1,1 and intended full-vocabulary top1 is.625,.5,.875,.625. Word-fragment both-endpoint correctness is1 throughout and full-vocabulary top1 is1,.75,.875,.625. These qualify16discovery states for future causal testing, not immediate fitting.

Clause completion fails discovery/lexical/structural gates; its passing joint cohort cannot rescue discovery. List markers pass but remain quarantined because of their relationship to the original sequence_copy holdout. The explicit eligibility manifest overrides the raw generic whole_task_holdout=False field.

Noun contrasts confound instrument and action, though all16pairs have equal length and final determiner. Subject splits reuse action cues. Word-fragment tests measure one next BPE token, not full-word correctness; scientific-domain cues and variable lengths remain confounds. Neither establishes a discovered causal unit.

Independent saved-logit audit: endpoint metric error0,16gates replayed, all64pairs same finaltoken, no exact prompt overlap with the three checked prior panels, no candidate-discovery versus non-discovery prompt overlap. Forward replay: capture relative error5.03e-7, logits8.11e-6. Source hashes unchanged; no fitting or reserved tests. Natural-document pool remains26,112.

Next: finite interventions for noun expectation and word fragments, comparing residual-derived directions against the old task PCA span, full residual and random controls, plus factorial cue controls. Require causal qualification before training. This advances the discovery branch but does not complete the bootstrap loop. H961 remains pending; broad goal active.
