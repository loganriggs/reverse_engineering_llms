## H953 — Fresh-context transfer separates task-effect compression from distribution fidelity

All H949 fits (four penalties, two seeds) frozen, together with their discovery-selected top-eight products. Evaluated all 212 full/rank1 directed interventions from H952, including native competence failures. 424 local teacher MLP evaluations plus 7,632 live downstream forwards. No parameters, subsets, or coefficients refit. Exact MLP finite changes are compared against frozen student changes, either alone or together with the exact residual change; native recipient computation is retained. This does not test replacing the whole MLP.

Representative full-input interventions, joint residual+MLP path. Errors are normalized RMS task-margin-change errors. KL ratio compares the predicted output distribution to the exact intervention, relative to omitting the MLP change; one is the zero-MLP-change baseline. Entries span two seeds. Every fit and rank1 condition remains in the JSON; the table emphasizes the predesignated exploratory .1 candidate and unpenalized control.

| Task | Context | Competent | Fit | Joint effect error | Joint KL ratio |
|---|---|---|---|---|---|
| agreement | alternate | False | lambda=0.0:all64 | 68.582–71.270% | 0.921–0.936 |
| agreement | alternate | False | lambda=0.1:all64 | 69.165–75.813% | 0.965–0.990 |
| agreement | alternate | False | lambda=0.1:top8 | 85.692–89.243% | 1.023–1.085 |
| agreement | baseline | True | lambda=0.0:all64 | 10.643–11.245% | 0.068–0.068 |
| agreement | baseline | True | lambda=0.1:all64 | 17.665–18.215% | 0.095–0.098 |
| agreement | baseline | True | lambda=0.1:top8 | 18.933–19.026% | 0.102–0.104 |
| agreement | distractor | True | lambda=0.0:all64 | 18.436–20.207% | 0.600–0.606 |
| agreement | distractor | True | lambda=0.1:all64 | 13.620–14.989% | 0.606–0.624 |
| agreement | distractor | True | lambda=0.1:top8 | 16.032–16.165% | 0.825–0.861 |
| sentiment | alternate | True | lambda=0.0:all64 | 37.156–37.564% | 0.167–0.167 |
| sentiment | alternate | True | lambda=0.1:all64 | 38.972–39.917% | 0.188–0.196 |
| sentiment | alternate | True | lambda=0.1:top8 | 38.451–39.153% | 0.186–0.193 |
| sentiment | baseline | True | lambda=0.0:all64 | 21.100–21.723% | 0.064–0.066 |
| sentiment | baseline | True | lambda=0.1:all64 | 23.879–25.496% | 0.085–0.095 |
| sentiment | baseline | True | lambda=0.1:top8 | 22.309–24.593% | 0.076–0.090 |
| sentiment | composition | False | lambda=0.0:all64 | 66.784–66.908% | 0.590–0.601 |
| sentiment | composition | False | lambda=0.1:all64 | 72.120–72.855% | 0.681–0.696 |
| sentiment | composition | False | lambda=0.1:top8 | 68.915–73.468% | 0.624–0.706 |
| sentiment | distractor | False | lambda=0.0:all64 | 61.962–63.043% | 0.390–0.404 |
| sentiment | distractor | False | lambda=0.1:all64 | 63.157–64.152% | 0.412–0.427 |
| sentiment | distractor | False | lambda=0.1:top8 | 60.855–61.544% | 0.386–0.396 |

Findings:

- Baseline lexical transfer remains useful: .1/top8 joint errors19% agreement and22–25% sentiment, KL ratios.102–.104 and.076–.090. Unpenalized full fits are better (agreement11%, sentiment21–22%); compactness does incur a transfer cost.
- On competent agreement distractors, .1/top8 task error16% looks good, but joint KLratio.825–.861 is far worse than complete .1 fits(.606–.624). Isolated-MLP KLratio is similarly .808–.843, so this is not only the residual bypass. Top-eight pruning discards distribution-relevant behavior that the task margin largely misses. These KL ratios quantify residual distribution error relative to a zero-change baseline, not a percentage of recovered computation.
- Alternate sentiment phrasing is competent but harder: compact joint errors38–39%, KLratio.186–.193.
- Negation, which passes binary discrimination but not the predefined full-vocabulary gate, has compact joint errors69–74%; isolated-MLP errors80–85%. Conflicting sentiment clauses also transfer poorly (joint61–62%) and fail native binary competence. Retain these as diagnostics; do not present them as successful task-general units.
- Alternate agreement has little causal signal at the fixed relay: joint reference RMS.0267 versus3.2203 in baseline, and native full-vocabulary competence fails. Its large normalized errors therefore need that absolute-effect context. Relative-clause agreement remains untested because no pairs were token-aligned.

The original H952 agreement distractor pairs change both subject and distractor number in opposite directions. They therefore cannot separate subject-number computation from distractor processing. This motivates a factorial follow-up rather than further fitting or selecting on this panel. Likewise the sentiment contexts mix lexical and compositional changes; success on template variants would not establish composition by itself.

Audits: native identity max absolute logits error2.53e-5; exact joint margin replay against H9522.14e-5; original checkpoint prediction replay exact; fresh whole-product algebra relative error2.17e-6. All expected forwards completed, input hashes unchanged, finite metrics, independent per-case reaggregation matches within3.34e-16. A first reused-evaluator attempt incorrectly compared fresh212-row predictions against old312-row cached predictions and failed before downstream evaluation. Fixed by replaying the original inputs separately; rejected log retained. No training or reserved-test access; natural document pool26,112.

Conclusion: compact supervised components transfer to some fresh contexts, but task-effect fidelity substantially overstates full-distribution fidelity on competent agreement distractors. Neither stable components nor compactness establish general reusable computation. The evidence favors a restricted approximation of selected mechanisms, with context-dependent missing effects.

Next: freeze all models and use a 2x2 subject-number × distractor-number intervention design. Independently swap subject, distractor, and both; report native competence, MLP/residual/joint effects, full-output KL, and compact-versus-full differences. Add a prospectively token-aligned relative-clause template using invariant past-tense embedded verbs. These interventions should distinguish the modeled task signal from omitted distractor-related computation; do not optimize checkpoints on the new panel.
