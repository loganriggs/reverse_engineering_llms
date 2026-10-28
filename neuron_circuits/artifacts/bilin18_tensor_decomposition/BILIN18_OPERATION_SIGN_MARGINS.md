## H748 — Sign failures concentrate at small target effects, but are not all negligible

This posthoc diagnostic stratifies H747's frozen interventions by absolute and relative target effect. No bases, examples, or competence gates change. All240 contexts remain in the record;120 competent narrative contexts are summarized here. Of these,102 have opposite native/full-intervention signs in the absent Kevin-versus-Brian contrast.

The original operation subspace preserves86/102 reversals; the pooled subspace93/102. Original failures have median absolute target effect0.0586 logits, maximum0.1464; successes have median0.3326. Fifteen of16 original failures occur in binding receivers. Pooled failures have median0.0741, maximum0.1252; eight ofnine occur in binding receivers.

| Minimum absolute target effect | Original successful/eligible | Pooled successful/eligible |
|---|---:|---:|
| 0 | 86/102 | 93/102 |
| 0.01 logits | 85/99 | 91/99 |
| 0.05 logits | 79/87 | 82/87 |
| 0.10 logits | 68/73 | 71/73 |
| 0.25 logits | 52/52 | 52/52 |
| 0.50 logits | 19/19 | 19/19 |

These thresholds were fixed in the diagnostic configuration before calculation, but the analysis itself follows inspection of H747. They are descriptive, not a new success criterion. The original all-context errors and failures remain unchanged. Shared templates and names mean counts are not independent trials.

Small absolute effect does not always imply a trivial relative error. Relative target margin is abs(full target effect)/abs(full minus native effect). The original failures have median0.070 but maximum0.693; even restricting to relative margin>=0.5 leaves one failure out of51. Both original and pooled bases miss that case. Thus near-zero sign sensitivity explains much of the discrepancy between RMS fidelity and sign counts, but not all of it.

Interpretation: do not discard the component because of raw sign counts, and do not declare a reliable operation switch based on low RMS error. The evidence is strong for approximation of substantial single-edge effects and weaker for exact binary sign transfer. The next more informative experiment changes actual query/key states, allowing all affected attention edges to respond, instead of surgically prescribing one edge coefficient.

Audits: all240 contexts retained; source hash unchanged; finite diagnostics; zero additional model forwards or reserved-test evaluations; natural document pool remains26,112.
