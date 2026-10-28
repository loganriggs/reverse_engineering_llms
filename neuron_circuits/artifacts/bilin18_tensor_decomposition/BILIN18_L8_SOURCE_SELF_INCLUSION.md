# Source self-inclusion versus remaining-context alignment

| Baseline / edit | Actual | Shuffled | Gaussian | Angle matched | Actual equal-site | Angle equal-site |
|---|---:|---:|---:|---:|---:|---:|
| intact / tangent | 0.651 | 0.282 | 0.259 | 0.261 | 0.291 | 0.242 |
| intact / add | 0.687 | 0.280 | 0.259 | 0.259 | 0.293 | 0.242 |
| intact / remove | 0.586 | 0.276 | 0.251 | 0.256 | 0.288 | 0.242 |
| removed / tangent | 0.463 | 0.278 | 0.251 | 0.252 | 0.275 | 0.242 |
| removed / add | 0.528 | 0.273 | 0.249 | 0.247 | 0.275 | 0.242 |
| removed / remove | 0.377 | 0.272 | 0.245 | 0.248 | 0.272 | 0.242 |

Locked gates: {'remaining_context_alignment': True, 'self_inclusion_collapse': False, 'same_secant_chart_retention': True}. Same-secant retention: 0.902; reversal error: 1.61e-05; joint metric difference: 2.48e-06.

One model, intervened source-removed baseline, trained upstream sources still correlated. Conditional row bootstrap, no semantic or downstream-quality claim. All source heads/all output coordinates. No fitted decoder. Fresh only within this continuation.
