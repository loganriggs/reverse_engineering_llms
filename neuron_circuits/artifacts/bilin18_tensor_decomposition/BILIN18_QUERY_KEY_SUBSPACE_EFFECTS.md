## H743 — Compact q2/k2 edits preserve downstream effects on held-out names

We propagated every H742 held-out-name projection through the actual L13H0 source contribution and downstream model. The entire grid was retained: pooled and single-format SVD bases, all tested ranks, three random-subspace seeds, and raw versus norm-preserving edits. No candidate was dropped based on this run's outcomes.

For each projected coefficient, two readouts were evaluated. First, retain the original source value and change only its coefficient, measuring the native intended-answer margin change. Second, substitute Kevin versus Brian through that source contribution and measure the absent-identity contrast. The target is the exact q2+k2 swap with native factor 1 retained, not a full-head replacement or the whole model's task behavior.

Norm-preserving, name-held-out results:

| Basis fit / evaluation format | Rank per q2 and k2 state | Absent effect sign transfer | Relative absent-effect edit error | Relative original-value effect error |
|---|---:|---:|---:|---:|
| Pooled / narrative | 1 | 13/16 | 18.5% | 18.9% |
| Pooled / narrative | 2 | 13/16 | 16.3% | 16.9% |
| Pooled / narrative | 4 | 16/16 | 4.1% | 4.5% |
| Pooled / list | 1 | 4/14 | 73.0% | 71.4% |
| Pooled / list | 2 | 14/14 | 12.8% | 16.2% |
| Pooled / list | 4 | 14/14 | 8.4% | 8.7% |
| Narrative / narrative | 2 | 16/16 | 6.7% | 6.6% |
| List / list | 1 | 14/14 | 13.7% | 15.6% |
| Narrative / list | 2 | 7/14 | 61.8% | 60.0% |
| List / narrative | 2 | 0/16 | 85.0% | 84.0% |

Sign transfer excludes the two directed cases whose native and target signs already agree; their effect errors remain included. Absent-effect edit error is RMS prediction error divided by RMS change from the native absent effect to the exact factor-2-swapped effect. Original-value effect error is RMS deviation from the exact factor-2 swap's intended-answer margin change, divided by RMS of that change. These are different denominators and are not task accuracy.

Random rank-four subspaces transfer 0/48 eligible narrative signs and 2/42 list signs across three seeds, with about 99% and 96% absent-effect edit error. Both formats therefore contain specific compact edit structure beyond equal-rank random projection. Raw and norm-preserving variants, higher ranks, all seeds and both exchange directions remain archived.

The format restriction survives causal testing: a basis fitted solely on narrative examples transfers poorly to lists, and conversely. The pooled rank-four model may combine multiple format-specific directions. Moreover, the downstream experiment tests the same surgical coefficient-to-source pathway characterized in H740/H741, so its agreement with coefficient fidelity is expected under the validated fixed-map mechanism. It is necessary implementation/causal validation, not an independent proof of abstract semantics or a new population sample.

Audits: 5,760 full-model example forwards completed. Full-rank absent effects exactly reproduced H741. Native absent effects reproduced H740 within 3.10e-6; native-value identity margin error was at most 4.29e-6; zero-coefficient absent contrast was exactly zero. All source/data hashes remained unchanged and results were finite. No new natural documents or original reserved-test evaluations were used.

Interpretation: a compact, causally effective operation-edit subspace is now a viable candidate in this head. It is more constrained evidence than high-capacity reconstruction: it survives held-out names, equal-rank random controls, normalization controls, exact projection checks and downstream intervention. It remains a supervised paired-edit construction on reused templates, not a verified manifold, unsupervised task label, universal operation axis or sufficient account of the entire network.

Next: freeze the pooled rank-four query/key bases and compare them with the already-fixed random rank-four controls on a newly locked panel that varies names and wording independently, retaining narrative/list forms. Check native competence and source/final-position compatibility before interpreting interventions; keep failures and exclusions visible. Use full factor-2 exchange and zero/native controls, with no refitting or rank selection on new outcomes. This is the needed confirmation before tracing upstream causes or naming a broader computational unit. Sentiment's separate diagnostic remains outstanding.
