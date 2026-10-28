## H964 — Entity binding and sequence copying have stronger attention-output intervention effects

Screened all18 attention-output sites for entity binding and sequence copying, selecting layers using discovery pairs only. Replaced the last-token attention output while preserving the native attention value bus. The same-block MLP and downstream computation respond naturally. Compared donor swaps with identity and paired norm-matched random edits, and same-answer discovery donors where available. No featurizer training.

| Family | Selected attention layer (zero-based) | Discovery signed donor margin change | Random-edit RMS | Lexical candidate donor effect | Best discovery MLP effect (H963) |
|---|---|---|---|---|---|
| Entity binding |14|3.755|0.274|4.579|0.175|
| Sequence copying |8|2.042|0.304|2.931|0.153|

Entity-binding structural/joint effects are3.993/4.844, with native competence gates passing. Sequence-copy structural/joint effects are2.282/3.173, but native competence gates fail: these remain intervention diagnostics, not evidence of successful task transfer. Same-answer controls are available in structural cohorts only (RMS0.181 entity,0.150 copy); missing controls elsewhere limit specificity claims. Four discovery pairs per family support exploratory localization, not strong uncertainty estimates. Candidate lexical cohorts do not guarantee vocabulary-disjoint evaluation.

Attention edits cause much stronger task-margin changes than the previously screened MLP writes. This does not estimate additive fractions of computation: edit magnitudes differ across modules, and attention edits also affect the same-block MLP. Neither whole-write swaps nor shared layer involvement establish reusable features. Full-output KL(patch||native) measures perturbation size, not agreement with a known correct counterfactual.

Audits:1,430/1,430 whole-model forwards, finite outputs, inputs unchanged, frozen discovery-selected layers, independently recomputed selection and exact aggregate means. Capture replay maximum logit difference2.86e-5; identity2.65e-5; random norm mismatch1.78e-7 relative. Independent manual model implementation matches nontrivial hook-based donor patches exactly at layers0,8,17, preserving native attention-bus state. Zero reserved-test evaluations. Sequence copying remains a whole-task training holdout. Natural-document pool remains26,112.

Next: lock multi-task causal targets around MLP12 for agreement/sentiment/negation/pronoun, MLP13 for capital recall and attention14 for entity binding, retaining attention8 copying and MLP8 spatial inversion as training holdouts. Distinguish shared computation from generic output readout with held-out combinations and selective interventions before claiming meaningful featurizer manifolds. H961 frozen context-bridge predictor evaluation remains pending.
