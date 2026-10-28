## H976 — Frozen composition-site ablation and period rescue transfer beyond the original quartet cohort

H975 screened64 prospective quartets with frozen candidate sites/formulas. This experiment applies the exact same adjective-last-subtoken MLP3 local/total interaction calculation and periodMLP4 native-write rescue to all cohorts, without layer selection or outcome-based example filtering. Includes paired norm-matched random upstream factorial edits, random period correction controls, and native-rescue identity. New quartet activations provide algebraic coefficients: this tests mechanistic site/formula transfer, not a frozen learned feature encoder.

Eight cohorts pass native competence, including the original anchor and seven beyond it. The frozen local-term removal causes clear answer-margin loss in allseven new passing cohorts; native period rescue recovers a substantial but incomplete part of that loss.

| Passing cohort | Local ablation margin loss | Upstream random RMS | Period rescue margin recovery | Native-reference KL remaining |
|---|---|---|---|---|
| Film anchors (replay) |0.264|0.014|56.1%|0.262|
| Film new3 |0.301|0.032|40.3%|0.440|
| Movie anchors |0.264|0.015|53.1%|0.283|
| Movie new3 |0.303|0.026|39.2%|0.439|
| Book anchors |0.298|0.024|55.7%|0.240|
| Book new1 |0.405|0.022|53.5%|0.243|
| Book new2 |0.327|0.029|51.8%|0.281|
| Book new3 |0.327|0.026|42.1%|0.409|

Book/new1 is a useful prospective case: four new adjective pairs, allsingle-token, allfour endpoints binary-correct in every quartet, full-vocabulary intended top1 rate68.75%. Ablation and rescue transfer there without the multi-subtoken alignment complication. Other new cohorts broaden lexicons but reuse adjective pairs across templates; they are not seven independent lexical datasets. Native gate allows one of four quartets to fail binary competence; those examples remain included, not filtered.

Failed native cohorts also show similar mechanistic effects, but are retained as diagnostics only. Overall-output-wording variants have100% contrast accuracy but poor full-vocabulary top1, so their intervention results do not establish successful native task transfer. New2/new3 include multi-token adjectives; the frozen last-subtoken intervention is applied unchanged and does not claim to capture all adjective tokens.

Interpretation: the adjectiveMLP3-local / periodMLP4 route is not restricted to the original four adjective pairs in film reviews. The site and algebraic formula survive prospective vocabulary and noun-context changes. This is a stronger causal target for evaluating featurizers, but each test still has access to the full teacher quartet to derive its local term and native rescue. A compact encoder or reusable latent operator has not yet been learned or shown to transfer.

Audits:4,096/4,096 whole-model forward examples. Native/rescue identity logit difference2.57e-5; exact weight-plus-bias write replay relative6.40e-7; local/inherited polynomial identity1.97e-15; random norm mismatch1.81e-7. Anchor ablation/rescue margins independently replay H974 within1.01e-5; summary aggregation independently checked. Inputs unchanged and outputs finite. No training, layer retuning or original reserved-test access. Natural-document pool26,112.

Next: test whether the computation uses reusable low-dimensional input subspaces. Build adjective and negation input subspaces from the original discovery quartets only, freeze them, and test how well the projected bilinear cross term reproduces local intervention effects on the prospective panel. Compare rank1 versus full discovery spans, matched random subspaces, and output-direction-only baselines. Keep this span-adequacy test explicitly separate from deployment of an encoder; if it succeeds, train and validate a state-based representation without access to test quartets. H961 frozen context-bridge predictor evaluation remains pending.
