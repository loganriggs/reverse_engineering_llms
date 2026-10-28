# M14: simple removed-energy predictors do not explain the block correspondence

2026-10-28. Uses cached M13group interventions only: no new model forwards, featurizer fitting, task examples, matching or group selection. Coefficients are fitted solely on the ten-family training signatures; the six-family evaluation signatures remain untouched by fitting.

## Baselines and comparison scope

Predict signed answer-margin loss from removed relative squared write energy. Three fixed baselines: a single signed through-origin coefficient for allgroups/families; one signed through-origin coefficient pergroup; and a constant training-family mean effect pergroup. The last ignores energy and controls for general group susceptibility. No intercept/regularization/nonlinearity search. All32groups retained.

Unlike M13cross-seed prediction, these baselines use the target model's own removed energy and training effects. They are within-seed magnitude/general-susceptibility controls, not equivalent cross-seed estimators. Both have access to target-model training effects, but their evaluation inputs differ. This is not a universal model-class comparison.

Relative squared error is normalized by evaluation squared margin effects; zero prediction scores1. It is not centered explained variance or fraction of reverse-engineered computation.

## Actual block groups

| Target seed | Shared energy coefficient | Per-group energy coefficient | Per-group mean |
|---|---:|---:|---:|
| 71 | 0.999 | 0.897 | 0.901 |
| 72 | 0.998 | 0.947 | 0.882 |

M13's unchanged geometry-based prediction of seed72effects scored.551, compared with these.998/.947/.882baselines. Thus simple linear removed-energy scaling and average susceptibility do not explain the strongest observed cross-seed correspondence. They still explain some effects; these results do not rule out nonlinear magnitude, contextual sensitivity, common readout directions, or other nonsemantic mechanisms.

## Concentration caveat remains

For seed72actualblocks, omitting antonym gives errors1.067(shared energy),.993(per-group energy),1.003(group mean), versus the fixed M13geometry prediction.682. Equal-family mean errors are1.060/.992/1.031 versus geometry.723. These follow the previously declared concentration diagnostics; no predictions are refitted after omission.

Scalar per-group energy baselines score.929/.885 for seeds71/72; shuffled-block baselines.859/.946. The learned block grouping is not universally special under every measure. Its evidence is specifically the stronger geometry-based cross-seed prediction in M13, conditional on the tested panel and reconstruction.

## Interpretation and stopping rule

The surviving hypothesis is limited: some learned block subspaces have reproducible causal effects that cannot be reduced to these simple energy/mean baselines. We still lack verified semantic manifolds, reusable computational primitives, and a stable one-block-to-one-task mapping. Repeated analysis of this tiny development panel cannot supply that evidence by itself.

Stop adding regressors, thresholds or matching schemes to this panel. Next prepare a fixed context-generalization challenge using the same allowed task operations with changed prompt contexts. Freeze model weights, M13geometry assignments and training amplitudes. Specify native competence gates before examining ablation-transfer outcomes; report failed cohorts rather than silently replacing them. Distinguish context transfer from novel lexical/task discovery, and keep original excluded task families unused. If accepted contexts show the same role correspondence, that would be stronger evidence than another fit on the old14pairs.

## Audit and reproduction

Independent Torch64 least-squares solves reproduce all18NumPy closed-form baseline fits and evaluation predictions, family-normalized scores and antonym-omission scores. Maximum relative coefficient difference9.73e-16. M13reference values are unchanged. No new independent data or native-model verification is claimed.

Run `analyze_role_energy_baseline.py`, then `audit_role_energy_baseline.py`.

[Fixed baseline protocol](validation/role_energy_protocol.json), [all coefficients/predictions](validation/role_energy_baseline.json), [audit](validation/role_energy_audit.json), [M13 role correspondence](GROUP_CAUSAL_ROLES.md).
