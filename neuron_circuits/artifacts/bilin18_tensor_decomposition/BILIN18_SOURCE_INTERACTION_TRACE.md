## H793 — Factorial interactions appear at the second source before growing at the query

Previous turn was progress: H792 established substantial per-example non-additivity in combining L1 source identities, without a clear average agreement bonus. H793 traces the exact same factorial interventions through downstream computation to localize where the interactions form.

For every H791 item/format, replay the complete4x4 source-identity factorial in one batch. At both source positions and the final query token, capture block-scaled residual input, attention-normalized input, attention output, pre-MLP residual, MLP-normalized input, MLP output and block output for layers1–17. Compute I(a,b)=z_ab-z_a0-z_0b+z_00 for all matching and mixed nonreceiver donor pairs. The two L1 output replacements occur before their outputs are captured. Detailed early-layer captures are saved for subsequent interaction-removal experiments.

### Architectural controls pass

The factorial residual is exactly zero immediately after the L1 output replacements: the two edited positions initially contain separate additive changes. The first source position is exactly independent of the later source identity at every traced stage, as required by causal attention.

At the final query, L2 attention combines the two source changes additively to numerical precision: aggregate interaction/joint-change vector RMS is2.70e-6. The query state entering L2 is unchanged by the earlier-position L1 output replacements. Because this model's attention has no softmax coupling across keys, the first L2 query read sums independent changed-source contributions. This is an architectural consistency check, not evidence that later attention must remain additive.

### Where interaction develops

For matching donors, interaction vector RMS divided by actual joint-change vector RMS is:

| Position and stage | Vertical bars | Colon/newline |
|---|---:|---:|
| Second source, L2 attention output | 1.807 | 2.112 |
| Second source, L2 block output | .868 | .997 |
| Query, L2 attention output | .0000019 | .0000042 |
| Query, L2 MLP-normalized input | .0157 | .0109 |
| Query, L2 MLP output | .0625 | .0489 |
| Query, L2 block output | .0469 | .0400 |
| Query, L3 attention output | .602 | .520 |
| Query, L3 block output | .512 | .483 |

The second source can attend to the first while its own query representation is changed. It therefore becomes a strong early site of source interaction, whereas the final query initially receives an almost additive L2 attention update. At the query, RMS normalization of that combined update introduces a nonzero factorial residual, and the subsequent bilinear MLP output also has a nonzero residual. These observations identify stages of formation; they do not uniquely divide the MLP's residual into propagated normalization effects and newly created bilinear effects.

By L3 attention, the query interaction is much larger relative to its joint change. This is consistent with nonlinear source states being read later, alongside changed query/routing computations. It does not prove that the second source alone causes that growth: intermediate untracked token positions and other pathways could also contribute.

Query block-output ratios subsequently reach about1–2.4 in parts of the network and end around1.14/1.17 at L17 for matching donors. Ratios above1 reflect cancellation in joint-versus-single changes; they are not “more than100% causal contribution.” Different stages have different scales, and a large interaction norm is not evidence that removing it will alter the relevant output contrast.

### Verification

512 model forwards. Final candidate-logit changes replay H792 within1.05e-5, confirming that tracing preserves the measured computation. Identity logits replay within1.67e-5 across batches. L1 post-replacement factorial residual and first-source future independence are exactly zero. L2 query-attention interaction has maximum component .000793 at large activation scale, but relative RMS only2.70e-6; the relative architectural audit passes. All metrics finite and source/model/input/reference hashes unchanged. Original reserved test untouched; natural-document pool remains26,112.

The full trace contains both matching and mixed donors, both formats and all stages; the table focuses on matching donors for readability. These reused competent copying prompts remain a localization panel. The trace is descriptive and cannot establish causal mediation, necessity, a discrete agreement detector, or a learned computational DAG.

### Next causal discriminator

Use the saved factorial captures to replace selected joint-state module outputs with their additive predictions z_a0+z_0b-z_00. Predeclare L2 attention output at the second source and L2 MLP output at the query, individually and jointly, with exact identity and reference replays. Read the resulting final candidate specificity and interaction vectors. This will test whether early source-to-source composition or early query-local mixing actually contributes to the observed output non-additivity, rather than assuming that the largest traced norm is the relevant computation. If neither explains the effect, test later query attention and other positions rather than declaring a minimal route.
