## H553 — Distributed MLP response transfers; a single layer is insufficient

Freeze the H552 selection (L8 MLP, best single-layer discovery MAE) before evaluating 48 newly constructed prompts: eight lexical pairs each in prose-code copying, record-list copying, and binding. Copying uses colors, binding uses names. Use the same L8H3 source-edge intervention and five virtual positions. Save all 240 frozen predictions before 960 live/clamped evaluations. Conditions: live, L8-only MLP clamp, all MLP writes L8–L17 clamped, and those plus attention writes L9–L17 clamped. No outcome filtering.

Native competence: prose copying 16/16 candidate and full-vocabulary correct; record copying 15/16 candidate, 13/16 full-vocabulary, 7/8 both-endpoint pairs correct; binding 16/16 candidate, 14/16 full-vocabulary. All pass the existing competence gate. The unsuccessful record endpoint remains included.

| Family | Clamp | Frozen-prediction direction agreement | Prediction MAE | Mean absolute effect |
|---|---|---:|---:|---:|
|binding|live|100.0% (64)|0.01200|0.07523|
|binding|single_8|100.0% (64)|0.01042|0.07574|
|binding|freeze_mlp|100.0% (64)|0.00589|0.07588|
|binding|freeze_both|100.0% (64)|0.00000|0.07615|
|copy_prose|live|0.0% (64)|0.14114|0.09356|
|copy_prose|single_8|6.2% (64)|0.08268|0.03603|
|copy_prose|freeze_mlp|100.0% (64)|0.00955|0.04047|
|copy_prose|freeze_both|100.0% (64)|0.00001|0.04758|
|copy_record|live|0.0% (64)|0.20256|0.12297|
|copy_record|single_8|6.2% (64)|0.13733|0.06002|
|copy_record|freeze_mlp|100.0% (64)|0.02225|0.05734|
|copy_record|freeze_both|100.0% (64)|0.00000|0.07959|

Both copying templates have 0/64 live direction agreement with the frozen prediction. Clamping only L8 lowers prediction MAE by 41% for prose and 32% for record, supporting the locked magnitude prediction, but restores direction in only 4/64 per template. All-MLP clamps restore 64/64 in each. Binding remains 64/64 direction-consistent even when live. Thus the distributed response transfers to new wording, whereas a compact single-layer explanation does not.

This is stronger evidence that an apparently identity-aligned attention write can have an opposing live effect after downstream computation. It is not evidence that the model becomes inaccurate under every intervention; effect-direction agreement and task accuracy are different quantities. Nor does the result yet establish abstract task specificity: copying uses colors and binding uses names, with different contexts and source distances. Head, source role and intervention class are also fixed, not a random sample of the model.

Both-clamped replay agrees within 1.82e−5; native-position self replay within 2.77e−5. These errors are small relative to the copying effects. Predictions, source and manifest remain unchanged and outputs finite. An initial runner attempt stopped before intervention evaluation due to a manifest-hash variable colliding with a hook function; renaming the hook fixed it, with identical prompts and config. There was no outcome-based redesign.

### Updated hypothesis set and discriminating tests

These are alternatives and compatible mechanisms, not calibrated probabilities. “Higher” means the evidence currently favors pursuing the idea, not that it has been established.

| # | Hypothesis | Priority / evidence | Discriminating test |
|---|---|---|---|
| 1 | Computation depends on a reader and its downstream consumers | Higher: live copying reverses frozen effects | Predict consumer responses on held-out contexts |
| 2 | Several MLP layers jointly transform the write | Higher: no single clamp restores discovery; all-MLP transfers | Cumulative and leave-one-live interventions |
| 3 | Same-layer L8 response provides a substantial first transformation | Higher: best single-layer discovery, transfer MAE reduction | Exact L8 input/output response decomposition |
| 4 | Context gates bilinear MLP response to the same write | Plausible | Match write direction across contexts, compare finite response |
| 5 | Token vocabulary, rather than task, explains copying/binding contrast | Unresolved major confound | Copy names and colors in identical templates |
| 6 | Physical source–destination distance explains contrast | Unresolved | Matched-distance templates and actual text-position controls |
| 7 | Rotary phase interacts with contextual key/query content | Supported locally by H549–H550 | Cross content and phase on fresh matched cases |
| 8 | A universal monotone distance preference explains all readers | Weakened by H550 | Prospective distance sweeps without fitting |
| 9 | A single MLP layer explains the reversal | Weakened by H552–H553 | Locked single-layer transfer and joint-layer tests |
| 10 | Residual normalization creates most of the response reversal | Open | Separate normalization-induced input change from bilinear response |
| 11 | Bilinear first-order cross terms dominate the finite response | Open | Exact linear-versus-quadratic response decomposition |
| 12 | Large interventions create off-distribution nonlinear reversals | Open | Positive/negative amplitude sweep approaching zero |
| 13 | Late layers counteract earlier transformations | Supported in discovery, not yet transferred | Boundary clamps and layerwise signed response accounting |
| 14 | Attention–MLP feedback rather than MLP alone is essential | Plausible: attention clamps also help | Factorial clamps at selected boundaries |
| 15 | Readout saturation creates apparent reversals | Weakened: exact frozen readout still disagrees | Compare residual/logit effects before and after bounded readout |
| 16 | Color copying has a special lexical subspace | Open | Hold templates fixed across colors, names and common nouns |
| 17 | The selected source is atypical among copying sources | Open | Repeat source-role intervention at first and second occurrences |
| 18 | A few neurons or a low-dimensional MLP response span the transformation | Open; not implied by whole-write clamps | Fit on discovery, test intervention reconstruction on held-out prompts |
| 19 | Effects are mainly additive cancellations rather than new feature construction | Open | Dose-response and exact path/response decomposition |
| 20 | Prompt-family surface conventions explain the transfer | Still possible despite two formats | New templates with matched tokens, positions and competence |

Next H554 prioritizes hypothesis 5: use the same name pairs for copying and binding, and cross name versus color vocabulary within identical copying templates. Freeze heads, layer and intervention conditions. Only after that distinction should the result be labeled task-specific. In parallel conceptual planning, retain amplitude/normalization decomposition as the route to specifying what the downstream consumers compute. Physical-position validation is still outstanding.

H552–H553 completed 3,008 interventions and added 48 constructed prompts. No new FineWeb documents; total 26,112. Broad research goal active and incomplete.
