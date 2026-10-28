## H555 — Copying reversals persist at small positive and negative amplitudes

Reuse all 80 H554 prompts. Take the selected source-edge delta induced by assigning the key the destination's rotary position, then scale this fixed vector by −1, −1/4, −1/16, 0, +1/16, +1/4 and +1. Negative scaling is a vector intervention, not an actual alternative rotary position. Test live, L8-only MLP clamp, all downstream MLP clamps, and all downstream attention/MLP clamps. Lock 560 finite frozen-readout predictions before 2,240 interventions. No selection by observed effect or competence.

| Family | Live agreement at −1/16 | Live agreement at +1/16 | All-MLP-clamped agreement at ±1/16 | Live absolute effect per unit amplitude, −1/16 / +1/16 |
|---|---:|---:|---:|---:|
|binding|100.0% (16)|100.0% (16)|100.0%|0.05693 / 0.05702|
|color_prose|0.0% (16)|0.0% (16)|100.0%|0.17400 / 0.17256|
|color_record|0.0% (16)|0.0% (16)|100.0%|0.19655 / 0.19728|
|name_prose|46.7% (15)|46.7% (15)|100.0%|0.02993 / 0.02998|
|name_record|31.2% (16)|31.2% (16)|100.0%|0.05747 / 0.05645|

Color copying retains zero direction agreement in both templates at every nonzero scale and both signs: all 192 color-copying comparisons oppose the frozen-readout direction. Thus those reversals do not require the original large perturbation. Binding retains full direction agreement throughout. All-MLP clamps recover every direction across families and scales.

Name copying remains mixed at small scales: prose has 7/15 directions agreeing at each sign, record 5/16. One prose effect per sign is below the fixed 1e−4 sign-scoring threshold and is excluded from direction scoring, but remains in magnitude statistics. At +1 amplitude the name formats agree on 11/16 each; at −1 they agree on 5/16 and 3/16. This shows finite-amplitude effects can change individual signs even though the broad MLP dependence persists. H554 pooled four virtual positions, whereas this experiment scales one fixed direction; their pooled percentages should not be treated as repeated measurements of the same statistic.

The near-zero positive/negative mean absolute responses per unit amplitude are close. This supports studying a local response operator, but does not establish infinitesimal linearity, bound per-example curvature, or rule out normalization effects. The smallest perturbation here is 1/16 of a specific edge delta, not a universal smallness guarantee in residual norm. Numerical errors remain much smaller than the mean copying responses; exact zero controls and all-clamped replay are retained.

Next H556: decompose the same-layer MLP response into the effect of input normalization and the bilinear linear/quadratic terms, using measured native and perturbed inputs. Check reconstruction exactly before claiming a mechanism. Compare color/name copying and name binding; then test whether a response operator learned on one vocabulary/context predicts another. Keep downstream layers, source-role controls and physical-position experiments as unresolved branches rather than treating the L8 response as sufficient.

Audits: zero replay within 2.77e−5; fully frozen prediction replay within 2.20e−5; all predictions, manifest and source unchanged; finite results. H554–H555 complete 3,840 interventions and add 32 constructed prompts. No new FineWeb documents; total 26,112. Broad goal remains active and incomplete.
