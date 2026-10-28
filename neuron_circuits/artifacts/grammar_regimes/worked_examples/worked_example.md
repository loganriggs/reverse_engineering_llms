# Worked neuron-patching example

**Rule:** relation_0 returns a permuted label for the subject's first syllable.

**Counterfactual:** subject first syllable 2 -> 3.

**Old answer:** `answer_3`. **Intended new answer:** `answer_6`.

**Base prompt:** `<bos> <mira> <agent> syllable_2 syllable_2 syllable_6 <patient> syllable_1 syllable_4 syllable_3 <asks> relation_0 <means>`

**Changed prompt:** `<bos> <mira> <agent> syllable_3 syllable_2 syllable_6 <patient> syllable_1 syllable_4 syllable_3 <asks> relation_0 <means>`

Selection rule: first fixed-RNG candidate naturally answered correctly in both forms by the pure-rule model and switched successfully under its native-32 patch; intermediate-load models and outcomes were not used. There were 2048 eligible pure-rule examples, of which 2048 were native-patch successes. This is deliberately an illustrative success, not an unbiased estimate of effect size.

## What the interventions did

The values below are probabilities assigned to the intended new answer. The unchanged base prompt should stay low; changing the text itself is the natural upper reference.

| arbitrary fraction | intact model understands both prompts? | unchanged base | text changed naturally | all 192 channels patched | top 32 native patched | top 32 rotated patched | top-32 delta energy |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 0% | yes | 0.000 | 1.000 | 0.939 | 0.924 | 0.252 | 0.994 |
| 5% | yes | 0.005 | 0.975 | 0.072 | 0.048 | 0.010 | 0.969 |
| 20% | no | 0.000 | 0.541 | 0.028 | 0.003 | 0.002 | 0.871 |
| 50% | yes | 0.000 | 1.000 | 0.027 | 0.005 | 0.001 | 0.863 |

## The largest pure-rule channels

Each single-channel effect is recomputed through the downstream residual stream and final RMSNorm. Positive values push this recipient toward the intended new answer; negative values oppose it.

| channel | mean activation change | best simple held-out label | label R² | exact target-margin change |
|---:|---:|---|---:|---:|
| 78 | +15.480 | relation&structured_answer | 0.935 | +5.540 |
| 27 | +4.689 | relation&structured_answer | 0.848 | +0.553 |
| 143 | -4.648 | relation&structured_answer | 0.899 | +1.367 |
| 35 | +4.367 | relation&structured_answer | 0.755 | +0.624 |
| 91 | +3.794 | relation&structured_answer | 0.862 | +1.043 |
| 154 | +2.683 | relation&structured_answer | 0.723 | +0.414 |
| 152 | +2.576 | relation&structured_answer | 0.749 | +0.662 |
| 188 | -2.350 | relation&structured_answer | 0.827 | +0.267 |

## How to read this example

- `text changed naturally` shows that the intact model understands both prompts; it is not an intervention.
- `all 192 channels patched` adds the discovery-set mean activation change without changing the words.
- `top 32 native` asks whether the trained neuron basis contains a compact version of that transformation.
- `top 32 rotated` keeps the same number of coordinates after an arbitrary basis change. A native advantage is evidence for neuron alignment.
- The channel labels are restricted descriptive fits, not proof that a neuron has only that meaning. Exact single-channel margin effects are the causal measurement.
- This single prompt is illustrative. The aggregate interpolation figures, not this example, support the population claim.
