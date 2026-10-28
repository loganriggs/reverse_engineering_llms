# Neurons as context-dependent discriminators

## Short answer

Joint neuron codes are clearly present, but the strongest proposed definition of polysemanticity—one neuron's causal effect reversing when a partner is toggled—is uncommon in these tasks. Depth creates non-additive interactions; most of the memory interactions remain answer-conditioned and sign-stable, which is more consistent with compositional verbs than arbitrary multiplexing.

## Self-bilinear memory: fixed tasks across 1–4 layers

All 24 models converged. Values are means over three seeds. `Interaction` is the median exact removal-context interaction fraction for top causal pairs. `Complement` is joint answer information beyond the better individual neuron, after a permutation-null correction.

| task | layers | accuracy | interaction | random-pair interaction | complement bits | within-example sign flips | answer-conditioned sign stability |
|---|---:|---:|---:|---:|---:|---:|---:|
| lookup only | 1 | 1.000 | 0.0000 | 0.0000 | 0.107 | 0.000 | nan |
| lookup only | 2 | 1.000 | 0.0047 | 0.0003 | 0.041 | 0.000 | 1.000 |
| lookup only | 3 | 1.000 | 0.0221 | 0.0028 | 0.085 | 0.000 | 1.000 |
| lookup only | 4 | 1.000 | 0.0265 | 0.0063 | 0.032 | 0.000 | 0.922 |
| lookup after parity training | 1 | 1.000 | 0.0000 | 0.0000 | 0.085 | 0.000 | nan |
| lookup after parity training | 2 | 1.000 | 0.0045 | 0.0016 | 0.067 | 0.000 | 1.000 |
| lookup after parity training | 3 | 1.000 | 0.0289 | 0.0031 | 0.067 | 0.000 | 1.000 |
| lookup after parity training | 4 | 1.000 | 0.0767 | 0.0037 | 0.053 | 0.000 | 1.000 |
| parity-2 | 1 | 1.000 | 0.0000 | 0.0000 | 0.072 | 0.000 | nan |
| parity-2 | 2 | 1.000 | 0.0040 | 0.0015 | 0.073 | 0.000 | 1.000 |
| parity-2 | 3 | 1.000 | 0.0208 | 0.0039 | 0.114 | 0.000 | 0.980 |
| parity-2 | 4 | 1.000 | 0.0489 | 0.0030 | 0.200 | 0.000 | 1.000 |

The one-layer interaction is approximately `10^-6`, validating the additive negative control. Nevertheless, one-layer activations contain complementary joint information. Therefore a distributed joint code does not by itself imply context-dependent causal meaning.

## Grammar transformers

Open graph markers have less than 90% group accuracy and are not used as strong mechanistic evidence. The one-layer bilinear cells are especially underfit.

| MLP | layers | structured interaction | mixed-exception interaction | arbitrary interaction | structured sign flips | arbitrary sign flips |
|---|---:|---:|---:|---:|---:|---:|
| swiglu | 1 | 0.024 | 0.024 | 0.033 | 0.001 | 0.003 |
| swiglu | 2 | 0.043 | 0.048 | 0.067 | 0.023 | 0.047 |
| swiglu | 3 | 0.073 | 0.055 | 0.093 | 0.013 | 0.077 |
| swiglu | 4 | 0.081 | 0.054 | 0.064 | 0.023 | 0.035 |
| bilinear | 1 | 0.020 | 0.032 | 0.021 | 0.000 | 0.001 |
| bilinear | 2 | 0.027 | 0.034 | 0.046 | 0.001 | 0.010 |
| bilinear | 3 | 0.039 | 0.040 | 0.043 | 0.002 | 0.004 |
| bilinear | 4 | 0.094 | 0.041 | 0.048 | 0.054 | 0.003 |

## Verdicts

1. **Validated:** the exact factorial metric detects zero interaction in the architecture where additivity is guaranteed, then detects increasing interaction when downstream nonlinear layers are added.
2. **Validated:** multiple neurons jointly add answer information even when their causal writes are additive. This is distributed coding, not sufficient evidence of polysemanticity.
3. **Supported:** depth creates context-dependent composition. Targeted pairs interact more than random pairs, and the gap generally grows with depth in self-bilinear memory.
4. **Not supported as a general rule:** arbitrary facts do not consistently cause more pair interaction than structured rules. The ordering varies with depth and architecture.
5. **Mostly falsified in its strongest form:** partner-dependent causal sign reversal is rare—zero in the memory panels and usually below 5% in grammar, with the largest reliable exploratory value around 7.7% for depth-3 arbitrary SwiGLU.
6. **More plausible interpretation:** most measured neurons behave like stable discriminating verbs composed into joint codes. Multilayer models add interaction and cancellation, but the current evidence for widespread contextual multiplexing of individual neurons is weak.

## Red-team limits

- Grammar depths use one data/model seed; depth-4 has earlier replications, but the new interaction-depth trend is exploratory.
- The audit examines the top eight first-order neurons per example and all 28 pairs among them. Higher-order interactions can exist without strong pair effects.
- Interaction is defined on an answer margin and depends on the chosen neuron basis. Random-pair controls address selection, not rotation dependence.
- Intact margins grow strongly with depth in the self-bilinear models, so raw `J` values are not comparable across depth. The primary depth plot uses a dimensionless interaction fraction; even that normalization should not be read as a complete scale correction.
- Top neurons and their pair effects are measured on the same causal panel. The exact interactions are valid, but the targeted-versus-random advantage is exploratory until pair selection and evaluation use separate examples.
- The sign-flip threshold is 0.05 raw margin units. This makes cross-depth flip rates conservative at shallow depth and easier to exceed at large-margin deep models; the zero memory flip result remains notable despite this bias.
- Complementary information is observational. The exact factorial intervention is the causal test.
- Answer-conditioned sign stability has small groups for the eight-class grammar and should be replicated with larger panels.

## Figures

- `memory_depth_synergy.png`: replicated fixed-task memory depth sweep.
- `grammar_depth_synergy.png`: structured versus arbitrary grammar interactions.
- `information_vs_causal.png`: why joint information and causal context dependence differ.
- `factorial_examples.png`: three concrete 2×2 neuron interventions.
