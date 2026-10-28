# Causal synergy and contextual-polysemanticity audit

This protocol was fixed before inspecting the new depth-sweep interaction results.

## Question

Do individual neurons implement stable discriminations that compose additively, or does a neuron's causal discrimination depend on which other neurons are active?

The latter is called **contextual polysemanticity** here. It is distinct from a broad but stable predicate. A neuron meaning “has wings” may combine with “animal” to identify birds and with “machine” to identify airplanes without changing its own operation. By contrast, a neuron is contextually multiplexed when its causal effect changes sign, target, or discrimination according to its partner state and no stable predicate explains the change.

## Models and fixed comparisons

1. Grammar transformers at depths 1–4, width 96, for structured (`rho=0`), mixed (`rho=0.5`), and arbitrary (`rho=1`) tables. SwiGLU and bilinear MLPs are analyzed separately. Seed 0 is the exploratory complete-depth sweep; existing depth-4 replications remain external confirmation for earlier metrics.
2. Residual self-bilinear networks at depths 1–4, 128 random binary facts, and 64 neurons per layer. Two fixed tasks are used at every depth: arbitrary lookup only, and the same lookup task plus the same bank of two-fact parity queries. Seeds 7, 19, and 31 are replication units.

Only correctly predicted examples enter answer-margin causal summaries. Accuracy is reported so changes in interaction structure are not confused with failure to learn.

## Exact factorial intervention

For a neuron pair `(A,B)` and a fixed example, evaluate a 2×2 mask grid while recomputing all downstream layers:

```text
m11: A and B present
m10: A present, B absent
m01: A absent, B present
m00: A and B absent
```

The exact interaction is

`J_AB = m11 - m10 - m01 + m00`.

It is zero for an additive causal effect on the measured answer margin. Two grids are evaluated:

- **removal context:** all other neurons remain present;
- **construction context:** all other neurons are absent.

The normalized interaction fraction is `|J| / (|effect_A| + |effect_B| + |J| + epsilon)`. We also report whether A's effect changes sign depending on B and vice versa. Large interactions or sign changes establish context dependence, not semantic incoherence by themselves.

For every example, the eight highest `|activation × margin-gradient|` neurons define 28 targeted pairs. Twenty-eight uniformly sampled neuron pairs form a matched search-control baseline. Selection uses intact first-order scores; every reported interaction comes from fresh finite forward passes.

## Observational information synergy

Neuron activations are median-binarized on a balanced panel. For each targeted pair, estimate

`S_AB = I(Y; A,B | context) - I(Y; A | context) - I(Y; B | context)`.

Grammar estimates condition on relation; memory families have fixed query type. A target-permutation null is subtracted. Positive synergy means the joint activation contains answer information absent from the sum of individual information. This is descriptive until supported by the factorial intervention.

## Primary reported profile

- median and upper-quartile exact interaction fraction;
- fraction of pairs with interaction fraction above 0.25;
- fraction with a causal sign flip;
- targeted-minus-random interaction advantage;
- adjusted observational information synergy;
- interaction transfer: sign agreement for the same pair across examples;
- performance and number of eligible examples.

## Predictions

- A one-layer lookup model with a linear readout after its bilinear channels should be an additive negative control: removal-context pair interactions should be approximately zero.
- Additional nonlinear layers can create context dependence even when individual output writes are fixed.
- If capacity pressure creates contextual polysemanticity, mixed/arbitrary grammar tasks should show greater targeted interaction and lower cross-example sign stability than structured grammar at the same depth.
- If depth mainly composes stable predicates, interaction magnitude may grow but pair signs and effects should transfer within the same rule/query family.

## Interpretation limits

Pairwise interaction is coordinate- and margin-dependent and does not capture every higher-order code. A large `J_AB` can be an interpretable conjunction, normalization, or arbitrary cancellation. “Polysemantic” is used only when context dependence is paired with poor stability or transfer. The top-eight search is a bounded audit, not an exhaustive discovery of minimal circuits.

## Logged post-pilot clarification

The preregistered whole-minus-parts information statistic was strongly negative in the first run because the causally selected neurons often carry redundant individual answer information. This does not invalidate the statistic, but it makes it a poor graph of the user's “joint code adds a discrimination” idea. Before drawing conclusions, the audit was extended to report **complementary information**, `I(Y;A,B)-max(I(Y;A),I(Y;B))`, with the same within-context permutation correction. The original synergy remains reported. Complementary information is nonnegative before bias correction and measures what the pair adds beyond its better member; it is not labeled unique PID synergy.

The first transfer summary also pooled interaction signs across different correct answers. That can label a coherent answer-conditional output role unstable. The final report therefore includes answer-conditioned sign consistency, requiring at least two material examples per answer, while retaining the pooled statistic for transparency. The within-example factorial sign-flip test is unchanged.
