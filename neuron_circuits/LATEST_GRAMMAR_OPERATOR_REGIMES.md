# Latest result: reusable rules and mixed memory use different operator regimes

## The question

The previous experiments separated semantic reuse, neuron-axis privilege, and
constraint coupling. The new experiments ask a more direct computational
question:

> Across contexts, how many distinct downstream causal maps does a layer use?

For layer `l`, define the centered answer-logit Jacobian

```text
J_l(x) = d(centered answer logits) / d(post-product MLP activation at layer l).
```

This is not an activation similarity. Each row predicts what a small channel
intervention will do to the answer logits at that exact context. Manual traces
reproduce native logits exactly, and one-sided interventions at 1% and 5% of
activation norm are predicted at 0.997–1.000 energy fidelity.

Raw neuron coordinates have a bilinear scale gauge: multiplying a channel by
`s` and dividing its down-projection by `s` leaves the model's function intact.
The main comparisons therefore multiply Jacobian columns by channel RMS, or use
the full natural activation covariance. A deliberate function-preserving gauge
edit changed raw effective rank by 4.60 but RMS-normalized rank by only 0.0007.
The locked `1e-4` invariance threshold technically failed: float32 roundoff
changed logits by `5.5e-5`, and RMS rank moved by `7e-4`. We record that literal
threshold as failed while treating the contrast with the 4.60 raw-rank change as
strong practical gauge-robustness evidence.

## Main result

On jointly learned, attribute/answer-balanced prompts:

| Regime | Layer-0 RMS effective rank | Layer-0 rank-16 reconstruction | Final-layer rank |
|---|---:|---:|---:|
| Pure reusable rules | 16–23 | 0.75–0.80 | 8–9 |
| 50% arbitrary facts | 151–160 | 0.18–0.19 | 18–19 |
| 100% arbitrary facts | 125–143 | 0.18–0.21 | 14–17 |
| Untrained model | 9–10 | about 1.00 | 5–7 |

The high-rank arbitrary regime is learned; it is not generic random-network
complexity. Half-arbitrary load is already saturated: its early rank is at least
twice the pure-rule rank and at least 80% of the fully arbitrary rank in both
paired seeds.

The geometry compresses with depth. Different contexts use many early causal
operators, but those operators converge toward a small late answer-writing
interface.

Absolute effective-rank values depend on the matched panel size and sampling
protocol. Only within-experiment comparisons use the same rank ceiling and
stratification; these are not universal circuit counts.

## Capacity does not restore one shared operator

At fixed 50% arbitrariness, width 64, 96, 192, and 384 models were evaluated on
the same correctly learned prompts within each of three data seeds.

- Mean Layer-0 absolute rank stays high: 122, 120, 128, and 134.
- Mean final-layer rank stays compact: 18, 18, 18, and 19.
- Layer-0 rank divided by MLP channels falls from about 0.95 to 0.17.
- Selected exceptions and unselected training facts have nearly identical
  early ranks.
- Their top operator subspaces do not increasingly separate with width.

Extra width therefore **dilutes** the same diverse operator family across more
capacity. It does not recover a single reusable rule program under this
objective. This coexists with the older result that wider models can have more
privileged native axes. Axis privilege and operator reuse are different
properties.

## The strongest coexistence test failed

An obvious rescue hypothesis was that the mixed model has two internal regimes:
a compact rule circuit for genuine generalization and a high-rank lookup circuit
for exceptions.

That is not what the current model does. On 120 relation/answer-matched examples
over 20 resamples:

| Seed | Mixed held-out rules | Mixed trained unselected | Mixed exceptions | Pure model on identical held-out prompts |
|---:|---:|---:|---:|---:|
| 0 | 73.9 | 74.4 | 74.2 | 13.4 |
| 1 | 73.9 | 71.5 | 76.7 | 21.4 |
| 2 | 73.7 | 76.3 | 77.8 | 28.3 |

The held-out examples were never trained, were unrandomized, were answered
correctly by both models, and used identical prompts in the mixed-versus-pure
comparison. Correct rule behavior in the mixed model is implemented with the
mixed model's broad early operator regime, not the pure model's compact one.

This survives stronger relevance metrics. At Layer 0, mixed held-out rules versus
the pure model on the same prompts have:

- full activation-covariance rank roughly 66–67 versus 13–21;
- softmax-Fisher/covariance rank 28–34 versus 2–4;
- correct-versus-rival margin/covariance rank 21–23 versus 4–6;
- a greater than twofold Fisher-rank gap on the 120 highest-margin correct
  predictions in every seed.

Thus the result is not explained by unused intervention directions, softmax-null
directions, low-confidence lucky guesses, or independently rescaled neurons.

One nuance matters. Under RMS, covariance, and margin metrics, mixed held-out and
trained rules look similar. Under softmax Fisher weighting, held-out rules have
higher rank than trained rules because output confidence/calibration changes the
local behavioral metric. “Same regime” is therefore metric-relative:

- use covariance or margin geometry to study the implemented operator;
- use Fisher geometry to study locally relevant probability changes.

## What the operator modes mean

Simple additive explanations were frozen before fitting and evaluated on new
pairs and a held-out template. At Layers 0–1, relation/rule variables, current
answer and rival, token digits, entity addresses, and their bounded additive
combination explain only about 0–5% of covariance-weighted margin-operator
variation. At Layer 2, output role reaches about 17%. At Layer 3, output role
alone reaches 93%.

So the late interface is easy to name: it is mostly an answer-versus-rival
actuator. The early mixed operator family is not explained by the tested
semantic or additive address vocabulary.

We then tried to treat the internal state as a compact controller. Linear PCA
codes of the local bilinear activation, final-token residual, or complete
residual sequence—up to rank 128—predict essentially none of the early chart on
new facts. Adding every quadratic product among the leading 16 coordinates does
not help. The failure repeats when discovery and evaluation use the same prompt
template, so it is not merely a template-coordinate rotation.

This falsifies a **smooth linear/small-quadratic controller in those
coordinates**. It does not imply irreducibility: the exact weights and state
still deterministically generate the Jacobian. A gated, hash-like, or
higher-order tensor program remains plausible.

The predeclared controller “transfer” condition passes only because validation
and held-out R2 are both near zero. That is a vacuous threshold pass, not
evidence for a useful controller. The analogous bounded-feature transfer result
is substantive at the high-R2 final layer and vacuous in the early near-zero
layers.

## Updated regime map

The controlled models now show at least three distinct organizations:

1. **Reusable-rule regime.** A small family of early causal operators, strong
   semantic transfer, and privileged native patches.
2. **Global mixed/lookup regime.** A broad context-specific operator family used
   for trained exceptions, trained rule-consistent facts, and even successful
   held-out rule predictions. Width reduces capacity density but not absolute
   diversity.
3. **Compact output interface.** Both regimes converge toward a small late
   answer-writing family whose variation is mostly explained by answer/rival
   role.

These are not labels for an entire neural network. One model can move through
all three organizations with depth.

## What is genuinely falsified

- Raw Jacobian rank as a coordinate-invariant metric.
- Arbitrary-memory geometry being merely an untrained random-network effect.
- More width automatically recovering the pure reusable-rule implementation.
- A clean separation between exception operators and unselected training-rule
  operators inside the mixed model.
- Correct held-out rule behavior implying the pure-rule internal mechanism.
- A simple additive semantic/address explanation of the early mixed charts.
- A linear PCA or small top-16 quadratic controller of those charts.

## What is not falsified

- A compact nonlinear gated program.
- A higher-order sparse tensor factorization derived from the exact weights.
- A different learned coordinate system such as a supervised sparse bilinear
  dictionary.
- The possibility that a different training objective or data curriculum makes
  width separate rules from exceptions.
- Transfer of these synthetic regimes to all natural-language computation.

## Next decisive experiment

Do not fit another generic probe. Factor the exact recursive Jacobian as a
weight-defined tensor program:

1. Start at Layer 3, where the answer-role description is a positive control.
2. Recursively include Layer-2 bilinear derivatives and exact QK/value attention
   derivatives.
3. Compress only the fixed tensors with Tucker/TT/sparse bilinear factors; let
   the current state supply contractions or gates.
4. Freeze factor ranks and routing rules on discovery facts.
5. Predict complete held-out Jacobians and finite interventions on new facts,
   templates, seeds, and rule loads.

The decisive outcomes are:

- **same small tensor program, different state contractions:** the high sample
  rank is a local atlas generated by a compact shared algorithm;
- **mixed models require substantially larger tensor rank and description
  length, with causal factor-removal effects:** arbitrary storage increases the
  program itself;
- **discovery fit without held-out causal prediction:** the factorization is a
  lookup or compression artifact, not reverse engineering.

## First exact tensor-chain result

The first step of that experiment is now complete. For a final-position MLP
intervention, causality makes the derivative chain exactly factor as

```text
J_l(x) = O(x) A_3(x) ... A_(l+1)(x) D_l,
```

where `D_l` is the fixed source down-projection, every `A_k` is the exact
final-position residual-to-residual block Jacobian, and `O` is final
RMSNorm/unembedding. Closed-form RMSNorm, softmax QK/value, and bilinear
derivatives reproduce autograd with fidelity 1.000000.

This rejects a tempting but wrong explanation of the pure model. Its compact
operator family is not one constant global chain: replacing every factor by its
discovery mean gives only 0.02–0.08 Layer-0 fidelity. Reuse lives in coordinated
low-dimensional factor variation.

The mixed factors are much more diverse. `A1/A2/A3/O` ranks are approximately
`47–53 / 49–51 / 44–45 / 40–41`, compared with pure values around
`28 / 16–22 / 9–22 / 9–11`. Freezing only `O` retains 0.92–0.93 in the mixed
model, while freezing only `A3` drops to 0.17–0.20. No one varying factor with
all others mean is sufficient.

The exact last-block split is

```text
A3 = (I + ML + MR) @ (I + Q + V),
```

where `ML` and `MR` are the two bilinear product-rule arms, `Q` is softmax-pattern
response, and `V` is the current-position value response. In mixed models:

- freezing `ML` or `MR` retains only about 0.74;
- freezing `Q` retains about 0.93;
- freezing `V` retains 0.995–0.998;
- retaining only one varying bilinear arm reaches only 0.68–0.69;
- Q rank rises from 6–9 to 35, and bilinear-arm ranks rise from 7–16 to 31–34.

The predeclared claim that both arm ranks at least double in every seed narrowly
fails for one arm (`31.4` versus a `31.6` threshold). The causal result is still
clear: paired bilinear gating dominates, QK routing is material, and the local
value derivative is nearly dispensable for this chart family.

This gives two simultaneously correct descriptions:

1. **Local-chart view:** the mixed model uses many context-specific causal maps.
2. **Weight-program view:** those maps are generated by a short exact algebra of
   two bilinear arms plus QK routing, followed by a reusable readout.

The unresolved object is now the gating code—the left/right coefficients and QK
patterns selected by each context—not the derivative law itself. Factor freezing
is an exact tangent-chain recombination, not yet a literal finite component
ablation; finite path-specific validation remains a required red team.

### Native bilinear channels inside the exact program

Each final-block bilinear channel contributes one exact rank-one residual
operator, including its two product-rule arms. Ranking channels once by their
complete downstream answer effect gives sharply different sufficiency curves:

| Regime | top 16 | top 32 | top 64 | top 128 | per-site top 32 |
|---|---:|---:|---:|---:|---:|
| Pure rules | 0.87–0.97 | 0.94–1.00 | 0.98–1.00 | about 1.00 | 0.98–1.00 |
| 50% arbitrary | 0.37–0.39 | 0.49–0.52 | 0.69–0.71 | 0.93–0.94 | 0.73–0.75 |

All 192 channels recompose the exact Jacobian. Removing the fixed top 32 causes
a material loss in every run. The top-32 bank beats random-32 controls, but the
locked 0.10 advantage narrowly fails in mixed models (about 0.083–0.087).

The 0.22–0.24 gain from per-site selection is the clearest current operational
definition of multiplexing. A native channel does not change its weights or
become a different geometric direction. It remains a fixed bilinear
discrimination/operator. What changes is the coalition of channels selected by
the context. Pure rules reuse a small global coalition; mixed memory draws from
a broad bank and routes different coalitions at different sites.

Per-site top-32 is only a contribution-norm oracle and may be nonoptimal under
cancellation. It is evidence that routing matters, not yet a deployable selector
or a semantic explanation of each channel.

The coalition identities sharpen this result. Pure-rule per-site top-32 sets
overlap by 0.51–0.71 on average and draw from an entropy-effective bank of 50–74
channels. Mixed sets overlap by only 0.26–0.27 and draw from about 138 of 192
channels. A discovery-fixed top-32 bank captures 63–76% of pure site selections
but only 37–39% of mixed selections.

The mixed selector is not a hidden relation-module switch: same-relation overlap
advantage is only 0.001–0.004, and same-answer advantage only 0.003–0.013. Pure
coalitions are also not consistently relation-organized; answer identity is the
stronger, though still modest, organizer (0.047–0.072 overlap advantage).

A red-team caught an implementation bug before this conclusion was accepted:
boolean matrix multiplication measured whether two sets had *any* overlap and
made almost every value exactly `1/32`. Converting indicators to integer counts
reversed the initial negative result. The regression suite now includes an exact
overlap-count check.

### Finite path-integrated validation

The QK/value/bilinear ontology now reconstructs actual finite edits, not only
local tangents. Integrating the nine exact last-block derivative terms along real
Layer-0 activation edits reaches 1.000000 fidelity at both 5% and 50% of
activation norm.

| Regime | Identity projection | Direct bilinear | Direct QK | Bilinear×attention |
|---|---:|---:|---:|---:|
| Pure rules | 0.52–0.68 | 0.31–0.47 | 0.003–0.017 | -0.004–0.008 |
| 50% arbitrary | 0.16–0.18 | 0.73–0.78 | 0.011–0.028 | 0.048–0.089 |

Norms expose hidden redirection and cancellation. In mixed models QK has
0.12–0.13 relative norm despite only 1–3% signed projection, and interaction
terms have 0.24–0.31 norm. The local value term has near-zero signed projection
but 0.026–0.040 norm. The locked “value ≤2%” claim therefore fails: value is
small, not strictly negligible.

The attribution changes by less than the locked 0.15 tolerance when edit size
grows tenfold. Arbitrary load therefore shifts finite computation from the
identity route into bilinear gating and bilinear–attention cooperation.

## Artifacts

- [Four-panel figure](artifacts/grammar_regimes/grammar_operator_regimes.png)
- [Load and gauge red team](artifacts/grammar_regimes/GRAMMAR_JACOBIAN_GAUGE_REDTEAM.md)
- [Fixed-task capacity sweep](artifacts/grammar_regimes/GRAMMAR_JACOBIAN_CAPACITY_SWEEP.md)
- [Held-out generalization comparison](artifacts/grammar_regimes/GRAMMAR_MIXED_GENERALIZATION_CHART.md)
- [Natural-metric red team](artifacts/grammar_regimes/GRAMMAR_MIXED_NATURAL_METRIC_REDTEAM.md)
- [Simple explanation test](artifacts/grammar_regimes/GRAMMAR_MIXED_CHART_EXPLANATION.md)
- [Internal controller test](artifacts/grammar_regimes/GRAMMAR_MIXED_CHART_CONTROLLER.md)
- [Same-template controller control](artifacts/grammar_regimes/GRAMMAR_MIXED_CHART_CONTROLLER_SAME_TEMPLATE.md)
- [Exact block-factor chain](artifacts/grammar_regimes/GRAMMAR_EXACT_FACTOR_CHAIN.md)
- [Exact last-block QK/value/bilinear split](artifacts/grammar_regimes/GRAMMAR_LAST_BLOCK_COMPONENT_CHAIN.md)
- [Native-channel frontier report](artifacts/grammar_regimes/GRAMMAR_LAST_BLOCK_CHANNEL_FRONTIER.md)
- [Native-channel frontier figure](artifacts/grammar_regimes/grammar_last_block_channel_frontier.png)
- [Coalition-structure audit](artifacts/grammar_regimes/GRAMMAR_CHANNEL_COALITION_STRUCTURE.md)
- [Finite path-integrated decomposition](artifacts/grammar_regimes/GRAMMAR_FINITE_PATH_COMPONENT_DECOMPOSITION.md)
- [Mixed cross-initialization native effects](artifacts/grammar_regimes/GRAMMAR_MIXED_GATE_CROSS_INIT.md)
- [Mixed cross-initialization margin red-team](artifacts/grammar_regimes/GRAMMAR_MIXED_GATE_MARGIN_ALIGNMENT.md)
- [Mixed residual-map and Jacobian conjugacy](artifacts/grammar_regimes/GRAMMAR_CROSS_INIT_LINEAR_CONJUGACY.md)
- [Pure-rule cross-initialization native effects](artifacts/grammar_regimes/GRAMMAR_PURE_GATE_CROSS_INIT.md)
- [Pure-rule residual-map and Jacobian conjugacy](artifacts/grammar_regimes/GRAMMAR_PURE_CROSS_INIT_LINEAR_CONJUGACY.md)
- [Pure-versus-mixed recurrence comparison](artifacts/grammar_regimes/GRAMMAR_CROSS_INIT_REGIME_COMPARISON.md)
- [Operation-family tournament, original gated run](artifacts/grammar_regimes/GRAMMAR_OPERATION_FAMILY_TOURNAMENT.md)
- [Operation-family high-count replication](artifacts/grammar_regimes/GRAMMAR_OPERATION_FAMILY_EFFECTFUL_REPLICATION.md)
## Cross-initialization boundary: programs recur more than neurons or full tangents

Same-manifest independent initializations now separate three possible meanings of
“the same computation.” Native final-block channels do not match one-to-one on
held-out facts/templates in either pure or 50%-arbitrary models. Pure models do
show more functional-subspace recurrence: top-64 overlap is 0.120 against a
0.063 random-subspace expectation, versus 0.038 against 0.031 in mixed models.
Late post-product gate states are much more decisive: a global linear map reaches
mean Layer-3 held-out-template R2 0.61 in pure models and about -2.60 in mixed
models.

This does not extend to every local direction. The same state maps fail to
conjugate complete centered-answer Jacobians (final-boundary cosine about 0.24
pure and 0.08 mixed; earlier boundaries are near zero). The analytic
channel/factor chain was checked directly against autograd, and the malformed
constant-final-marker version of the state audit was corrected before
interpretation by fitting across all token positions.

Combined with the earlier successful cross-run transfer of selected pure-rule
semantic edits, the result favors **intervention-family-relative recurrence**:
finite semantic programs can be shared in a distributed coordinate space while
native neuron labels and arbitrary off-task tangents remain noncanonical. The
comparison uses only two gate-passing candidate initializations per regime and
is therefore a strong design clue, not a population scaling law. See
`artifacts/grammar_regimes/GRAMMAR_CROSS_INIT_REGIME_COMPARISON.md`.

## Operation-family tournament: where recurrence actually holds

A frozen tournament now compares scaled native permutations, global gate maps,
relation-conditioned maps, and a weight-quotiented residual-write map. Fits use
unlabelled anchors; finite effects are evaluated on template-6 identities and
scored only when the target model's own edit has Fisher norm at least 0.1. The
first run failed its 64-row causal-signal gate, so its positive pure result was
not accepted. A separately locked high-count replication passes that gate for
every family and layer.

Pure semantic edits are highly recurrent: global-map effect fidelity is
0.990/0.998/0.998 at Layers 1--3, and the write quotient reaches
0.923/0.985/0.988. At Layer 3 this is behaviorally meaningful—mapped accuracy
is 0.915 against a 0.943 target-own ceiling. Generic natural prompt-pair deltas
also transfer early but fall to 0.469 global fidelity late, so the final semantic
program is more stable than the unrestricted natural-difference space.

Mixed models separate semantics from facts. Relation-conditioned semantic
effects retain 0.68--0.70 fidelity, but mapped answer accuracy reaches only
0.21 at Layer 3; the target model's own edit reaches only 0.31. Exact arbitrary
fact swaps matched on relation and structured answer do not recur. Global,
relation-conditioned, and write-quotient fidelity all remain below the locked
0.30 gate; the best is 0.272, without the required improvement over global.
Mapped answer accuracy remains at most 0.126. The same network can therefore
contain a partly shared semantic effect geometry and run-specific
arbitrary-memory routes.

The late native-permutation result is a useful negative red team: pure native
fidelity rises to 0.887, so the locked claim that a distributed map must beat
native matching by 0.15 at the best layer fails. “No canonical neurons” is too
absolute; late codebook-relative axes can converge even though early layers and
mixed memory do not.
