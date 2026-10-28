# Bilin18 breadth-first whole-model MLP atlas

## Bottom line

The complete 18-layer pass supports a crossword-style interpretation strategy:
map everything cheaply, use convergent signals to prioritize, and only then
trace selected circuits.

The shared-forward atlas processed 2,048,000 discovery and 512,000 held-out
FineWeb positions for every layer—46.1 million layer-position observations—in
124 seconds on an RTX 5080. It reproduces the earlier independently collected
layer 0–2 and 15–17 measurements closely, providing an internal validation of
the new one-pass implementation.

The depth profile is U-shaped rather than a monotonic transition from clean to
polysemantic neurons. Coarse token-class alignment falls rapidly in the first
four layers, remains on a broad low plateau through the middle, and returns near
the output.

| Layer | Median held-out lift | Median signed gap (SD) | Fraction ≥2× lift | Fraction ≥1 SD gap |
|---:|---:|---:|---:|---:|
| 0 | 1.821 | 0.885 | 40.8% | 41.6% |
| 1 | 1.640 | 0.693 | 25.1% | 23.5% |
| 2 | 1.512 | 0.556 | 15.7% | 12.6% |
| 3 | 1.487 | 0.530 | 16.6% | 11.2% |
| 4 | 1.328 | 0.405 | 5.4% | 3.2% |
| 6 | 1.323 | 0.344 | 5.2% | 2.6% |
| 8 | 1.297 | 0.325 | 3.7% | 1.9% |
| 10 | 1.318 | 0.330 | 4.4% | 2.2% |
| 12 | 1.319 | 0.305 | 4.1% | 2.3% |
| 14 | 1.285 | 0.280 | 2.8% | 1.8% |
| 15 | 1.289 | 0.312 | 5.3% | 4.2% |
| 16 | 1.301 | 0.396 | 7.5% | 9.7% |
| 17 | 1.370 | 0.547 | 14.4% | 19.3% |

These labels cover only ten lexical classes. The middle-layer plateau means
“poorly explained by this vocabulary,” not “meaningless.” Selected examples
already show contextual distinctions such as document boundaries, syntactic
roles, number formats, capitalization, and discourse state.

## Causal layer priority

Every MLP was separately replaced by a position-specific FineWeb mean and the
rest of the network was recomputed. A smaller zero-ablation check was also run
on the leading candidates.

The robust priority set is layers **0, 1, 2, 3, and 17**. Their ordering depends
on the intervention:

- position-specific mean replacement ranks 1, 0, 2, 3, 17;
- zero ablation on 200 independent examples ranks 0, 1, 17, 2, 3;
- the corresponding zero-ablation CE increases are +2.56, +1.10, +0.69,
  +0.14, and +0.13.

Layer 1's +8.30 CE mean-replacement effect is not a clean measure of unique
information: its mean write is actively harmful when substituted for a real
state. Zero ablation is the safer confirmation. The agreement on the same
five-layer set, rather than exact rank, should guide the next budget.

Middle layers 7–15 individually have small mean-replacement effects of roughly
+0.025 to +0.054 CE. This does not prove they are dispensable together; residual
networks can contain redundant or compensating modules. It does make them lower
priority for expensive neuron-by-neuron causal work until interactions are
tested.

## Weight-side breadth

Every one of the 82,944 native MLP channels was also folded through its Down
matrix and the unembedding. Output directions are broad at every depth:

- median vocabulary effective support remains about 16,000 tokens;
- unconditioned output energy has effective support 4,200–4,492 channels;
- median direct-logit norm varies only modestly, about 90–105.

Output-weight magnitude alone therefore does not identify a small circuit.
Input-conditioned activation, causal loss effects, and connection routing are
needed jointly.

## Breadth-first allocation policy

The initial pass suggests ranking future work with three separate quantities,
not one “importance” scalar:

1. **Module necessity:** finite mean/zero replacement CE and KL.
2. **Within-module availability:** activation variation times downstream write
   or logit norm, reported as a percentile within that layer.
3. **Explanation opportunity:** held-out lift/gap and cross-corpus stability.

High necessity plus high availability identifies likely causal workhorses. High
necessity with low coarse explanation—especially layer 1—is the strongest case
for tracing incoming attention and prior-MLP connections. High explanation but
low necessity is useful for understanding interfaces but should not consume the
same causal budget.

The first deep-trace allocation should therefore be:

- **Layer 1:** why mean substitution is catastrophic; identify the small set of
  state-dependent writes and incoming attention/MLP0 routes responsible.
- **Layer 0:** direct lexical and attention-constructed quadratic separators,
  using the existing QK×OV causal method.
- **Layers 2–3:** test whether the layer-1 code is reused across both layers and
  whether their similar class profile hides distinct computations.
- **Layer 17:** compare native overcomplete channels with the learned 64-general
  and 128-square replacements.
- **Layer 16:** secondary output-interface priority because its coarse alignment
  rebounds and its mean replacement costs +0.13 CE.

## Red-team limits

- Ten lexical classes systematically underdescribe the middle of the network.
- Activation tails are observational. Only selected early-layer attention edges
  have full downstream causal validation so far.
- Mean and zero replacement answer different counterfactuals and can reverse
  exact rankings. Neither gives a Shapley-style unique contribution.
- Native-neuron scores are within-layer priorities; raw magnitudes should not be
  compared across differently scaled residual layers.
- FineWeb is the activation discovery distribution. WikiText and TinyStories
  should be added to neuron-label stability before assigning durable semantics.
- Weight contractions identify available paths, not paths actually used on an
  example. Connection ranking must be validated by finite edge removal.

## Artifacts

- `summary.json`: all-layer aggregate curve
- `layer_0.json` … `layer_17.json`: all 82,944 channel metrics and selected tails
- `output_geometry/results.json`: full output-weight census
- `layer_importance.json`: position-specific mean interventions
- `layer_importance_zero_check.json`: zero-ablation red-team check
