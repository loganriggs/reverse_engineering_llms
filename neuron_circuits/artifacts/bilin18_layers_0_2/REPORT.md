# Bilin18 layers 0–2: weight-first interpretation report

## Bottom line

The first three MLPs do not split cleanly into “monosemantic neurons” versus
“uninterpretable superposition.” The evidence instead supports a depth-dependent
change in the object that is easiest to interpret:

- **Layer 0 is often a direct quadratic lexical separator.** Its two learned
  factors frequently agree on a coarse token class, and a single named
  source-pair usually dominates selected examples. Local attention can construct
  or reverse the separator's sign.
- **Layers 1 and 2 still contain clear native neurons**, including number,
  auxiliary, preposition, determiner, and capitalization-related separators.
  They are less common under the same frozen audit vocabulary.
- **Higher-layer neurons more often operate on an upstream learned code.** In
  the traced examples, several embedding, previous-MLP, and local-attention
  terms cooperate. By layer 2, the layer-1 MLP write is the main carrier in all
  selected sites, while local attention typically modulates or binds it.

This is compatible with “neurons as verbs,” but the verb increasingly acts on
a contextual state rather than directly on a token. Dense causation is not by
itself a failure of interpretation: an understandable separator can be assembled
from several understandable sources.

## Model and evidence

- Checkpoint: `Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd`
- Parameters: 545,902,902
- Architecture audited: squared bilinear attention and native bilinear MLP
  product channels
- Layers: 0, 1, and 2
- Channels: 4,608 per layer; 13,824 total
- Final activation census: 8,192,000 discovery plus 2,048,000 held-out FineWeb
  positions per layer
- Corpus source: `HuggingFaceFW/fineweb`, `sample-10BT`, documents
  50,000–64,847; 20,000 tokenized rows of length 513
- Token manifest SHA-256:
  `437ef40fe28fdaebd511819e23e3266811db64989cdea3c321c86686d989814b`
- Weight census: every channel, without selecting on activation examples
- Exact named-source study: four preselected units per layer, four held-out tail
  sites per unit, 48 sites total

The final viewer corpus is disjoint by document range from the earlier
development corpora. Labels were selected on the discovery split and scored on
the held-out split. The same frozen ten-class vocabulary was used in every
layer.

## Population result

The table reports the more usable half of channels, ranked by product standard
deviation times Down-column norm. “Lift” is the mean absolute product activation
on the discovery-selected class divided by the mean over all held-out tokens.
“Signed gap” is the held-out mean difference between the discovery-selected
positive and negative classes, in product standard deviations.

| Layer | Median held-out lift | Channels ≥2× lift | Median signed gap | Channels ≥1 SD gap | Left/Right same class |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.824× | 40.89% | 0.883 SD | 41.36% | 52.04% |
| 1 | 1.626× | 24.91% | 0.694 SD | 24.26% | 49.22% |
| 2 | 1.516× | 16.28% | 0.559 SD | 13.37% | 44.88% |

The decline is monotonic on all five activation measures. A separate, entirely
weight-based test points the same way. Folding normalized token embeddings
through each channel's Left and Right factors gives:

| Layer | Median embedding lift | Median embedding signed gap | Left/Right same class |
|---:|---:|---:|---:|
| 0 | 1.397× | 0.622 SD | 40.63% |
| 1 | 1.290× | 0.542 SD | 31.03% |
| 2 | 1.292× | 0.526 SD | 27.00% |

The weight-only and activation results therefore agree that direct
current-token/class alignment becomes less characteristic with depth. They do
**not** show that layer-2 channels are meaningless. The class vocabulary is
coarse, and contextual distinctions can score poorly even when they are clean.

## What the exact source decomposition says

For a channel with factors `L(x)` and `R(x)`, the residual input was retained as
named additive sources. The product was expanded exactly into symmetric pairs:

`L(x)R(x) = Σ_s L(x_s)R(x_s) + Σ_{s<t}[L(x_s)R(x_t)+L(x_t)R(x_s)]`.

The available sources were embedding and attention 0 at layer 0; those plus MLP
0 and attention 1 at layer 1; and those plus MLP 1 and attention 2 at layer 2.

| Layer | Selected sites | Median largest-pair share | Median effective pair support | Dominant pair families | Sign flips after removing local attention |
|---:|---:|---:|---:|---|---:|
| 0 | 16 | 72.0% | 1.74 | embedding×embedding (7), embedding×attention0 (7), attention0×attention0 (2) | 6/16 |
| 1 | 16 | 37.8% | 3.98 | embedding×attention1 (6), MLP0×attention1 (4), embedding×MLP0 (4), MLP0×MLP0 (2) | 0/16 |
| 2 | 16 | 47.5% | 3.32 | MLP1×MLP1 (9), MLP1×attention2 (7) | 0/16 |

These are selected extreme sites, not population estimates. Within that sample,
layer 0 is usually explained by one dominant lexical or lexical-context pair.
Layer 1 has the broadest source support. Layer 2 is not an arbitrary soup: every
dominant pair contains MLP1, suggesting that the preceding MLP supplies a reused
interface and attention 2 often contextualizes it.

The factor reconstruction and source-pair reconstruction errors were exactly
zero after bfloat16 rounding calibration. The maximum relative vector-component
reconstruction error was 0.3505%, below the predeclared 0.5% gate.

## Concrete examples

### Layer 0 unit 874: quotation-context separator

Positive tails include a quotation mark beginning speech after a newline;
negative tails include a quotation mark ending a phrase such as `"business"`.
At two positive sites, removing attention 0 changes the product from about
`+3,650` to `-151`. The dominant term is embedding×attention0. This is a clean
example of attention constructing the context needed by a native quadratic
separator: the quote token alone is insufficient.

### Layer 1 unit 1690: capitalization-related contextual separator

Capitalized positive sites such as `T`/`H` can be dominated by MLP0×MLP0, while
negative suffix-like sites use attention 1 more strongly. The unit has a stable
observable preference, but its causes are not a single token direction. Local
attention changes magnitude substantially without being necessary for the sign
at the selected sites.

### Layer 2 unit 223: numbers versus “a row”

Positive tails include prices and numeric quantities; negative tails repeatedly
include the word `row` after `a`. It has 7.68× held-out lift and a 2.44-SD signed
gap. Its selected products are dominated by MLP1×MLP1 or MLP1×attention2. The
neuron is well aligned to an observable discrimination even though its input is
an upstream contextual code rather than raw number-token weights alone.

### Layer 2 unit 4027: auxiliary/conjunction state

Positive tails include `can and should` or `can or should`; negative tails
include forms of `is` and `are`. It has 6.59× held-out lift and a 2.86-SD signed
gap. Positive sites are often MLP1×attention2, while negative sites are often
MLP1×MLP1. This is closer to a contextual grammatical operation than a
one-token detector.

## Interpretation regimes supported so far

1. **Direct lexical quadratic separators:** strongly represented in layer 0.
2. **Attention-constructed local discriminators:** clear in layer 0, where
   finite local-attention removal can reverse product sign.
3. **Distributed but named source-pair programs:** increasingly important in
   layers 1 and 2.
4. **Upstream-code consumers:** especially clear in layer 2, where MLP1 appears
   in every dominant selected pair.
5. **Unexplained by the audit vocabulary:** common and increasing with depth;
   this is not equivalent to polysemantic or random.

This falsifies the simplest hypothesis that higher layers should merely contain
the same kind of token-aligned neuron with more sophisticated labels. It also
falsifies the equally simple claim that dense cross-layer causation makes native
neurons uninterpretable. The surviving hypothesis is hierarchical: early
neurons often separate lexical states directly, while later neurons separate
states produced by reusable upstream codes and attention routing.

## Red-team findings and limits

- **Frozen coarse labels:** all depth comparisons use the same labels, so the
  monotonic decline is real under this audit. The labels may systematically
  underdescribe contextual higher-layer structure.
- **Independent evidence types:** the weight-only embedding fold agrees with the
  held-out activation trend, reducing the chance that the result is only a
  corpus-tail artifact.
- **Large, disjoint census:** 10.24M positions per layer made the population
  summary nearly identical to the earlier 307,200-token census. Sampling noise
  is not a plausible explanation for the headline depth trend.
- **Selection bias in mechanics:** the 48 exact source traces are deliberately
  interesting neurons and extreme sites. Their sign-flip and dominant-pair
  counts must not be generalized to all 13,824 channels.
- **Availability is not use:** QK, V, later-MLP, and unembedding contractions
  show that a write direction is readable. They do not show that a real input
  uses that path. Raw gains also cannot be compared across consumer families
  with different dimensions and scales.
- **Local removal is partial causality:** the finite local-attention removal is
  exact for the target factor computation, but the current study did not yet
  run balanced targeted edge removals through final next-token loss with random
  controls. Strong semantic names therefore remain hypotheses rather than full
  end-to-end circuits.
- **Bilinear gauge:** Left and Right can swap and rescale inversely. Their pair
  and product are the stable explanatory object; either factor alone is not a
  canonical neuron meaning.
- **RMSNorm:** named residual contributions are evaluated after the actual
  normalization scale. Finite interventions must continue to recompute
  normalization rather than treating raw weight products as logits.

## Viewer

Serve the directory because the page loads a separate 20 MB data file:

```bash
python3 -m http.server 8765 \
  --directory artifacts/bilin18_layers_0_2/neuron_viewer
```

Then open `http://127.0.0.1:8765/index.html`. The viewer exposes every native
channel in layers 0–2, held-out population position, positive and negative
FineWeb tails, weight-folded vocabulary tails, downstream weight availability,
and exact source traces for the selected mechanistic examples. “No targeted
source trace” means only that the expensive trace was not run for that unit.

## Next decisive experiment

Freeze a balanced set of layer-0 quotation/punctuation neurons, layer-1
capitalization/determiner neurons, and layer-2 number/auxiliary/preposition
neurons. For each, use the QK×OV ranking to remove predicted source edges and
matched random edges, then measure the native factor, channel product, final
next-token logit contrast, and loss. Cross-context transfer should test whether
one edge-level explanation predicts held-out examples of the same proposed
verb. That would distinguish a reusable causal discriminator from a visually
coherent but epiphenomenal activation.

