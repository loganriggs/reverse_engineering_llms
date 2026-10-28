# Bilin18 layers 0–2 interpretation plan

## Goal

Determine what native bilinear MLP neurons in layers 0–2 discriminate, how
attention supplies their context, and whether the useful explanatory object is
a token-aligned neuron, a contextual discriminator, or a distributed source-pair
code. Use the tensor-network weights as the primary evidence and activations to
measure which weight paths are actually used.

The final deliverable is a neuron viewer covering all 13,824 native product
channels. It is built only after the metrics and causal gates are frozen.

## Exact computation to expose

For MLP layer `l`, preserve the residual as named additive sources. Before the
MLP these are:

- layer 0: `embedding`, `attention0`;
- layer 1: `embedding`, `attention0`, `mlp0`, `attention1`;
- layer 2: `embedding`, `attention0`, `mlp0`, `attention1`, `mlp1`,
  `attention2`.

Learned block mixing coefficients are propagated exactly. RMSNorm contributes
one positive scale at each position, so it changes magnitudes but not the signs
of the two factor projections.

For channel `j`, compute source projections

`a_s = Left_j(source_s)` and `b_s = Right_j(source_s)`.

Then its product is the exact ordered source-pair expansion

`h_j = sum_s sum_t a_s b_t`.

Report diagonal terms (`embedding × embedding`, `attention × attention`, etc.)
and symmetric cross terms (`a_s b_t + a_t b_s`). A source-pair concentration
statistic describes whether one named interaction explains the product or many
cancelling interactions are required.

For each local attention layer, expose each source-token path into an MLP factor:

`QK1(destination, source) × QK2(destination, source) × OV_to_factor(source)`.

This is computed per head from native weights. It is validated by reconstructing
the actual factor and by finite removal of the implicated attention write.

## Weight-first measurements

For every native channel in layers 0–2:

1. **Quadratic separator:** retain `Left_j`, `Right_j`, their symmetric rank-2
   quadratic form, norms, angle, and conditioning.
2. **Embedding fold:** evaluate `(Left_j e_t)(Right_j e_t)` over the vocabulary.
   Record positive and negative token tails and coarse lexical selectivity.
3. **Write role:** analyze `Down[:,j]` against the unembedding and the next
   layer's Q, K, Q2, K2, V, Left, and Right rows. This identifies which later
   consumers can read the neuron without claiming they do so in data.
4. **QK×OV context paths:** rank heads and source-token paths into both factors
   at real activation sites.
5. **Source-pair tensor:** measure embedding-only, propagated-state,
   local-attention, and cross-term contributions on real inputs.

## Representation-regime tests

Each neuron receives continuous scores rather than one forced label:

- **direct lexical:** held-out variance explained by current-token identity and
  stability of the embedding-folded token ranking;
- **position/address:** held-out variance explained by position after
  conditioning on current token;
- **local relational:** gain from previous-token or QK×OV source features;
- **context-constructed:** sign or tail membership changes when the local
  attention write is removed;
- **distributed source-pair:** low concentration of the named source-pair
  expansion with reproducible cancellation structure;
- **output-role coherent:** a concentrated write role even when input causes are
  heterogeneous;
- **unexplained:** stable behavior not captured by the frozen vocabulary.

The primary depth question is whether layers 0→2 move from direct token
separators toward contextual/source-pair computations. Existing repository
results motivate—but do not settle—the registered expectation that current-token
determination decreases with depth.

## Causal gates

No semantic name is promoted to “mechanism” without:

1. full forward reconstruction against checkpoint logits;
2. source-sum and QK×OV factor reconstruction error below 0.5% relative;
3. all-one masks reproducing logits;
4. finite local-attention removal agreeing in sign with the path prediction;
5. targeted head/source-edge removal on a balanced held-out panel;
6. matched random-unit and random-edge controls;
7. next-token logit or loss effects, not activation change alone.

## Data protocol

- **Development:** at most 204,800 FineWeb tokens. Used for implementation and
  choosing a fixed metric vocabulary.
- **Mechanistic validation:** a disjoint 204,800-token FineWeb panel. Used for
  reconstruction and targeted interventions.
- **Final viewer census:** at least 10,000,000 fresh FineWeb tokens, streamed in
  batches. Fit and held-out document ranges are disjoint. Only moments,
  quantiles, class sums, source-pair summaries, and bounded activation tails are
  retained; full activations are never dumped.
- **Optional robustness:** a Pile panel is labeled out-of-distribution and is
  never mixed into the FineWeb headline.

The final census sample size, document range, tokenizer, manifest hash, and
checkpoint hash are saved with the artifact.

## Stages and stopping rules

### 1. Exact layer 0–2 tracer

Implement one reference forward that returns named residual sources, QK factors,
OV writes, MLP factors, products, and output writes. Stop if it does not match
the reference checkpoint or source reconstructions miss the 0.5% gate.

### 2. Moderate-data regime census

Run all channels on development and validation panels. Freeze metrics and select
examples using the development panel only. Continue only if held-out scores and
tail interpretations are stable enough to distinguish at least direct lexical,
context-constructed, and unexplained cases.

### 3. Targeted causal programs

For predeclared phenomena—quotation role, sentence/list punctuation, numbers
versus numeric-looking word pieces, determiners, and newly discovered layer-1/2
cases—remove predicted head/source edges. Compare neuron factors, logits, loss,
and matched controls.

### 4. Large-data frozen atlas

Stream at least 10M fresh FineWeb tokens through layers 0–2. Refit no metric
after inspecting held-out results. Produce compact per-neuron records and
aggregate depth comparisons with bootstrap uncertainty over documents.

### 5. Neuron viewer

Build the viewer from the frozen records. It includes layer/unit navigation,
positive and negative contexts, Left/Right/product evidence, embedding-folded
tokens, source-pair heatmaps, QK×OV paths, downstream weight consumers, causal
results, sample counts, and explicit evidence-strength badges. It must expose
failed or unstable interpretations rather than showing only attractive units.

### 6. Red-team report

Test label sensitivity, document duplication, rare-token domination, position
confounding, bfloat16 reconstruction, tail-selection bias, RMSNorm effects,
attention-path cancellation, and the distinction between weight availability
and causal use. The final report states which regime claims survived and which
were falsified.

