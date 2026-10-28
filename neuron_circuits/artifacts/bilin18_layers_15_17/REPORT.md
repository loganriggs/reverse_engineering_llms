# Bilin18 layers 15–17: output-first interpretation

## Bottom line

The last three MLPs occupy a different regime from layers 0–2.

1. Coarse native-neuron alignment reaches a minimum around layer 15 and then
   increases toward layer 17. The final layer is not simply the most entangled
   layer.
2. Individual `Down→unembedding` directions are extremely broad. Their median
   effective support is about 16,000 vocabulary tokens, and their largest token
   coefficient is only about 2.1% of direction norm.
3. Despite those dense output weights, activation-conditioned computational
   importance is sharply concentrated. The final-layer ranking has effective
   support about 46 channels, and its top 64 channels contain 69% of ranking
   mass.
4. A small fixed subset recovers most of the final MLP's next-token-loss benefit,
   but many more channels are required to reproduce the intact distribution.
5. Exact late-layer neurons are assembled from many earlier residual writes.
   The most useful description is usually “a native discriminator acting on a
   distributed upstream code,” not a direct token detector and not an
   uninterpretable random mixture.

## Data and architecture

- Checkpoint: `Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd`
- Parameters: 545,902,902
- Native product channels: 4,608 per MLP
- Final activation atlas: 2,048,000 discovery and 512,000 held-out FineWeb
  positions per layer
- Layers processed in output order: 17, then 16, then 15
- Output census: every channel folded through `Down` and the unembedding
- Exact source traces: four selected neurons per layer, positive and negative
  poles, 48 sites total
- Final-MLP replacement: two independent discovery/evaluation ranges; 400 and
  1,600 held-out next-token examples

## Input-side alignment returns near the output

The same frozen ten-class audit was used at every depth. Results cover the more
usable half of each layer's channels.

| Layer | Median held-out input lift | Channels ≥2× lift | Median signed gap | Channels ≥1 SD gap | Left/Right same class |
|---:|---:|---:|---:|---:|---:|
| 15 | 1.289× | 5.25% | 0.313 SD | 4.30% | 39.71% |
| 16 | 1.300× | 7.55% | 0.398 SD | 9.72% | 35.59% |
| 17 | 1.370× | 14.28% | 0.547 SD | 19.40% | 36.41% |

For comparison, layer 2 had median lift 1.516× and median gap 0.559 SD; layer 0
had 1.824× and 0.883 SD. The overall story is therefore not monotonic loss of
alignment. Direct lexical structure falls early, reaches a low point in the late
middle, and partly reappears at the final readout.

The rebound is robust to sample size: the 76,800-position development census
and the 2.56M-position final census give nearly identical results.

## Output weights are broad, not one-neuron/one-token decoders

| Layer | Median direct-logit norm | Median largest-token fraction | Median vocabulary effective support | Output-weight energy effective channel count |
|---:|---:|---:|---:|---:|
| 15 | 96.00 | 2.09% | 16,089 | 4,447 |
| 16 | 94.06 | 2.10% | 16,049 | 4,397 |
| 17 | 105.22 | 2.11% | 16,019 | 4,485 |

Raw output geometry is almost maximally spread over the 4,608 channels: the top
64 layer-17 columns contain only 2.35% of unconditioned
`Down→unembedding` energy. Thus a small effective layer, if present, cannot be
seen by looking for a few unusually large decoder columns.

The per-neuron output tails remain useful descriptions of *direction*. For
example, layer-17 unit 59 responds positively to capitalized continuations such
as initials and directly promotes common male-name tokens including ` Paul`,
` John`, ` Richard`, and ` Michael`. That is a plausible coherent name-related
interface. Many other channels have much less tidy activation/output agreement,
which the viewer intentionally exposes.

## Is the final bilinear layer effectively smaller?

Yes for recovering most of its task-loss contribution; no for faithful
emulation. A subsequent learned-replacement experiment makes this conclusion
substantially stronger than the fixed-channel results below.

Channels were ranked on a discovery split by:

`activation variance × squared norm(Down→unembedding)`.

The ranking replicated almost perfectly: top-64 overlap was 100%, top-128
overlap 98.4%, and top-256 overlap 97.7%.

| Retained layer-17 channels | Layer fraction | CE increase, replication | KL from full, replication | Top-1 agreement | Loss benefit recovered |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.69% | +0.181 | 0.180 | 82.9% | 56.3% |
| 64 | 1.39% | +0.121 | 0.127 | 83.7% | 70.9% |
| 128 | 2.78% | +0.086 | 0.091 | 84.9% | 79.3% |
| 512 | 11.11% | +0.054 | 0.055 | 87.6% | 87.1% |
| 1,024 | 22.22% | +0.035 | 0.037 | 88.8% | 91.5% |
| 2,048 | 44.44% | +0.020 | 0.019 | 92.5% | 95.2% |
| 4,096 | 88.89% | +0.001 | 0.002 | 97.6% | 99.7% |

Omitted activations were replaced by their discovery mean, which can be folded
into a replacement bias. Random fixed subsets were dramatically worse: random
2,048-channel subsets had CE increases of roughly 0.15–0.18, compared with
0.020 for the ranked subset.

The activation-conditioned importance score has effective support 45.6
channels. The top 32 contain 58.4% of its mass and the top 64 contain 69.1%.
This is the sense in which the nominal 4,608-channel layer behaves partly like a
much smaller bilinear layer. However, 64 channels still leave 0.127 nats of KL,
so they are not a faithful mechanistic replacement.

### Learned low-channel replacement

A new bilinear MLP was trained to match the complete final-MLP write using a
cosine objective plus a write-norm term. Its factors were normalized to quotient
out bilinear scaling gauge; evaluation compared functions and was therefore
invariant to left/right swaps, paired sign moves, and channel permutations.

Across two disjoint FineWeb replications, learned 16, 32, 64, and 128-channel
students had mean CE increases of +0.086, +0.073, +0.062, and +0.049. The
32-channel student used only 0.70% of the original MLP parameters and incurred a
mean 7.5% perplexity increase. On the same evaluation panels, fixed native
32/64/128-channel subsets had CE increases of +0.175/+0.119/+0.084. The learned
quadratic basis therefore compresses much better than native pruning.

This supports a small *distributional bilinear rank* for much of the final
layer's useful computation. It does not imply exact equivalence: the learned
128-channel student still had 0.055 mean KL from the full model and 84.9% top-1
agreement. See `../bilin18_final_mlp_student/REPORT.md`.

## How late neurons are constructed

The exact residual decomposition has many possible raw source pairs by layer
15–17. At selected extreme sites, the median largest raw pair accounts for only
5.7–8.4% of absolute pair mass, with raw effective support 38–55 pairs. This is
much more distributed than layers 0–2.

Because the number of named historical sources grows with depth, sources were
also grouped into embedding, early blocks, middle five blocks, recent three
blocks, local attention, and numerical rounding. On this fairer grouping:

| Layer | Median largest grouped pair | Median grouped effective support | Most common dominant grouped interaction |
|---:|---:|---:|---|
| 15 | 37.0% | 4.54 | recent × recent or middle × recent |
| 16 | 36.2% | 4.08 | middle × recent |
| 17 | 31.9% | 5.16 | middle × recent |

No selected late-layer product changed sign when the current layer's local
attention write was removed. The current attention still changes magnitudes,
but the dominant state was already carried by earlier MLP writes. By contrast,
six of sixteen selected layer-0 sites changed sign after removing attention 0.

The late-layer picture is therefore a long-timescale conjunction: a native
bilinear neuron combines several previously constructed states. This is denser
than the early lexical separators but still structurally describable.

Accumulated bfloat16 residual-addition error reached 1.65% before correction.
The final tracer preserves it as an explicit `rounding_residual` source. Its
median contribution is only 0.18–0.42% of absolute pair mass and never dominates
the selected examples. After including it, vector, factor, and product
reconstruction errors are exactly zero.

## Representative regimes

- **Layer 17 unit 59:** capitalized/initial-like input state with a male-name
  output direction. This is the cleanest input/output interface found so far.
- **Layer 17 unit 3504:** compression rank 1. It strongly detects punctuation
  and document-boundary states, suppresses common function words, and promotes
  several numeric/uppercase pieces. It is computationally important without
  having a single clean human concept—possibly an output-calibration or boundary
  verb.
- **Layer 17 unit 4094:** compression rank 11, strongly separates auxiliary-like
  states from numbers and writes toward several verb-like continuations. Its
  exact activation is assembled from many earlier MLP interactions.
- **Layer 16 unit 438:** positive on auxiliaries such as `may`/`will` and
  negative on table/date punctuation. Its selected sites have 22–86 raw source
  pairs of effective support, even though the native neuron has a stable coarse
  discrimination.

These cases show why “monosemantic or superposed” is too narrow. A channel can
have a stable discrimination and a broad output role while being causally
constructed from a distributed upstream code.

## Red-team conclusions

- The ten-class vocabulary underdescribes syntax, discourse, entity type, and
  long-context state. “Unexplained” is not evidence of randomness.
- Direct `Down→unembedding` products omit the nonlinear effect of final RMSNorm.
  The compression experiment recomputes RMSNorm and actual logits, so it is the
  stronger output-side result.
- Source-pair support mechanically grows with depth; grouped-source results are
  reported to avoid calling this growth alone “more polysemantic.”
- Exact source traces use 48 selected extreme sites. Their source organization
  is not an unbiased population frequency.
- The replacement test measures document-final position 127. Other positions
  and context lengths could require different channel subsets.
- The small-subset result concerns retained native channels. A learned smaller
  bilinear layer could do better; pruning without retraining could do worse on
  rare behaviors.
- Cross-entropy and KL answer different questions. The “small layer” claim is
  validated for most average loss benefit and falsified for close reproduction
  of the full model distribution.

## Artifacts

- `neuron_viewer/`: all 13,824 late-layer channels with activation and output
  evidence
- `output_geometry/results.json`: complete output-weight census
- `source_traces/results.json`: exact selected-site decompositions
- `viewer_atlas/layer_{15,16,17}/results.json`: large FineWeb censuses
- `../bilin18_final_mlp_compression/`: replacement experiment and independent
  replication
