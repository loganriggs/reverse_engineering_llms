# Layer 0/1 feature ontology after prospective circuit compilation

## Current answer

The early tensor transformer does not choose between neuron-aligned features and
dense superposition. It implements several different reusable objects:

1. **Native bilinear instructions.** Every MLP channel always multiplies the
   same unordered pair of learned predicates and writes through the same Down
   column. This algebraic neuron identity is exact.
2. **Generic attention transport.** A small frozen head set transfers lexical
   changes across new pairs and corpora. Layer 0 is especially anchored by head
   7; Layer 1 mixes several heads with carried state.
3. **Instance-conditioned native execution.** For a specified example, native
   neurons beat equal-rank dense rotations, but high fidelity remains broad and
   the selected identities change across contexts.
4. **Low-dimensional finite-edit paths.** A full token replacement follows an
   almost planar curved trajectory. One target-conditioned midpoint JVP predicts
   most of the realized native support and downstream effect.

These do not imply a fixed semantic neuron dictionary. Frozen sentiment,
gender, tense, scale, and polarity banks fail prospectively at the final query.

## Which explanatory object should be used?

| Goal | Best current object | Why |
|---|---|---|
| Name a recurrent lexical discriminator | Native neuron plus activation tails and source trace | Some Layer-0 channels genuinely separate numbers, quotations, or punctuation states |
| Explain how information moves | QK1/QK2/value edges and frozen transport heads | Head modules transfer better than semantic neuron identities |
| Predict an infinitesimal intervention | Base-state Jacobian | It reaches about 0.995 product cosine at 10% edit dose |
| Compile a full token edit | Midpoint/path contraction | Endpoint gradients fail; midpoint support Jaccard is about 0.70--0.87 |
| Edit a reusable human concept | Not yet solved | Same-family neuron banks and fixed final write spaces do not transfer |
| Describe one realized computation | Dynamic native neuron set | Native axes are privileged relative to rotations for that execution |

## Strongest causal result

At fixed rank 256, a midpoint router selected without seeing the finite endpoint
product change obtains 0.645--0.753 downstream Fisher fidelity. It retains at
least 85% of the realized local oracle in every Layer-0/1 and FineWeb/TinyStories
cell. A wrong-target midpoint falls to 0.160--0.295 fidelity and 0.118--0.138
support overlap. Serial-versus-packed numerical discrepancies are below 0.01 at
the 95th percentile.

## Strongest negative result

The ordinary endpoint Jacobian is only an infinitesimal explanation. At a full
token replacement it has 0.45--0.55 product cosine, does not reliably beat
base-active channels under lexical-pair bootstraps, and does not close most of
the realized-oracle gap.

Likewise, midpoint compilation is not evidence that the five semantic relation
families have privileged circuits. Exact-distance synthetic edits remain highly
compilable at Layer 0, and no semantic fidelity advantage survives clustered
inference. Natural semantic edits do have cleaner native support, especially at
Layer 1, but that may reflect token-manifold geometry rather than relation-level
abstraction.

The adaptive-chart result is now more sharply bounded. At rank 256, exactly
distance-matched random sphere endpoints gain 0.057--0.077 from prospective
native-plus-seven chart routing, comparable to or larger than the
0.000--0.064 gains for natural edits. Semantic-minus-random gain intervals all
cross zero, and native-chart choice rates are similar. The chart atlas is
therefore a generic output-conditioned effect compressor, not evidence that the
model contains several semantic coordinate systems.

A stronger relation-transfer test is also negative. Directions from other word
pairs in the same declared family were transported to the evaluated source at
the exact natural chord length and compared with equal-sized different-family
donor sets. Same-family native-fidelity advantages are only +0.024, +0.018,
+0.023, and -0.007 across the four layer/corpus cells, with every clustered
interval crossing zero. The signs of individual families change across layer
and corpus. This rules out a simple shared linear sentiment/tense/etc. vector
as the explanation of early native alignment under this assay.

## Best present model

```text
embedding/current context
        -> generic attention transport
        -> target-conditioned low-dimensional state path
        -> dynamic native bilinear instructions
        -> broad residual write
```

“Feature” can correctly refer to an observable discriminator, an algebraic
instruction, a transport module, a local sensitivity direction, or a finite
edit path. Those definitions have different uses and different transfer tests.
The main interpretability error is to demand that all five coincide with one
human-labeled neuron.

## Repeated-embedding control

An exact four-world route factorial confirms that the Layer-0/1 result is not
mainly caused by the architecture's repeated token-embedding injection. At
Layer 1, the carried core supplies 0.85--0.86 of the local product-edit
projection and 0.905--0.924 downstream fidelity; later direct embedding
injection supplies 0.14--0.15, while the persistent Layer-0 value bus supplies
about 0.002. This changes at depth: by Layer 17 the repeated injection and value
bus are major contributors, so full-depth path simplicity cannot be interpreted
as unchanged transport of one early feature.

## Independent-checkpoint replication and basis boundary

The central result now replicates in an independently trained 12-layer tensor
transformer with 3,072 bilinear channels. Target-conditioned native selection
reaches 0.50 fidelity with 64 channels, versus 128--256 after a frozen random
rotation; 0.90 still needs 1,024--2,048 native channels. Frozen semantic-family
banks remain weak even after quadrupling discovery contexts and sweeping to
rank 1,024.

The red team narrows what “native privilege” means. Every one of seven fixed
random rotations is weaker in the median, and a rotation selected on other
lexical pairs still trails native with pair-bootstrap intervals above zero.
But an oracle allowed to choose a different best rotation after seeing each
held-out downstream effect can match or beat native. Native neurons are
therefore a privileged **fixed instruction set**, not a uniquely optimal sparse
code among all adaptively routed dictionaries. Description length and router
information are part of the comparison.

## Output-conditioned adaptive charts

A prospective tensor contraction now explains most of the best-of-seven
rotation phenomenon. Local residual-write fidelity cannot choose the useful
rotated chart, even when given the exact finite endpoint. Propagating each
midpoint-selected chart through the complete suffix Jacobian and the
softmax-Fisher metric agrees with the finite winning chart on 71--86% of
examples and closes 92--98% of its advantage over a fixed rotation.

When native is included as an eighth option, this router reaches median
fidelity 0.836--0.871 and stays within 0.012 of the finite chart oracle. It
chooses native on 45--47% of Layer-0 examples and 29--33% of Layer-1 examples.
The rotated alternatives are effectively dense in native products, so this
does not weaken native neurons as literal causal parts. It adds a second useful
description: an output-conditioned effect chart for compressing a declared
counterfactual.

The numerical red team preserves an important caveat. A finite 1% suffix step
is too large in some examples and smaller float32 steps suffer cancellation,
so the original finite-difference gate fails. Batched and single JVPs agree to
about `3e-6`, while forward/reverse AD duality agrees to `2.52e-8` after
Cauchy normalization. The tangent is correct; its finite radius is limited.

## Chart-count and rank frontier

The adaptive chart result is not a free compression. Sweeping 32--512
coordinates and every subset of one, two, or four charts from a seven-rotation
bank shows that seven charts generally cannot replace 256 native coordinates
with 128, and never with 64. Two rotations capture only 34--75% of the full
rank-256 gain.

Four rotations are a reasonable operating point: the remaining three improve
rank-256 fidelity by at most 0.02, and choice among four-chart subsets changes
median fidelity by at most 0.045. The atlas is therefore useful for improving
an output-effect approximation at a fixed rank, not for claiming that the
model's representation has half as many features.

For interpretation work, report coordinate payload, chart identity, routing
compute, and stored chart definitions separately. Comparing `K=256` native
with `K=256` from an eight-chart dictionary without charging the router answers
a real effect-compression question, but not a minimum-description-length or
native-mechanism question.

## Revised separation of representation and interpreter compression

The latest controls require one additional distinction:

| Object | Model-native? | Semantic evidence? | Best use |
|---|---|---|---|
| Bilinear product neuron | Yes, exactly | Sometimes as a discriminator; no fixed family bank | Literal ablation and algebraic execution tracing |
| Dynamic native coalition | Yes, for one state/edit | Target-specific, not family-stable | Reconstructing one realized computation |
| Suffix-routed rotated chart | No; dense in native coordinates | No advantage over exact-distance random edits | Compressing one requested downstream effect |
| Same-family embedding direction | Input-derived hypothesis | Fails prospective transfer here | Negative control for simple linear semantic codes |
| Attention transport skeleton | Yes, reusable weights/heads | Generic lexical transport, not family-specific | Explaining where information moves |

The best current guess is therefore not “features are neurons” or “features are
rotated latent directions.” Native neurons are fixed quadratic instructions;
context selects a broad coalition; and an external interpreter can often
compress the resulting output effect in a routed chart that the model itself
does not sparsely execute. Human semantic labels have not yet identified the
selector or a transferable early relation code.

## Supervised feature census: what is neuron-aligned?

A declared 50-feature test now replaces the earlier handful of examples.
Across two disjoint 16,384-site FineWeb splits, the direct answer is: **the
tested familiar features are not neuron-aligned in Layers 0 or 1**. Among 32
adequately powered and behaviorally competent labels, Layer-1 median AUROC is
0.599 for one neuron, 0.767 for 64, and 0.788 for 256. Zero labels have a single
neuron within 0.03 of the best 64-neuron result.

The first result has a lexical confound. The current token predicts the next
token class, and this architecture reinjects its embedding at every layer. A
current-token unigram reaches 0.731; a separately fitted isolated-token product
code reaches 0.746. Context raises Layer 1 to 0.767 in the global metric. The
global census therefore mostly establishes a distributed token/morphology code.

The decisive control compares only positive-negative examples with the exact
same current token. Any deterministic embedding-only score is then constant
and has AUROC 0.5. On 46 labels with at least 500 such pairs:

| Object | Median conditional AUROC |
|---|---:|
| Full model output | 0.747 |
| Layer 0, one / 64 / 256 products | 0.518 / 0.561 / 0.586 |
| Layer 1, one / 64 / 256 products | 0.520 / 0.602 / 0.608 |

Contextual information is genuinely present by Layer 1, spread across a
coalition, and incomplete relative to the whole model.

## From readable to causal coalitions

Frozen 64-channel semantic decoders have a small finite causal effect. Across
96 held-out donor-recipient pairs sharing the current token, they move the
declared output class in the donor direction 63.5% of the time, versus 49.0%
for random supports. The mean shift is only 0.0114 and correlation with the
full Layer-1 product patch is 0.218. Across eight feature families, the
cluster-bootstrap interval for directional advantage still crosses zero.

Fixed output-aware refinements fail. Down×unembedding weights reach 51.0%
directional accuracy, and feature-average suffix gradients reach 57.3%. The
same neuron does not have one context-independent amount of semantic leverage.

The recipient-conditioned test succeeds sharply. For each pair, compute the
exact derivative of the final semantic score with respect to all Layer-1
products, multiply it by the donor activation delta, select the 64 largest
positive contributions, and perform the finite patch. This works on 95/96
pairs; predicted and finite shifts correlate 0.922. The support changes almost
completely across contexts: same-feature median Jaccard is 0.0079 and no
channel occurs in half of a feature's twelve contexts.

The operational ontology is therefore:

```text
stable native neuron       = fixed bilinear instruction and write column
stable weight scaffold     = recurring direct output-role preferences
distributed evidence       = feature information readable from many products
dynamic causal coalition   = channels with leverage for this state and objective
full natural computation   = dynamic coalition plus opposing/cancelling paths
```

This is a concrete kind of multiplexing. Neuron identity is stable, but which
neurons jointly implement a semantic discrimination depends on the input and
requested output contrast. The verb is better attached to the **local
coalition and effect**, not to each member neuron.

The dependency is genuinely joint. A 2×2 factorial replaces the activation
delta and suffix gradient with disjoint-discovery feature means:

| Activation delta | Suffix gradient | Directional finite patches |
|---|---|---:|
| feature mean | feature mean | 0.594 |
| pair-specific | feature mean | 0.573 |
| feature mean | pair-specific | 0.729 |
| pair-specific | pair-specific | 0.990 |

Thus the suffix state is the more informative single input, but neither is
sufficient. The feature implementation is the pairing `delta_h * d(output)/dh`.

That gradient field is not compact in one small frozen basis. A 256-gradient
discovery span stratified across all 32 positions captures 0.633 median held-out
gradient cosine and 66.2% of exact top-64 first-order leverage. This decisively
beats a random rank-256 span (0.235 cosine) but misses every locked compactness
threshold. The short explanation should therefore be a state-conditioned tensor
program for computing sensitivities, not a 16- or 32-dimensional semantic axis
bank.

## Prospective path compilation

The local coalition need not be an endpoint-only post-hoc story. Adaptive
interior Gauss integration of exact tensor derivatives predicts finite
token-edit product changes. At Layer 1, a 4→8→16→32→64 stopping rule catches
all seven sixteen-node failures, uses 35 nodes on average, and recovers 97.9%
of effect-weighted output KL over 64 FineWeb edits. Median recovery is
effectively exact.

This supplies an internal error signal and a prospective compiler for a known
edit. The unsolved step is semantic discovery: given a natural behavior rather
than a specified token replacement and output contrast, identify which
counterfactual and objective define the useful local chart.

## New result: the selector is an executable operator program

The exact downstream selector has now been decomposed rather than treated as a
black-box gradient. Static compression remains weak: a rank-256 frozen gradient
span recovers only 0.633 median held-out cosine. In contrast, exact backward
operators grouped by MLP and attention interaction counts give 0.952 cosine and
92.6% of the finite causal shift with an 8×8 truncation; a 12×12 truncation is
effectively exact.

The mechanism is concentrated near the source. Attention responses in Blocks
2--9 recover 0.969 median cosine and 94.2% of exact finite shift, versus 0.753
and 78.3% for Blocks 10--17. No single block is necessary or sufficient.

For same-position objectives, both QK factors together recover 0.926 cosine;
the current-value branch alone reaches 0.740. But this is not a universal
“feature in queries” result. When a token is edited 8, 32, or 64 positions
before the output, query-only and MLP-only paths to that earlier position are
structurally zero. Key+value transport reaches 0.812/0.760/0.759, and adding the
full early Q/K/V roles raises this to 0.927/0.944/0.924.

Finally, role canonicalization does not collapse the mechanism to a few heads.
A frozen ordering over all 144 suffix layer-head edges needs 64 edges to exceed
0.90 median cosine at every tested distance. Seventy-two reach about 0.96 and
strongly beat random sets, but 32 remain at 0.63--0.71. The current best object
is therefore a structured moderately broad routing program that selects a small
state-specific native-neuron payload.

This head scaffold is independently identifiable. Two disjoint discovery panels
produce top-64 Jaccard 0.730 and complete-rank correlation 0.879. On a third
panel, their 54 shared edges alone retain 0.936/0.927/0.900 cosine at distances
8/32/64, compared with random-64 p95 values of 0.668/0.599/0.557. The routing
program is therefore reusable rather than an arbitrary half-graph, although it
is still much broader than a classic small circuit.

## Extension through Layer 3

The same exact analysis has now been run at the Layer-0, Layer-1, Layer-2, and
Layer-3 MLP product boundaries on all 96 held-out semantic pairs. It does not
show a transition to a handful of neuron-aligned concepts. Median positive-mass
K90 remains approximately constant at 1,038, 1,042, 1,053, and 1,049 native
channels out of 4,608.

What does simplify is the downstream program. The next-eight-block native
operator composition rises from 0.953 median gradient cosine at Layer 0 to
0.986 at Layer 3. For the concrete numeric example where a comma after
`$1,000` is contrasted with a comma before `doubts`, Layer 3 still needs 1,045
helpful channels for 90% of positive first-order mass; the best 1/16/64/256/
1,024 channel finite shifts are +0.015/+0.175/+0.427/+0.879/+1.281. Yet the
compact Q/K and next-eight-block programs already match its exact downstream
sensitivity at cosines 0.974 and 0.975.

This example also illustrates cancellation. The full donor product patch shifts
the semantic score by only +0.389, much less than the positive-only 1,024-channel
patch, because the natural product delta contains channels that oppose the
requested output contrast. A large positive K90 is therefore evidence about a
distributed helpful coalition, not a claim that all those channels are jointly
necessary for the model's unedited behavior.

The cross-position red team supplies the complementary causal grammar. At an
earlier token, query-only and MLP-only effects at that position are exactly zero;
keys and values must transport sensitivity. Key+value alignment improves across
the four source layers, while Layer 0 needs more than the nearest eight blocks
for 0.90 fidelity. “Feature” is consequently best represented here as an
input-conditioned coalition plus a composable role program, not as a scalar
stored at one neuron or one attention factor.

## Prospective atlas failure and the MLP-Jacobian backbone

A moderate stored atlas does not replace that role program. A 64-chart-per-
feature bank and a corrected 256-state full semantic-Jacobian bank were routed
using token identity, source and final states, complete current-position forward
trajectories, and native eight-block attention patterns. The best locked
prospective results remain only 0.03--0.14 gradient cosine. The 256-chart oracle
span reaches 0.63--0.65, so the bank contains useful geometry, but the forward
signatures do not supply its coefficients.

Directly swapping local operator states gives the cleaner ontology. A nearest
same-position context reproduces essentially none of the suffix gradient.
Keeping exact target-state MLP responses while omitting attention retains
0.66--0.78 cosine across source Layers 0--3. Keeping only exact attention
responses retains 0.00--0.08. Wrong-context attention reduces the exact-MLP
program to 0.47--0.67, demonstrating modulation and destructive interference
rather than irrelevance.

Thus the most useful current “feature” object has three levels:

```text
native product channels        distributed state-specific payload
bilinear MLP Jacobians         suffix-wide computational backbone
near-source attention roles    contextual routing/modulation of that backbone
```

This is more structured than arbitrary superposition, but less localized than
a neuron dictionary. Because the MLP Jacobian is an explicit contraction of
the bilinear weights with the current state, it is still directly available to
weight-based reverse engineering. The unresolved question is whether its
spectral outliers form reusable clean circuits while its bulk supplies contextual
corrections, or whether high-rank bulk is required throughout.

## Dynamic Jacobian answer and extension through Layer 3

The dynamic MLP operator is moderately high rank through most of the network,
not a small set of universal singular modes. Blocks 1--12 have effective ranks
of roughly 444--562 out of 1,152. An output-conditioned best-mode selection
still needs 230--273 modes for 90% local response energy. Block 17 is the clear
exception: effective rank 69 and task K90 36.

When these truncated operators are composed through the exact suffix, task-
selected rank 64 reaches 0.674/0.732/0.767/0.835 median gradient cosine from
source Layers 0/1/2/3. Rank 256 reaches 0.919/0.935/0.944/0.960. Thus the third
and fourth analyzed source layers are easier to compress, but still not a tiny
neuron-aligned code.

The more promising ontology appears in the resultant directions. On 32 held-
out examples of eight known token properties, exact local MLP responses become
strongly feature-aligned late (same/different similarity ratios 2.28 at Block
12 and 2.85 at Block 17), even though the task-selected singular-mode spans are
nearly feature-agnostic (1.01 and 1.05). The same feature can be reached through
different local coalitions. That is a falsifiable and weight-computable form of
compositional representation, not a claim that every direction is meaningful.

### Important correction from 32 competent features

The late direction result does not by itself establish a semantic dictionary.
On four held-out contexts for each of 32 predeclared features that the model
actually predicts, semantic response directions become 60.2% identifiable by
Block 17 (3.1% chance). But a shared vocabulary-permutation null preserving the
entire feature-overlap Gram matrix reaches 86.7%. Semantic recurrence is also
weaker than this matched null at Blocks 12 and 17.

The clean late direction therefore primarily identifies the requested output
objective. It may still be useful operationally, but it is not sufficient
evidence for the model's native ontology. The stronger semantic evidence remains
the held-out forward census plus causal cross-context transfer; future ontology
claims should require both and retain a Gram-matched objective null.

### Causal transfer resolves the remaining ambiguity

The 32 competent features were also tested with discovery-frozen channel lists
on 64 disjoint positive-donor/negative-recipient pairs. Activation-only,
mean-gradient-aware, scaled, and mean-paired-contribution selectors all fail to
transfer reliably. In contrast, the recipient-specific top-64
`activation change × exact suffix gradient` coalition moves the desired score
positively on every pair at every source layer.

Those successful coalitions share only about two channels across the two
contexts of a feature (Jaccard 0.014--0.018). This is above the different-
feature and random baselines, so there is a weak feature scaffold, but roughly
97% of membership is context-dependent. “Feature” is best treated as a rule for
selecting a causal coalition, not a persistent set of neurons.

### The selector itself is compressible

The context-specific endpoint selector can be approximated prospectively from
the weights and recipient states. Exact MLP Jacobians plus only the next eight
attention Jacobians recover 0.951/0.966/0.978/0.988 gradient cosine from source
Layers 0/1/2/3, select top-64 supports with 0.652--0.816 Jaccard, and preserve
93.3--98.1% of the exact finite effect. Strict minimum tested windows are
8/8/6/5 blocks.

Inside this window, QK routing alone preserves 95.2--98.4% of the full effect;
query roles dominate keys, and both tensor QK factors contribute. Fixed native
head rankings are weaker: half the edges works approximately, but random halves
are close and discovery rankings vary. The robust coordinates are the QK/query
verb and relative downstream depth, especially offsets one and two—not a fixed
list of “feature heads.”

### Multiple positions and contextual interactions

The same recurrence now selects one top-64 coalition across two earlier token
positions and source Layers 0--3. With balanced single-edit strengths, a
next-twelve attention window reaches 0.967--0.997 exact-gradient cosine and the
correct finite direction in every tested cell. The model does not always divide
the budget evenly: two position pairs are winner-take-most, while 95+119 requires
both sources.

More importantly, the earlier token's context-dependent interaction with the
later token is recovered by subtracting the two suffix programs. This
differential program reproduces exact interaction gradients and finite Möbius
effects without creating an explicit high-order polynomial or positing a
dedicated conjunction neuron. A multi-token feature can therefore be a stable
relation between state-conditioned operators rather than a persistent channel.

### Objective discovery correction

A label-free varimax rotation of the late rank-16 output core is almost perfectly
reproducible and far sparser than native or random axes, but is not more aligned
with human token predicates than matched random rotations. It is a legitimate
coordinate system for sparse output control, not evidence that individual axes
are privileged semantic concepts. For semantics, the subspace plus a supervised
readout remains better supported than any native coordinate basis.

## Extension beyond Layer 1

The supervised census now covers Layers 0--4 on the same 16,384 discovery and
16,384 held-out FineWeb sites. Median held-out AUROC for a fixed 64-channel
coalition is 0.747/0.767/0.767/0.777/0.779. A single channel remains at
0.637/0.599/0.602/0.625/0.607, and none of the 32 competent labels is within
0.03 of coalition performance using one neuron.

Because this architecture re-adds the token embedding in every block, the key
red team ranks sites only against other sites with exactly the same current
token. The 64-channel within-token AUROC is still
0.558/0.581/0.591/0.600/0.622. The third, fourth, and fifth analyzed layers are
therefore adding context-sensitive feature information, not merely preserving
token identity.

The compositional causal result extends in parallel through source Layer 5.
Exact weight-derived local MLP and QKOV derivatives are applied sequentially,
so there is no need to expand the model into a seventh- or higher-order global
polynomial. A next-eight-attention program plus the exact MLP backbone reaches
0.959/0.980/0.985/0.988 median gradient cosine from Layers 2/3/4/5. Despite
this increasingly short downstream description, native positive-contribution
K90 stays near 1,040 channels at every boundary. Greater depth simplifies the
causal *program* without turning it into a small list of monosemantic neurons.
