# Latest result: a sparse interaction grammar, not monosemantic axes

The final Layer-11 bilinear MLP now has a concrete, causally tested description
for its shared token-edit backbone.

## Short version

The model's 768-dimensional state naturally occupies a much smaller useful
subspace for this behavior. In a 32-dimensional ordinary-language state basis,
the final MLP's contribution to a pooled 32-dimensional write backbone is an
exact family of quadratic forms. Those forms cannot be rotated into 32
independent scalar features without losing much of the behavior. But they can
be reduced to all diagonal terms plus roughly 32 stable pairwise interactions.

```text
768D native state
   → 32D natural state manifold
   → diagonal terms + about 32 pair interactions
   → shared 32D write backbone
   → output distribution
```

So the current answer to “are the neurons monosemantic?” is: **that is not the
right primary object here**. Boundary information is distributed through the
32D state space; random rotations decode the simple token categories at least
as well as the canonical axes. The privileged object is the sparse interaction
graph defined by the restricted bilinear tensor.

## Evidence

- Exact weight-derived quadratic coefficients match native coefficients to
  relative error `8.2e-7`.
- A natural-state rank-32 program reaches causal fidelity
  `0.812 / 0.746 / 0.808` on FineWeb / WikiText / TinyStories.
- Diagonal-only execution reaches only `0.537 / 0.350 / 0.574`.
- Adding 32 selected coordinate pairs restores
  `0.809 / 0.720 / 0.811`.
- Disjoint FineWeb panels select almost the same graph: top-32 Jaccard `0.939`.
- One pooled graph works for newline, period, comma, and ` the`; target-specific
  graph overlaps are `0.64–0.83`.
- On 96,000 natural FineWeb sites, selected edge products have median held-out
  category AUROC `0.939`, versus `0.745` for random edges. Frozen semantics
  transfer with median `0.931` on WikiText and `0.960` on TinyStories.
- On 9,600 natural sites per corpus, replacing the exact pooled write component
  with the selected graph recovers `0.940 / 0.920 / 0.938` of its KL effect.
  The full restricted tensor recovers `0.953 / 0.939 / 0.961`; diagonal-only and
  random-edge programs recover `0.59–0.72`.
- Without running the finite replacement, the local final-output Jacobian ranks
  its sitewise KL error with Spearman `0.979–0.985` and predicts top-token flips
  with AUROC `0.999–1.000`. The theoretical Fisher scale is accurate to within
  about 2% median. The errors are broad rather than outlier-only: the largest
  predicted 10% contain only `33–36%` of total KL.

## What this means

This is a tractable form of multiplexing. Individual coordinates are reused and
not uniquely semantic. Their pairwise products implement a small set of
discriminations or “verbs,” mostly involving boundary and punctuation structure.
An endpoint coordinate does not need one English meaning for the interaction to
be a clean computational object.

The result also clarifies why the earlier linear probe failed on WikiText. The
needed information was present, but the readout hypothesis was wrong: the native
module reads state quadratically. Using the actual tensor algebra repairs the
OOD failure.

## What is still missing

- The 32D pooled write backbone is not the complete final layer. Common edits
  such as comma and ` the` require broad output tails for high fidelity.
- The graph is established for one checkpoint. Individual axes/edges must not be
  called canonical across training runs until functional cross-checkpoint
  alignment succeeds.
- We know the final operator and its state contraction, but not yet the shortest
  upstream attention/MLP program that constructs the 32 state coordinates.
- Aggregate write reconstruction is not yet a blind prediction of which
  individual prompts will fail under edge deletion.

## Next experiment

Trace the high-degree interaction hubs backward through Layer-11 attention and
Layer-10 MLP using exact weight-derived QK/value and bilinear contractions.
Freeze a small upstream source program on FineWeb, then predict hub and edge
activations—and their finite distributional errors—on WikiText and TinyStories.
If it succeeds, the sparse grammar extends upstream into a multi-layer circuit.
If it fails while the final graph remains valid, the graph is an interface with
context-specific upstream routing rather than a complete circuit.

## Upstream verdict

The latter outcome is closer to what happened. The final graph is a valid sparse
interface, but not a literal sparse graph carried unchanged from middle depth.
Layer-6 perturbations are transported by a context-specific 32-by-32 Jacobian;
the fixed final coordinates only become almost identity carriers near Layer 10.

The encouraging part is that this context specificity is not opaque. With
attention response frozen, the exact chain of RMSNorm and bilinear-MLP
derivatives computed from weights and natural activations predicts finite
transport above 0.999999. It also predicts about 0.98 of the complete native
response because downstream attention response is small for this route.

The endpoint rank-32 space is not the whole internal path. Projecting back into
it after every MLP retains about 0.88–0.90, so transient off-subspace directions
are materially useful. Across contexts, the local Jacobians have a leading
spectral outlier 2.53 times a shuffled null but effective rank about 155. The
best present description is therefore **structured local charts feeding a
sparse late grammar**, not one fixed sparse circuit throughout the network.

The full chart can now include attention. Explicit QK-pattern and value/OV
derivatives raise one-sided finite native fidelity from roughly 0.97–0.98 for
the MLP-only chain to about 0.996. The final sparse grammar is therefore not
merely correlated with upstream state: its response can be forecast from the
actual tensor weights through the complete local QK/value/MLP path.

The spectral modes of these charts have a second important property. Their
leading span is causally useful and rotation invariant, while their semantic
axes are not. Random rotations, ICA, and varimax show that many coordinate
systems can expose different clean token predicates without changing causal
sufficiency. The graph and operator subspaces are stronger ontological objects
than the English label attached to any one axis.
