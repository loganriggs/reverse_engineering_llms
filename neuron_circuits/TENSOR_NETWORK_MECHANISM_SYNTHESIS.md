# Tensor-network mechanism synthesis

## Current best hypothesis

The evidence no longer fits a one-dimensional spectrum from monosemantic
neurons to arbitrary memorization. A better decomposition is:

```text
evidence carriers
    ↓
feature construction / address code
    ↓
state-conditioned operator grammar
    ↓
compact behavioral interface
```

Each boundary can have a different kind of interpretability. Evidence carriers
may be simple token-value vectors. Feature construction may be a broad
superposition of cross-token products. The operator grammar can use a small
vocabulary of recurring bilinear operations even though its matrices and
coefficients are context-specific. The late behavioral interface can again be
low-dimensional and semantic.

The latest supervised-anchor comparison sharpens the distinction. Thirty-two
competence-gated human labels yield reproducible head-level causal geometry
when estimated from 50 FineWeb contexts each, but they do not cleanly recover
the label-blind weight motifs. Motif members share unsigned causal salience;
a Gram-matched nonsense ontology reproduces that effect. Their signed semantic
profiles show only a nonsignificant trend. Thus “used on similar examples” is
not equivalent to “implements the same semantic feature.”

## What is established in the controlled grammar model

### Pure rules

- Eight causal gate directions are sufficient at every layer.
- An answer-semantic subspace becomes sufficient by Layer 1.
- Removing value routes from digits irrelevant to the current rule has
  essentially no effect.

### Rules plus arbitrary facts

- Layer 0 attention broadcasts identity-bearing digit values.
- The first bilinear MLP forms relation×digit and digit×digit products.
- Products between different source tokens are causally essential.
- Digit-specific Layer-0 keys are almost dispensable; digit values are not.
- Layers 0--2 use a broad 32--64-dimensional gate program, including on
  unrandomized rule facts.
- The finite Layer-0 identity correction then propagates through a recurring
  downstream vocabulary dominated by the two bilinear product-rule arms.
  Thirty-two path types retain 0.82--0.89 of the correction and 128 retain
  0.98--0.99.
- The last layer collapses into a seven-dimensional answer-control quotient.

The key distinction is between a **broad operand/code** and a **small operator
vocabulary**. A compact list of operation types does not imply that the state,
matrix contractions, or stored facts are compact in the same basis.

## Connection to the FineWeb tensor models

The natural-language results are consistent with the same stage structure.
They are not yet a direct replication because their intervention families and
architectures differ.

In the 18-layer model's semantic-attention atlas:

- Layer 0 attention changes are value-dominated: value receives 0.808 signed
  projection on FineWeb and 0.793 on TinyStories.
- The changed source token supplies about 0.88 of the Layer-0 attention change.
- By Layer 6, the two QK terms together receive about 0.61--0.64 while value
  falls to about 0.28--0.29.

In the independent 12-layer model's newline audit:

- Layer 1 is current-value dominated: roughly 0.56--0.60 locally across three
  corpora.
- From Layer 2 onward, final-query routing is usually dominant, rising above
  0.9 in the late layers.
- Exact source-edge studies find conditional sparsity but no small global
  source-position skeleton.

Together these suggest a tentative natural-language pipeline:

```text
early value transport
    ↓
bilinear feature construction
    ↓
context-dependent query routing through a broad operand code
    ↓
compact late write/output modules for some behaviors
```

This is a hypothesis about computation stages, not evidence that FineWeb facts
are stored exactly like the synthetic random table.

## Which ontology is correct for which use case?

| Question | Best current object | Inadequate shortcut |
|---|---|---|
| Where did token information enter? | QK/value source decomposition | Top activating neuron |
| Which interactions construct an address? | Exact bilinear source-pair terms | Attention weights alone |
| How many directions preserve a finite effect? | Discovery-frozen causal subspace frontier | Activation variance alone |
| What operation types carry a correction? | Integrated tensor-path grammar | One static edge list |
| What controls the output decision? | Late residual-write or answer quotient | Exact upstream state reconstruction |
| Are facts mutually load-bearing in parameters? | Margin-gradient/finite-edit constraint graph | Activation co-occurrence |
| Is a mechanism reusable? | Frozen OOD finite-effect prediction | Same semantic label or probe score |

There is no contradiction in saying that a model has simple evidence vectors,
a distributed address, reusable bilinear verbs, and a compact semantic output
interface. These descriptions refer to different boundaries.

## Important red-team lessons

1. The original codeword-swap result contained a four-of-eight-relations bug;
   complete coverage materially lowered exact-effect fidelity.
2. Eight-point path integration looked exact in the median but missed a rare
   large intervention by 10.4%. Thirty-two points reduced the worst error below
   `8.4e-6` without changing the path conclusions.
3. Pair count alone was insensitive: top 32 terms could be near ceiling in both
   regimes even though semantic digit masks sharply separated them.
4. Random controls can be too easy in an already compact regime. A failed
   pooled random-bank threshold did not negate the stronger semantic
   intervention.
5. A recurring path label such as `I→ML→MR` names an operation type, not a fixed
   matrix. The matrix is still state- and model-dependent.
6. Correct answer, exact eight-logit effect, exact residual write, and exact
   internal state are different fidelity targets.

## Next decisive cross-scale experiment

Freeze a small suffix path-type bank using one family of natural edits in the
500M model, without fitting intervention outcomes on the evaluation set. Test
it on other edit families and on WikiText/TinyStories, comparing:

1. the frozen path-type bank with native state-conditioned matrices;
2. the same number of random path types;
3. a per-example oracle path bank;
4. the complete analytic tensor path;
5. a native-neuron bank with matched downstream compute.

If the frozen operator grammar transfers while coefficients and source routes
remain local, the controlled ontology scales: reusable verbs act on multiplexed
operands. If only the per-example oracle works, the apparent grammar is too
coarse. If native neurons win at matched fidelity and description length, the
subspace/path description is unnecessary in that regime.

The strongest version should be blind: freeze path types, thresholds, and
coefficient calculations before revealing finite intervention effects.

## Cross-scale result: compactness survives, but the ontology changes

The proposed MLP-suffix bridge is now complete in both a 190.5M and the target
545.9M FineWeb tensor model. It produced a positive result and then survived
three red teams, but not under the strongest initial interpretation.

With attention responses frozen, four consecutive bilinear MLP derivatives
factor as a product of residual and bilinear updates. Treating `Left` and
`Right` derivative arms separately gives 81 paths, but that distinction is not
canonical: exchanging the two factors leaves a bilinear neuron unchanged. The
symmetry-invariant quotient has only 16 paths, one for every choice of residual
`I` or full bilinear derivative `B` at four layers.

In the 190M model, four frozen groups retain 0.878--0.904 and eight retain
0.949--0.961. In the 546M high-count replication, four retain 0.958--0.963 and
eight retain 0.989--0.993 over 256 sites per corpus. Per-example oracles add
less than 0.01, the rankings transfer to WikiText and TinyStories, and exact
finite interventions agree with the tangent programs. A separate endpoint
red team shows the 546M result is stable from rank 32 to 64.

The main red-team correction is that **interaction order explains more than
path identity**. The residual plus all four one-layer bilinear paths is already
sufficient. Degree-two interactions are material, but the selected layer pairs
do not beat matched random pairs. Random endpoint subspaces are also fairly
compact, although natural PCA subspaces are consistently more compact and use
a different top ordering.

The early-layer replication adds a complementary basis result. Native product
coordinates beat any one frozen random rotation for target-conditioned causal
patches, but a suffix-Fisher tensor contraction can prospectively select among
seven rotated charts and nearly match the finite per-example chart oracle. The
rotated patches are dense in native coordinates. Thus native neurons remain the
right literal parts, while output-conditioned adaptive charts can be the shorter
description of one effect. Basis privilege is relative to whether the requested
object is a fixed intervention vocabulary or a routed effect approximation.

The best revised account is:

```text
natural activity selects a particularly contractive operand slice
        ↓
residual + low-order state-conditioned bilinear updates
        ↓
compact late effect interface
```

This is weaker and more defensible than “a small semantic verb vocabulary.”
The executable path types are canonical only after quotienting exact weight
symmetries, and much of their simplicity comes from a convergent residual
expansion. Increasing a counterfactual bilinear gain continuously broadens the
required program, providing a concrete mechanism for movement toward
superposition.

## Revised next decisive experiment

The present result freezes every suffix attention response. The attention-
inclusive manual Jacobian already shows that QK and value routes close the
remaining native gap, but it has not been compressed into the same invariant
path language. The next test should:

1. expand each late block into residual, full bilinear, QK-routing, and value/OV
   terms while quotienting factor and head symmetries;
2. freeze interaction-order and layer-family truncations on FineWeb;
3. evaluate finite native effects on larger FineWeb, WikiText, and TinyStories
   panels;
4. compare equal execution budgets against native-channel banks and spectral
   Jacobian modes;
5. report separate fidelity curves for endpoint coordinates, final logits,
   cross-entropy, and top-1 behavior.

If low interaction order remains sufficient only on natural charts, the useful
ontology is a data-conditioned convergent tensor expansion. If attention makes
the expansion broad or site-specific, the MLP result is a local subsystem fact
rather than an end-to-end model ontology.

### First attention-boundary result

The first boundary test keeps the frozen four-path MLP program but recomputes
all suffix attention. The complete MLP-only route still explains 0.961--0.973
of the native finite effect, while the four-path approximation explains
0.910--0.933. Adding the exact finite attention correction reaches
0.961--0.968.

Attention is not negligible: its norm is 0.165--0.197 of the native effect even
though its signed projection is only 0.045--0.080. It behaves more like a
redirection/cancellation operator than an aligned extra feature. The remaining
four-path MLP error is about the same size as the entire attention correction,
so neither side alone is the residual problem.

This preserves the layered ontology:

```text
compact low-order MLP backbone
        +
material, mostly transverse attention correction
        →
native late causal effect
```

The correction is currently an oracle finite difference. It must be replaced
by an executable two-QK/value path program before it counts as an explanation.

### Executable attention result

That replacement is now implemented. A manual full-sequence JVP for the
model's two-QK attention matches autograd to `2.74e-7`, and all 32 late-layer
`I/A` paths recompose it to `9.33e-7`. Four frozen paths retain 0.997--0.998
across FineWeb, WikiText, and TinyStories. Factoring each singleton response
shows that the QK-pattern term alone retains 0.998--0.999, whereas the current-
value term is close to the identity baseline. The two QK branches are balanced
and exchangeable; the stable object is QK redirection, not a privileged branch.

This is an interaction-order result, not a named-circuit result. Both the
layer-path and QK/value four-term selections narrowly fail their locked
advantage over matched random selections. Many small banks work because the
whole expansion is strongly contractive.

### Description sparsity is not execution sparsity

The same local bilinear operators were decomposed into all 4,608 native CP
channels at Layers 13--16. In Layers 13--15, the best discovery-static 512
channels retain only 0.220--0.396 of the exact local derivative; even 1,024
retain only 0.355--0.502. Layer 16 is more concentrated (0.840 and 0.870), but
still needs 2,048 tested channels to cross 90%. Layers 13--15 need 4,096.

So “four paths” does not mean “four neurons.” A path label invokes a dense,
state-conditioned bilinear matrix assembled from many native channels. The two
ontologies have different uses:

- path/operator types give a short algebraic program and expose interaction
  order;
- native channels localize implementation in the architecture and matter for
  sparse edits or compute removal;
- endpoint subspaces summarize behavior without identifying execution;
- QK/value factors separate routing from payload.

The next test asks whether the dense channel collection has a compact spectral
or state-predictable operator representation. Oracle low rank alone would show
coordinate compression; prediction from the local state is required for an
executable intermediate representation.

That test finds a depth transition rather than a universal compression. In
Layers 13--15, rank-64 spectral atlases retain only 0.27--0.48 and rank 256
only 0.51--0.66. Layer 16 is sharply different: one direction carries 73% of
discovery energy, rank 64 retains 0.86--0.89, and the rank-90 is 56 rather than
224--257. State-predicted coefficients nearly match hindsight projection and
transfer OOD at every layer.

This does not yet make the modes semantic. Bilinear local derivatives are
linear functions of the current state, so predicting fixed spectral
coordinates is structurally expected. The next canonicalization asks whether
each operator mode is itself low rank between input perturbation and output,
and whether its coefficient has an exact compact state read vector. That would
turn a dense mode into a testable read–transform–write circuit; failure would
leave it as behavioral compression only.

The centered/factorized red team resolves the Layer-16 spike. Its leading mode
is exactly aligned with the frozen mean operator, which alone retains
0.72--0.76 of local derivative energy. That common matrix has inner rank-90 of
three. But the centered variation is broad: its top mode carries only 0.133
and its rank-90 is 206. An exact weight-derived 64-mode, inner-rank-8
read–transform–write program reaches 0.81--0.83; inner rank four narrowly
misses the locked 0.80 gate.

This suggests a useful new ontology: **simple background operation plus broad
contextual correction**. It avoids calling the average circuit a full
explanation. Whether the correction matters depends on the downstream
readout, so the next assay composes both pieces with the exact Layer-17
normalization and output chart rather than scoring local Frobenius energy.

On that causal boundary, the broad correction collapses. The exact composed
derivative matches autograd to `4.69e-7`; centered rank-90 drops from 206
locally to 26 at the output-facing state. A rank-16 endpoint atlas retains
about 0.98, and the executable 64-mode/inner-rank-8 local program also retains
about 0.98 across all corpora. The mean circuit alone reaches 0.87--0.91, while
contextual modes add a real 0.08--0.12.

This is the clearest example so far of why local monosemanticity, execution
sparsity, and causal understanding should not be conflated. Thousands of
native channels and hundreds of local variation modes can implement a compact
downstream control program. The immediate red team is to restore the dynamic
Layer-17 two-QK response; after that, output-chart and logit/CE tests determine
how much of the compression is specific to this behavioral quotient.

Dynamic Layer-17 attention is material—the frozen/dynamic derivative fidelity
is only 0.93--0.95—but it does not destroy the compact program. An independently
validated full two-QK trace leaves the direct endpoint rank-90 at three; the
executable inner-rank-8 local program remains at 0.978--0.982. Thus routing
changes the numerical map while preserving the low-dimensional causal
interface.

The next endpoint should be chosen by use case. Full logit tangents test model
behavior broadly; cross-entropy gradients test the actual next-token objective;
top-1 changes test decisions but discard calibrated probability information.
These need not agree, and disagreement is itself evidence that “the correct
representation” is quotient-dependent.

They do disagree. The same inner-rank-8 program that retains about 0.98 in the
pre-final-MLP state chart retains only 0.87--0.89 in full logits, 0.77--0.78 in
local Fisher geometry, and 0.76--0.86 in true-token loss derivatives. Even the
full 64-mode program reaches only 0.83--0.92 on loss. The complete chain is
autograd-validated, so the gap is an amplification effect in the final MLP and
readout rather than a missing derivative term.

This adds an important practical rule: select the quotient before selecting
the circuit basis. A Frobenius-optimal local operator atlas is useful for
mechanistic geometry but is not automatically a good loss-preserving
compression. The next comparison is a larger replication followed by a basis
learned under logit/Fisher/CE weighting, with the local basis retained as a
held-out baseline.

The larger replication and tail sweep show that this is not repaired by a
small loss-aware reorder. CE-aware greedy selection improves discovery but
generalizes slightly worse than the local-energy prefix. Although the local
operator energy rank-90 is only 69, held-out loss needs 512--768 modes to cross
90%; 1,024 modes reach 0.985--0.991. The final readout is using a broad
low-energy tail.

That yields a concrete reverse-engineering warning: **variance is not
importance**. Spectral outliers can be clean, real circuits and still omit many
small directions that matter after a high-gain downstream map. A useful atlas
should therefore pair every representation coordinate with an adjoint or
behavioral sensitivity, rather than interpreting the forward spectrum alone.

The same loss effect is broad in native neurons. Marginal and coordinated
greedy rankings both require 4,096 of 4,608 Layer-16 channels to cross 90% on
held-out corpora. Greedy reaches 0.93 with 1,024 channels on discovery but only
0.80--0.81 held out, exposing strong subset-selection overfit. A small stable
native bank is therefore not supported.

This remains path- and direction-conditional. The input directions were the
Layer-16 natural PCA chart. The next falsifier replaces them with tangents
actually delivered from the Layer-12 boundary through dynamic Layers 13--16.
If those are sparse in native channels, the model can execute sparse programs
inside a broad counterfactual operator; if they remain broad, the dense
implementation claim becomes substantially stronger.

They remain broad. Exact Layer-12 directions propagated through dynamic
Layers 13--16 still require 4,096 native Layer-16 channels for 90% held-out
loss-path fidelity. An eightfold larger selection panel does not change that
result. At 2,048 channels the coordinated bank reaches only 0.66--0.77.

This strengthens the “dense implementation of a short program” account. It
also adds distribution dependence: the same FineWeb bank is substantially
worse on WikiText than on TinyStories at small budgets. The next distinction
is between **per-context density** and **union density**. A broad fixed bank may
mean every context uses many channels, or that each uses a different sparse
coalition. Contribution participation, cancellation, and per-context subset
tests can distinguish those without pretending that a reweighted 32D span is
native execution sparsity.

The per-context audit resolves that distinction. Individual examples are not
native-sparse: norm-effective support is roughly 1,500--1,900 channels and
target-projection support roughly 600--900. Absolute signed-projection mass is
about five, revealing large cooperating and cancelling contributions. The
broad fixed bank is therefore not merely the union of tiny native coalitions.

Yet a freely reweighted OMP description uses only five channels for 90% and
nine for 99%. This is the CPU/intermediate-representation lesson in concrete
form: a short observer-side description can summarize a dense executed
calculation. It cannot be substituted for the model's actual coefficients or
used to claim a sparse causal circuit.

The resulting ontology now has three distinct code lengths:

1. **Program length:** few residual/bilinear/QK operation types.
2. **Native execution support:** thousands of participating channels with
   cancellation.
3. **Behavioral description length:** a handful of freely reweighted vectors
   for one context and one 32D endpoint.

All three are correct for their use cases; conflating them produces most of the
apparent contradictions in the earlier analyses.

A first attempt to turn the short per-context descriptions into one shared
vocabulary does not succeed cleanly. Four atoms from a trained 128-atom
dictionary retain only 0.65--0.70, barely above matched random overcomplete
dictionaries and not consistently above raw training prototypes. Eight atoms
reach 0.86--0.90. These atoms are therefore not ready to be called concepts.

The more canonical tensor-network representation is now the exact pairing

```text
upstream tangent transport J_s
        ×
Layer-16 bilinear Jacobian B(x_s)
        ×
downstream loss adjoint g_s
        →
32D behavioral effect
```

Expanding `B(x_s)` in native CP channels produces the dense cancellation web.
Contracting it first with `g_s` produces one effective backward read vector;
contracting that with `J_s` produces the behavior. Forward, native-CP, adjoint,
and endpoint views are exact rearrangements with different interpretive uses.

The exact atlas makes that comparison quantitative. A rank-64 basis learned
from forward tangent rows retains 0.70--0.85 of the held-out loss effect,
substantially above state, raw-adjoint, effective-adjoint, and random bases.
Forward reachability is therefore the best tested shared coordinate system,
but it is not a complete small interface: even rank 256 reaches only
0.80--0.90 depending on corpus.

Directly optimizing the projector for `J P h` does not repair this. Stochastic
Adam selected the PCA initialization on validation. Deterministic full-batch
Grassmann descent raises training fidelity by about 0.046 but validation by
only 0.001, and gives essentially no OOD gain. This makes “PCA used the wrong
metric” an inadequate explanation, although non-convex global optimality is
not proved.

There is also an important endpoint caveat. The current `g_s` is the gradient
of true-next-token cross entropy, so it contains an observer-supplied target
token direction. Breadth across this quotient may combine model computation
with the diversity of target-token readouts. The next tensor red team should
split the exact CE adjoint into its probability-weighted prediction term and
its target-logit term, then compare both with target-independent logit/Fisher
interfaces. That distinction determines whether the broad atlas describes
inference, training influence, or both.

The split rejects the simplest label-confound account. Target-logit adjoints
are broader than probability-weighted prediction adjoints, but both are
nontrivial, and the model's own top-1-versus-runner-up margin is broader still
under the tested rank-64 chart. The output-facing calculation is not compact
merely because the evaluator stops supplying a true token.

At the same time, CE should not be treated as a native internal signal. Its
prediction and target terms are individually several times larger and strongly
aligned, leaving CE through cancellation. Output-token identity organizes the
target adjoints but does not determine them: same-token directions cluster,
yet a frozen token prototype is a poor cross-context substitute. The next
canonical factorization should therefore be

```text
output covector q_s
        ×
context-dependent suffix pullback S_s^T
        ×
Layer-16 bilinear Jacobian and upstream tangent
```

This asks whether there are many mechanisms or one/few shared suffix operators
acting on many clean output questions. It is a more useful ontology test than
counting the rank of a mixture of token readouts.

A single shared suffix operator is not enough. Zero-centered regression,
identity-centered rank-768 regression, and a rank-64 local-state CP tensor all
fail behavioral sufficiency. The local-state model is not completely empty:
shuffling state damages prediction and target effects. But it misses the
sequence-wide information consumed by attention and does not explain margins.

The exact weights expose the shorter description. Backward propagation through
the final suffix has four path types:

```text
final residual or final MLP response
              ×
direct mixed-state or attention response
```

The two direct paths retain 0.897--0.991. Adding the residual-attention path
raises every tested component and corpus to 0.957--0.990. The final-MLP path is
essential; attention is a smaller distribution-dependent correction. Thus the
suffix is not one static map, but it is generated by a short conditional
program.

The four paths match a same-graph full VJP to `1.15e-7`. Their `2.38e-4`
difference from an earlier saved adjoint is therefore a cross-run float32
provenance effect under cancellation, not an omitted derivative branch. The
predeclared saved-adjoint exactness gate is still reported as failed.

This is the sharpest current version of the CPU analogy. A processor
instruction can have a stable algebraic definition while producing many dense
register-level vectors on different inputs. The explanatory object is the
instruction grammar plus operands and routing state. Calling every realized
vector a new feature loses the program; calling the path label a three-neuron
circuit loses the implementation.

Opening the essential final-MLP path reveals that native sparsity is not a
property of the module alone. Probability and target-logit reads reach 90%
with 32--64 channels. The model's top1-versus-runner-up margin needs 4,096
channels on FineWeb/WikiText and 512 on TinyStories. The same module therefore
contains both a neuron-concentrated score regime and a dense decision-precision
regime.

The dense margin is not a union of unrelated sparse circuits. Top-1 and
runner-up top-128 banks overlap in 107 channels. Each logit is reconstructed
well, but their difference is not: union fidelity is only 0.65--0.83 at 128
channels per side. The margin is a smaller cancellation-sensitive quantity,
with absolute projection mass around five versus around two for each score.

This adds a fourth code length to the ontology:

1. algebraic program length;
2. native execution participation;
3. observer-side description length;
4. **precision-conditioned support**—how many terms are needed after large
   shared components cancel in a chosen contrast.

Precision-conditioned support is especially relevant for argmax boundaries,
calibrated probabilities, and safety margins. It should not be mislabeled as
additional semantic concepts.

Symmetry-aware tensor contraction does not automatically solve the problem.
The native final-MLP weights contract exactly into an
output-adjoint×input×input symmetric tensor. Inside a 64D PCA quotient, 32
modes preserve roughly 0.94--0.99 of that quotient. Inside a 128D quotient, 64
modes preserve roughly 0.89--0.98. The factorization itself is real and exact.

However, the quotients retain little native behavior: individual score
fidelity is only 0.20--0.44 at 64D and 0.32--0.52 at 128D, while margins can
have negative fidelity. Increasing dimension also increases tensor energy rank
from 38 to 92 rather than revealing a stable tiny core.

This separates two more notions of “canonical”:

- **algebraically canonical:** respects bilinear factor-swap symmetry and
  exactly represents a specified quotient;
- **behaviorally adequate:** preserves the native effect under the downstream
  metric that matters.

The symmetric tensor is algebraically canonical but the PCA quotient is not
behaviorally adequate. A short description of the wrong quotient is not a
reverse-engineered model.

The failed quotient was not the end of the tensor story. It had imposed one
state-PCA basis on both arguments of the bilinear derivative. Along the actual
causal path those arguments have different roles:

```text
natural context x
        ×
propagated tangent d
        -> Down[(Left d)(Right x) + (Left x)(Right d)]
        -> endpoint covector
```

Giving context, tangent, and output their own rank-64 natural bases changes
the result completely. The contracted tensor is exact to `6.4e-7` and retains
0.89--0.95 of individual prediction/target effects on held-out FineWeb,
WikiText, and TinyStories. Swapping the context and tangent bases destroys the
result even though the dimensional budget and learned subspaces are
unchanged. Tensor roles, not merely dimension, are mechanistic information.

The decision margin remains at 0.60 on FineWeb/WikiText and 0.81 on
TinyStories. This is consistent with the native-channel result: the two score
computations are compact, but accurately subtracting them is a stricter
numerical endpoint.

The role tensor has a small multilinear spectrum—90% energy needs only 15--17
modes in each unfolding—and Tucker-16 retains at least 0.85 of the score paths.
Yet the coordinates are not unique. Across discovery splits, output core
overlap is above 0.92 while context/tangent overlap can fall to 0.64--0.69;
behavior changes by only a few hundredths. Behavior-optimized rank-16 cores
show the same pattern: similar external performance, materially different
subspaces, and only a modest margin gain.

The current reverse-engineering target is therefore an executable equivalence
class rather than a privileged neuron list or unique rotation. A complete
claim should name four standards separately:

1. **algebraic exactness**—does the contracted program equal the specified
   projected native computation?
2. **behavioral coverage**—which scores, losses, margins, or interventions
   does it preserve, at what fidelity?
3. **distributional scope**—does it transfer across sites and corpora?
4. **coordinate identifiability**—do independently estimated physical
   subspaces or axes agree?

The final-MLP tangent program passes the first three for individual scores at
moderate fidelity and fails the fourth on its input-side coordinates. That is
a useful reverse-engineered object, but not a monosemantic basis.

## Semantic output and routed context are different interfaces

The stable output core is semantic without having privileged scalar axes.
Uniform-vocabulary weight reads give median AUROC 0.828, well above isotropic
directions, while rotations inside the same selected subspace perform almost
as well as the native axes. Natural target/top1 decoders transfer across all
three corpora at median AUROC 0.91--0.97 when allowed to use the full subspace.
The semantic carrier is therefore the output space plus the declared readout,
not a unique list of sixteen neuron-like meanings.

Three output-role metrics must also be separated. Raw unembedding reads answer
which tokens a direction can promote before context; natural sampling answers
which questions the corpus asks; and the final-RMSNorm Jacobian supplies the
local causal metric. Raw-versus-exact correlations of 0.89--0.93 rise to
0.97--0.99 after the RMS pullback, and RMS-aware labels match exact labels on
14/16 target and 16/16 top1 modes. This turns an apparent semantic conflict
into a metric distinction.

On the input side, the context coefficients act jointly as a control register.
Contracting them with the tensor yields a per-site 16×16 tangent-to-output
operator whose median 90%-energy rank is three. Four context modes preserve
0.97 of the operator and eight preserve 0.993, yet random rotations within the
same context subspace are just as semantically labelable as the tensor axes.
The register is compact and semantic as a whole, not decomposed into canonical
scalar instructions.

The router red teams expose a short composition across interfaces. Coarse token
class predicts 0.79--0.81 of the operator, exact token lookup 0.81--0.89, and a
linear map from the checkpoint's own embedding 0.85--0.90. The embedding map
retains 0.81--0.83 on token identities absent from discovery and beats
class/position-matched response permutations. A rank-eight map is sufficient,
and even a map restricted to the tensor context span remains useful. But the
full router's physical input span and tensor context span align only at mean
canonical cosine 0.26.

This gives a more precise replacement for the hope of one canonical feature
basis:

```text
lexical embedding geometry
        -> low-rank learned router
        -> noncanonical joint context register
        -> contracted low-rank local operator
        -> stable semantic output subspace
        -> context-metric output effect
```

Each arrow is shorter than the native execution and transfers OOD, while the
coordinate systems on adjacent boundaries need not coincide. Reverse
engineering can therefore succeed compositionally even when no global neuron
or feature dictionary is uniquely correct.

The architectural source intervention adds a necessary correction to that
diagram. At Layer 17 the explicit original-embedding reinjection is scaled by
8.0, but it is not the causal carrier of operator variation. Position-mean or
cyclic substitution of that term, including exact recomputation of final
attention, preserves more than 0.99999 of the operator. The carried state alone
preserves 0.988--0.995, while replacing it drops fidelity to 0.54--0.70 and
cyclically substituting it drops fidelity to 0.38--0.58.

Thus the first arrow should be read as prediction of a compiled code, not
direct execution by the embedding wire:

```text
native token embedding --predicts--> compiled carried-state router
                                      |
                                      v
                            joint context register -> local operator
```

Earlier blocks and repeated embedding injections create the correlation; the
final carried residual owns the local causal effect. This is a concrete case
where a simple semantic probe is true, transferable, and mechanistically
useful without identifying the intervention site. Predictive coordinates and
causal carriers are complementary explanatory objects.

## First-four-layer extension: compose local operators instead of expanding a giant tensor

The exact Layer-1 backward program now generalizes unchanged to source MLP
Layers 0--3. For a source layer \(s\), begin with the output cotangent and apply
the native local adjoint of Blocks 17 down through \(s+1\), finally pulling the
result through `Down_s`. This costs one local operation per remaining block. It
does not require materializing the seventh-, ninth-, or higher-order polynomial
that would result from algebraically expanding all bilinear interactions.

On 96 held-out FineWeb semantic edits, this compositional recurrence is exact
at all four boundaries. The native product payload remains broad: roughly
1,040 of 4,608 helpful channels are needed for 90% of positive local linear
mass at every layer. But the suffix description becomes more local with depth.
Full attention in the next eight blocks plus all native MLP responses reaches
0.953/0.969/0.972/0.986 median gradient cosine from Layers 0 through 3.

Cross-position tests identify the relevant verbs. Same-position effects are
query-heavy, partly for structural reasons. For an earlier source token,
query-only and MLP-only paths are exactly zero; keys and values transport the
signal, and later queries refine it. At distance 64, key+value cosine improves
from 0.728 at Layer 0 to 0.869 at Layer 3. The next-eight-block program is above
0.90 at Layers 1--3 but only 0.86--0.87 at Layer 0, so the earliest boundary
requires a longer correction.

The emerging breadth-first ontology is therefore:

```text
dense state-specific channel payload
        -> position-conditioned Q/K/V transport or refinement
        -> short sequence of nearby native MLP/attention adjoints
        -> smaller long-range residual correction
        -> output effect
```

This is a shorter mechanistic description than a path list and a more faithful
one than a neuron dictionary. The next compression target is not a higher-order
global tensor; it is a reusable parameterization of each local operator and of
the state-dependent rule that composes them.

### Why a stored chart atlas is not enough

Two prospective atlas experiments test the obvious shortcut: store gradients
from discovery contexts and retrieve or interpolate them from similar forward
states. The first stores 64 gradients per semantic objective. The corrected
version stores the complete eight-objective semantic Jacobian at each of 256
shared states.

Both fail as interpreters. The larger feature-specific span can represent about
0.63--0.65 cosine of a held-out gradient if the held-out gradient itself chooses
the coefficients, but KNN and continuous-kernel routers recover only 0.03--0.14.
Even complete current-position trajectories with cosine above 0.7 can have
nearly orthogonal suffix gradients. The missing metric is sensitivity geometry,
not ordinary activation proximity.

A local-state swap identifies where that geometry comes from. With the final
output adjoint fixed to the target example, substituting every suffix block's
state with a path-nearest same-position context yields essentially zero gradient
alignment. Exact target-state MLP responses alone retain 0.66--0.78 cosine;
exact target-state attention responses alone retain only 0.00--0.08. Inserting
wrong-context attention lowers the exact-MLP program to 0.47--0.67, so attention
acts as a consequential modulator even though its pure path family is small.

This refines the compositional diagram:

```text
forward state at every suffix block
        -> exact bilinear MLP Jacobian (backbone)
        -> near-source Q/K/V Jacobian modulation
        -> repeated cross-composition
        -> semantic gradient / causal coalition
```

The distinction between omission and substitution is essential. Omitting late
attention responses while keeping exact MLP states gives the successful
next-eight-block program. Replacing late MLP states with plausible neighboring
states destroys it. A component can therefore be dispensable as an additive
path but indispensable as the state on which another path is conditioned.

The highest-information next step is a spectral and causal decomposition of the
dynamic MLP Jacobian

\[
J_{\mathrm{MLP}}(x)
=D\,\mathrm{diag}(Rx)L + D\,\mathrm{diag}(Lx)R,
\]

including the RMSNorm pullback. This object is directly computable from weights
and state, and can be tested for reusable spectral outliers over a distributed
bulk. If a small set of modes composes causally across blocks, it supplies the
missing shorter ontology; if not, the distributed bulk is part of the actual
algorithm rather than atlas noise.

## Early-layer result: a privileged instruction set, not sparse semantic circuits

A prospective basis tournament now resolves the main ambiguity in the Layer-0
and Layer-1 neuron atlas. For the same semantic token edits, native bilinear
product coordinates were compared with exact dense orthogonal coordinates.
Family banks were frozen on one FineWeb context and evaluated through the full
nonlinear suffix on a held-out FineWeb context and TinyStories.

The native basis wins decisively. At rank 128, native family banks reach
0.21--0.40 downstream fidelity across the two layers and corpora, while dense
rotations reach only 0.06--0.09. This is not a decoder-norm artifact: the
advantage survives contribution scores divided by Down-column norm. It also
survives leaving the evaluated lexical pair out of bank construction and an
eight-seed rotation null.

The semantic grouping is real but modest. Banks trained on the other members
of a declared family beat the p99 of 2,000 random regroupings on both corpora
and layers. Layer 1 also beats a global all-other-edit bank. Layer 0 sometimes
does not, and a generic high-norm bank is competitive on TinyStories. Thus
Layer 0 contains semantic organization plus a large lexical/generic component;
Layer 1 has clearer cross-pair family reuse.

Crucially, basis privilege does not imply a sparse semantic circuit. Native
per-example oracles need 128--256 channels for 50% fidelity and 1,024--2,048
for 90%. A frozen leave-one-pair-out semantic-family bank needs 2,048 of 4,608
channels for 50% and 4,096 for 90%. The correct picture is:

```text
privileged native quadratic predicates
        + context-dependent selection and cancellation
        -> broad reusable semantic transformation
```

This adds a useful distinction to the tensor ontology:

1. **atom privilege:** does the architectural coordinate beat rotations?
2. **instance sparsity:** how many atoms explain one edit when selected with
   knowledge of that example?
3. **family reuse:** can a bank selected from other examples transfer?
4. **family sufficiency:** how large must that frozen bank be for a declared
   behavioral fidelity?

Layers 0 and 1 pass the first three and fail a strong sparse version of the
fourth. Neuron inspection is therefore informative about the model's local
instruction set, while tensor programs and state-conditioned routing are still
required to explain complete behavior.

A stricter recurrence test refines the bank itself. Channels that appear in
the top-256 contribution sets of at least two other family members form median
cores of only 69 neurons at Layer 0 and 65 at Layer 1. These strict cores have
positive OOD causal effects (0.115 and 0.206 on TinyStories), but recover only
part of the 256-channel bank and do not universally retain half its effect.
Layer-1 recurrence beats an equal-size magnitude control; Layer 0 recurrence
does not consistently do so. The best current decomposition is therefore:

```text
small recurrent native family core
        + large pair/context-specific native tail
        -> broad semantic edit effect
```

This is more informative than either “the feature is one neuron” or “the
feature is fully distributed.” There is a reusable neuron-aligned kernel, but
the requested fidelity determines whether that kernel is an adequate circuit
or merely the stable center of a much broader computation.

The frozen result also transfers prospectively. Seventeen new substitutions
were evaluated after all Layer-0/1 banks were fixed. Correct semantic-family
banks beat wrong-family medians by 0.106--0.167 after contexts were clustered
within lexical pair; pair-bootstrap intervals exclude zero and family-label
permutations give `p<0.001`. Native banks continue to beat dense rotations.

This does not turn the banks into discrete modules. Correct-family top-1
classification is only 0.41--0.65, a global edit bank remains competitive, and
which individual pairs are easiest is unstable across corpora at Layer 1. The
appropriate term is **overlapping family-biased gate distribution**. It is a
valid reusable feature for retrieval, ranking, and partial intervention, but
not a complete exclusive circuit for the semantic relation.

## Spatial scope correction: storage, transport, and query readout are different features

The Layer-0/1 bank result above is authoritative for a distributed-prefix
intervention: its channel scores and patches include every causal position
through the prediction position. A later audit initially treated that quantity
as if it described only the final query position. Computing both scopes exactly
shows that they are different mechanisms, not competing estimates of one number.

Across the complete prefix, Layer-0 family writes are 72--78% embedding-sourced
and 22--28% attention-sourced. At the final unchanged token, the embedding delta
is structurally zero and attention supplies the entire change. At Layer 1, the
complete-prefix family write combines carried state (0.46--0.47), repeated
embedding injection (0.38--0.41), and attention (0.12). At the final position it
is instead about 0.22 carried and 0.78 newly computed attention.

Fresh final-position-only banks do not reproduce the distributed semantic
result. Even with two discovery contexts, rank-256 family banks reach only
0.064--0.180 fidelity on the seventeen prospective pairs, do not reliably beat
wrong families, and overlap their distributed counterparts by only 0.03--0.11.
Per-example oracles still reach 0.73--0.76, so the negative result is reuse
failure rather than absence of native causal coordinates.

The revised early-layer diagram is therefore:

```text
edited-token embedding and local MLP code
        -> attention QKOV transport
        -> context-specific final-position native-channel readout
        -> later suffix effect
```

The first and third MLP codes need not share a neuron dictionary even in one
layer. A position-free label such as “the sentiment neuron” erases this
distinction.

## Weight-level attention skeleton versus state-level execution

Exact tensor expansion makes the transport stage unusually legible. At Layer 0,
the final-query attention change comes entirely from the edited source edge.
Value change contributes 0.67--0.80 of the ensuing finite MLP write; QK1 and QK2
supply the remainder. At Layer 1, the edited source contributes 0.52--0.54 and
later positions 0.43, while earlier context contributes almost nothing. The MLP
response is about one quarter carried state, one half value transport, and one
sixth QK routing.

Raw weight magnitude is not an execution map. Top-three heads ranked by a
state-free, scale-canonicalized QK-tensor/OV capacity score recover only
0.12--0.19 of the attention delta and have near-zero rank correlation with
realized importance. Contracting the same weights with the current token states
and selecting three heads recovers 0.84--0.93.

There is nevertheless a reusable weight-level skeleton. Three heads calibrated
on the original substitutions recover 0.69--0.75 at Layer 0 and 0.63--0.66 at
Layer 1 on new pairs and TinyStories. Exhausting all 84 three-head subsets places
these frozen sets in the 77th--99th percentile. Family-specific head sets do not
consistently improve them. The best description is **generic transport anchors
plus state-selected assistants**, not one head per semantic family and not a
fully dynamic unstructured attention cloud.

This supplies a practical three-level weight ontology:

1. static tensor geometry enumerates possible operators;
2. small discovery panels identify recurring transport modules;
3. exact state contraction identifies which edges and heads execute now.

Each is useful for a different reverse-engineering budget. Only the third is a
complete local execution account.

A fixed output-space reinterpretation does not rescue a semantic query feature.
Family/global SVD bases fit from two contexts reach only 0.02--0.05 prospective
fidelity at rank 6. Expanding discovery to eight contexts per original pair and
rank 24 raises some cells to 0.09--0.17, but all locked transfer/specificity
gates still fail. Three of four pair-clustered family effects are
non-significant; Layer 1 on TinyStories retains only a partial label signal.

Thus the current division of labor is unusually clear:

```text
stable generic attention transport skeleton
        + current token/context state
        -> changing final-position bilinear-channel support
        -> changing MLP write direction
```

The appropriate next model is conditional and algebraic: predict the bilinear
product/write from QKOV factors and the pre-MLP state. Repeating unconditional
SAE, neuron-bank, or PCA searches at this boundary is unlikely to expose the
missing invariant.

That conditional test succeeds. The exact finite bilinear expansion splits the
query readout almost 50/50 between `ΔL·R(base)` and `L(base)·ΔR`; the
`ΔL·ΔR` term is negligible. Thus the changing base state gates the transported
edit through both native factor banks. No fitted semantic decoder is required
to state the computation.

The resulting product code is **dynamically native-axis privileged**. At rank
256, target-aware native selection reaches 0.66--0.78 downstream fidelity and
beats an exact dense rotation by 0.07--0.18. Activation magnitude alone reaches
0.62--0.66, while static decoder norm reaches only 0.02--0.13. Target-aware
native S50 is 64--128 channels; rotated S50 is 128--256. Yet native S90 remains
2,048 channels.

This resolves the apparent contradiction between the failed semantic bank and
successful neuron interventions:

```text
same semantic family  -/-> same query neurons
current bilinear execution -> privileged native neurons
```

Native neurons are the model's instruction-level products, but semantic
identity does not determine a fixed instruction subset. Context supplies the
operands. For reverse engineering, inspect factor values and their products per
state; for editing a reusable concept, operate earlier at the transport/router
level or learn the conditional selection rule.

The product factors provide a limited but real “verb” interpretation. Among
causally important channels, 52--55% are at least 80/20 dominated by one cross
term, compared with roughly 48% across the full layer. The 5--7 point enrichment
survives pair bootstraps. Channels that recur within one corpus keep the same
dominant factor side at 0.667 agreement, significantly above shuffled
0.60--0.625 baselines.

The semantic reading must remain modest. Only 4--9% of the top-256 identities
overlap across corpora. Operand-side agreement transfers for Layer 0 but falls
to chance at Layer 1. Nearly perfect contribution-sign recurrence is an
imbalance artifact, and isolated embedding-nearest tokens usually do not name
the contextual predicates. Therefore:

```text
bilinear neuron = interpretable algebraic verb
bilinear neuron ≠ automatically a fixed human-semantic conjunction
```

The weights specify the predicate pair; the state supplies its current
meanings. A semantic label is earned only when those operand roles predict new
contexts or interventions.

The next contraction closes part of that gap. Exact attention-head writes can
be mapped through RMSNorm into changes of each neuron's Left and Right factors.
At Layer 0, head 7 is the most common dominant source for causally important
operands across FineWeb and TinyStories; frozen transport heads account for
0.53--0.68 of their absolute source attribution. At Layer 1 the corresponding
coverage falls to 0.34--0.49, carried state becomes material, and the dominant
source is much less stable across corpora.

This conclusion is not an artifact of one integration convention. Float64
straight-path integration and permutation-Shapley attribution recompose the
factor endpoints to `8.24e-14` and `1.83e-16`, agree on dominant sources, and
return the same frozen-head shares to reported precision. The original float32
locked exactness failure remains recorded, but is diagnosed as numerical rather
than mechanistic.

The strongest current Layer-0/1 description is therefore:

```text
repeated embedding/current state
        -> reusable generic attention transport heads
        -> context-dependent Left/Right factor changes
        -> native bilinear product neurons selected for this example
        -> broad downstream write
```

The hierarchy becomes less edge-local by Layer 1: operand sources remix even
while product coordinates retain instance-conditioned native-basis privilege.
This is positive evidence for neurons as algebraic execution atoms, but negative
evidence for a universal dictionary of semantic neuron identities.

## Prospective execution: local gradients versus finite circuit compilation

The conditional account now makes a successful prospective prediction, but
only after distinguishing two notions of locality. A checkpoint-native JVP at
the original token is exact for small continuous edits: at 10% source-to-target
embedding dose it reaches 0.995 product-direction cosine. A full token
replacement leaves that tangent regime. Endpoint cosine falls to 0.45--0.55,
causal fidelity to 0.29--0.41, and pair-clustered tests do not show a reliable
advantage over simply selecting channels already active in the base state.

One edit-conditioned relinearization at the midpoint changes the result. It
predicts the full product direction at 0.976--0.996 cosine, overlaps the
realized top-256 circuit by 0.70--0.87 Jaccard, and recovers 0.645--0.753 of the
downstream effect when its selected actual deltas are patched through the
complete suffix. A matched wrong-target midpoint loses both support and causal
fidelity, and the correct midpoint beats base activity under lexical-pair
bootstraps on both FineWeb and TinyStories.

The tensor path is compact without being exactly quadratic. Across held-out
interpolation nodes, a quadratic vector fit reaches 0.996--0.997 fidelity and
two singular directions contain 0.995--0.9995 of path energy. Yet a linear
secant fit is already 0.963--0.977, and the midpoint derivative misestimates
endpoint norm by 34--39%. The robust invariant is therefore **direction and
native support along a nearly planar curved edit path**, not an exact quadratic
coefficient formula.

This adds a fourth useful interpretability object:

1. fixed neurons for genuinely repeated identities;
2. generic heads for repeated transport operations;
3. endpoint Jacobians for infinitesimal sensitivity;
4. interior path contractions for finite counterfactual circuit compilation.

These objects answer different questions. Asking all of them to be “the
feature” creates false contradictions; choosing among them should be driven by
whether the goal is naming, transport, local prediction, or finite editing.

The midpoint result is not, by itself, evidence for semantic abstraction.
Cross-family token replacements and exact-distance random endpoints show that
finite path compilation is largely generic, particularly in Layer 0. No
semantic-edit advantage in downstream fidelity survives lexical-pair
bootstrapping.

There is nevertheless a natural-manifold effect. Relative to exact-distance
random sphere endpoints, semantic token edits improve top-256 support overlap
by 0.057--0.065 at Layer 0 and 0.077--0.125 at Layer 1, with pair-clustered
intervals above zero. Direction cosine improves similarly, while behavioral
fidelity does not. Natural edits are therefore more cleanly expressed in the
native instruction basis, especially after one layer of contextual processing,
without being proven to use a shared semantic circuit.

An ideal natural-language control is partly ill-posed: the checkpoint embedding
geometry places morphological and semantic neighbors close together. Matching
`walk→walked` in distance and token statistics returns `walks` or `walking`, not
an unrelated word. Exact-distance synthetic controls leave the token manifold;
unrelated token controls change distance. The honest conclusion is a manifold
alignment result with semantic specificity unresolved, not a forced binary
answer.

The subsequent chart and relation controls resolve part of that uncertainty.
Prospective adaptive-chart gains are at least as large for exact-distance random
sphere endpoints as for natural edits, and the chart router stays near the
finite random-endpoint oracle. Adaptive rotations should therefore be treated
as generic output-effect coordinates supplied by the interpreter, not latent
semantic bases used sparsely by the model.

Natural native-coordinate fidelity is sometimes higher than transported random
token-difference controls, but the effect is not universal after expanding from
two to eight control directions. More decisively, transporting directions from
other pairs in the same semantic family does not outperform matched
different-family directions under pair-clustered inference. Thus the early
model supports exact native quadratic instructions and state-specific edit
programs, while a shared linear sentiment/gender/tense/scale/polarity code is
not detected.

The present hierarchy is:

```text
fixed model mechanism          native quadratic neurons + attention operators
realized computation           state-conditioned broad native coalition
compact effect description     suffix-routed chart, generally non-native
human-semantic reusable code   not yet identified in Layers 0--1
```

These objects are all legitimate interpretations, but they answer different
questions. Conflating a short effect description with a native representation
would make the generic chart result look like semantic superposition; requiring
one fixed neuron bank for semantics would miss the reproducible conditional
native execution.

## Supervised ontology and the feature/coalition boundary

The first explicit 50-feature FineWeb census adds a bridge between weight
algebra and human labels. It gives three successively stricter answers.

1. **Global readability:** Layer-1 native products predict 32 powered lexical
   labels at 0.767 median AUROC with 64 channels, versus 0.599 with the best
   single channel.
2. **Lexical red team:** isolated-current-token product codes already reach
   0.746. Much global readability is distributed token/morphology information,
   amplified by repeated embedding injection.
3. **Exact-token conditional readability:** when every positive-negative pair
   shares the current token, the full model scores 0.747, Layer 0 rank-64 scores
   0.561, and Layer 1 rank-64 scores 0.602. One neuron is nearly chance at
   0.520.

The causal test separates availability from use. A fixed discovery-frozen
semantic support has a small donor-directed effect, but neither direct output
weights nor a feature-average suffix gradient finds a universally better bank.
The exact recipient-state suffix gradient does. A local rank-64 positive
coalition survives finite intervention on 95/96 pairs, with 0.922 correlation
between first-order and finite shifts.

The support is almost completely context-specific. Same-feature median Jaccard
is 0.0079, near the 0.0070 random scale; roughly 680 distinct neurons appear
over twelve contexts and there is no half-context core. Stable direct-output
supports show more semantic reuse but are weaker causal controllers.

This yields a sharper model of “features as input-conditioned coalitions”:

```text
stable weight roles
      +
distributed context evidence
      +
recipient-specific suffix sensitivity
      ->
small controllable coalition for one declared semantic effect
```

The small coalition is real and finite-causal, but it is not a persistent set
of feature neurons and does not reconstruct the whole natural state change.
Selecting absolute local contributions correlates only 0.280 with the full
Layer-1 patch because the full patch contains helpful, opposing, and nonlinear
effects. This is analogous to a constraint/cancellation web at the activation
level without claiming that homogeneous KKT results apply unchanged to the
complete transformer.

The local selector also cannot be reduced to one context-dependent input. With
feature means for both activation delta and suffix gradient, finite direction
is correct on 0.594 of patches. Pair delta alone with mean gradient gives 0.573;
mean delta with pair gradient gives 0.729; pair-specific values for both give
0.990. The relevant object is their elementwise contraction.

A position-stratified 256-gradient discovery basis shows genuine but broad
reuse. Its full span reaches 0.633 held-out gradient cosine versus 0.235 for a
random rank-256 span, but preserves only 0.313 top-64 support Jaccard and 0.662
of exact first-order leverage. Compactness lives in the executable tensor
program that recomputes the suffix sensitivity, not in a small global feature
subspace.

## Prospective versus endpoint-conditioned explanations

The strongest semantic selector above reads the donor activation delta and
recipient suffix gradient, so it is an upper bound. A separate experiment shows
that finite product changes themselves are prospectively compilable. An
adaptive interior quadrature rule uses only disagreement among 4, 8, 16, 32,
and 64-node derivative estimates. It catches every remaining sixteen-node tail
failure and reaches 0.979 effect-weighted KL recovery at Layer 1, with 35 nodes
selected on average.

Together the experiments isolate the unsolved step:

```text
specified edit -> prospective tensor path compiler        validated
specified effect -> endpoint-informed local coalition     validated upper bound
natural prompt -> semantic objective/edit discovery       unresolved
```

The next useful work should target that last arrow rather than search for one
globally correct neuron dictionary.

## Exact early-to-output suffix program

The attention-inclusive test is now complete for Layer-1 semantic interventions
in the 18-layer FineWeb model. A small static gradient basis does not work:
even 256 discovery gradients recover only 0.633 median held-out cosine. The
same field has a substantially shorter **executable** description.

Each Block-2--17 backward map splits exactly into residual identity, bilinear
MLP response, attention response, and attention-after-MLP response. Grouping
the exponentially many paths by operator counts shows:

| Program | Median gradient cosine | Exact finite-shift fraction |
|---|---:|---:|
| MLP-only axis | 0.717 | 0.748 |
| MLP×attention rectangle 4 | 0.300 | 0.389 |
| MLP×attention rectangle 8 | 0.952 | 0.926 |
| Rectangle 12 | 1.000 | 1.000 |

Attention has little independent semantic axis, but its repeated
cross-composition redirects the MLP-carried adjoint. Attention VJPs in Blocks
2--9 alone recover 0.969 median gradient cosine and 94.2% of the exact finite
shift; Blocks 10--17 recover 0.753 and 78.3%. No individual block is sufficient
or necessary.

## Position changes the correct feature ontology

For same-position semantic patches, the early attention correction is mostly
QK routing: QK-only reaches 0.926 median cosine, while current-value sensitivity
reaches 0.740. This is not universal. On genuinely cross-position token edits,
MLP-only and query-only gradients at the earlier source are exactly zero.

| Source-to-output distance | Key+value | Full early Q/K/V program |
|---:|---:|---:|
| 8 | 0.812 | 0.927 |
| 32 | 0.760 | 0.944 |
| 64 | 0.759 | 0.924 |

Q/K/V roles are stable architectural verbs, but the useful coalition is selected
by the source/destination relation. A feature is not well described as “living
in queries” or “living in values” without specifying the causal task.

## Is the role program a sparse head circuit?

Only partly. A discovery-frozen ranking over the complete 16-block × 9-head
suffix transfers to new FineWeb rows and all three distances. Sixty-four of the
144 layer-head edges are the first tested budget to exceed 0.90 median gradient
cosine everywhere. Seventy-two reach about 0.96 and decisively beat random
72-edge sets, while 32 reach only 0.63--0.71.

```text
broad native neuron evidence
        ↓
compact operator types (MLP, Q, K, V)
        ↓
structured but moderately broad layer-head coalition
        ↓
small state-specific causal channel patch
```

The last patch can contain only 64 Layer-1 channels and work causally, while its
selector depends on a much broader suffix program. Payload sparsity and routing
sparsity are different quantities.

The head scaffold is also reproducible. Two independent discovery panels give
top-64 Jaccard 0.730 and full-ranking correlation 0.879. Their 54-edge
intersection retains 0.936/0.927/0.900 gradient cosine at distances 8/32/64,
well above matched random sets. This rules out the explanation that many
arbitrary half-graphs are equally good. The remaining challenge is to compress
the stable 54-edge core within heads—by QK/source structure or continuous
operator modes—without reintroducing a context-specific oracle.

## Dynamic Jacobian spectrum: outliers are clues, not the whole program

The explicit bilinear derivative gives a compositional method that avoids
expanding a seventh-order polynomial. For a pre-MLP state `x`, contract the two
bilinear factors with the normalized state, form the local MLP Jacobian, include
the RMSNorm derivative, and compose its VJP with the residual and attention
operators at each subsequent block. This weight formula reproduces autograd to
`1.75e-6` relative error.

The spectrum is not generally tiny. Blocks 1--12 have effective ranks around
444--562. Even an output-conditioned oracle needs roughly 230--273 singular
modes for 90% local response energy. Task-selected rank 256 composes to 0.919,
0.935, 0.944, and 0.960 gradient cosine when interpretation begins at Layers
0, 1, 2, and 3. This is substantial compression, but it is a medium-rank
program rather than a handful of circuits.

Block 17 is qualitatively cleaner: effective rank 69, task K90 36, and a
leading singular value 4.58 times the strongest matched null. The bulk-plus-
outlier hypothesis is therefore depth-dependent. The early/middle bulk cannot
be discarded merely because outliers exist; the final output-facing block is
much closer to the hoped-for small circuit.

## A feature can be stable while its singular modes are not

Across 32 independent sites for eight supervised token properties, fixed local
mode lists are weakly reusable. Rank-64 response subspaces selected for one
example capture only about 1.01--1.08 times more energy on another example of
the same feature than on different features. Yet exact response directions at
Blocks 12 and 17 are 2.28 and 2.85 times more similar within a feature.

That combination is the current best concrete example of an input-conditioned
coalition:

```text
different contexts
    -> different sets of local bilinear modes
    -> context-specific cancellations and reinforcement
    -> a recurring feature-level transformation direction
```

This is not neuron alignment, and it is not arbitrary opacity. The recurring
direction is a compact semantic object; the weight-defined dynamic Jacobian is
the executable mechanism that produces it. Reverse engineering should look for
both levels and test their composition causally.

### Red team: a recurring backward direction need not be a native feature

A 32-feature extension corrects the strongest possible reading of the result
above. All 32 token properties passed a prior held-out competence gate, and
their semantic response directions become highly identifiable with depth.
However, a shared vocabulary permutation that preserves the full objective
Gram matrix produces *even cleaner* late identities: Block-17 prototype
accuracy is 0.867 for the null versus 0.602 for semantics.

Thus a backward direction can recur because the same analyst-defined output
contrast is being transported through an increasingly output-facing map. That
is sensitivity geometry, not sufficient evidence that the model represents the
contrast as a native concept. The semantic object must also be visible in
forward behavior and survive frozen causal transfer across contexts.

This yields a cleaner division of labor:

```text
forward held-out prediction       does the model encode useful information?
matched backward-objective null   is recurrence more than objective identity?
cross-context causal transfer     does the proposed feature perform reusable work?
```

The dynamic Jacobian remains the right executable mechanism, but ontology must
be established by these three tests together.

## Causal ontology: a weak feature scaffold selects a context-specific coalition

The three-way test is now complete for the 32 competent labels at source Layers
0--3. Discovery-frozen activation coalitions predict labels but fail causal
transfer. Adding the average suffix-gradient sign, shrinking the intervention
to 10%, and learning the mean paired contribution `E[delta * gradient]` all
fail to produce a reliable global feature circuit.

The positive control succeeds completely. Selecting the top 64 channels from
the held-out recipient's exact `delta * gradient` produces a positive semantic
shift on all 256 layer-pair cells. Its mean shifts are 0.285, 0.403, 0.390, and
0.479 across Layers 0--3. The derivative agrees with an independent autograd
path to better than `8.1e-7` relative error.

Yet two contexts for the same feature share only about two of those 64 channels.
Same-feature Jaccard is 0.014--0.018, modestly above different-feature
0.009--0.012 and random 0.007. This gives the shortest current description:

```text
feature label                  weak prior over useful channels
actual activation change      available evidence in this edit
recipient suffix Jacobian      how this context will read the evidence
elementwise contraction        the small causal coalition
```

The feature is therefore neither one neuron nor an arbitrary dense cloud. It is
a stable *selection problem* whose answer is sparse but changes with the input.
The dynamic tensor program explains how to compute that answer.

## The context-specific selector has a short executable program

The failure of global neuron lists does not imply that every intervention needs
the full opaque model. Retaining all native state-conditioned MLP Jacobians but
only the first eight downstream attention Jacobians recovers 0.951--0.988
median exact-gradient cosine from source Layers 0--3. Its prospective top-64
channel patch preserves 93.3--98.1% of the exact selector's finite effect and
has the correct direction on every one of 256 tested layer-pair cells.

Under a joint gradient, tail, support, direction, and effect gate, the minimum
tested attention window shrinks from 8/8 blocks at source Layers 0/1 to 6/5 at
Layers 2/3. The stable algorithm is therefore:

```text
declared output effect + donor activation change + recipient forward states
    -> exact bilinear MLP Jacobian recurrence
    -> short near-source attention correction
    -> rank native channels by predicted delta × gradient
    -> small finite-causal patch
```

This is prospective channel selection for a specified task. It is not yet
automatic discovery of which semantic question to ask.

## Verb ontology versus implementation ontology

The successful attention correction factorizes exactly into Q1/K1/Q2/K2/value
roles. For same-position semantic effects, combined QK routing preserves
95.2--98.4% of the full near-window finite effect. Query roles carry most of
that signal; keys and value/OV are corrections. Both tensor QK factors are
needed.

A frozen native-head list is much less canonical. Half of the 72 near-window
layer-head edges retains 88.6--94.7% of the effect, but random half-edge sets
are close and independent rankings are unstable. Relative depth is the stable
coordinate: the first two downstream blocks rank highest at every source, with
panel correlations far above early head-index correlations.

This gives different legitimate representations for different uses:

| Object | Best use | Main limitation |
|---|---|---|
| Supervised forward feature | Discover human-readable regularities | Predictive, not necessarily causal |
| Output/sensitivity direction | Ask a precise behavioral question | Can encode any arbitrary objective |
| Endpoint channel coalition | Minimal reliable intervention payload | Recipient- and edit-specific |
| MLP + QK near-window program | Recompute that coalition prospectively | Needs objective and forward states |
| Frozen head subset | Cheap approximate execution | Weakly canonical and redundant |

There is therefore no single universally “correct” feature ontology. Correctness
means predicting the relevant observations and interventions for the declared
use case; the tensor program links these levels without pretending they are the
same object.

## Multiple sources and interactions compose without polynomial expansion

The recurrence now works across two edited positions and through source Layers
0--3. For a joint edit, concatenate the source-product changes, run the same
state-conditioned suffix VJP, and rank the flattened `delta × gradient` vector
under one shared causal budget. With single-edit strengths balanced, the
next-twelve-attention program reaches 0.967--0.997 median exact-gradient cosine,
recovers 0.695--0.910 of the exact top-64 support by Jaccard, and preserves the
finite effect in every layer/pair cell.

The stronger result concerns a genuine interaction. Let `A` be an earlier edit
and `B` a later edit. Causal masking makes the Layer-0--3 product change caused
by `A` independent of whether `B` is present at the later position. The
interaction selector is therefore

```text
delta_A × (suffix_gradient_in_B_context - suffix_gradient_in_base_context).
```

Its finite target is the corresponding Möbius effect:

```text
(effect of patch A when B is present) - (effect of patch A in the base context).
```

Subtracting two next-twelve tensor programs recovers the exact differential
gradient at 0.967--0.997 median cosine and 0.695--0.910 top-64 Jaccard. Every
exact-selected patch has the intended finite interaction direction; the short
program retains 0.862--1.029 of that finite effect. Thus a conjunction can be a
relation between two executable programs rather than a scalar conjunction
feature stored in one neuron.

### Red team: behavioral balance is not circuit balance

Matching the two single-edit output norms raises the median minority-position
share of the top-64 support from 0.055 to 0.172, but does not force cooperation.
The later source alone reproduces about 97% of the sparse joint effect for
63+95 and 63+119. For 95+119, the best single source reaches only 74%, so both
sources are material. A forced 32+32 quota is sometimes worse than the natural
allocation, and the support-majority source predicts the stronger finite route
on 95% of sites.

This separates three quantities that should not be conflated:

1. how much each input changes behavior when edited alone;
2. how much sparse causal leverage each source has for a specified joint effect;
3. whether the effect of one source is modulated by the other source's context.

The third is the interaction circuit. It can be accurately reconstructed even
when the second is winner-take-most.

## Automatic objective discovery finds a sparse control basis, not privileged concepts

The stable rank-16 late output core was rotated without semantic labels. Varimax
finds axes with held-out vocabulary-loading kurtosis 2.960, versus 0.777 for the
native HOSVD axes and random-rotation p95 0.717. These axes are highly stable:
eight random half-vocabulary refits have median/minimum alignment 0.994/0.989.

But the semantic red team fails. Varimax vocabulary-label AUROC is 0.803 versus
0.828 for native axes and random p95 0.842. Its random-split semantic score trails
matched random p95 by 0.053 median. Natural target and top-1 labels are also no
better than matched rotations. We have therefore discovered a canonical sparse
*control* basis, not a canonical human-semantic basis.

This makes the use-case dependence concrete:

| Desired object | Defensible coordinate |
|---|---|
| Sparse vocabulary manipulation | Stable varimax output axes |
| Human semantic prediction | Complete output subspace plus supervised decoder |
| Minimal intervention for one example | Context-specific `delta × gradient` coalition |
| Multi-token interaction | Difference of state-conditioned tensor programs |
| Architectural explanation | MLP Jacobians plus Q/K/V roles and relative depth |

The same physical computation admits several correct summaries. A basis becomes
ontological only for the observations and interventions it predicts better than
matched alternatives.

## Metric and depth corrections to the current ontology

Two tempting shortcuts now have direct negative evidence. First, a stable sparse
late-output basis does not induce a stable sparse set of early neurons. Frozen
top-64 supports for residual-output directions capture only about 1.5--2.2% of
held-out upstream energy or contribution mass, while endpoint-specific supports
capture 11--19%. Second, the most extreme raw-logit Fisher axis is almost exactly
the leading eigenvector of the unembedding Gram matrix. Whitening the output
metric changes both the spectrum and the axes. “The output feature direction”
is therefore incomplete unless its metric is named.

The positive replacement is a metric-declared subspace plus a conditional
execution map. The residual-dual Fisher geometry has a stable four-dimensional
subspace (90.5% energy; split overlap 0.994), but the early channels executing
one of its directions change nearly completely with context. This is compatible
with the earlier feature result: what recurs is a behavioral question or
subspace, while the local coalition is selected by the current state.

## Layers 2--5: more context, not more neuron alignment

The 50-label census has now been extended from Layers 0--1 through Layer 4 on
16,384 discovery and 16,384 held-out FineWeb sites. For the 32 sufficiently
common labels the median 64-channel AUROC increases from 0.747 at Layer 0 to
0.779 at Layer 4. The same comparison restricted to positive/negative examples
with the *identical current token* rises from 0.558 to 0.622. Repeated embedding
injection therefore cannot explain the depth gain.

Single neurons do not catch up: median single-channel AUROC remains 0.60--0.64,
and no competent feature has a single neuron within 0.03 of its best coalition.
For example, `suffix_ion` improves from 0.607 at Layer 0 to 0.754 at Layer 4
using 64 channels, while its best single neuron is only 0.551 at Layer 4.
Layer-2 `colon_semicolon` reaches 0.816 with 64 channels versus 0.485 with one.
These are concrete examples of distributed contextual features rather than a
generic claim based on averaged probes.

The exact causal program has also been extended through source Layer 5. Write
`g_l` for an incoming adjoint and let the current-state local maps be
`J_M,l(x_l)` and `J_A,l(x_l)` for the MLP and attention paths. The recurrence is
implemented blockwise as

```text
u_l       = g_l + J_M,l(x_l)^T g_l
g_(l-1)   = lambda_l [u_l + J_A,l(x_l)^T u_l].
```

For the bilinear MLP, `J_M,l` is obtained directly by contracting the fixed
Left/Right/Down weights with the current normalized state; attention is split
exactly into Q1/K1/Q2/K2/value derivatives. No global polynomial is formed.
The next-eight-attention approximation reaches cosine 0.933 at source Layer 0
and 0.988 at Layer 5, while the exact role recomposition error is `8.32e-8`.

This makes the scaling claim precise. Computation can be represented as a chain
of low-level verbs whose parameters are fixed but whose realized operators are
state-conditioned. The chain scales linearly in depth for a fixed set of probes.
It does not yet provide a short global ontology, because the operator states and
native supports remain highly context-specific and moderately high rank.

## Direct search for an operator vocabulary

The bilinear MLP gives an unusually clean test. For normalized state `z`,

```text
J_MLP(z) = Down [diag(Right z) Left + diag(Left z) Right],
```

so the local Jacobian is exactly linear in `z`. Random output/input contractions
`u^T J_MLP(z) q` therefore define a weight-only sketch of the complete
third-order operator tensor. Combining that map with FineWeb state covariance
produces label-free operator-state modes.

This succeeds at the subspace level. With 4,096 contractions, independently
estimated top-64 subspaces overlap 0.836--0.864 through Layers 0--5. Their
held-out operator R2 is 0.410--0.489, narrowly above equal-rank state PCA. But
matched individual axes have only 0.47--0.55 cosine and weakly recurring
semantic profiles. The stable object is again a subspace, not a list of axes.

The decisive causal test is less favorable. Replacing the state conditioning of
Blocks 1--5 by rank-64 operator reconstructions yields source-gradient cosine
0.467 at Layer 0 and 0.801 at Layer 3. Rank 256 yields 0.735 and 0.906; rank 512
yields 0.884 and 0.957. Finite directions are usually correct much earlier than
complete gradients, so a successful edit sign is not enough to call the
mechanism reconstructed.

A suffix-aware alternative does not rescue a tiny basis. Weighting the operator
tensor by actual incoming residual-output adjoints gives stable high-probe
subspaces, but they tie the general Frobenius basis and state PCA on both their
own objectives and semantic transfer. Keeping raw adjoint magnitudes makes the
basis unstable and worse.

The resulting hierarchy is:

| Representation | What is stable | Appropriate use | Failure mode |
|---|---|---|---|
| Native neurons | Weak feature-biased membership statistics | Sparse endpoint edits | Identities change with context |
| Operator-state top-64 subspace | Projector under weight + FineWeb metric | Coarse local approximation | Only ~40--49% held-out operator R2 |
| Operator-state rank 256--512 | Medium-rank local/casual approximation | Faster approximate suffix analysis | Still loses early-source program detail |
| Exact local contraction rule | Weights plus current state | Mechanistic execution and prediction | Requires the actual state trajectory |

This is not “no ontology.” It says the shortest faithful early-layer ontology is
algorithmic: fixed bilinear tensors, RMSNorm geometry, QKOV routing roles, and
the current state. Global coordinates are lossy summaries whose validity depends
on the intervention budget and source depth.

## A prospective compositional language through Layer 5

The remaining circularity was that an operator basis reconstructed
`J_MLP(z)` only after seeing the same state `z` that defines it. A prospective
test now predicts each Layer-0--5 operator from signals available before its MLP:
the mixed residual and the context-routed attention write.

This works without expanding the global polynomial. For each block, discovery-
frozen PCA bases encode the two inputs, their reconstructions are added, and the
native RMS normalization produces the state that contracts the fixed bilinear
tensor. No decoder is fitted from these coordinates to the target operator.
Rank 64 per input recovers 80--94% of the rank-64 operator oracle's held-out
gain; rank 256 per input recovers 96--99%. Replacing the attention write with
one from another sequence at the same position collapses the result, showing
that the interface carries specific context rather than only position or scale.

The end-to-end causal result is deliberately less flattering. On semantic
adjoints, 64+64 coordinates give only 0.676--0.746 local response cosine and
0.475 source-gradient cosine when five approximate blocks lie downstream.
With 256+256 coordinates, local response cosine rises to 0.880--0.920 and
composed source-gradient cosine is 0.811/0.833/0.883/0.924 from source Layers
0--3. Finite edit direction is already reliable at the smaller rank, but exact
causal shape and magnitude require the larger code.

The current layer-spanning picture is therefore:

```text
compressed incoming residual (operand carried forward)
                    +
compressed attention write (context-specific routed update)
                    ↓
               RMS-normalized state
                    ↓ contracts fixed Left/Right/Down tensor
          state-conditioned local MLP operator
                    ↓
             repeat at the next layer
```

This is the compositional alternative to forming a seventh- or exponentially
higher-order symbolic polynomial. It scales linearly with the number of layers
for a fixed probe or intervention. It is also not yet a semantic ontology:
the useful coordinates are dense, individual axes are unstable, and roughly
512 input coordinates are needed for strong five-block causal fidelity. The
next ontology question is whether those dense coordinates factor into recurring
attention-source roles, grammatical state variables, or sparse nonlinear
features without losing the causal guarantees above.

## Attention resolves into routers and conditional typed edges

Exact head decomposition supplies one level below the medium-rank state
interface. The nine native head writes sum to attention within `4.52e-7` relative
error, but their allocation changes with depth. One head explains roughly
70--81% of the attention-dependent operator effect at Layers 0--1 and 97% at
Layer 5, while the best head explains only 24--28% at Layers 2--4. The middle
regime is multi-head rather than merely noisier.

This does not yield nine semantic modules. Initial one-head semantic probes kept
the complete residual and inherited nearly all of its predictability. After
conditioning on the residual, broad gains mostly vanish. Six frozen apparent
roles fail exact finite ablation on untouched FineWeb, and none reaches 0.60
causal AUROC at any removal scale from 10% to 100%. A stable head identity is
therefore not equivalent to a stable output meaning.

The more successful ontology lives inside heads. Normalize each head's source
transport against the value-norm-weighted sources it could causally read, and
its distance/source-type enrichment profile is highly reproducible: every head
at Layers 0--5 matches itself across data splits. Frozen examples such as
Layer-2 Head-4 reading colon/semicolon sources survive exact edge removal on new
FineWeb. Against equal-count sources matched on distance and value norm, that
route has 8.0x the local operator effect and 5.95x the final-logit KL. Similar
positive results occur for opening delimiters and suffix-bearing sources.

The positive result is sparse and conditional. Only three of eight declared
routes beat controls on 60% of individual sites, even though six have large
aggregate effects and five have bootstrap-robust downstream effects. The right
description is therefore not `head = concept`, but:

```text
head = reusable routing operator
typed edge = (source category, relative-distance/context, transported value)
natural computation = sparse context-specific mixture of typed edges
```

This supplies three valid, non-interchangeable ontologies:

| Question | Useful object |
|---|---|
| Reproduce attention exactly | Native heads and Q1/K1/Q2/K2/value contraction |
| Approximate the next MLP operator | Dense residual plus attention state coordinates |
| Explain a large human-legible conditional effect | Typed source edge within a head |

The next compression test is whether a fixed small vocabulary of typed edges
reconstructs full held-out attention behavior, or whether the positive typed
edges are only isolated interpretable exceptions within a broad residual route.

## A fixed routing vocabulary survives the full suffix

The isolated typed-edge examples do extend to a vocabulary. Every causal edge
was assigned one of 1,323 possible `(head, distance bin, source category)`
types. Discovery ranked these types by their effect on a weight-derived local
operator sketch; held-out evaluation froze the list and recomputed RMSNorm.
Thirty-two types recover 0.53--0.95 of the local attention-dependent operator,
and 128 recover 0.84--0.99. Layers 2--4 again require the broadest mixture.

This is not just local curve fitting. On 2,048 untouched FineWeb positions per
layer, the experiment replaced only the selected local attention write,
recomputed the layer's bilinear MLP, and ran the exact suffix. The 32-type
programs recover 0.70/0.84/0.52/0.40/0.55/0.76 of the output-KL effect across
Layers 0--5; 128 types recover 0.79/0.96/0.79/0.70/0.80/0.91. Local recovery
tracks downstream recovery with Spearman 0.901.

The negative result matters. A preregistered claim that 128 types would recover
at least 80% at every layer fails, most clearly at Layer 3. The vocabulary is a
graded approximation, not a solved circuit decomposition. The positive result
is also not merely a few semantic anecdotes: fixed programs generalize and
retain substantial causal behavior through the complete suffix.

There are now two defensible resolutions of the same attention graph:

| Cost being minimized | Better description | What it preserves |
|---|---|---|
| Number of executed native edges | Fine `(head, distance, category)` types | Selective causal intervention |
| Number of human-readable predicates | Closures such as head-distance or head-category | More behavior with a broader rule |

All three fine coordinates matter when the execution budget is held fixed. Yet
closing over any one coordinate usually improves behavior because more edges
are admitted. “The correct ontology” is therefore not a unique basis absent a
declared use case. The stronger invariant is the compositional grammar: native
heads route values; source category and distance condition those routes; the
resulting write changes a state-conditioned bilinear operator.

The next red team executes fixed programs in multiple layers simultaneously.
That distinguishes a genuinely compositional language from a set of local
descriptions that work only while every surrounding layer remains native.

## Closed-loop composition works, but only at the larger program scale

The multi-layer red team now executes each program at every position in a
progressively deeper Layer-0--5 prefix. Later QK scores, values, normalized
states, and bilinear MLPs are recomputed from the approximation's own trajectory.
This is the scalable compositional construction: apply the same local grammar
repeatedly, rather than symbolically expanding the network into a high-order
global polynomial.

At 128 types per layer, the construction is unexpectedly faithful. The complete
six-layer program obtains about 0.974 normalized KL recovery on independent
replication, with 0.185 nats absolute KL error, CE +0.174, and 0.799 top-1
agreement. Across four untouched panels its top-1 agreement is about 0.79--0.82.
An all-types program exactly reproduces the native trajectory, while random 128-
type programs collapse. Thirty-two types are not enough: they retain positive
normalized KL recovery but only about 25% top-1 agreement.

The tuple is genuinely bound. Randomizing source category while preserving
head-distance counts reduces joint recovery to 0.17--0.38; randomizing distance
gives 0.69--0.72; randomizing head gives 0.70--0.84. Moving each intact type set
to the wrong layer gives 0.55. Thus the executable object is specifically
`(layer, head, distance, category)`, not a bag of generally useful token filters.

There is also a simpler partial account. Selecting types only by absolute QK
mass gives 0.890 recovery, and including value norm gives 0.913, versus 0.975 for
the destination-operator score. The latter's preregistered +0.10 advantage is
not met. The 128 operator types keep just 12--38% of edge instances but capture
about half of transport magnitude. Most of the success is therefore structured
dominant-route pruning; the tensor-aware MLP destination adds the final material
increment in KL and raises top-1 agreement from 0.65 to 0.79.

The current compact picture is:

```text
many possible causal edges
        ↓ sparse concentration by signed bilinear QK routing magnitude
high-transport (layer, head, distance, source-type) routes
        ↓ destination projection + RMSNorm + bilinear MLP sensitivity
operator-relevant executable program
        ↓ repeat on the program's own evolving states
six-layer approximation with useful prediction fidelity
```

This is stronger than an activation atlas and weaker than a semantic source
code. It says which structured routes execute and compose, but the human token
categories are still imposed by us. The next ontology problem is to learn a
source-type partition from the model's value and downstream-effect geometry,
then test whether that learned vocabulary shortens the program or improves
transfer without sacrificing the exact causal controls.

## Model-derived variables: useful geometry without a unique dictionary

The obvious weight-only attempt fails. Equal-size spherical clusters of token
embeddings, value projections, or six layers of key/key2/value roles all trail
the 21 human token categories substantially in closed-loop behavior. Exact
embedding clustering does not repair the gap. Static weight similarity is
therefore a poor definition of substitutable source variables for this program.

Realized causal profiles work much better. Represent each token by its routing
magnitude over `(layer, head, distance)`, or by its effect on the receiving
normalized bilinear operator, and 21 learned categories reach 0.94--0.944 KL
recovery. That approaches the human program's 0.967. But independent fits have
low categorical agreement despite almost identical behavioral quality. The
model supports many good hard codes rather than one privileged token partition.

A nonnegative factorization makes the distinction explicit:

```text
token × routing-coordinate profile
        ≈
token × soft variable memberships
        ·
variable × read-function uses
```

Several read-function factors are recognizable as distance primitives reused by
many heads and layers: current token, previous token, 2--4 back, 4--8 back, or
long-range sources. This validates the proposed separation between a variable
and the functions that read it. Yet the factors are not sparse semantic atoms.
A typical token uses eight of 21 variables and a typical factor touches 102 of
378 layer/head/distance coordinates.

Two independent hard argmax views of these factors are nearly unrelated (NMI
0.053) but both give excellent six-layer programs: 0.962/0.970 recovery and
0.785/0.809 top-1. Soft memberships are more stable, but at 128 predicates their
program is much worse; 256 predicates are needed to reach 0.975 recovery and
0.778 top-1. The hard view is currently a better execution code, while the soft
view is a better statement of recurring geometry.

Searching the residual after human categories produces stable candidate axes,
including apparent tokenizer-boundary and Layer-0 routing distinctions. They
fail prospective held-out reconstruction, however. These are saved as discovery
objects with 50 examples per pole, not promoted to features.

## Ontology is relative to an interpreter and a cost function

The resulting MDL language separates:

1. **Variables**: token/state properties or learned membership vectors.
2. **Read functions**: layer, head, distance, and contextual guards. A
   previous-token head is one reuse of the `[1,2)` read primitive.
3. **Transforms**: value/output projections, RMSNorm, and bilinear tensor maps.
4. **Invocations**: bindings of variables, reads, and transforms in a context.
5. **Residual behavior**: KL from the native model.

These costs cannot be collapsed innocently. `Keep all edges` has a one-line
description and exact behavior but performs the full computation. An opaque hard
token table is sparse and accurate but unreadable. A human lexical grammar is
short only for an interpreter that already treats suffix, punctuation, and
whitespace predicates as cheap. A soft table describes stable geometry but has
millions of literal membership bits and dense fractional execution.

The current use-case frontier is therefore:

| Goal | Current best object |
|---|---|
| Human explanation | Human token variables + native head/distance functions |
| Sparse approximate execution | Hard 128-type programs |
| Higher-fidelity surrogate | Soft-256 or a larger hard program |
| Discover recurring geometry | Continuous routing profiles / soft factors |
| Exact causal prediction | Native state-conditioned QKOV, RMSNorm, and bilinear contractions |

Under one deliberately crude literal-table code, the best independent hard NMF
partition repays its opaque dictionary after about 11.6 million evaluated
tokens; soft-256 crosses the human program around 96.9 million. These are
sensitivity calculations, not universal constants. The next scientific step is
grammar induction: find short predicates that generate the stable memberships
and read patterns, then test those predicates prospectively against literal
tables at matched behavior and execution.

## Readable grammar search exposes a variable--invocation tradeoff

A learned readable tree is highly reproducible, but its main rules are not a
new ontology: frequency, alphabetic status, common function-word classes,
punctuation, token length, and capitalization largely refine the existing human
vocabulary. At a fixed 128-route budget, increasing from 21 to 42 leaves causes
recovery to collapse from about 0.94 to 0.72 because one useful route is split
across many declared types. Scaling the route budget to 256 repairs recovery to
0.954.

The decisive red team gives the original 21 human categories the same budget.
Human-21/256 reaches 0.992 recovery and 0.893 top-1, beating readable-42/256 by
0.038. Thus the scaled grammar result is not evidence for newly discovered
variables. It is evidence that declaration granularity and invocation count are
coupled costs.

Simple context refinements tell the same story. Previous-token role is genuine
signal—it beats a matched random context code by a wide margin—but crossing it
with every source category wastes the sparse route budget. Selective splits are
better than full crossing yet do not beat the static human categories. For the
current attention program, these contextual predicates are best treated as
guards on invocations rather than new variables.

## Variables, functions, and realized effects are different objects

The next atlas holds the readable source variables fixed and asks how they are
used. Let a native `(layer, head)` define the current-value OV operator, while
distance and category define read invocations. The resulting graph is
many-to-many: a used head reads a median six variables, while one variable has
about four effective head functions. Functions that read the same variable have
nearly orthogonal OV maps, yet routing-profile similarity predicts OV-map
similarity even after controlling for layer.

This yields a stronger program grammar:

```text
source variable
      + contextual read guard (layer, head, distance)
      + argument value transported from the source state
                    ↓
             native OV operator
                    ↓
 destination state + RMSNorm + bilinear MLP + suffix
                    ↓
          context-specific realized output effect
```

A finite test makes the final distinction necessary. Four Layer-0 heads all
read the same `function_word` variable, and removing each route family produces
material, mostly nonaligned downstream logit effects. But near and far
invocations of one head are no more directionally aligned. Therefore “function”
cannot mean a fixed vector added to the residual stream. It means a stable
operator applied to arguments. The native head is a valid exact operator
boundary; its individual calls remain context-dependent.

This is good news for the proposed MDL view. A previous-token head is not a
special ontology category. It is one compact function declaration plus many
bindings to variables and arguments. Readability should be scored on the
declaration, the binding rule, and prospective predictions of its calls—not on
whether every call points in the same logit direction.

## Reusable function templates exist at one resolution but fail as substitutions

Including the persistent Layer-0 value bus gives an exact two-argument value-side
operator for every early head:

```text
(current normalized residual, persistent Layer-0 value bus)
                         ↓
             head-specific value/write map
                         ↓
                  residual direction
```

This family has several different intrinsic sizes. Twenty-two linear templates
recover 90% of raw operator weight energy, while natural response signatures
need 45 of 54 dimensions. The pairwise natural geometry is nearly identical on
two independent FineWeb panels, but an eight-way categorical clustering is not
stable enough to call canonical. Thus “22 weight templates,” “45 natural
response coordinates,” and “54 native heads” are all correct answers to
different compression questions.

The leading output directions look much more recurrent than random vectors.
Matched controls show this is learned: it exceeds random vectors inside the
same output subspaces and permutations of input maps across output projections.
But global covariance whitening removes most of the population advantage. The
best description is a shared anisotropic output geometry with a few outlier
pairs, not a clean dictionary of output verbs.

The strongest pair, L4H2/L5H7, supplies the decisive causal red team. Their
directions have raw cosine 0.705, whitened cosine 0.350, and natural-response
cosine about 0.46. Nevertheless, transplanting the directions is asymmetric
and not functionally equivalent. One direction is worse than angle-matched
learned controls in the earlier head; in the later head it beats controls but is
still substantially worse than deleting the component.

This separates three notions that should not be collapsed:

| Object | Valid use | Invalid inference |
|---|---|---|
| Shared output geometry | Coordinate prior, visualization, low-rank weight description | Heads are interchangeable |
| Natural response similarity | Group functions with similar corpus behavior | Same intervention in a new layer |
| Finite causal substitutability | Replaceable executable module | Universal semantic identity |

The result argues for typed functions. A declaration needs not only an output
direction but an input coordinate, gain/scale, routing guard, argument
distribution, destination layer, and downstream metric. Two functions may share
one primitive while differing in the remaining fields.

## Function compatibility is a destination-conditioned quotient

The strongest full-operator pair reveals a more useful relation than clustering.
L2H5 and L3H8 were selected before finite evaluation because they jointly have
the strongest bus-aware operator resemblance and replicated natural-response
similarity. Exact transplantation then gives an asymmetric result: the earlier
L2H5 operator is the best of all 53 norm-and-angle-matched replacements for the
later L3H8 on both text halves, while L3H8 is an ordinary replacement for
L2H5. The relation persists from scale 0.05 to 1.0 and is quadratic at small
scale.

This is not a general similarity law. Across the full candidate bank, raw
operator and natural-response similarities have essentially no correlation with
finite compatibility; output-direction similarity is only weakly predictive.
The successful pair is a genuine outlier, but intrinsic geometry cannot rank
the remaining substitutes.

The correct object appears only after composing with the destination:

```text
candidate operator difference
        ↓ native target router and natural arguments
destination write perturbation
        ↓ target state, later attention/MLPs, unembedding
next-token logit tangent
        ↓ Fisher metric of the native output distribution
destination-conditioned compatibility geometry
```

For L3H8 this 53-candidate Fisher geometry is highly reproducible across text
halves. One sensitivity mode explains more than half its energy, and L2H5 has
the smallest loading of every candidate on that mode. The leading eight-mode
subspace is stable, although 90% energy requires 13--18 modes. Reversing the
target reproduces the pattern even more compactly: one dominant mode and rank
5--9 for 90% energy. Raw weight geometry correlates only weakly with either
destination metric.

An unrelated L4H1 target preserves the stable dominant direction and stable
leading subspace but needs rank 27 for 90% energy. Its geometry-selected L5H8
replacement ranks fourth/fifth rather than first. Thus the general regime is
not “all compatibility spaces have rank eight.” It is “one strongly dominant,
reproducible sensitivity direction plus a destination-dependent corrective
tail.”

Trying to name the L3H8 dominant direction supplies another warning. It retains
69--76% of Fisher energy on later text, yet a continuation/speech interpretation
seen in one contiguous document does not survive randomly scattered FineWeb.
The scattered panel instead emphasizes capitalization, suffixes, token
continuations, ellipses, and formatting. The mode is causally stable while its
simple semantic story is not.

This gives a principled answer to “multiple correct ontologies”:

| Equivalence relation | Correct object | Typical size here |
|---|---|---:|
| Same implementation | Native head/value/QK tensors | 54 functions |
| Similar raw value-side map | Frobenius operator basis | rank 22 for 90% |
| Similar natural response | Corpus response geometry | rank 45 for 90% |
| Similar output direction | Whitened residual direction family | continuous; few outliers |
| Substitutable at one destination | Downstream Fisher quotient | one dominant mode plus rank 5--27 tail |

None is universally “the true feature basis.” The destination-Fisher quotient
is the right ontology for safe replacement and causal compression. The raw
operator basis is appropriate for weight compression. Natural response geometry
is appropriate for descriptive behavior. Human variables and typed routes
remain preferable for readable explanations.

## The tensor network predicts new destination features prospectively

The destination quotient is no longer merely a post-hoc measurement. For three
previously untested heads spanning Layers 0, 2, and 5, candidate operator
differences were propagated through the checkpoint's exact analytic Jacobian.
Only afterward were 1% finite replacements evaluated. Analytic Fisher geometry
and diagonal ranks agree with the finite effects at approximately 1.000 on both
text halves; raw operator norms are essentially uninformative.

This supplies an executable definition of a feature:

```text
operator difference
    -> native target router and arguments
    -> destination-conditioned state tangent
    -> repeated attention/MLP transfer operators
    -> output-Fisher response
```

It also falsifies a tempting readable simplification. The attention-free path
plus every path with one downstream attention use is insufficient, especially
for early targets. Exact expansion by attention order initially becomes much
worse: individual order groups grow and later cancel. Full all-order
recomposition is numerically exact, with 90% response fidelity reached only at
orders 10--12 from Layer 0, 8--9 from Layer 2, and 5 from Layer 5.

The high order is not evidence that the model explicitly executes a twelve-step
symbolic chain. It is a warning that polynomial path expansion is a poorly
conditioned coordinate system. The compact program is instead the recurrence

```text
delta x_(l+1)
    = B_l [ lambda_l delta x_l
            + A_l(state_l, bus; lambda_l delta x_l) ]
```

where `A_l` is the exact dynamic two-QK attention Jacobian and `B_l` is the
residual-plus-bilinear-MLP Jacobian. Local attention corrections are modest,
usually 14--26% of the identity branch norm, and no individual layer has extreme
cancellation. Their repeated composition creates the ill-conditioned expanded
paths. Layerwise conditioning profiles are highly stable across text halves and
are strongest immediately after the perturbed destination.

This distinguishes two notions of simplicity:

| Representation | Short? | Numerically stable? | Good use |
|---|---:|---:|---|
| Enumerated causal paths | No | No for early layers | Exact algebra, limited debugging |
| Layerwise transfer recurrence | Yes | Yes | Execution and causal prediction |
| Destination-Fisher quotient | Yes after aggregation | Yes | Replacement ranking/compression |
| Human variable/function program | Potentially | Not yet established | Scientific explanation |

The remaining gap is semantic rather than causal: we can now predict which
operator differences matter, but have not yet named the transported tangent
variables in a compact human vocabulary.

## The transported state uses rotating, moderately broad charts

The next attempted compression was a static rank-64 residual vocabulary shared
across targets and contexts. It fails both locally and causally. Held-out local
energy ranges from 0.17 to 0.63 and final Fisher fidelity from 0.10 to 0.43.
Target-specific PCA does not rescue it. A matched native-activation PCA performs
similarly and sometimes better, showing that the modest success is largely
generic residual anisotropy rather than a discovered causal dictionary.

A same-context oracle changes the result. Fitting a residual chart to each
context's candidate tangent bundle raises rank-64 Fisher fidelity to 0.71--0.93,
at least 0.20 above the static basis in every tested cell. The leading charts
have only 0.10--0.23 mean squared overlap between contexts. This is evidence for
an input-conditioned coalition/chart view rather than neuron-aligned or static
subspace features.

But the stronger “small rotating basis solves representation” claim is false.
Local rank90 varies from 53 to 397, and even rank 128 yields only 0.83--0.97
Fisher fidelity. The correct present statement is:

```text
static neuron or residual dictionary       rejected at rank 64
context-conditioned residual chart         strongly better
uniform tiny dynamic chart                 rejected
destination-conditioned behavioral quotient compact and predictive
```

These are not contradictions because they optimize different equivalence
relations. The intermediate state must preserve information for many possible
future uses; the destination-Fisher quotient identifies only distinctions seen
by one requested replacement and output metric.

## A prospective chart generator exists, but its axes are not canonical

The oracle chart's implicit description cost can be reduced. A
candidate-independent generator starts from the native next-token distribution,
constructs probability-weighted score adjoints, and transports them backward
through exact attention/MLP VJPs. Summing their residual-axis covariance across
positions yields a suffix-Fisher chart at each boundary.

This chart never sees the 53 operator-replacement tangents used for evaluation.
At rank 64 it preserves 0.63--0.85 of their final Fisher response and beats the
static residual basis by at least 0.20 in every target/boundary cell. Rank 128
reaches 0.74--0.93. Thus native state plus tensor weights and a declared output
metric can generate a useful dynamic coordinate system prospectively.

The literal coordinates remain noncanonical. Independent stratified Fisher
sketches overlap by only 0.44--0.60, even with more exact top-token coverage.
Yet rank-64 causal fidelity is highly reproducible across sketches (Spearman
0.958; maximum absolute change 0.035). The stable object is therefore closer to
a behaviorally equivalent subspace family or quotient than a fixed list of
directions.

This supplies another case where “correct ontology” depends on use:

| Question | Appropriate object |
|---|---|
| Which literal residual axes recur? | No stable rank-64 answer yet |
| Which context-conditioned directions can affect outputs? | Suffix-Fisher chart |
| Which replacements are behaviorally interchangeable? | Candidate destination-Fisher quotient |
| Which variables are understandable to people? | Still requires semantic primitives |

## A readable variable/function circuit survives prospective red teams

The first attempt to name late read functions failed in the right way. On
nearly 200,000 FineWeb tokens, route profiles for Layers 7--10 are exceptionally
stable and almost rank one. But human source categories are slightly worse than
frequency bins and frequency-preserving shuffled labels. Rich readable trees
predict extreme effects, yet add essentially nothing beyond distance, position,
and frequency. Stable routing geometry is therefore not automatically semantic
structure.

Prospective relation tests on later text isolate two exceptions. L9H8
selectively reads newline/period boundary states; L7H8 and L9H7 selectively read
an adjacent function-word relation. Both survive distance and frequency
matching and exact finite edge removal through all remaining layers. Attractive
proper-name, long-range-capitalization, repetition, and induction stories fail.

The adjacent-function-word family demonstrates typed reuse. L7H8 and L9H7 use
the same address variable but have different OV maps, low write-subspace
overlap, and nearly orthogonal downstream effects. The reusable object is not
the whole head. It is the relation `adjacent_function_word`, bound to two
different transforms.

L9H8 yields the first strongly readable transported variable. Its weight mode
suggests US geography versus British spelling. Observational held-out probes,
finite edge removal, and a single-token counterfactual all confirm a
US/UK-spelling-style state at boundaries. The strongest red team swaps a cue
spelling while preserving the rest of the prefix and evaluates four disjoint
spelling pairs. Across three pair folds and two text splits, the native model
moves correctly in 95--98% of examples and the single L9H8 boundary edge moves
correctly in 67--78%. The declared contrast beats 31 random balanced contrasts
in every cell. A same-head non-boundary edge carries about thirty times less
signed shift; another head at the boundary moves weakly in the opposite
direction.

The resulting mechanistic decomposition is:

```text
prefix spelling evidence
        -> contextual style variable at a boundary residual state
        -> L9H8 boundary address and value transform
        -> context-conditioned suffix computation
        -> altered probabilities for future US/UK spellings
```

This does not make L9H8 a monosemantic “dialect head.” The same native head can
have other routes and consumers, and its final output-effect subspace remains
broad. It establishes something more precise: one typed invocation of the head
transports a readable variable with prospective, finite, and counterfactual
evidence.

## Updated overall ontology

The accumulated evidence supports a hierarchy rather than one universal feature
basis:

| Level | Object | Evidence/use |
|---|---|---|
| Variable | Contextual information such as spelling/style state | Supervised and counterfactual prediction |
| Address function | Boundary read or adjacent-token relation | QK/value routing and matched edge tests |
| Payload transform | Native OV/write subspace bound to the variable | Weight geometry and cross-split write capture |
| Invocation | Variable + address + transform at one layer/context | Exact finite intervention |
| Consumer quotient | Output distinctions relevant to a declared task | Destination/output Fisher or logit contrast |
| Behavioral surrogate | Any shorter program preserving that quotient | CE/KL/fidelity evaluation |

Neuron, head, residual direction, and semantic concept are therefore not
competing answers to one question. They are possible boundaries at different
levels. A useful reverse-engineering account states which boundary it uses,
which equivalence relation defines correctness, and what residual behavior its
short description leaves unexplained.

## A complete readable route now connects lexical evidence to output behavior

The spelling/style result is no longer only a readable variable at one late
head. Exact upstream decomposition, native edge replacement, and downstream
mediation expose a short circuit:

```text
cue token's persistent Layer-0 value code
  + smaller contextual-value and QK reinforcement
        -> L8H2 direct cue-to-boundary edge
        -> contextual boundary value
        -> L9H8 boundary-to-destination edge
        -> spelling-sensitive suffix consumers
```

The numerical separation is unusually sharp. At Layer 8, Head 2 supplies
`98--99%` of the builder projection and the cue source supplies approximately
all of its relevant effect. At Layer 9, swapping only the contextual value is
sufficient, while QK changes slightly oppose the original whole-prefix change.
A fully finite L8H2 edge edit changes disjoint spelling targets in `84--91%`
of examples; freezing L9H8's boundary edge removes `51--58%` of that effect,
whereas matched head and source controls remove effectively none.

Tensor factorization also prevents a misleading semantic gloss. L8H2 does not
manufacture an abstract dialect scalar from scratch. Roughly `68--73%` of its
finite effect comes from the persistent Layer-0 lexical value bus, `7--8%`
from its contextual value, and `19--24%` from QK changes. These factors
cooperate. The circuit turns lexical token evidence into a contextual style
state; the style abstraction belongs to the composed route and downstream
equivalence class, not necessarily to one native tensor factor.

The circuit transfers to eleven spelling pairs excluded from discovery in
both directions—new cues to old targets and old cues to new targets. That
supports a family-level program rather than a twelve-item lookup table. It
does not establish global monosemanticity: both heads can implement other
invocations, and nearly half of the upstream spelling effect reaches the
output through consumers other than the tested L9H8 edge.

## The terminal consumer is operator-sparse and position-distributed

The downstream remainder is now localized. A complete Layers 9--17 screen and
terminal source factorial show that it is not a repeated boundary-head chain
and not a final-MLP computation. At the last residual boundary, `66--74%` of
the spelling effect is already in carried destination state and `27--35%`
enters through final attention; the final MLP contributes approximately zero.

The apparently broad final-attention branch then collapses under a
head-by-source atlas. L17H2 reading post-boundary positions supplies `87--89%`
of positive nonboundary attention allocation. Freezing that native edge group
removes `19--24%` of the complete effect, while identical source relations in
H1 and H6 remove approximately zero. The state is nevertheless distributed
over positions: no fixed distance bin supplies more than `27%`.

The most compact current route is therefore:

```text
lexical spelling code
  --L8H2 cue read--> boundary style state
  --L9H8 boundary read--> destination-carried state
  --residual transport + L17H2 reads of post-boundary echoes--> output quotient
```

This resolves an apparent contradiction. The final consumer is sparse as an
operator (`head × relation`) but distributed as an argument field (many source
positions). Neuron/head alignment and distributed representation are not
opposite global regimes; they can describe different tensor indices of the
same computation.

A subsequent value-subspace test sharpens and partially retracts the semantic
gloss. The post-boundary H2 value change is genuinely almost scalar: a
discovery-frozen rank-1 direction explains `73--83%` of its raw energy and more
than `99.8%` of its finite spelling effect, including on eleven new spelling
pairs. But this direction is not spelling-specific. Sixteen unrelated lexical
swaps produce a rank-1 axis with cosine `0.959`; cross-family explained energy
is `0.51--0.71`.

The corrected route is therefore:

```text
semantic spelling evidence upstream
    -> boundary-specific typed transport
    -> generic low-rank edit field over later positions
    -> L17H2 terminal read
    -> spelling quotient at the output
```

This is an important negative result for semantic tracing. Causal membership
in a spelling circuit does not imply that every intermediate variable is a
spelling representation. Generic transport modes can carry many semantics;
the semantic identity can reside in the source intervention and destination
readout while the middle code is shared.

The generic field is itself created by a simple native tensor component. Before
the Layer-17 H2 value projection, rank 1 explains only `21%` of spelling
residual edits and `7%` of unrelated edits. The H2 `c_v` matrix maps those broad
directions into value axes that are `78%` and `39%` rank one. Its leading
left-singular vector predicts the spelling value axis at cosine `0.955`; using
that weight-only direction in finite forwards recovers `99.6--99.8%` of the
value-dependent spelling effect, including new lexemes.

This partially validates the random-matrix-outlier strategy. L17H2 has a clear
top-singular/RMS outlier (`1.721` versus a Gaussian edge of `1.333`), and the
outlier is the operative compression direction. But it ranks only `60/162`
globally and second within Layer 17. Weight outliers are therefore candidate
functions, not self-labeling circuits. The crossword strategy remains
necessary: module importance narrows the layer, relation analysis narrows the
head invocation, and the outlier supplies the short value coordinate.

## Natural edits reveal two terminal representation regimes

The occupied-outlier rule was next tested without injecting the known spelling
circuit and without supplying a human semantic output axis. Natural token swaps
were propagated through the full model, and each Layer-17 head was scored
against the model's own changed next-token distribution.

The rule transfers but the operation does not remain fixed. Spelling edits
select H2 on discovery and held-out data. A pooled set of unrelated swaps
selects H6. Both heads contain real value-matrix spectral outliers, but rank only
`60/162` and `73/162` globally. Weights supply candidate functions; occupancy,
relation, and destination type choose the invocation.

The attractive pooled-H6 interpretation then fails under disaggregation. H6 is
positive on `14/16` individual pairs but top-three on only `8/16`; the winning
head varies across eight of nine heads. Pair-frozen head choices also transfer
poorly. H6 is therefore a useful average component, not a universal lexical
edit primitive.

Per-example and joint-coalition tests expose a sharper regime distinction:

| Panel | Stable single head | Median complete-head support | Joint heads for 90% terminal-branch fidelity | Leading weight mode preserves head order |
|---|---:|---:|---:|---:|
| spelling family | H2 on `81--86%` | `1.35--1.60` | `2` | yes (`r=0.98`, winner `89--90%`) |
| heterogeneous swaps | no head above `25%` | about `3.0` | `5` | no (`r=0.52--0.64`, winner `30--45%`) |

The output-conditioned tensor selector remains nearly exact for both regimes,
but that is an explanation of a declared behavior, not autonomous semantic
discovery. Routing magnitude alone is already sufficient for almost all of the
spelling-family positive effect; on heterogeneous edits it loses substantial
effect and does not reliably identify the winning head.

This suggests that feature organization is conditional on the population and
equivalence relation. A coherent semantic intervention family can admit a short
native operation program. A mixture of unrelated edits requires a conditional
coalition even in the same layer. Pooling the latter can manufacture a stable
average head that is not the mechanism for most individual examples.

The joint-head assay has a deliberate scope limit. It reconstructs the final
post-boundary attention branch, whose mean alignment fraction with the complete
natural edit is only `7%` for spelling and `2--3%` for unrelated swaps. The
branch has positive cosine with whole-model behavior but is not a global circuit
replacement. Its all-head minimum relative closure gate also fails because
float32 summation differences near `4e-5` are divided by branch energies as low
as `4.5e-12`; the failure remains recorded rather than silently relabeled.

The updated working ontology is therefore distribution-aware:

```text
native tensor modes             available verbs
counterfactual state + relation occupied invocations
downstream Fisher/logit quotient behaviorally relevant invocations
joint finite reconstruction     sufficient coalition under that quotient
semantic/counterfactual family  candidate human variable
```

The same weights may support a monosemantic-looking program on one coherent
family and a multiplexed coalition on a heterogeneous population. These are not
contradictory global claims about the model; they are different conditional
program lengths.

## Correction: semantic coherence, causal transfer, and branch localization are separate gates

The first spelling-versus-heterogeneous comparison suggested that a coherent
semantic family might generally produce a shorter terminal program. A locked
replication on five predeclared lexical families rejects that simple rule.
Sentiment, gender, tense, scale, and polarity all require median `4--6` heads
for the Layer-17 post-boundary attention branch; none has a stable majority
head, and leading weight modes do not preserve complete-head ordering.

That negative result initially had a serious capability confound. The same
families had already failed state-delta recurrence and relation-direction
transfer, so a human label did not guarantee one model variable. A new disjoint
cross-pair test resolves the confound only partially:

| Family | Correct disjoint target shift | Terminal attention k90 |
|---|---:|---:|
| spelling | `95--98%` | `2` |
| sentiment | `79--86%` | `5.5` |
| gender | `84--86%` | `4.5` |
| tense | `69--81%` | `5.0` |
| increasing scale | `39--46%` | `4.5` |
| polarity | `54--74%` | `5.0` |

Sentiment, gender, and tense therefore are genuine transferable variables under
this behavioral test, yet they do not use spelling's sparse terminal-attention
organization. Transfer strength and terminal head count correlate only
`-0.433`, failing the locked threshold.

The decisive correction comes from terminal source localization. Fresh
Layer-17 attention supplies `38%` of the spelling contrast but only `8%` of
sentiment, `3%` of gender, and `2%` of tense. Their carried residual shares are
`80%`, `70%`, and `94%`; gender additionally assigns `27%` to the final MLP.
Across all families, fresh-attention share correlates `-0.919` with attention
head `k90`.

The broad head coalitions therefore mostly describe small generic side branches,
not the main representation of those semantic variables. Spelling is special
because the audited final-attention interface materially refreshes its
document-level state. For the other validated variables, reverse engineering
must follow the carried residual backward before asking whether their primary
implementation is neuron-aligned, subspace-aligned, or multiplexed.

This yields three gates for any feature-organization claim:

```text
1. capability: does the model exhibit transferable behavior for the variable?
2. localization: does the audited branch materially carry that behavior?
3. organization: within that material branch, what coalition is sufficient?
```

Skipping gate 1 mistakes human categories for model variables. Skipping gate 2
mistakes generic collateral computation for the feature's implementation.
Only after both pass does neuron/head/subspace sparsity answer the intended
representation question.

## New synthesis: semantic variables compile into stable verbs and low-rank payloads

The carried-residual question is now resolved far enough to distinguish several
representation regimes. Natural spelling, sentiment, gender, and tense edits all
transfer to disjoint lexical targets, but they become destination-aligned at
different depths. By Layer 14, one prediction-position vector recovers `0.51`,
`0.87`, `0.80`, and `0.62` of their complete effects. Equal-token-budget controls
show that late positions matter, while a unique post-boundary relay field does not.

Exact block factorials then separate three meanings of “the feature is in this
layer”:

```text
carried/reinjected destination state
        + fresh attention read
        + realized bilinear-MLP write
        -> next destination state
```

Gender has a sharp L12 attention construction pulse (`0.444` projection). Spelling,
sentiment, and tense cross 50% destination sufficiency mostly through carried state,
although they have earlier or later attention pulses. The final layer again differs:
spelling uses a large fresh-attention correction, while gender uses a material final
MLP correction.

Four frozen attention pulses reduce to strikingly short native programs:

| Variable/site | Frozen heads | Held-out branch fidelity | State fidelity | Interpretation |
|---|---|---:|---:|---|
| spelling L9 | H8 | 1.000 | 0.950 | head dominates the state change |
| sentiment L9 | H7+H1 | 0.945 | 0.547 | output-selective subspace |
| gender L12 | H4 | 1.000 | 0.954 | head dominates the state change |
| tense L15 | H5 | 0.993 | 0.681 | output-selective subspace |

The state control is essential. A head that reconstructs the entire attention delta
will score well on many output contrasts; that does not make the head semantically
monosemantic. Sentiment and tense supply stronger evidence of semantic selectivity:
their true contrast exceeds the 95th percentile of matched random vocabulary
contrasts by `0.185` and `0.261` despite incomplete state reconstruction.

The tensor factorization supplies a shorter common account. Across all four programs,
the cue swap barely changes QK routing; almost all program credit goes to changed
values. The heads behave like stable read verbs whose arguments are contextual
payloads. Those payloads use different relational operands: direct cue reads for
gender and tense, a post-boundary field for sentiment, and a broad between/boundary/
post-boundary relay for spelling. This retracts the overly narrow natural-spelling
story in which the latest boundary alone stores the variable. The boundary edge is
a real branch, not the complete computation.

The payload itself is often extremely short in weight coordinates. Leading singular
modes of the native OV maps need rank `2/16/1/1` per head for spelling, sentiment,
gender, and tense. Random nonzero singular modes fail badly at the same budgets.
This gives a concrete hierarchy:

```text
semantic edit
  -> distributed contextual state and source-role field
  -> stable head/read verb
  -> one or a few native OV payload modes (except broader sentiment)
  -> destination-aligned variable
  -> carried state plus later corrections
  -> output behavior
```

No single basis is “the true feature basis” at every boundary. Source-role fields are
best for operand localization; heads are best for read operations; OV modes are best
for payload compression; destination residual directions are best for late semantic
interfaces; neurons remain appropriate for exact implementation coordinates.

The architecture adds one more legitimate ontology. Tense's L15H5 payload allocates
roughly half its effect to current values and half to the persistent Layer-0 value
bus. A carrier can therefore be weak when intervened on alone yet material inside a
specific downstream tensor contraction. Whole-stream sufficiency and invocation-level
factor credit answer different questions.
# Update: label-blind weight motifs are local routing verbs, not global head types

The new label-blind circuit pass represented all 162 attention heads by symmetry-invariant random-function sketches of their bilinear address and value-bus-aware write operators. Independent pairwise geometries replicated (`r=0.812`), but forced global partitions did not (best ARI `0.153`). Stable structure was local: mutual top-three-neighbor edges produced 23 nontrivial components, with 81 heads isolated at this threshold.

The persistent layer-0 value bus was a major architectural confound: `32.8%` of native motif edges shared head index versus a `10.6%` random-pair baseline, and only `44.8%` of edges survived removing the bus. Semantic interpretation was therefore restricted to bus-robust components.

Two held-out roles replicated. `L2H5/L3H5/L3H8` route prior exact occurrences (`11.12×` QK-address enrichment; `0.84×` OV-write enrichment). Two early-layer families route the immediately previous token (`20.44×` and `20.06×` address enrichment; about `1.00×` write enrichment). These are clean verbs whose apparent object-level diversity comes from reuse across contexts. A proposed multiscale-boundary family failed the stronger uniformity and magnitude-equalized causal tests and remains split/unexplained.

Full interpretation and red teams: `artifacts/bilin18_tensor_decomposition/BILIN18_UNSUPERVISED_CIRCUIT_INTERPRETATION.md`.

## Stronger controlled anchors: features are invoked operations, not bare modes

The four controlled edits now supply a compact ground truth for evaluating the ontology: spelling, sentiment, gender, and tense. Exact destination accounting finds different construction pulses: spelling at L9H8, sentiment mainly L9H7 plus L9H1, gender at L12H4, and tense at L15H5. These frozen programs reproduce held-out construction effects with fidelity `0.9998/0.9447/0.9999/0.9933`.

This alone was not enough: spelling and gender heads also preserve many random state contrasts. The stronger family-swap red team holds the program and its low-rank OV basis fixed while changing the controlled edit. Matching-family output fidelity is `0.999/0.949/1.000/0.998`; every program beats its best non-target family by `0.46--0.86` in output space. Thus the result is not just that a high-energy head can compress arbitrary changes.

The simplest current feature object is:

```text
source-role condition
  + stable attention read verb
  + occupied low-rank OV payload mode(s)
  + downstream output quotient
  = invoked semantic operation
```

Each term matters. A head alone is too generic, a singular mode alone has no semantic name, and an output direction alone omits how the variable was read. For spelling, gender, and tense, one or two native modes are nearly sufficient; sentiment needs a broader rank-16 payload. Equal-rank input bases are mostly orthogonal across families, while the Layer-0 value bus produces genuine shared carrier effects—especially for tense. This is structured multiplexing rather than either one-neuron monosemanticity or arbitrary dense superposition.

This also clarifies how the supervised anchors should be used with unsupervised weight clusters. They are external tests of a discovered verb or invocation, not labels to fold into the discovery objective. A cluster can correctly group the same routing verb across different payloads without representing one human concept, and different verbs can participate in the same semantic operation.

## Broad-regime correction: there is more than one feature-construction regime

The four lexical calibration edits were followed by a preregistered screen of 17 families. A full-vocabulary semantic-output red team retained seven operational families: local and attractor agreement, IOI, capital recall, plain and negated sentiment, and repeated-sequence retrieval. In-context key-value retrieval was rejected despite 0.875 candidate accuracy because exact top-1 was only 0.188.

The surviving families do not share one construction branch:

- local agreement is predominantly an MLP computation at its early crossing;
- plain sentiment is predominantly carried compiled state;
- agreement-with-attractor, IOI, capital recall, negated sentiment, and repeated retrieval receive material fresh-attention contributions.

The material attention pulses nevertheless have compact implementations. Discovery-frozen top-three programs achieve 0.896--1.000 held-out fidelity and rank in the 98.8--100% percentile of every equal-size subset. Their tensor-edge grammars are also stable across discovery and held-out pairs.

The operands distinguish regimes. IOI reads the changed giver token, capital recall reads the changed country, and repeated-sequence retrieval reads the two changed payload occurrences. These positions recover about 94%, 100%, and 99% of their respective selected program effects. Sentiment is relayed: the adjective positions recover only 11% for plain and 22% for negated sentiment, while later context positions carry most of the selected program.

The resulting ontology is explicitly plural:

    direct discriminating MLP
    carried compiled variable
    direct relational attention read
    cue-seeded relay attention read

All can be concise explanations. They answer different causal questions and should not be collapsed into a single neuron-alignment score.

### Correction: compact head identity is mostly a backbone, not a semantic label

A prospective comparison asked whether the six material broad-task programs could
be recovered without answer labels. Static random-function scale reached `0.845`
mean held-out fidelity; input-conditioned head-write magnitude reached `0.953` and
the 94th percentile among all equal-size subsets. The older label-blind weight-motif
graph was also enriched for the selected heads after a two-test correction.

That positive result did not survive the most important specificity control.
Replacing the true changed tokens with arbitrary tokens while preserving the edit
positions and counts still selected programs with `0.948` mean fidelity. A
full-vocabulary suffix score chose nearly the same heads as raw state magnitude and
did not improve fidelity. Thus broad semantic tasks often travel through a small
set of high-gain heads, but the *identity* of those heads does not specify which
semantic variable is traveling through them.

The revised hierarchy is:

```text
task-defined relation and changed contextual state
  -> occupied payload direction/subspace
  -> generic high-gain head/read backbone
  -> destination variable and downstream output quotient
```

Head sparsity remains useful for implementation tracing and intervention budgeting.
It is not, on its own, evidence for head-level monosemanticity. The next specificity
test belongs at the payload-subspace level: a genuine family basis should transfer
across held-out members of that family while rejecting matched arbitrary edits even
when both use the same heads.

That payload test reveals two representation regimes. Rank-two bases transfer for
agreement and plain/negated sentiment (`0.968--0.993` program-output fidelity), but
not for IOI, capital recall, or repeated retrieval (`0.136--0.412`). This is not a
failure of interpretation: the latter operations must transport whichever entity or
token is requested, so one fixed output direction would erase the very information
they need to preserve.

Factoring the tensor computation as a function validates this account. Discovery-
mean route coefficients applied to new held-out contextual values recover `0.894--
0.999` of the direct-read program effects. Freezing the values instead destroys the
identity tasks, and permuting route coefficients across heads drops mean fidelity to
`0.208`. A shared route multiplying only the contextual value change works as well as
separate base/source routes.

The most economical plural ontology is now:

```text
class-like variable: fixed low-rank payload direction/subspace

content variable: stable typed read operator(value argument)

both may execute through the same generic high-gain head backbone
```

This is the precise sense in which neurons or heads can behave like verbs. A verb is
not polysemantic merely because it accepts many objects. Interpretation should name
the operator signature, argument source, and output contract rather than demanding
one object concept per unit.

### Natural-language validation and recursive semantic generalization

The typed operators were next searched over three disjoint 512-row FineWeb panels
using only frozen route vectors and a weight-aware edge-magnitude score. Exact route
geometry did not pass every global gate: capital and one repetition role remained
below `0.90` median cosine, and the IOI template did not consistently beat all
head-permuted assignments by 5%. This prevents a claim that every synthetic program
is a rigid natural tensor type.

The natural functions were nevertheless highly reproducible. The repetition route
finds literal source-to-next-token copy in `38--51%` of its top edges, versus
`0.5--2.6%` for other templates. The capital route consistently emits source-linked
entity/lexical continuations. Crucially, the first semantic names were red-teamed:
the narrow auxiliary-control and strict-locality labels failed on the second panel,
even while their richer semantic profiles transferred.

A third panel then tested broader frozen descriptions:

| Controlled seed | Rejected narrow name | Validated broader natural function |
|---|---|---|
| agreement with attractor | auxiliary-only controller | nearby predicate lexicalization |
| indirect object | strictly local IOI lookup | short-range name-valued association without echo |
| capital recall | capital-specific fact lookup | source-linked entity lexicalization |
| repeated sequence | toy repetition circuit | long-range lexical copy |

All third-panel profile cosines are `0.975--0.999`. This is concrete support for
recursive semantic organization: apparently different examples can be related uses
of one broader function. It also supplies a stopping rule. Broadening a label is
accepted only after it predicts a new disjoint panel; semantic elasticity without
fresh validation is not evidence.
