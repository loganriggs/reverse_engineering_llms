# Ontology MDL framework

The experiments now contain several behaviorally valid descriptions that optimize different resources. A single scalar ‘interpretability’ or MDL number would hide the choice of interpreter. This document defines the costs separately and reports the current frontier.

## Program language

A mechanistic description has four reusable declarations:

1. **Variables** describe source token/state properties. Examples are `function_word`, a learned categorical token table, or a soft routing membership vector.
2. **Read functions** specify where a variable is read: layer, native head, relative-distance predicate, and any contextual guard. `previous_token` is the distance interval `[1,2)` and can be reused by many heads.
3. **Transforms** specify an operator applied to arguments: native value projection, output projection, RMSNorm geometry, and the receiving bilinear tensor contraction. A transform is not assumed to emit one fixed direction.
4. **Invocations** bind a variable, read function, transform, transported source value, and destination context. The arguments explain why two calls to one native head can have different realized effects.

The behavior residual is the KL from the native next-token distribution. Its ideal coding penalty is `N × KL / ln(2)` bits over `N` token predictions.

A function declaration may itself be factored into an input coordinate, gain,
output direction or subspace, argument types, and a destination interface. The
bus-aware experiments show why this matters: sharing one output direction does
not make the complete functions interchangeable.

## Why one MDL number is insufficient

- A mask saying `keep all edges` is extremely short but performs the full native computation. Description length must not be confused with runtime compression.
- An arbitrary hard token table can be execution-sparse and behaviorally strong, yet is not readable or stable enough to be a scientific variable.
- A soft factorization can expose reusable geometry, but its real-valued memberships and fractional execution are expensive unless another short generator explains them.
- Human rules are short only relative to an interpreter that already knows lexical predicates such as punctuation, suffix, or leading whitespace. Readability is therefore an explicit primitive prior, not a basis-invariant fact.

## Current measured frontier

| Program | Route predicates/layer | Mean KL | Top-1 | Variable declaration | Runtime proxy | Literal table bits* |
|---|---:|---:|---:|---|---|---:|
| human_hard128 | 128 | 0.2318 | 0.767 | 20 ordered readable clauses / 38 feature references | hard sparse; 0.22 edge fraction | 254897 |
| human_hard192 | 192 | 0.0987 | 0.846 | same readable variable grammar | hard sparse; larger invocation bank | — |
| human_hard256 | 256 | 0.0549 | 0.893 | same readable variable grammar | hard sparse; larger invocation bank | — |
| readable_tree42_hard256 | 256 | 0.3277 | 0.720 | stable 42-leaf readable tree | hard sparse; fragmented invocations | — |
| nmf_hard128_a | 128 | 0.2686 | 0.785 | opaque categorical token table | hard sparse; — | 254897 |
| nmf_hard128_b | 128 | 0.2168 | 0.809 | opaque categorical token table | hard sparse; — | 254897 |
| nmf_soft128 | 128 | 1.9002 | 0.358 | real-valued multiplexed membership table | dense fractional; 1.00 edge fraction | 8044732 |
| nmf_soft256 | 256 | 0.1742 | 0.778 | real-valued multiplexed membership table | dense fractional; 1.00 edge fraction | 8046715 |
| native_all | 1323 | 0.0000 | 1.000 | no source dictionary needed for mask | native dense; 1.00 edge fraction | 0 |

`*` Literal bits use a deliberately crude standalone encoding: five bits per hard token label, eight bits per independent soft membership, plus an enumerative code for selected route types. They are not claimed as the final scientific readability prior.

Under that literal-table assumption, the token-count crossovers where lower behavioral KL repays extra description are:

- `nmf_hard128_a`: never, because it has no behavioral-bit advantage.
- `nmf_hard128_b`: approximately 11,614,908 evaluated tokens.
- `nmf_soft256`: approximately 96,854,632 evaluated tokens.

## Recommended MDL views by use case

| Use case | Charge heavily | Current preferred ontology |
|---|---|---|
| Human reverse engineering | Unreadable variables/functions | Human hard categories plus native head/distance functions |
| Fast approximate execution | Edge invocations and residual KL | Hard 128-type programs |
| Faithful behavioral surrogate | Residual KL | Soft-256 or a larger hard program |
| Discover recurring geometry | Instability across splits | Continuous routing profiles / soft memberships |
| Exact mechanistic execution | Any unexplained residual | Native QKOV + RMSNorm + bilinear contractions |

Function templates add another use-case frontier:

| Function question | Current defensible object |
|---|---|
| Compress raw value-side weights | Rank-22 linear operator basis for 90% energy |
| Describe recurring output geometry | Globally anisotropic leading-direction family |
| Describe natural corpus behavior | Stable 45-dimensional response geometry |
| Execute or transplant a function | Native typed head operator with its arguments and destination |

The last row is stricter than the first three. The strongest geometrically
matched pair failed bidirectional finite transplantation, so a cheaper geometric
code cannot currently replace the native executable declaration.

The full-operator follow-up refines that statement. Function reuse is sometimes
directional: an earlier operator can be an excellent replacement at a later
destination without the reverse being true. A reusable declaration therefore
needs a typed interface and a destination-specific error metric.

For causal replacement, charge the residual in the destination Fisher quotient:

```text
replacement cost
    = declaration bits
    + binding/interface bits
    + invocation bits
    + downstream Fisher residual bits
```

For the three tested destinations, this residual geometry is stable and dominated
by one direction, with 5--27 dimensions needed for 90% energy. A raw
Frobenius residual is substantially cheaper to calculate but correlates only
0.17--0.32 with the causally relevant geometry. Consequently, raw-weight MDL is
appropriate for storage compression, while Fisher-weighted MDL is appropriate
for behavioral substitution.

## Compositional transfer beats path enumeration

The destination-Fisher coordinates can now be predicted prospectively from the
tensor network. On three fixed new destinations, the exact analytic Jacobian
ranks finite compatibility at 0.999--1.000, while raw operator norm does not.
This resolves the previous next test positively at the causal-prediction level.

It simultaneously rejects enumerated attention paths as the program language.
For early destinations, low-order path sums have huge error and only recover
after orders 8--12 cancel. Charging separately for every expanded path would
assign a long, unstable code to a computation that has a short recurrent
description.

The program language should therefore include a reusable transfer primitive:

```text
transfer(layer, state, value_bus, tangent):
    mixed = residual_scale(layer) * tangent
    routed = attention_jvp(layer, state, value_bus, mixed)
    return residual_mlp_jvp(layer, mixed + routed)
```

The MDL cost is then

```text
transfer declaration bits
  + layer/state binding bits
  + initial typed-tangent bits
  + number of recurrent invocations
  + destination-Fisher residual bits
```

It is not the sum of all algebraically expanded path descriptions. This is the
same reason a loop is normally shorter than a list of every execution trace.

The state carried by this loop cannot currently be charged as one global
rank-64 vocabulary. That code retains only 10--43% of causal response and is
matched surprisingly closely by ordinary native-state PCA. A per-context chart
is much better (71--93% rank-64 Fisher fidelity), but its basis rotates strongly
and is fit from the very tangent bundle it describes.

MDL must therefore charge for the chart generator:

```text
dynamic-state code
    = generator declaration bits
    + native-state/context arguments
    + generated chart dimension
    + coordinates in that chart
    + destination-Fisher residual bits
```

Treating each oracle chart as free would hide nearly all of the explanatory
cost. Conversely, charging a full literal matrix per context would erase its
computational regularity. The scientific target is a short prospective
generator, likely expressed through the local tensor Jacobians and routing
state.

The joint sequence-field experiment shows why runtime and declaration costs
must remain separate. At a matched 128 runtime coefficients, the joint chart
reaches mean causal fidelity 0.767 while the position-collapsed chart reaches
0.080. Joint rank 64 also beats collapsed rank 4 despite using 64 rather than
512 coefficients. Yet a literal joint basis contains 128 times as many entries
per axis as a residual-only basis. The favorable code is therefore not “send
the joint eigenvectors”; it is “invoke the shared Fisher/tensor generator and
send a small number of coefficients.”

This motivates an explicit function-oriented code:

```text
program cost
    = variable declaration bits
    + read-function declaration bits
    + transform declaration bits
    + invocation/binding bits
    + runtime coordinate bits
    + output-Fisher residual bits
```

For example, `previous_token` should be charged once as a read function and
then reused with different variables and transforms. A raw list of every edge
that implements it is a valid circuit trace but a poor program. Conversely, an
opaque context-specific joint eigenfield is cheap at runtime but remains costly
to declare unless its spatial and residual factors are generated by shorter
primitives.

The first read-function audit makes the binding distinction empirical. Stable
output demand is concentrated in only 3--9 positional modes and strongly
overlaps current/recent-token spans, but one residual transform shared across
all calls loses much of the joint field's behavior. Allowing four current-token
invocations to bind separate rank-32 transforms nearly closes the gap at late
boundaries while keeping the same 128 runtime coefficients. It does not close
the early boundary, where noncurrent sites help.

Accordingly, “same function” must not mean “same complete vector operation”:

```text
declare current_token(source, destination)
invoke current_token(q0, q0) with variable v0 and transform f0
invoke current_token(q1, q1) with variable v1 and transform f1
...
```

The declaration is shared; variables, transforms, and invocation bindings are
separate costs. Calling the whole bundle one feature would hide the reuse.
Calling each invocation an unrelated feature would hide the common function.

The next selector result adds a dynamic address cost. Output-sensitive demand
chooses stable context-specific sites, including nonlocal content tokens, and
improves early-boundary causal fidelity. A simple unsigned QK rollout does not
generate those early addresses: it succeeds only near the output. Therefore an
MDL code may declare `attention_rollout` cheaply for late reads, but must pay for
the richer signed tensor recurrence for early reads until a shorter faithful
generator is found.

The signed-role decomposition shortens that early generator differently for
two use cases:

```text
address code  = recurrent key roles + recurrent value roles + top-k selector
payload code  = recurrent query roles + recurrent value roles + local chart
```

On the present panel, key+value exactly reproduces the top-eight address set,
whereas query+value produces the best self-contained causal chart. The two
programs share value transport but answer different questions. MDL should not
force a single canonical “feature definition” when one application needs
source discovery and another needs behavioral substitution. It should charge
for the declared interface and residual under the intended use-case metric.

The depth-localization red team refines “exactly” above: two score sketches did
produce the identical top-eight set, but a third high-sample seed changed one
near-tied site in several contexts despite demand cosine above 0.997. The stable
description is the weighted K+V demand field, not its arbitrary rank-eight
threshold.

For behavioral substitution, four nearest-boundary Q+V role invocations are
already within 0.10 of the all-layer program on every tested target. The MDL
frontier therefore has separate rows:

| Use case | Preferred object | Residual/error |
|---|---|---|
| Discover likely source sites | Weighted K+V demand field | Ranking/field error |
| Execute sparse reads | Top-k sites plus local bases | Downstream Fisher residual |
| Substitute tested behavior | Four-layer near-boundary Q+V recurrence | Causal fidelity loss |
| Reproduce exact local derivative | Full signed Q/K/V + MLP recurrence | Numerical derivative error |

None dominates the others without specifying the consumer of the explanation.

Native-head selection adds another quotient-dependent frontier. A frozen top
eight Q+V head set reconstructs the suffix adjoint substantially better than all
sixteen matched random sets, but does not beat their behavioral fidelity
distribution at the preregistered 95th-percentile gate. Therefore:

- charge exact or Fisher-weighted adjoint residual when building a derivative
  emulator or gradient-level circuit atlas;
- charge only downstream behavioral residual when building a task-specific
  surrogate, where many native-head sets may be equivalent;
- do not count “eight heads suffice” as eight named mechanistic primitives
  unless their identities beat matched random subsets under the relevant use
  case.

This is a concrete example of two correct MDL descriptions. The derivative
description prefers the ranked native heads. The behavioral description can
replace them with a cheaper equivalence-class declaration plus any adequate
binding from that class.

One such generator now works partially: propagate stratified next-token Fisher
adjoints backward through the exact tensor network and diagonalize their
position-aggregated residual covariance. It is candidate-independent and raises
rank-64 causal fidelity from 0.10--0.43 for static PCA to 0.63--0.85.

Its MDL accounting must distinguish the chart subspace from its arbitrary
internal coordinates. Independent sketches yield only 0.44--0.60 literal-axis
overlap while producing highly correlated rank-64 behavioral fidelity. Charging
for exact basis vectors would overstate the scientifically stable content;
charging only for rank would understate the remaining behavioral residual.
The appropriate code records the generator, metric, rank, and residual—not a
claim that every eigenvector has an invariant meaning.

## Next test

The suffix-Fisher generator now resolves the cross-position part positively in
runtime coordinates, but not in readability or total description length. The
next search should factor its joint sequence fields into predeclared read
functions—current token, previous token, fixed offsets, causal-prefix summaries,
and multiscale intervals—paired with residual transforms. A proposed primitive
earns readability only if it predicts held-out chart structure and finite causal
effects better than matched opaque position bases, while paying for every
variable, function, binding, coordinate, and residual.

## A semantic primitive now survives the full MDL ladder

The late read-function search first produced an instructive failure. Route
profiles were almost perfectly stable and nearly rank one, but human token
classes did not beat frequency bins or frequency-preserving shuffled labels.
The shortest supported program at that stage was only:

```text
route magnitude ~= head-specific distance curve * source prevalence
```

Rich contextual trees also failed to earn their extra declarations. Although
they predicted extreme edge effects with median AUROC 0.93, distance, absolute
position, and endpoint frequency already supplied nearly all that accuracy.
MDL must charge a readable predicate that merely renames a correlated address
variable; zero incremental held-out gain means no semantic compression credit.

Prospective matched tests and finite interventions then isolated two better
primitives:

```text
read_boundary_state(source_boundary, destination) using L9H8
read_adjacent_function_word(previous_token, destination) using L7H8 or L9H7
```

Both predict exact edge effects after matching distance and endpoint frequency,
and exact removal reaches final logits. Proper-name, capitalization, repetition,
and induction stories failed the same procedure. This suggests a practical
primitive-admission rule:

```text
admit primitive P only if
    held-out conditional prediction gain(P) > complexity charge(P)
    and finite intervention effect(P) beats matched controls
```

The adjacent-function-word result demonstrates why variables and functions need
separate code lengths. L7H8 and L9H7 share the same readable address relation,
but their exact OV cosine is negative, their write subspaces overlap only about
0.15, and their paired final effects have cosine about 0.13. A compact program
therefore declares the argument once but pays for two transforms:

```text
variable  v = adjacent_function_word
function  f7 = transform_L7H8(v)
function  f9 = transform_L9H7(v)
```

Calling both one feature underfits the computation. Declaring two unrelated
lookup tables overpays for the shared variable/address grammar.

L9H8 supplies a stronger end-to-end example. Its weight-derived leading mode
suggested a US-geography/British-spelling axis. A human-readable code now
survives five progressively stricter costs:

1. **Weight declaration:** the leading OV mode is unusually concentrated and
   decodes toward US geography versus British spellings.
2. **Observational encoding:** the boundary write classifies matched US/UK
   spelling prefixes on held-out FineWeb.
3. **Finite use:** removing that edge selectively changes a declared
   US-minus-UK output contrast.
4. **Counterfactual transfer:** swapping one cue spelling changes the edge's
   effect on *disjoint* spelling pairs in the expected direction.
5. **Specificity:** the effect beats random target-token contrasts, a
   frequency-near non-boundary edge, and another head at the same boundary.

The resulting typed program is:

```text
variable   style_state(prefix)          # US/UK spelling tendency here
address    latest_boundary(prefix)
function   L9H8_value_transform(style_state)
consumer   downstream spelling logits and correlated domain predictions
```

This is not free of residual cost. The native write space is reusable across
splits, but its final-logit effect subspace is broad and context-dependent.
The MDL code should therefore charge a short semantic variable/function
declaration plus the tensor-network suffix recurrence and output residual. It
should not replace the whole head by one scalar “dialect neuron.”

The counterfactual result also clarifies readability. A variable name is useful
when it predicts new interventions, not merely when decoded top tokens look
coherent. Here “spelling/style state” predicts held-out data, disjoint lexical
pairs, and controlled token swaps. “Proper-name continuation” did not survive
prospective matching and must remain an example-level anecdote.

## Revised next test

The immediate target is no longer merely to find a readable feature. It is to
measure how far the successful typed program generalizes:

- Does L9H8 carry other source-style variables such as formatting, register,
  locale, or code/natural-language state through the same boundary address?
- Is the spelling/style variable already explicit before L9H8, and which MLP or
  attention transforms create it?
- Which downstream consumers read the transported state, and can they be
  replaced by a shorter declared contrast without losing natural-text CE?
- Can the same variable be found prospectively from weights in another model or
  another head, without inspecting token poles first?

These tests distinguish a reusable typed boundary function from a single
post-hoc circuit success and provide the next MDL comparison between native
tensor execution and a human-readable program.

## The spelling variable now has an explicit two-hop implementation

The upstream-origin tests turn the earlier typed declaration into a finite
native circuit:

```text
lexical spelling evidence at the cue
    -- Layer 8 Head 2, direct cue read -->
contextual state at the latest boundary
    -- Layer 9 Head 8, boundary read -->
future US/UK spelling preferences
```

This description survives three additional MDL charges. First, an exact QK /
current-value / persistent-bus factorial shows that L9H8's effect comes from
the contextual boundary value rather than a routing proxy. Second, an exact
upstream telescope localizes most construction to Layer 8 instead of repeated
embedding injection or a diffuse early history. Third, a fully finite edge
replacement followed by a downstream edge freeze validates the composed path
through every intervening nonlinearity.

The implementation is much sparser than the generic dynamic-chart results
would predict: L8H2 explains `98--99%` of the Layer-8 builder projection and
its direct cue edge explains approximately all of that head's relevant write.
Yet the correct feature boundary is still not “Head 2 means dialect.” The
upstream tensor factorization assigns most effect to its persistent Layer-0
lexical value bus, with smaller cooperative current-value and QK terms. The
readable program is therefore:

```text
variable   lexical spelling code(token)
address    cue_position -> latest_boundary
function   L8H2 lexical-evidence write
variable   contextual spelling/style state(boundary)
address    latest_boundary -> prediction_position
function   L9H8 style-state write
consumer   future spelling contrast plus uncharacterized residual consumers
```

Eleven new single-token spellings transfer in both directions between cue and
target vocabularies. This earns a reusable family-level variable declaration;
it is shorter than enumerating the original twelve pairs. L9H8 freezing removes
only `51--58%` of the L8H2 effect, so an honest MDL program must still pay for
the residual downstream consumers rather than claiming circuit closure.

## New next test

The most informative expansion is now breadth across variable families, not
another spelling refinement:

- prospectively declare formatting, code/natural-language, register, and
  locale-independent lexical contrasts without inspecting new weight poles;
- ask whether L8H2 performs the same cue-to-boundary function for them, or
  whether another head binds the shared address to a different payload;
- map the remaining `42--49%` of the L8H2 spelling effect not mediated by
  L9H8, using downstream-freeze screening followed by finite confirmation;
- repeat weight-only discovery in another checkpoint before admitting
  `lexical_evidence_bus` as a model-family primitive.

This distinguishes three increasingly broad descriptions: one spelling
circuit, a reusable lexical bus with typed consumers, and a general
boundary-state architecture.

## The terminal spelling consumer separates operator sparsity from state sparsity

The previously uncharacterized downstream remainder is now partly resolved.
The final block receives the spelling edit mainly through two branches:

```text
carried destination residual                    66--74%
Layer-17 destination attention                  27--35%
Layer-17 MLP                                    approximately 0%
```

Within attention, the initial broad description was misleading. One native
operator invocation—L17H2 reading positions after the latest boundary—accounts
for `87--89%` of positive nonboundary allocation and `19--24%` of the complete
finite edit. Yet its state support is not sparse: six distance bins contribute,
and the largest supplies at most `27%`.

This yields two valid but noninterchangeable feature descriptions:

| Use | Shortest supported object |
|---|---|
| Select which native operation matters | `L17H2(post_boundary_state)` |
| Locate where its argument is represented | A repeated post-boundary state field |
| Reproduce the declared output change | Carried residual plus the H2 read and smaller residual terms |
| Claim global semantics for the head | Not supported |

An MDL account should charge once for the shared H2 read function and then for
the positional field or recurrence that supplies its arguments. Charging one
unrelated feature per source edge overcounts the program; calling H2 a single
global spelling feature undercounts its other possible invocations.

The next discriminating test is whether post-boundary H2 values share one
transported spelling direction. Cross-position value-factor alignment and
finite donor/recipient patches can separate literal state echoes from distinct
position-specific transforms that merely agree on the output quotient.

That test now has a split verdict. The field is literally low rank but is not
literally a spelling variable. A discovery-frozen rank-1 value axis captures
`73--83%` of raw H2 change and essentially all output-relevant H2 spelling
effect, including for eleven new spelling pairs. Yet a heterogeneous panel of
sixteen unrelated token swaps produces an axis with cosine `0.959` and strong
cross-family reconstruction.

The MDL declaration must consequently separate payload semantics from
transport format:

```text
semantic payload       spelling evidence at cue/boundary stages
transport format       generic approximately scalar edit field
read function          L17H2(post_boundary_field)
consumer quotient      US-versus-UK output contrast
```

Calling the scalar field “spelling style” is too expensive semantically: it
fails to predict its reuse for unrelated token edits. Calling it a generic edit
bus is shorter and currently predicts more interventions. The next cost to pay
is a weight-derived explanation of where that edit axis comes from—pre-value
residual dynamics, the H2 value matrix, or normalization.

That cost is now paid. The edit field is not rank one before the value map.
Instead, a native `c_v` singular outlier canonicalizes broad residual changes
into a shared value coordinate. Its weight-only left singular vector recovers
almost all of the finite H2 spelling effect without activation fitting.

The outlier is mechanistically real but not globally distinctive: it ranks
`60/162` among all value-head slices. A viable discovery code is consequently
conditional rather than weight-only:

```text
important destination module
    + readable source relation
    + occupied weight outlier
    + downstream quotient agreement
    -> candidate function invocation
```

This also refines the random-matrix heuristic. An outlier can provide the short
implementation once the circuit context is known, while magnitude alone may
produce dozens of stronger but irrelevant candidates. MDL ranking should charge
for the contextual selectors rather than pretending the singular vector names
itself.

## Conditional program length is a representation regime

The natural-edit experiments add a distributional qualification to the MDL
framework. The same Layer-17 tensor bank supports a two-head median program for
the coherent spelling family and a five-head median program for a deliberately
heterogeneous edit mixture. A pooled H6 summary of the latter is reproducible
at the population level but fails as a per-pair primitive.

There are consequently several correct descriptions, each with a different
use case and code length:

| Question | Correct object | Required information |
|---|---|---|
| What operations can this layer implement? | Native head tensors and stable singular subspaces | Weights |
| Which operation is strongly occupied? | Routed write magnitude for the current counterfactual | Weights plus states and source relation |
| Which operation matters for this behavior? | Output-adjoint/Fisher-ranked invocation | Above plus a declared consumer quotient |
| What compact program reproduces the branch? | Joint finite head coalition and k90 | Quotient plus recomputation |
| What human variable recurs across examples? | A prospectively tested semantic/counterfactual family | Dataset primitives and fresh examples |
| What is typical over a population? | Average causal field such as pooled H6 | Explicit population distribution |

The population-average and instance-program ontologies must not be conflated.
H6 is positive for `14/16` unrelated pairs, so it is a legitimate average
cooperator. It is top-three for only `8/16`, so it is not a faithful universal
implementation. Conversely, H2 is a strong spelling-family invocation without
being globally synonymous with spelling.

For MDL comparison, the distribution is part of the problem statement. A
single-head spelling program is short only because the intervention family and
output quotient are coherent. Combining unrelated queries can increase the
shortest sufficient program without changing any weights. “The model is
monosemantic” and “the model is multiplexed” are therefore incomplete claims
unless they name the query distribution, mechanism boundary, and accepted
behavioral error.

The practical reverse-engineering sequence is now:

```text
enumerate native verbs from tensor structure
    -> measure contextual occupancy and address relation
    -> apply a declared downstream quotient
    -> select a small conditional coalition
    -> validate it jointly with finite recomputation
    -> test proposed human names on fresh counterfactual families
```

This sequence supports mechanistic explanations before every intermediate has
a human label. It also exposes the remaining hard problem: discovering useful
consumer quotients and semantic families without choosing them after seeing the
mechanism.

## Charge separately for capability, localization, and organization

The matched semantic-family experiments show why an MDL score cannot begin by
compressing whichever module was convenient to audit.

Sentiment, gender, and tense pass a disjoint cross-pair behavioral test, but the
Layer-17 post-boundary attention branch carries only `8%`, `3%`, and `2%` of
their final contrast. Their broad `4--6`-head programs are consequently cheap
descriptions of a minor side branch, not descriptions of the primary semantic
mechanism. Spelling sends `38%` through that interface and admits a two-head
program, so terminal attention is a defensible boundary for spelling.

An honest description length should therefore include three distinct costs:

| Cost | Question | Failure means |
|---|---|---|
| Capability | Does the proposed variable predict held-out counterfactual behavior? | The human grouping may not be a model variable in this task |
| Localization | How much of that behavior crosses the chosen mechanism boundary? | Compression of this branch says little about the main feature |
| Organization | How many typed components jointly reproduce the material branch? | The chosen component vocabulary is too coarse or the code is distributed |

These costs define different legitimate use cases. A population scientist may
care about average causal allocation. A circuit engineer needs a jointly
sufficient local program. A semantic interpreter needs transfer across examples
and lexemes. A model compressor needs CE/KL preservation even if no coordinate
has a human name.

The next MDL target for sentiment, gender, and tense is not a better final-head
dictionary. It is the carried residual path: locate the earliest layer boundary
at which disjoint-family behavior becomes recoverable, then factor the writes
that create and maintain that state. Only there is a neuron-versus-coalition
comparison about the primary variable rather than its collateral tail.

## New frontier: fixed variables versus typed functions

The broad task battery falsifies a tempting but incorrect MDL shortcut. A short
three-head list is not necessarily a short semantic description. Arbitrary tokens
inserted at the same edited positions recover `0.948` mean held-out fidelity from
head identity alone, versus `0.953` for the true counterfactual. Those heads are a
high-gain backbone shared by many possible payloads.

The next two representations separate cleanly by use case:

| Conditional description | Extra continuous metadata | Mean held-out fidelity | Best use |
|---|---:|---:|---|
| Three native head identities | 0 | high, but nonspecific | Locate/rerun the native branch |
| Rank-two residual payload basis | 2,304 basis coefficients plus per-example coordinates | 0.630 | Class-like variables; output-selective compression |
| Typed read template | 3 route coefficients; 6 for two-source repetition | 0.954 | Transport new contextual identities |
| Head-permuted read template | same | 0.208 | Matched structural control |

These counts are conditional on the checkpoint weights, contextual value vectors,
source-role definition, and destination interface being shared. They are not
standalone network-compression bit counts.

The apparent weakness of the rank-two basis is structured. It reaches `0.968--
0.993` for agreement and plain/negated sentiment, but only `0.136--0.412` for IOI,
capital recall, and repeated retrieval. The latter tasks cannot use one fixed
direction without discarding which name, capital, or color is being transported.

For those identity tasks, a functional code is dramatically shorter and more
faithful:

```text
declare read(heads, source_role, quantized_route)
invoke read with current contextual value
```

The route template contains three scalars for one source role and six for the two
repeated payload locations. Two-bit symmetric quantization already retains
`0.922/0.990/1.000/0.915` fidelity for agreement, IOI, capitals, and repetition.
Four bits are indistinguishable from floating point at this resolution. One-bit
sign codes fail the all-family gate because agreement falls to `0.686`.

This produces two legitimate notions of a concise feature:

1. **State feature:** a fixed low-rank subspace whose coordinates vary.
2. **Functional feature:** a fixed typed operator whose value argument varies.

The state definition is useful for classification, steering, and patching a stable
latent variable. The functional definition is useful for retrieval, copying,
binding, and explaining why semantically different objects use the same circuitry.
Neither should be treated as the unique canonical ontology.

Layout transfer adds a limited but important generalization test. Inserting one or
two tokens between the operand and prediction position preserves exact behavior for
agreement, IOI, and repeated retrieval. Original route templates retain `0.714--
0.998` fidelity, and altered-layout route cosine averages `0.966`. This rejects an
absolute-position lookup. The templates remain close paraphrases with similar
relative structure, so arbitrary syntactic transfer is not yet established.
