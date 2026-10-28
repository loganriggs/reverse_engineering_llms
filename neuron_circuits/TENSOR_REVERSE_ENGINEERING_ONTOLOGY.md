# An ontology for reverse-engineering the tensor transformer

## Bottom line

There is not one uniquely correct decomposition of a neural network. There are
several explanatory objects, and they answer different questions.

For Bilin18 layer 1, the current evidence supports this picture:

1. **Native neurons can be clean local predicates.** The weight-first `-ing`
   group is a good example: its input selectivity transfers across corpora and
   its joint ablation is specific.
2. **The computation after those predicates is distributed.** Restoring the 32
   strongest downstream targets repairs only 9.7% of the `-ing` effect, and a
   one-neuron intervention fans out through roughly 2,000 downstream channels.
3. **Language uses a much easier part of a bilinear operator than arbitrary
   residual directions do.** Activation-refined CP-2048 has 86% natural-value
   fidelity but only 36% Gaussian-tangent fidelity.
4. **Natural behavior and global causal response contain complementary
   information.** A frozen 80% language / 20% global-tensor blend improved KL
   and both causal metrics on disjoint FineWeb, WikiText, and TinyStories rows.
5. **That complement is not localized to a stable handful of inherited CP
   factors.** A 512-term subset recovers useful natural behavior, but not the
   Gaussian causal gain, and the identity of the selected factors is unstable.
6. **Ordinary output PCA is also insufficient.** High correction variance does
   not imply downstream importance; the first 32 directions explain 60% of
   value variance while making causal fidelity worse.
7. **Downstream weighting helps, but does not make rank 256 sufficient.** A
   learned rank-256 correction quotient beats hybrid PCA on aggregate, yet
   misses the preregistered behavior, causal, and transfer gates. Separate
   Gaussian and natural specialists also fail their gates.
8. **Output-conditioned interaction spectra expose compact semantic
   discriminators, not automatically complete circuits.** For five token-form
   directions, one or two eigenmodes preserve semantic discrimination and recur
   across independent SAE fits. The first intervention used to assess them has
   now been reclassified as a scalar-coordinate reconstruction test.
9. **The power-law bulk is not itself evidence of a circuit.** Its exponent and
   fit are nearly unchanged by coefficient sign randomization, and matched
   nonsemantic features also have spectral outliers. The circuit evidence comes
   from semantic prediction, cross-fit recurrence, and residual-path tests—not
   from a visually clean scree plot.
10. **Scalar-coordinate interventions and residual-direction interventions are
    different causal questions.** The former changes one SAE encoder coordinate
    along an arbitrarily chosen product-space dual. The latter changes the
    proposed eigenvector in the actual MLP input and recomputes all product
    channels. Only the second tests whether an eigenmode is a causal input path.
11. **Known spectral modes pass a local but partial residual-path test.** At a
    10%-of-input-norm edit, rank-4 modes for the five known categories change
    their selected feature 16.7–20.0 times more than matched low modes and cause
    2.2–5.4 times more downstream effect. SAE-feature-path mediation fidelity is
    about 0.41; substantial behavior remains distributed elsewhere.
12. **Scale and numerical precision are part of the mechanism definition.**
    Full component removal can move the input by several times its norm, while
    small product-coordinate edits are quantized in bfloat16. Local float32
    interventions are stable; large raw-basis edits cease to be feature-local.
13. **The first local result was enriched for extreme examples.** A 64-site-per-
    category stratified replication reduced known-category mediation from
    0.33–0.46 on high positives to 0.12–0.23 on random positives. Surface labels
    alone do not define when the route is used.
14. **A feature coordinate has multiple input routes.** Exact SAE gating and
    surface semantics both matter. High-active known routes mediate about 0.37;
    random-active routes 0.11–0.21; inactive routes approximately zero. Within
    active sites, category-labeled examples still differ from other uses of the
    same SAE feature.
15. **The reusable causal object is an ensemble, not the named feature alone.**
    On held-out high-active sites, a frozen category-specific core of 8 SAE
    features reaches 0.501 downstream fidelity and 16 reach 0.606. On an
    independent random-active semantic panel, 16 reach 0.417 and 32 reach 0.514.
16. **The ensemble recurs across independently trained SAE bases.** A seed-102
    frozen 16-feature program reaches 0.540 fidelity. Its decoded-write subspace
    matches the corresponding seed-101 category 5/5 (`p = 1/120`), and the
    seed-101 subspace predicts seed-102 edit effects at 0.636 median fidelity.
17. **Stable joint axes are not context-free neurons.** At their largest
    activations, finite coordinate removal is well predicted by diagonal roles,
    but at ordinary sites the full 32D interaction matrix is necessary. At
    layer 17, all-query fidelity rises from 0.238 for diagonal roles to 0.893
    for the full interaction model.
18. **Semantic probes can hide residual-stream multiplexing.** Across 14 token
    predicates, four contextual role directions explain 91.5% of layer-0 axis
    effects. Random sketches of the full residual write reduce that to 42.1%; a
    median of about 49 directions is needed for 90%.
19. **Late-layer selected subspaces are much more output-role concentrated.**
    The corresponding layer-17 residual-write effects put 71.2% of directional
    energy in the first role and 90.3% in four, with the role subspaces recurring
    on WikiText and TinyStories.
20. **The subspace is more canonical than its coordinates.** Random rotations
    of the selected 32D subspace can make individual roles lower-rank, but only
    by duplicating them: effective first-role diversity drops from 8.74 to 4.28
    at layer 0 and from 6.51 to 4.16 at layer 17.
21. **Clean early predicates do not imply clean end-to-end roles.** For twelve
    selected layer-0 axes, propagating finite edits through blocks 1–17 raises
    median directional rank90 from 16.5 in the local write to 25 in the final
    state. At 10% removal it rises from 13.5 to 25, so the effect is already
    present near the tangent regime.
22. **An arbitrary-readout interpreter is exactly available inside a chosen
    subspace.** The layer-17 weights define a linear 1,152→32×32 map from any
    output readout to its interaction matrix. Its algebraic error is `5.1e-7`;
    semantic and frequent-token score correlations are 0.80–0.90 across three
    corpora.
23. **Spectral strength prioritizes real operations, but the bulk and semantic
    labels are not sufficient.** The query-map bulk follows a power law under
    trained and shuffled weights. Yet the trained top four carry 22–26× the
    middle-spectrum natural variance and reach 0.944–0.957 score correlation.
24. **The natural state manifold is more fundamental than the original semantic
    basis at the final MLP.** State-SVD-32 beats semantic-32 on whole-write
    reconstruction and even on the strongest map components. Their simple 64D
    union reaches 0.961–0.964 write R² and approximately 0.996 top-component
    correlation.
25. **High local reconstruction still does not imply behavioral replacement.**
    Hybrid-64 has write cosine 0.991 but changes CE by 0.021–0.058 nats. Median
    KL first falls below 0.01 around 768 dimensions; near-exact behavior needs
    roughly 1,024 of 1,152 dimensions.
26. **Observational and causal interpreters need different context.** A 32D
    quadratic score map can identify an active operation. Finite edits require
    the cross term with outside context; a 64D hybrid tangent interpreter raises
    the failed fourth component from correlation 0.290 to 0.991 and generalizes
    across sixteen components and three corpora.
27. **The compact quotient can be executed without native neurons.** Contracting
    Hybrid-64 and using natural-write output factors yields a rank-128 quadratic
    program with about 0.75M counted multiplies/token versus 15.9M native. It
    adds only 0.0012–0.0015 KL to the exact Hybrid-64 program.
28. **Efficient approximation and high-fidelity replacement remain different
    goals.** The best tested high-speed point is roughly 21× cheaper at KL
    0.041–0.047; lowering KL to 0.027–0.032 reduces the advantage to about 1.8×.
    No preregistered point was both near-exact and strongly accelerated.
29. **Spectral organization changes sharply with depth.** Under the same
    state-SVD-64 audit, layer-0/8/17 whole-write R² is 0.455/0.154/0.934 and
    top-four tensor-map energy is 0.147/0.068/0.375. Four of five universal-
    compactness hypotheses fail; the final layer is a distinct regime.
30. **Natural writes show three reproducible regimes.** Across FineWeb,
    WikiText, and TinyStories, layer 0 has intermediate effective rank 19–22,
    layer 8 has broad rank 140–226, and layer 17 has rank 2.7–3.5. The layer-17
    top-four write subspace transfers almost exactly across corpora, whereas the
    layer-8 subspace is substantially corpus-dependent.
31. **Thomas Dooms's power-law-plus-spikes observation is depth-conditional in
    this checkpoint.** Full-width raw tensor spectra have power-law-like bulks
    at all three layers and the nulls retain them. Layer 8 has no strict positive
    outlier; layer 0 has only a leading positive deviation; layer 17 has four
    positive deviations whose mean excess beats every sign and pairing null.
32. **The layer-17 full-width spikes are real causal operations, but not four
    complete semantic circuits.** Their output subspace overlaps the state-64
    query map at mean canonical cosine 0.979 and natural-write PCA at 0.930.
    Top input-eigenspace edits have 27,750–767,594× the scalar effect and
    377–1,863× the final-logit KL of low-mode edits. Yet one output vector fails
    the simple semantic AUROC gate and two exact input forms put under 50% of
    their energy in four modes.
33. **Cleaning is part of the explanatory object.** Uncentered activation-RMS
    weighting makes the final layer appear almost rank-one. A centered
    Gaussian-moment calculation restores the mean-state cross term but still
    reaches covariance cosine only 0.674/0.811/0.778 at layers 0/8/17.
    Empirical natural-write covariance is therefore the authoritative usage
    check, not a cleaned weight spectrum alone.
34. **A broad middle layer is not merely many fixed low-rank verbs.** Inside the
    same 512D state basis, leading natural output queries require 64 modes at
    layer 0, 256–512 at layer 8, and 4–16 at layer 17 to reach correlation 0.90
    and centered R² 0.80. The distinction survives WikiText and TinyStories.
35. **Late query compactness is partly a generic bottleneck, not semantics.** At
    layer 17 even random output readouts reach the observational score gate in
    four modes. Spectral compactness there identifies the natural write
    subspace efficiently, but does not by itself turn each direction into a
    human concept.
36. **Middle-layer multiplexing depends on the sparsity definition.** A typical
    layer-8 natural query uses about 193 modes for 90% of absolute contribution
    mass, so strict activity sparsity fails. Yet an oracle-selected top 16 raises
    signed-score correlation from 0.60–0.65 to about 0.93, drawing across
    378–398 possible modes with effective occupancy near 100. This is
    predictive conditional sparsity with a dense cancelling tail, not a fixed
    sparse circuit.
37. **Oracle sparsity has not yielded a compact router.** A frozen 64-cluster
    router reaches correlation 0.69–0.73, a 128D architecture-matched quadratic
    router 0.82–0.85, and a preregistered 128D/8-neighbor router 0.75–0.76,
    versus 0.93 for oracle top-16 selection. All transfer OOD, but none closes
    the fidelity or support-recovery gap.
38. **The best tested routing information is continuous and high-dimensional.**
    Quadratic routing improves monotonically from 16 to 128 state coordinates
    and recalls 50% of oracle support at 128D. More clusters saturate and nearest
    neighbors do worse, arguing against a small categorical context router. This
    is evidence about three router families, not a proof that no compact learned
    or hierarchical router exists.
39. **A large supervised router does not rescue a small address code.** A 2.14M-
    parameter 128D GELU router peaks at validation correlation 0.827 but reaches
    only 0.753–0.775 on frozen corpora, below the weight-derived quadratic
    router. Route predictability is not merely missing generic nonlinear
    capacity in the tested 128D quotient.
40. **Layer 8 has a sparse payload with a dense address.** In the quadratic
    router sweep, 192/512 state dimensions are first to pass correlation 0.85;
    256 are first within 0.05 of oracle and recover 62.5% of top-16 support; 384
    close the gap to 0.02. The fully reduced 256D score reaches 0.825–0.839 OOD.
    Sixteen selected modes are compact execution only after a roughly half-width
    state computation determines which modes to use.
41. **A hierarchy shortens the address computation, but only modestly.** A 64D
    coarse scan followed by 256D refinement of 128 candidates retains exact
    score correlation 0.866–0.881 and reduced-score correlation 0.804–0.825 at
    43.8% of flat-256 candidate projection multiplies. It misses the locked
    near-flat gates; shortlist 256 comes within 0.007–0.008 of flat fidelity but
    still costs 62.5% of its candidate projections.
42. **Immediate query fidelity and downstream causal fidelity separate.** On
    finite 5%-norm layer-8 edits, the post-hoc hierarchical-128 route predicts
    its oracle's direct query effect at correlation 0.858/0.853 on
    FineWeb/WikiText, but downstream softmax-Fisher fidelity falls from
    0.779 to 0.640. The flat-256 route reaches 0.828/0.798. A fixed first-16
    baseline can retain direct-effect correlation 0.868–0.930 while downstream
    fidelity is only 0.302–0.448. Explaining one local scalar is therefore not
    the same as identifying its causally equivalent downstream circuit.
43. **The next-layer direct and attention paths are nearly additive, but very
    unequal.** For routed layer-8 edits, clamping layer-9 attention while
    allowing the changed residual and recomputed MLP preserves 0.90–0.94 of the
    full downstream Fisher effect. The complementary attention-only hybrid has
    fidelity near zero and just 6–9% of the full edit's KL magnitude. Adding
    their logit effects reaches 0.999–1.000 fidelity, while the immediate
    layer-9 bilinear interaction carries only `2e-5`–`1e-4` of full effect
    energy at the 5% edit scale.
44. **The hierarchical router's OOD error is in the dominant direct path, not
    the small attention correction.** On WikiText, hierarchical-to-oracle
    Fisher fidelity is 0.739 for the direct hybrid but 0.930 for the attention
    hybrid. Flat-256 improves both. The direct/attention additive picture
    survives edits from 1% through 20% input norm; additive fidelity remains
    at least 0.998. At 20%, however, the hierarchy's attention-path match drops
    to about 0.80, so route equivalence itself is scale-dependent even though
    path composition remains almost linear.
45. **Downstream importance is not one definition.** A naive 16-draw Fisher
    sketch fails on sharply peaked predictions because repeated samples make
    its centered covariance degenerate. Exact quadrature over the top 16 output
    tokens plus stratified tail sampling repairs this: independent estimates
    reach impact correlation 0.995–0.996 and top-16 recall 0.938. Estimator
    design is therefore part of the mechanism claim, not a numerical detail.
46. **Locally important modes and individually downstream-sensitive modes are
    largely different sets.** Their median top-16 overlap is only 0.312.
    Fisher-ranked modes beat controls matched for local query contribution by
    107–169× exact single-mode KL with a 100% pairwise win rate. Conversely,
    local modes produce roughly seven times larger immediate query effects.
    Both are valid importance notions for different questions.
47. **Individual downstream importance does not define a joint circuit.** The
    signed sum of the 16 Fisher-ranked effects retains only 0.218–0.228 of their
    summed individual Fisher energy; the exact joint intervention retains
    0.202–0.226. The linear sum predicts the joint result at 0.956–0.971
    fidelity, proving destructive addition rather than nonlinear failure. The
    local-query set is constructively aligned instead, with joint energy
    1.23–1.49 times summed individual energy. The relevant joint object is the
    signed off-diagonal Fisher Gram matrix, not its diagonal.
48. **The full signed Fisher Gram recovers a strong joint causal program.** A
    greedy 16-mode program maximizing held-out joint Fisher energy per input
    norm reaches exact KL 0.0058–0.0060 at a 5% edit, versus
    0.00085–0.00108 for local-query modes and about `3e-5` for natural-component
    Fisher diagonals. Its held-out joint energy is constructive by factors
    2.79–5.03, and an independent Fisher sketch retains 90.4–96.8% of the
    optimized objective.
49. **The program is more identifiable in effect space than in native mode
    coordinates.** Independent sketches select only 40.6% of the same modes on
    WikiText, yet their held-out downstream-effect cosine is 0.954 and objective
    efficiency is 0.904. FineWeb gives 75% support overlap, 0.992 effect cosine,
    and 0.968 efficiency. Even the aggregate input direction is less stable
    (cosine 0.699/0.840) than the downstream effect. The current canonical
    object is therefore an equivalence class under the suffix computation.
50. **This joint causal advantage is finite-scale robust but not semantic.** The
    frozen program beats the local-query route by 3.21–7.19× exact KL from 1%
    through 20% edits and keeps effect-direction cosine 0.920–0.999 relative to
    its 1% tangent. The local route consistently changes the named layer-8
    query more. Moreover, the joint program is selected separately at each
    example using suffix gradients; it is an oracle, not yet a reusable router
    or a human-readable circuit.
51. **A universal frozen suffix metric is not the missing router.** A Fisher
    quotient trained on 64 disjoint FineWeb states captures only 9–10% of
    median per-site trace at rank 64. Even full rank reaches effect fidelity
    only 0.531/0.425 on FineWeb/WikiText. Identity, random-subspace,
    unweighted-projector, and eigenvalue-shuffle controls eliminate a strong
    learned-metric interpretation: none of five locked superiority gates pass.
52. **State-neighborhood retrieval also fails OOD.** A 128-state FineWeb bank
    reaches 0.606 fidelity in-distribution at K=16 but only 0.154 on WikiText;
    K=8 is negative OOD. Similar nearest-state cosine on both corpora does not
    predict metric transfer. State-conditioned suffix geometry is not locally
    reusable under this PCA/cosine KNN definition.
53. **The full tensor-network Jacobian recovers the causal program without
    backpropagation.** Analytically propagating all 512 mode tangents through
    learned residual coefficients, RMSNorm, bilinear MLPs, squared-bilinear
    QK×Q2K2 attention, the value bus, final normalization, and unembedding
    reaches oracle-effect fidelity 0.994/0.983 and support recall 0.875/0.844.
    Its 0.1%-norm JVP is exact and its prediction of the 5%-norm finite effect
    reaches 0.984–0.986 fidelity.
54. **Attention can be small as an isolated path yet important to program
    selection.** Clamping attention derivatives leaves a useful direct tensor
    model, but its fidelity varies from 0.702 to 0.932 on a fresh panel. Adding
    exact attention JVPs makes the result uniformly near-exact. Marginal path
    magnitude and derivative importance for choosing a cooperative coalition
    are therefore different causal questions.
55. **The exact state-conditioned mode-removal Gram is compact.** Its first four
    directions carry 0.780–0.794 of energy and already yield finite-effect
    fidelity 0.997/0.982 on FineWeb/WikiText. Rank 8 reaches 0.999/0.994;
    rank 16 carries 0.926–0.940 of energy and is effectively exact. This gives
    a short contrast space for the chosen eigenmode-removal perturbations, not
    an intrinsic-rank statement about the complete Layer-8 causal operator.
56. **Mode-removal compactness is not specific to natural output queries or
    extreme sites.** On random sites, random orthonormal readouts have rank-4
    finite fidelity 0.954/0.990 versus 0.995/0.939 for natural readouts, and
    rank 16 is effectively exact throughout. Since every readout supplies a
    complete rotated 512-mode basis, this is a perturbation-family result, not
    evidence for four semantic axes.
57. **The basis-invariant unit-input causal operator is substantially broader.**
    Replacing query-dependent mode-removal amplitudes with unit perturbations in
    the fixed 512D natural-state basis reduces rank-4 energy to 0.512/0.455 and
    rank-16 energy to 0.746/0.690. Rank 64 is first to reach approximately 0.9
    (0.918/0.880). An orthogonal basis rotation changes rank energies by at most
    `2.4e-7` and preserves the leading output subspace at overlap at least
    0.999999, so this correction is genuinely basis invariant.
58. **The leading middle-layer causal axes are local, not stable verbs.** Across
    random sites, rank-4 Fisher-output subspace overlap is 0.002–0.005, top-token
    Jaccard is 0.000–0.016, and even the corresponding input-sensitivity
    subspaces overlap only about 0.010—near the random `4/512` baseline. Removing
    softmax weighting raises logit-subspace overlap to 0.165–0.174 but still
    misses the preregistered recurrence gate. Coarse prediction class explains
    little of the rotation.
59. **Softmax geometry can create apparent local compactness.** A top-256
    principal-submatrix lower bound for the pure output Fisher already carries
    0.488/0.389 of trace in four directions, close to the full unit-input
    response's 0.512/0.455. Low local Fisher rank therefore cannot by itself be
    attributed to a compact internal semantic circuit.
60. **Context-local axes are nevertheless real causal operations.** Directly
    editing the four leading unit-input directions yields tangent-to-finite
    Fisher fidelity 1.000 at 1% input norm and 0.995–0.997 at 5%. Their ten most
    affected token probability signs validate perfectly, and their median KL is
    about 48–50× that of matched random input directions. The useful object is
    a local causal verb over current alternatives, not a globally recurring
    neuron or axis.
61. **Exact output conditioning recovers readout recurrence, not an internal
    verb.** Freezing four frequent native top predictions (` the`, `.`, `,`, and
    newline) raises within-versus-between Fisher-subspace overlap by
    0.057–0.077 and the same-prediction cross-corpus advantage is 0.062. But
    unweighted-logit advantage is only 0.004–0.005 and input-subspace advantage
    only 0.001–0.003; within-family input overlap is 0.012–0.013 versus the
    random `4/512 = 0.0078` expectation. Shared prediction identity therefore
    mostly aligns the softmax-weighted behavioral metric, not the upstream
    Layer-8 sensitivity directions.
62. **Natural semantic edits do not reveal a reusable global Layer-8 direction.**
    Across 19 one-token substitutions grouped into sentiment, gender, tense,
    scale, and polarity families, even the same lexical edit has state-delta
    cosine only 0.046/0.074 on FineWeb/TinyStories. Different words in the same
    semantic family do not beat different-family controls at the locked gates,
    and changes in input, logit, and Fisher sensitivity projectors have
    essentially zero recurrence. Fixing edit position and distance and selecting
    the two largest-KL contexts per pair do not rescue any claim.
63. **The local causal operator is smooth even though it is not globally
    canonical.** A one-token semantic edit preserves the rank-4 input/logit/
    Fisher subspaces at overlap 0.989–0.997 in variable-distance contexts and
    0.985–0.995 at fixed distance. Thus nearby contexts share a local tangent
    chart; unrelated contexts do not share its axes. The semantic state delta
    itself places only 0.009–0.012 of its energy in the four strongest input
    sensitivity directions, near the random `4/512` scale.
64. **Semantic changes cross Layer 8 mainly through the residual sequence, not
    its final MLP.** A full residual-sequence boundary patch reaches
    0.934–0.956 Fisher fidelity, while the final-position residual alone reaches
    0.469–0.472 and the final MLP write only 0.107–0.134. Exact factorial
    recomposition and signed Shapley projection allocate about 0.917–0.934 to
    the residual sequence, 0.060–0.078 to the attention value bus, and only
    0.007–0.009 to repeated embedding injection.
65. **The residual carrier is a near-linear broadcast over positions.** The
    edit affects exactly the source and sixteen later positions. The final
    position is individually strongest in 86–88% of examples but is incomplete;
    intermediate positions alone reach 0.531–0.627 fidelity. Oracle top-four
    positions reach 0.860–0.907, top eight reach 0.955–0.972, and the signed sum
    of all individual-position effects predicts the whole residual patch at
    0.996 fidelity. The recurrent positional skeleton is final, source,
    immediately post-source, and late intermediates—not a fixed feature axis.
66. **The attention broadcast has exact local tensor structure but only partial
    downstream sufficiency.** Every changed Layer-0--8 attention write is
    exactly decomposed into `(QK1)(QK2)V` edges and QK1/QK2/value Shapley terms.
    Early layers are predominantly value-driven, layer 6 is routing-heavy, and
    only one to three heads dominate each local attention delta. Finite patches
    of the eight locally strongest edges reach 0.110/0.139 median downstream
    target projection at layer 4 on FineWeb/TinyStories. A whole recurrent head
    and all direct edited-source edges both fail their stronger sufficiency
    gates, so local concentration is not a complete circuit.
67. **Local edge rank is a useful search ordering, not a downstream importance
    measure.** At layer 4, local ranks 1--8 beat ranks 9--16 by 3.87×/6.61×,
    and expanding to 64 edges raises downstream target projection to
    0.294/0.241. But local rank-band projection correlates with downstream
    effect only 0.388 on FineWeb and 0.647 on TinyStories. The top 64 retain
    only about a quarter of the complete semantic-edit effect, and insertion
    versus removal agreement again weakens OOD. The route is therefore sparse
    relative to all attention edges but remains nonlinear and context-sensitive
    in the downstream network.
68. **A reusable broadcast skeleton exists between global and per-example
    circuit notions.** Sixty-four fixed edge addresses selected only on the
    unused second FineWeb context per lexical pair carry 0.183 of the held-out
    FineWeb effect and 0.108 on TinyStories at layer 4. They recover 62% and 45%
    of the respective per-example oracle-64 routes, with 59% OOD retention.
    Cyclic head shifts preserve source and destination yet retain substantial
    OOD effect; increasing 64 to 128 edges helps FineWeb but hurts TinyStories.
    Thus positions and some head identities form a reusable relay skeleton,
    while context-conditioned edge weights are required for high fidelity.
69. **The reusable relay transports value content, not primarily changed
    routing.** Exact eight-world QK1/QK2/value Shapley interventions recompose
    every frozen and oracle route within `6.8e-8`. At layer 4, value carries
    0.869/0.874 of the frozen route's median absolute downstream attribution on
    FineWeb/TinyStories; QK1 and QK2 together miss the locked 0.15 nontriviality
    gate. Frozen-64 and per-example oracle-64 factor vectors have cosine
    0.999/0.997, and the factor mix transfers across corpora with absolute-share
    L1 distance 0.060. The missing frozen-to-oracle fidelity is therefore mainly
    edge/content selection, not a different QK-versus-value algorithm.
70. **A large negative value-bus coefficient is not behavioral evidence of
    cancellation.** Layer 4 computes
    `V_mixed = 5.1875 V_current - 4.1875 V_bus`, which superficially resembles
    a subtraction code. Exact sixteen-world QK1/QK2/current-V/bus-V Shapley
    interventions instead give current-V 0.690/0.656 and bus-V 0.151/0.200 of
    frozen-route absolute attribution. Their signed target projections agree,
    with a value-path cancellation ratio of 1.000 on both corpora. The four-
    factor mix is OOD-stable and matches the oracle route. Parameter signs alone
    cannot diagnose cancellation: the represented vector directions and the
    downstream metric determine whether two paths oppose or cooperate.
71. **The transported current-value code recurs lexically and semantically.**
    Against one disjoint FineWeb reference per token replacement, raw source
    current-V deltas have same-pair cosine 0.546/0.581 and retrieve the correct
    pair at 0.737/0.579 on FineWeb/TinyStories. After the frozen 64-edge route,
    same-pair cosine remains 0.438/0.434 and retrieval 0.632/0.579. Different
    lexical pairs in the same hand-labelled family exceed different-family
    cosine by 0.142/0.221 after routing. Routing does not improve the family gap,
    so the relay preserves a structured code rather than creating the observed
    abstraction.
72. **Large embedding-injection weights do not imply direct causal dominance.**
    Layer 4 mixes `0.4629 x_residual + 8 x_embedding`, yet an exact 32-world
    QK1/QK2/residual/embedding/value-bus intervention attributes only
    0.009/0.004 of the frozen route's absolute downstream effect to the direct
    layer-4 embedding path. The incoming residual carries 0.689/0.645 and the
    inherited bus 0.151/0.201. This mix is OOD-stable and agrees with the oracle
    route. The recurrent value code is therefore already present in the state
    arriving from layers 0--3; whether those layers construct it or accumulate
    earlier embedding injections remains unresolved.
73. **The incoming residual admits an exact, highly concentrated upstream
    telescope.** Expanding
    `x_(l+1)=a_l x_l+b_l x0+attention_l+MLP_l` yields fourteen initial,
    repeated-embedding, attention, and MLP terms that reconstruct the Layer-4
    mixed-state delta within `5.1e-7`. Analytic RMSNorm/current-V integration
    reconstructs the value delta within `1.7e-6` and the frozen routed write
    within `7.1e-7`. MLP3 contributes about 0.653 of local routed projection,
    MLP2 0.19--0.20, and MLP1 0.07--0.08; four terms carry 96% of absolute
    local attribution. Finite group patches confirm that learned writes, not
    explicit embedding terms, carry the downstream route.
74. **That clean MLP hierarchy is generic transport, not semantic-specific
    structure.** A null preserves every source context, edit position, and
    target-token count but deranges targets across semantic families. Its
    absolute component profile has cosine 0.999 with the semantic profile,
    MLP3 share rises from 0.662/0.667 to 0.682/0.682, and top-four concentration
    remains 0.961--0.967. Downstream route strength is also comparable. Thus
    the telescope identifies where arbitrary token changes are processed; the
    earlier recurrence tests identify semantic structure in the directions
    carried there. Carrier localization and content interpretation are distinct
    explanatory tasks.
75. **Layer-3 native channels contain reusable edit addresses, but no small
    sufficient native circuit.** All 4,608 product channels exactly reconstruct
    the MLP3 write delta, while oracle top-64 recover only 0.381/0.402 local
    fidelity and top-256 only 0.571/0.605. The latter nevertheless reach
    0.646/0.713 downstream fidelity to the complete MLP3-only patch. Top-64
    channel sets recur above random for identical lexical pairs and semantic
    families, and semantic versus cross-family-null sets have Jaccard below
    0.10. Thus the native basis carries structured routing information, but the
    local edit remains broad and the downstream network quotients much of it.
76. **The reusable native address prior is partly semantic and partly generic.**
    Frozen top-256 pair/family sets selected only on unused FineWeb contexts
    reach 0.324/0.432 and 0.350/0.456 downstream fidelity on held-out
    FineWeb/TinyStories, far above random and a cyclic wrong-family control.
    Leave-one-pair-out family sets still reach 0.378/0.295 and beat the median
    and even per-example best wrong-family baselines in aggregate. But a global
    set beats family LOPO on TinyStories (0.431 versus 0.295), removing the same
    lexical pair costs 0.161 OOD fidelity, and increasing-scale fails at the
    family level. Native channel groups are therefore useful family-conditioned
    address priors, not disjoint monosemantic modules or complete circuits.
77. **Reusable Layer-3 structure is not an isolated full-tensor spike.** The
    complete symmetric-interaction-to-write spectrum has a smooth rank-33--512
    log-log fit (`R²=0.968`, nominal exponent 0.241), but every one of its top
    four extrapolation excesses is negative and top-four energy is only 0.021.
    The absence of a strict positive spike survives all fifteen frozen bulk-fit
    windows. Thus the transferred family address sets coexist with a continuous
    global spectrum: a clean/reusable computation need not be a global spike.
78. **“Power-law bulk” and “exceptional learned shoulder” are different
    claims.** Layer 3's top four singular values exceed every one of 32 sign or
    Down/form-pairing nulls at matched rank by 7--12%, and their energy exceeds
    every null. Yet their relative-bulk excess beats every null in only 60% of
    fit windows, and an exponential-rank curve predicts alternating held-out
    ranks 33--1024 with log-RMSE 0.022 versus 0.089 for the power law. The robust
    result is learned cross-channel coordination near the top of a continuous
    spectrum—not a uniquely identified RMT law or four separated circuits.
79. **Family-conditioned quadratic forms isolate a clean but small semantic
    core.** A leave-one-pair-out rank-4 family output basis contracts Layer 3
    into four exact symmetric interaction matrices. Their full eigensystems
    reconstruct the family projection within `2.7e-6`; 64 modes per axis reach
    0.893/0.938 downstream fidelity to that projection and 256 reach
    0.977/0.983. The family projection beats global and wrong-family rank-4
    controls, but explains only 0.118/0.188 of the complete MLP3 effect. Clean
    conditional spectral circuits exist without being the dominant carrier.
80. **A reusable native address set is not equivalent to a fixed semantic
    output feature.** Expanding the leave-one-pair-out family output basis from
    rank 4 to 32 raises full-MLP fidelity only to 0.224/0.299, versus
    0.378/0.295 for the same family’s 256 native channels. At ranks 16 and 32,
    the family basis no longer consistently beats global or best-wrong-family
    bases. The native group is therefore better modeled as a fixed bank of
    bilinear operations with context-dependent coefficients and output
    directions, not as neurons jointly writing one stable family vector.
81. **The native operation bank has a weight-visible output hierarchy, but its
    causal tail remains important.** SVD of the frozen family bank's 256 `Down`
    columns recovers 0.632/0.510 of the exact bank's downstream effect in four
    directions, 0.804/0.720 in sixteen, and 0.823/0.793 in thirty-two. Bottom
    singular and random directions are essentially inert. A discovery-
    activation PCA is consistently worse, so the hierarchy is genuinely
    weight-defined and transfers OOD. Nevertheless, rank 32 retains only
    0.231/0.241 full-MLP fidelity versus 0.378/0.295 for the complete bank:
    compact dominant axes coexist with a behaviorally relevant long tail.
82. **A reusable edit family is a coupled operator bank and write subspace, not
    merely a channel list.** Holding the exact family-bank write fixed, its own
    weight-SVD basis reaches 0.823/0.793 native-effect fidelity at rank 32,
    versus 0.649/0.708 for a global-bank basis and 0.395/0.412 for the best
    wrong-family basis. Family and global top-32 bases have mean canonical
    cosine only 0.413. This specificity transfers OOD. But rank 64 still reaches
    only 0.860/0.914 and loses about 0.10 full-MLP fidelity. The emerging module
    is therefore a medium-rank family-conditioned bilinear operator with
    context-dependent coordinates and a causal tail—not one neuron, one fixed
    feature vector, or a completely basis-free distributed web.
83. **The family operator has a compact, causally used input discriminator
    subspace.** Contracting each bank along its top four output axes and forming
    the rotation-invariant input metric `Σ_j A_j²` puts a median 0.656 of form
    energy in 32 directions. Its rank-32 eigenspace overlaps the inclusive
    correct-family metric by 0.734, versus 0.269 global and 0.408 best wrong.
    More importantly, projecting held-out residual changes through just four
    correct input directions recovers 0.791/0.898 of the exact output-core
    effect; 32 recover 0.976/0.967, while low and random controls are near zero.
    Raw concentration is not specific—sign nulls retain 0.635 top-32 energy—so
    family recurrence and held-out causal transfer, not the scree curve, carry
    the interpretation.
84. **The stable input object is broader than the signed top-four axes.** The
    correct rank-32 subspace retains 0.949--0.976 core fidelity from 10% through
    full edits, but rank four falls to 0.487 on FineWeb near the tangent regime.
    The most-overlapping sign null and an unweighted Left/Right factor-span
    basis approach the top-four effect, whereas Down/form pairing nulls collapse
    to 0.003/0.021 at full scale. Thus selected-channel input geometry supplies
    much of the subspace; learned pairing to output writes refines it. The top
    four capture only 3--5% of raw state-delta energy, so their effect is high
    gain rather than ordinary variance capture.
85. **Causal input rank defines several family regimes.** For the declared
    four-axis output core at full edit scale, polarity has `K90=4`, sentiment
    `K90=4`, tense `K90=8`, gender `K90=32--64`, and increasing-scale
    `K90=128--256`. Full-scale ranks transfer across FineWeb and TinyStories
    within a factor of two, but tangent ranks can differ sharply: FineWeb
    sentiment rises from 4 to 32. Causal K90 is therefore a useful conditional
    complexity measure, not an intrinsic neuron count or a single model-wide
    monosemanticity score.
86. **The leading input axes inherit substantial factor geometry, while learned
    channel pairing makes them executable.** The most-overlapping sign null has
    rank-32 overlap 0.896 with the observed metric and its top four recover
    0.696/0.810 of the full-scale core. An unweighted Left/Right span reaches
    0.590/0.805. In contrast, Down/form-pairing nulls have overlap 0.244 and
    causal fidelity 0.003/0.021. Exact signed top-four axes are therefore not
    canonical; a broader factor-defined input subspace plus learned pairing is.
87. **A weight-defined quadratic verb becomes a context-conditioned linear
    program near a natural state.** For projected edits, the exact identity
    `Δq=2δᵀAx+δᵀAδ` separates the current-context cross term from within-edit
    interaction. At 10% scale, the linear term has norm share 0.918/0.945 and
    downstream fidelity 0.990/0.998. At full scale, quadratic share rises to
    0.472/0.369 and FineWeb linear-only fidelity falls to 0.337. Replacing the
    current state `x` with a discovery mean loses 0.933/0.607 fidelity. The
    float32 recomposition gate narrowly misses at `2.1e-5`, but float64 verifies
    the identity to `5.2e-14`; this is numerical cancellation, not mechanistic
    failure. Stable weights and live contextual operands are complementary
    parts of the circuit.
88. **Full-scale quadratic computation is sparse inside the right quotient,
    but its sparsity varies by family.** In each causal input basis, frozen
    signed coordinate pairs reach quadratic-effect `K90` at 16/528 for
    sentiment, 16/136 for tense, 256/2,080 for gender, 1,024--4,096/32,896 for
    increasing-scale, and approximately 4--10/10 for polarity. All ranks are
    OOD-stable within the declared factor-four gate. At budget 256, coefficient-
    magnitude ranking matches discovery-activation ranking within 0.009 and
    beats random by 0.69--0.73, making the pair program largely weight-visible.
89. **“Verb” does not mean “off-diagonal interaction” universally.** Gender's
    off-diagonal-only program reaches 0.879/0.626 quadratic fidelity, while
    sentiment and tense off-diagonal-only patches are negative or near zero;
    increasing-scale uses both diagonal and off-diagonal terms, and polarity is
    dominated by a tiny signed cancellation program. Off-diagonal tensor energy
    ranges only 0.31--0.52. The reusable notion is a sparse signed pair program
    in a family-specific causal basis; whether its pairs represent feature
    strength, cross-feature interaction, or cancellation is mechanism-specific.
90. **The leading family input modes are semantically useful but not canonical
    concept axes.** FineWeb selects one of eight frozen modes and a signed or
    magnitude token predicate; those explanations transfer to TinyStories at
    AUROC 0.797--0.994 and beat isotropic random directions by 0.145 in median.
    Yet a frozen random rotation of the same eight-dimensional subspace trails
    by only 0.015 median, and increasing-scale's chosen mode predicts polarity
    much better than scale. Thus semantic information belongs primarily to the
    family-relevant subspace; the metric eigenvectors are convenient weight-
    defined coordinates, not neuron-like atomic meanings.
91. **Four family subspaces contain genuine lexical semantics, but their
    current-token evidence is inherited from the embedding.** With disjoint
    manually frozen words, sentiment, gender, tense, and polarity reach
    0.864--0.996 AUROC on both TinyStories and WikiText. Direct projection of
    the repeatedly injected current-token embedding is equally good or better,
    so this is not evidence that Layer 3 newly constructs those concepts.
    One token later the current embedding no longer supplies the label:
    sentiment remains 0.814/0.690, polarity 0.898/0.838, and tense is
    0.476/0.720, while gender reverses below chance. Lexical inheritance and
    contextual transport are distinct mechanisms even along the same axis.
92. **Increasing-scale is a causal high-rank family regime without a clean
    leading semantic axis.** Its frozen leading mode reaches only 0.596/0.566
    on unseen scale words and is an almost perfect detector of a polarity pole.
    A seed-trained decoder over the correct family subspace reaches K80 at rank
    32 and 0.964/0.954 by rank 256, but random and best-wrong rank-256 spaces
    reach 0.951/0.946 and 0.959/0.947. Its offset-one score is only 0.461/0.599.
    High-rank decodability therefore confirms that information is present, not
    that the selected tensor bank uniquely represents, transports, or uses it.
93. **“Semantic representation” needs at least four separate qualifiers.** A
    direction may classify tokens; a subspace may contain a linearly decodable
    concept; a coordinate may persist after the lexical token; or a weight-
    defined operator may use that coordinate causally and specifically. The
    present Layer-3 results realize all separations: sentiment/polarity are
    broadly shared lexical axes, gender is a compact token-local discriminator,
    tense has a compact family-specific axis with corpus-dependent persistence,
    and increasing-scale requires a broad causal program without a clean owned
    semantic coordinate. None of these outcomes is summarized faithfully by a
    single monosemanticity score.
94. **The proposed shared Layer-3 backbone is a cluster, not a universal
    semantic basis.** The top-eight mean-projector subspace overlaps sentiment,
    increasing-scale, and polarity family spaces by 0.899--0.928, but gender
    and tense by only 0.069/0.147. The cluster-to-outside projector-affinity gap
    remains 0.265--0.426 as family rank varies from 16 to 64. What is reusable
    is therefore a weight-stable lexical-operator cluster plus distinct
    residual families, not one global semantic spine.
95. **A shared/residual quotient can be exact without defining canonical
    modules.** Rotating each family top-32 input space into shared-like eight
    and residual 24 coordinates preserves its projector to `5.6e-7`, but the
    mean-projector rank-8/rank-9 ratio is only 1.042 and causal labels change at
    rank 16 for gender and scale. The bilinear cross term retains 14--15% write-
    norm share at rank eight. Orthogonality of state coordinates does not imply
    causal additivity, and a convenient cut is not an ontological boundary.
96. **Frozen unseen-word interventions recover two robust causal regimes and
    one conditional regime.** Across ten disjoint FineWeb lexical pairs and
    repeated natural contexts, sentiment/polarity family patches are shared-
    like, while gender/tense are residual-dominant; every pair has the same sign.
    Scale is shared-like for all ten FineWeb pairs but splits across TinyStories
    pairs. These fidelities are relative to a family top-32 patch, and their
    projections onto the full semantic logit change are only 0.002--0.063.
    They identify internal organization, not task-level sufficiency.
97. **A regime label needs a sampling unit.** For TinyStories scale,
    occurrence-weighted shared-minus-residual fidelity is -0.029, while the
    lexical-pair-balanced median is +0.811: two frequent pairs dominate token
    occurrences, whereas four rarer pairs are shared-like. Occurrence weighting
    answers expected corpus behavior; pair balancing answers behavior of a
    typical chosen contrast; context-level unanimity asks a stronger mechanism
    question. None can silently stand in for the others, and manually selected
    pair bootstraps do not imply population sampling.
98. **Thomas-Dooms-style outliers emerge after a natural-write output question
    is fixed even when the global tensor lacks strict spikes.** Each frozen,
    family-labeled output-4 space was rotated by its exact restricted tensor Frobenius Gram,
    making the result invariant to the original basis. Median top-four input-
    interaction energy is 0.046--0.078 versus 0.021 globally. All five family
    subspaces beat every one of 64 equally canonicalized random output-4 spaces,
    including after dropping the tensor-energy-leading axis. This comparison
    establishes exceptional natural-write geometry, not yet semantic
    specificity; a later matched label-shuffle control is required for that.
99. **Exceptional learned coordination is more robust than extrapolated power-
    law spikes.** With 64 sign and 64 coefficient-pairing nulls per axis, four
    families retain at least three of four canonical axes at empirical
    `p<=0.05` under both nulls; gender retains two. The initial claim that all
    20 axes exceed every sign-null maximum fails. Only two of 20 semantic axes
    have positive excess above the fitted power-law curve. Null comparison is
    therefore a useful candidate test; “above a power law” has poor recall in
    this layer.
100. **Spectral circuit status is conditional on the operator being queried.**
    A global tensor spike asks which output operations dominate Frobenius
    energy across the whole layer. An output-conditioned spike asks whether a
    named write subspace has exceptional low-rank input structure. A natural-
    state spectrum asks which of those operations are actually used. These can
    disagree without contradiction. None establishes a complete circuit until
    held-out finite interventions connect the weight form to downstream
    behavior; conversely, validated causal computation may remain inside a
    continuous global spectral shoulder.
101. **Conditioned spectra find causal neighborhoods better than they rank the
    program inside them.** On disjoint unseen-word edits, every correct family
    output-4 space beats the best wrong-family and the five most concentrated
    random-subspace controls. Omitting its tensor-energy-leading axis loses at
    least 0.20 family-patch fidelity for all five families. Yet axis zero alone
    exceeds 0.50 fidelity on both corpora only for polarity; two axes clear 0.70
    only for sentiment and polarity; rank-64 input truncation ranges from 0.694
    to 0.979; and top-four spectral concentration correlates only 0.293 with
    held-out absolute necessity. The family patch itself has at most 0.247
    full-MLP fidelity. Spectra are therefore excellent breadth-first candidate
    generators, not a causal importance ordering or a certificate of circuit
    sufficiency.
102. **Conditioned spectral organization undergoes a sharp late-layer phase
    change.** With identical semantic edits, median conditioned top-four energy
    is 0.071/0.061/0.054 at layers 0/3/8 but 0.507 at layer 17. Strict fitted
    spikes rise from 1--2 of 20 to 19 of 20, and family/global-top-four overlap
    rises from 0.035--0.043 to 0.840. Yet same-family output-space recurrence is
    only 0.022 from layer 0→3, 0.018 from 3→8, and 0.005 from 8→17. Late
    concentration is a new shared readout regime, not persistence of the same
    family modules through depth.
103. **The five Layer-17 “family” spectra collapse into one generic operator
    neighborhood.** Median cross-family output overlap is 0.764 and the
    rotation-invariant input-metric top-32 overlap is 0.932. Family identity is
    not recoverable from trajectories inside the global top-four space
    (LOPO accuracy 0.158, permutation `p=0.669`), and the held-out local
    projection gap is negative. On unseen words, however, neither correct nor
    wrong family output-4 patches are sufficient: the correct patch has only
    0.082/0.127 aggregate full-MLP fidelity, while best-wrong fidelity to that
    restricted patch is 0.567/0.462 and varies sharply by family. The layer has
    a common spectral bottleneck plus a large contextual tail, not five clean
    semantic circuits or one sufficient universal circuit.
104. **Isotropic output controls seriously overstate semantic evidence.** A
    matched null preserves all 19 natural counterfactual write trajectories and
    group sizes while shuffling their family labels. No tested layer's observed
    semantic grouping exceeds the label-null 95th percentile in conditioned
    concentration. At layer 17, shuffled groups reproduce median top-four
    energy 0.512 versus 0.507 observed, subspace overlap 0.757 versus 0.764, and
    20 versus 19 strict spikes. Its label-null 95th percentile is about twelve
    times the isotropic-random threshold. The spike is a property of selecting
    the natural-write manifold; the supplied semantic partition adds no
    detectable spectral concentration.
105. **Empirical-null pass counts need a fixed resolution.** With 32 null draws,
    an empirical `p<=0.05` requires zero exceedances; with 64 draws it permits
    two. Layer-3 robust-axis counts therefore change across otherwise matching
    runs even though family concentration replicates within 0.01. Cross-depth
    claims should use fixed pooled nulls, effect sizes, or quantiles—not compare
    thresholded axis counts from different Monte Carlo resolutions. The matched
    natural-write label null carries the semantic-specificity conclusion here.
106. **The unrestricted Layer-17 object is a dominant bottleneck plus a second
    compact tail, not outliers plus featureless noise.** The four strict raw-
    weight outliers capture 0.847--0.857 of centered natural-write variance and
    0.909--0.939 of the variance captured by each corpus's optimal four-space.
    Keeping only those four directions nevertheless gives median final-
    distribution KL 0.126--0.156. The orthogonal 14--15% tail has effective
    rank only 4.2--6.2; its frozen FineWeb rank-16 space captures 0.944--0.993
    of the OOD-optimal tail-16 variance. Adding it lowers KL to 0.041--0.054,
    still far from an exact replacement. Variance dominance, causal necessity,
    and behavioral sufficiency are different orderings.
107. **The residual tail contains causal bilinear operations that the strict
    global-outlier rule misses.** Its leading four output axes best match raw
    tensor spectral ranks 5--7, but the raw top 64 recover only 0.666 of their
    subspace; natural covariance weighting raises this to 0.818. Their raw
    input forms place only 0.124--0.285 of Frobenius energy in four modes, yet
    fixed-5%-input-norm top-mode edits produce 37,000--101,000 times the median
    low-mode scalar effect, reach final logits on all corpora, and are predicted
    by the exact quadratic formula at correlation 1.000. Raw parameter energy
    is therefore a poor measure of natural causal concentration.
108. **Covariance cleaning reveals a compact natural computation but does not
    by itself establish exceptional learned structure.** Rewriting the same
    four forms in frozen FineWeb second-moment coordinates and separating the
    isotropic trace raises top-four energy to 0.990--0.994 and lowers effective
    rank to 1.1--2.6; rank-four score correlations remain 0.94--1.00 OOD.
    However, 128 sign nulls reproduce this concentration: only tail axis four
    has null-exceeding maximum strength (`p=0.031`). All four beat coefficient-
    pairing nulls (`p<=0.023`). Thus the learned pairing of output coefficients
    with input forms is robust, while exceptional signed cooperation is robust
    for only one coordinate. Input covariance itself creates most of the
    apparent low rank.
109. **The tail is more canonical as a tensor map than as four named axes.**
    Structural output roles are decodable from both bottleneck and tail, but
    the original tail-PCA axes do not beat rotations of the same output space
    semantically (`p=0.308`) or spectrally (`p=0.457`). A rotation-invariant
    four-output tensor Gram instead yields energy shares
    0.506/0.237/0.163/0.094. Its shared input basis reaches 0.990 tensor energy
    at rank eight and held-out median per-axis score correlation 0.957--0.982.
    A rank-16 program reaches score `R²` 0.951--0.986 and behavioral KL
    0.004--0.010, but fails the locked exactness gates. Cross terms are
    indispensable: diagonal-only rank-16 forms fall to `R²` 0.330--0.371.
110. **Tensor-canonical coordinates remain metric and distribution dependent.**
    FineWeb-defined canonical output axes match a disjoint FineWeb panel at
    0.993 and WikiText at 0.954, but rotate to 0.772 mean cosine on TinyStories.
    The four-dimensional output subspace and rank-eight interaction program
    transfer better than the named coordinates. Use the subspace/program for
    cross-domain mechanism claims; use individual axes only inside the declared
    covariance metric and corpus regime.
111. **“Clean circuit” now has at least five operational definitions.** A raw
    RMT outlier is distribution-free but low recall; a null-exceeding
    conditioned form identifies learned coordination; a covariance-cleaned
    form gives a short natural-state predictor; a basis-invariant tensor
    program gives a coordinate-free local mechanism; and a behavioral quotient
    preserves the final distribution. In Layer 17 these definitions select
    overlapping but nonidentical objects. The appropriate definition depends
    on whether the use case is discovery, description, intervention, OOD
    transfer, or replacement.
112. **The pulled-back final-logit Fisher metric is locally correct but does
    not make discrete tensor-axis selection behaviorally sufficient.** At 10%
    edit scale, quadratic Fisher predictions correlate 0.998--1.000 with exact
    KL and calibrate at 0.997--1.000; even at full scale correlation remains
    0.891--0.967. Greedy Fisher, score-MSE, and Frobenius selection nevertheless
    choose the same top-eight axis set, and rank 16 still misses the universal
    `KL<=0.005` gate. The failure is not primarily finite-softmax curvature.
113. **A continuously optimized behavioral subspace is real, mostly stable,
    and not identical to the variance core.** Two parity-split rank-eight fits
    initialized near the Frobenius basis agree at mean canonical cosine 0.974
    and reduce exact mean KL on FineWeb/WikiText/TinyStories from
    0.0217/0.0327/0.0078 to 0.0150/0.0196/0.0066. Random and deliberately
    orthogonal starts retain the held-out Fisher improvement. Roughly six
    canonical cosines are consistently near one; the last one or two can fall
    below 0.2 across splits or starts. The canonical object is therefore a
    stable behavioral core with a less identifiable boundary, not eight named
    axes.
114. **Behavioral rank has a plateau that raw rank does not reveal.** Increasing
    the continuously optimized subspace from rank 2 to 32 monotonically lowers
    held-out FineWeb Fisher error, but WikiText median KL plateaus near
    0.0052--0.0062 and next-token CE remains about 0.021--0.023 nats worse.
    None of ranks 2--32 reaches the predeclared universal fidelity gate. A small
    behavioral core is an excellent approximation, not a complete quotient.
115. **Bilinear dimension is not additive feature count.** Taking the union of
    the variance-preserving and Fisher-preserving rank-eight spaces restores
    score `R²` above 0.95 but worsens behavioral KL because it also introduces
    their cross terms. Suppressing those cross terms changes the final output
    materially (mean full-to-block KL 0.0028--0.0084), yet still does not
    consistently beat Fisher-eight. Thus “add another direction” can damage a
    bilinear program even when both constituent subspaces are useful. Circuit
    budgets must count interactions or blocks, not directions alone.
116. **The Layer-17 residual is conditionally compact even though no fixed
    global quotient is sufficient.** A linear router over 32 frozen natural-
    state coordinates selects one of four learned rank-eight tensor experts.
    Hard routing lowers mean exact KL relative to a parameter-matched global
    rank-32 program from 0.0123/0.0166/0.0054 to
    0.0088/0.0107/0.0035 on FineWeb/WikiText/TinyStories and reaches the locked
    median-`KL<=0.005` gate. Soft routing is only slightly better. Conditional
    execution rank and stored program rank are therefore different capacities.
117. **The routing gain is contextual, not an averaging or position artifact.**
    Shuffling hard assignments while preserving expert usage raises mean KL by
    roughly 3--5x. Uniform, global-mean, position-only, and every fixed-expert
    control fail on all three corpora. Expert subspaces have pairwise mean
    canonical cosine 0.727--0.806, so the router is not choosing redundant
    copies of one basis.
118. **Conditional compactness has an interior granularity.** With 32 total
    stored expert directions, the exact same training/evaluation protocol ranks
    `4x8` best on all three corpora; `2x16` helps less, `8x4` remains useful,
    and `16x2` fragments the program too far. The latter improves the local
    Fisher objective but worsens geometric-mean exact KL relative to `1x32`.
    Local sensitivity is a useful training metric, not a guaranteed ordering of
    very fine conditional decompositions.
119. **The routed function class is more canonical than its expert names.** Four
    independent `4x8` fits have only 3.7% coefficient of variation in
    geometric-mean hard KL, and permutation-aligned expert subspaces recur at
    cosine 0.814--0.868. Exact route labels agree on only 32--46% of sites after
    that alignment; pairwise adjusted Rand indices are 0.08--0.17. The same-
    context selected-subspace field beats shuffled-context nulls, but by only
    0.02--0.03 cosine and several cells miss 0.80. Treat four named experts as a
    useful quantization, not unique modules.
120. **Coarse routing semantics recur even when partitions do not.** On 32,000
    disjoint FineWeb sites, current-token identity predicts 46--52% of held-out
    routes and a FineWeb lookup transfers at 45--51% on WikiText and 47--59% on
    TinyStories. Structural shape alone predicts 33--38%; position remains near
    27--29%. After permutation alignment, token/shape/position semantic profiles
    match seed zero at cosine about 0.98. The stable ontology is lexical and
    structural routing geometry, not exact cluster membership.
121. **Exact loss and task loss define different quotients.** The Fisher-trained
    router transfers better than the fixed 120-step exact-KL fine-tune in every
    tested OOD cell, while adding observed-token CE does not make CE delta
    nonpositive. This specific optimization result does not prove Fisher is
    intrinsically superior: it establishes that finite exact-loss optimization
    can overfit and must be early-stopped and independently validated. Native-
    distribution imitation, observed-token utility, and conditional execution
    cost remain separate targets.
122. **A middle-layer sensitivity address is reusable, but its complete write is
    not a fixed verb.** A 64-prototype bank for the direct Layer-8 MLP Jacobian
    captures 0.33--0.38 of held-out tangent energy under state-only routing and
    beats matched random banks. The same address has only modest recurrent
    direct-output overlap and essentially no recurrent suffix-Fisher output
    role. The prototype says where the tensor is sensitive; the other input
    operand still determines what it writes.
123. **Fixing both the address family and route does not make the Layer-8
    derivative low-rank.** The exact restricted object has shape
    `output x four edit coordinates x full context`. Equal Tucker rank 128 gives
    held-out derivative `R²=0.577--0.758`; rank 512 gives 0.811--0.893; every one
    of eight frequent routes first clears 0.90 at rank 768. Full rank reconstructs
    exactly. Context-only and output-only truncations both fail, a `10^4` ridge
    sweep is stable, and a global FineWeb covariance slightly outperforms the
    route-conditioned estimate. Conditional local sensitivity and a compact
    complete conditional operator are different claims.
124. **Sparse native attribution coexists with broad native sufficiency.** In
    the exact Layer-8 CP/neuron basis, median per-state individual-energy support
    is only 20.5--103 channels. Nevertheless no fixed 512-channel set or
    per-state oracle 256-set reaches derivative `R²=0.90`; fixed K90 is
    2,048--4,096 of 4,608 channels. Training-ranked sets strongly beat random
    controls. Individual channel energies sum to only 0.533--0.794 of total
    write energy, showing a long positively cooperating tail rather than
    cancellation noise.
125. **Accounting for cooperation does not recover a small sufficient neuron
    set.** For the squared-write game, each channel's exact Shapley allocation
    is `<C_h, sum_k C_k>` and the allocations sum to total energy within
    `1e-7`. Aggregate train/test allocations recur at cosine 0.975--0.999, yet
    Shapley K90 remains 2,048--4,096 and its rank-256 gain over isolated-energy
    ordering is only 0.002 median `R²`. A train-fitted scalar and a 4x4
    address-coordinate calibration add only about 0.003--0.011 at rank 256;
    squared directional fidelity remains 0.576--0.722. The weak tail carries
    distinct write geometry, not merely missing gain or a coordinate rotation.
126. **“Sparse” now has at least four incompatible operational meanings.** Low
    effective support says a few terms dominate an additive attribution;
    sparse necessity says deleting a few terms matters; sparse sufficiency says
    keeping a few terms reconstructs the computation; sparse execution says a
    router actually evaluates only a few terms. Layer 8 is sparse by the first
    meaning and broad by the third. Interpretability claims must name the
    intended intervention and use case.
127. **The power-law bulk replicates independently; the four-spike late phase
    does not.** In a separately trained 12-layer, width-768 bilinear checkpoint,
    layers 0/5/11 again have bulk log-log `R²=0.956/0.967/0.947`, and sign and
    coefficient-pairing nulls retain the bulk. Layer 11 has only two positive
    fitted deviations, while Layer 5 has the strongest single deviation
    (`+0.217`) and largest raw top-four energy (0.053). Final-layer four-spike
    concentration is therefore a checkpoint-specific regime, not yet a scaling
    law.
128. **Independent raw outliers are naturally used but incomplete.** In the
    12-layer checkpoint, raw top-four spaces recover 0.567/0.684/0.874 of the
    optimal natural rank-four variance at layers 0/5/11. The Layer-5 leading
    spike alone recovers 0.894 of optimal rank-one variance. Raw four-spaces
    beat fixed random spaces behaviorally, but retaining them still incurs
    median final-distribution KL 0.160/0.089/0.169. The independent model
    reproduces “useful circuit candidates, not complete circuits,” not the
    original depth ordering.
129. **Covariance cleaning is an estimator with a bias/variance choice, not an
    intrinsic tensor property.** With only 1,024 states for 768 dimensions,
    held-out whitened second-moment mismatch reaches 8--23 and creates
    catastrophically negative quadratic-score `R²`; mismatch predicts failure
    at correlation 0.923. At 8,192 states with ridge 0.1, median rank-four score
    correlation/`R²` becomes 0.895/0.746, 0.740/0.535, and 0.974/0.948 across
    layers 0/5/11. No sample/ridge setting makes all twelve forms reach 0.90.
    Late cleaned compactness is robust; middle cleaned compactness is not.
130. **Raw, cleaned, natural, and behavioral spectra are four different
    definitions of a circuit.** Raw spikes are distribution-free discovery
    priors. Regularized covariance-cleaned spectra summarize a declared input
    distribution. Empirical score fidelity includes higher moments and
    covariance-transfer error. Final KL asks whether discarded writes matter
    after the suffix. The independent replication shows that these orderings
    can disagree without any algebraic contradiction.
131. **The independent final layer replicates a natural bottleneck plus a
    stable residual tail, but not a small sufficient circuit.** On disjoint
    FineWeb, WikiText, and TinyStories, the weight-only Layer-11 raw four-space
    captures 0.693--0.753 of centered write variance at 0.863--0.899 of the
    corpus-optimal rank-four value. After removing it, residual writes have
    effective rank 14.6--21.2; a frozen FineWeb tail-16 space transfers with
    mean canonical cosine 0.730--0.911 and 0.852--0.972 of optimal tail-16
    variance. This geometry is reproducible across corpora even though the
    exact axes need not be semantic neurons.
132. **Variance-compact tails can remain behaviorally broad.** Retaining only
    raw four gives median final-distribution KL 0.157--0.188. Adding the stable
    tail-16 lowers it to 0.083--0.108, but tail-64 still leaves 0.060--0.075 and
    tail-128 leaves 0.042--0.059. Natural rank-20 performs about as well as the
    weight-anchored 4+16 space. Thus the low-variance directions discarded by
    PCA remain collectively relevant after final RMSNorm and unembedding.
133. **Behavioral metric alignment is real but does not explain away the broad
    tail.** Exact-KL optimization, selected on interleaved FineWeb validation
    data, improves validation by 4.5--8.5% and usually transfers modestly while
    sacrificing a little write variance. Yet selected rank-68 spaces remain at
    median KL 0.054--0.072, far above the 0.02 near-sufficiency gate. The
    strongest warranted conclusion is a metric mismatch plus unresolved broad
    or routed computation, not a hidden fixed low-rank circuit.
134. **Keeping the spectral outliers is not the bottleneck.** Across exact-KL
    fits, forcing the raw four-space into the learned rank-20 or rank-68 basis
    costs at most about ten percent and can improve OOD behavior. The outliers
    are therefore useful anchors, but the remaining problem is how the tail is
    selected or composed by context. The next discriminator is an independently
    fitted router, not another claim that four global directions are complete.
135. **Single-expert conditional routing gives a weak, real compression gain but
    not a sufficient circuit.** Four learned experts share the raw four-space
    and each add a 16-dimensional tail; a hard upstream-state router therefore
    retains 20 directions per token from a bank with at most a 68-dimensional
    union. Hard routing lowers mean KL from fixed-rank-20 values of
    0.107--0.136 to 0.086--0.121, and shuffled or position-only assignments are
    worse. It does not match the fixed or learned rank-68 union, close 70% of
    the oracle gap, or approach the 0.02 sufficiency gate.
136. **Conditional execution support is smaller than the global union only in
    a relative, partial-model sense.** With the trained bank frozen, routed
    top-two expert unions use 36 directions and come within 10% of the full
    union on all three corpora. They do not beat matched fixed rank 36 by 20%,
    do not close the oracle gap, and remain at mean KL 0.078--0.109. Top-three
    and top-four continue to help. This is useful approximate compression, not
    evidence that the native computation executes a complete sparse circuit.
137. **Router performance is reproducible while its decomposition is not.** Three
    independent fits have hard-router KL coefficient of variation only
    0.2--1.6%, and every seed beats fixed rank 20. Yet the learned 68-dimensional
    unions overlap the primary union by only about 0.60, matched expert tails by
    0.59--0.62, aligned hard routes agree on only 34--47% of tokens, and top-two
    route sets have Jaccard 0.41--0.59. Expert identities and route labels are
    not canonical explanatory objects.
138. **The unstable routers are not merely gauge-equivalent implementations of
    one recovered function.** Pairwise selected-projector overlap is only
    0.47--0.55; reconstructed writes disagree by 18--22% of their native
    reconstruction error; prediction-to-prediction symmetric KL is 19--27% of
    native approximation KL; and approximate top-one agreement is about 90%.
    What recurs is the aggregate error and per-example difficulty profile
    (`r=0.84--0.95`), not the reconstructed write or output distribution.
139. **“Correct” has to be indexed by use case.** The raw four-space is correct
    as a weight-only discovery prior and natural-variance bottleneck. The stable
    tail PCA is correct for descriptive covariance compression. Exact-KL fitted
    spaces and routers are better lossy behavioral compressors. None is yet a
    correct mechanistic decomposition, because neither fixed nor routed
    replacements are sufficient and the routed functions fail restart
    equivalence. Stable performance alone licenses an engineering regime claim,
    not interpretation of individual experts or examples.
140. **A dominant weight-only output mode induces stable, exact input addresses,
    but stability does not make them spike-specific circuits.** At independent
    Layer 5, contracting the exact bilinear tensor with its leading output mode
    and covariance-cleaning the resulting quadratic form produces input
    addresses with split-half canonical cosine 0.986--0.990. Their exact
    restricted derivative operators transfer across FineWeb, WikiText, and
    TinyStories, and rank 64 explains 0.83--0.99 of operator energy. Yet
    nonleading and sign-scrambled controls are often even more compact. The
    addresses are canonical sensitivity probes, not unique evidence that the
    spectral spike owns their computation.
141. **Compact operator rank can conceal broad native-channel sufficiency.** The
    four leading cleaned address operators have singular K90 of 2, 112, 63, and
    218, while exact native-channel Shapley K90 is 35, 804, 549, and 1,428.
    Dominant channels explain much of attribution, but a fixed top-128 set
    reaches only 0.45--0.97 reconstruction R2. Individual channel energies sum
    to less than total energy and negative Shapley mass is tiny, identifying a
    broad cooperative tail rather than cancellation-dominated storage.
142. **A recurrent channel backbone is real but structural rather than
    spike-specific.** Channels 196 and 202 occur in every leading-address top
    eight, the four-address normalized attribution matrix has 0.946 first-mode
    energy, and recurrence strongly beats coordinate-permutation and random
    natural-address nulls. Sign-scrambled controls recur even more strongly,
    however. These channels are privileged coordinates of the layer's natural
    computation, not semantic evidence supplied by the outlier alone.
143. **Only one member of the apparent core has a robust simple predicate.** On
    fresh held-out data, channel 202 is suppressed at newline/boundary states
    with signed AUROC 0.929--0.984; literal ordinary-token to newline edits
    suppress it in 93.6--99.2% of contexts. Channel 196 ranges from 0.606 to
    0.926 observational AUROC and is suppressed by the same edit in only
    21.0--59.6% of contexts. Moreover, channel 202's two linear factors already
    have AUROC 0.883--0.971, so the product mostly sharpens an inherited
    boundary predicate rather than creating a uniquely bilinear AND concept.
144. **Joint-ablation KL superadditivity is not nonlinear-circuit evidence.** The
    196/202 pair has 1.3--1.5 times the sum of individual KLs, but summing the
    two independently measured logit changes reproduces 95.2--100.9% of the
    exact pair KL. The nonadditive logit residual is only 5.1--7.2%, while the
    individual effects have positive Fisher cosine 0.35--0.43. The apparent
    interaction is almost entirely the quadratic softmax metric applied to
    aligned additive effects.
145. **Causal privilege, semantic coherence, and circuit interaction remain
    separate axes.** The 196/202 pair causes 16--34 times the exact KL of
    activation-energy-matched controls, despite raw tensor-energy ranks 3,071
    and 3,072. Channel 202 has a reproducible boundary-related description;
    channel 196 does not. Their aligned causal effects do not form a special
    nonlinear pair circuit. Natural-use rank finds important coordinates,
    held-out substitutions test meanings, and additive-logit decomposition
    tests composition; none substitutes for the others.
146. **The power-law bulk is universal across depth in the independent model,
    but outlier strength is not a circuit-quality score.** All twelve layers
    have bulk log-log fit R2 0.924--0.975. Leading log excess has Spearman 0.46
    with natural rank-one efficiency, 0.24 with cleaned-address operator K90,
    and 0.04 with native-channel Shapley K90. RMT reliably describes a spectral
    regime; it does not by itself identify localized mechanisms.
147. **The strongest outliers live in broad middle-layer programs.** Layers
    5, 6, and 4 have the largest leading excess, natural rank-one efficiencies
    0.87--0.91, median operator K90 84--158, and median native-channel K90
    667--972. Thus a strong, naturally used outlier can prioritize an important
    family of computations while its implementation remains distributed.
148. **The compact-program phase is late, not the strongest-outlier phase.** The
    early/middle/late median operator K90 values are 88.5/157.5/1.5 and native
    channel K90 values are 965/972/64. Late median leading excess is only 0.037,
    below the middle value 0.125. Depth and downstream role predict local
    program form better than raw outlier magnitude in this checkpoint.
149. **The negative outlier/localization result is not an address-aggregation
    artifact.** Selecting the first, minimum, median, or maximum of four cleaned
    addresses gives excess-versus-channel-support Spearman from -0.12 to 0.24.
    One of the three strongest layers still has first-address K90 268, and
    leave-one-layer-out excess/use correlations remain only 0.37--0.59.
150. **High sign-null recurrence identifies structural coordinate dominance.**
    In eight of twelve layers, all four independent channel-sign nulls select
    the same top native channel, and sign-null top-eight recurrence usually
    exceeds the clean-address recurrence. Repeated selection is evidence that a
    coordinate is privileged, but semantic coordination requires evidence that
    does not survive destruction of the learned signs.
151. **The robust compact late phase begins at Layer 11, not Layer 10.** On an
    independent 8,192-state covariance slice, Layer-10 address K90 changes from
    `[1,1,2,60]` to `[1,1,193,234]`, and its weakest address-subspace canonical
    cosine is 0.587. Layer 11 remains `[1,2,1,2]` on both slices. Random natural
    directions can also be compact at Layer 10. The earlier two-layer phase
    claim was therefore too broad.
152. **Algebraic operator compactness and behavioral sufficiency still differ at
    the final layer.** The full exact derivative predicts finite MLP-input edits
    with at most 0.28% relative downstream KL error. But for Layer-11 address 3,
    rank two captures at least 90% Frobenius energy while leaving about 30%
    relative behavioral KL; even rank four leaves about 22%. Top-256 native
    channels lower the median to 8.7% but miss the every-corpus 10% gate. Small
    discarded directions can be downstream-important even in a K90-two map.
153. **The four compact Layer-11 programs have intervention-validated boundary
    semantics.** All four recurrent current-newline descriptions survive literal
    ordinary-token-to-newline substitutions on FineWeb, WikiText, and
    TinyStories, with directional fidelity 0.78--0.98. Two address scores rise
    and two fall at boundaries. Their position-demeaned polarity-aligned mean
    detects newline OOD at AUROC 0.981--0.992.
154. **The boundary family is a shared-input, partially multiplexed fan-out.**
    Program-balanced analysis assigns 0.982 of top-rank input energy to one
    common natural-metric covector; all pairwise input cosines are 0.96--0.99.
    Output sharing is partial: the balanced common fraction is 0.634 and
    pairwise output cosines range 0.10--0.89. Native top-32 channel sets have
    median Jaccard 0.314. The clean reusable object is a boundary-reading
    backbone feeding several partly distinct output verbs.
155. **Shared semantics do not collapse the realized activation code to one
    scalar.** After polarity alignment, activation PC1 explains only 0.49--0.57
    across the three corpora, and median residual correlation after removing
    the newline label is 0.27--0.41. The same abstraction is carried with
    additional context-dependent variation rather than by four redundant copies
    of one activation.
156. **Stable axes can span an unstable or ill-conditioned explanatory space.**
    The four Layer-11 native edit axes match across covariance slices at absolute
    cosine 0.988--1.000, yet their Gram condition numbers are 411--610 and the
    weakest full-subspace canonical cosine is 0.810. Axis reproducibility,
    subspace robustness, and numerical independence are separate stability
    claims.
157. **The Layer-11 address family lies on a real boundary-changing causal
    path.** Ordinary-token-to-newline replacements project 0.435--0.534 of their
    final-layer state-change energy into the four-address span, versus about
    0.004 for random four-spaces. Exact recomputation through the projected
    state recovers write-delta R2 0.835--0.909 across FineWeb, WikiText, and
    TinyStories. This exceeds natural-state PCA despite similar raw state
    coverage and approaches a separately fitted boundary-delta PCA upper bound.
158. **Natural-path localization does not imply final-logit sufficiency.** The
    address projection still leaves 0.514--0.580 of the isolated MLP-path KL,
    and the Layer-11 MLP path accounts for about 9.7--14.5% of the complete
    token-replacement KL. The recovered program is a genuine late component of
    a larger boundary circuit, not the whole causal explanation.
159. **The shared input covector and shared edit direction are different tensor
    legs.** Although the four top derivative operators read nearly the same
    boundary covector, the leading principal direction of the native address
    edits captures only 0.4--1.4% of real boundary state changes and 0--3% of
    their MLP write. Collapsing “what context gates the derivative” with “which
    upstream perturbation invokes it” gives the wrong one-dimensional circuit.
160. **Low-energy address contrasts carry the operative discrimination.** In the
    native edit code, PC1 contains 77.98% of address-vector energy but only 4.56%
    of natural derivative-operator energy. PC3 contains 2.81%/25.71%, and PC4
    only 0.19%/52.84%. Address energy and mean nested causal increment correlate
    at Spearman -0.40 rather than positively.
161. **Weak contrast directions are causally amplified, not numerical junk.**
    PC3 adds 62--66% write R2 and PC4 adds 7--15% across corpora. Together they
    provide 66--74% of captured state change and 78--92% of captured write R2.
    Removing the weakest fourth direction materially degrades both write and
    logit reconstruction even though its address singular value is only 0.087.
162. **This is an explicit discriminator-code regime.** A large redundant
    component identifies a reusable boundary family; small contrast coordinates
    distinguish which response is required; the bilinear map amplifies those
    distinctions into partly separate output verbs. The code is semantically
    coherent at the family level, distributed at the native-coordinate level,
    and incomplete at the whole-model behavioral level. Those are compatible,
    not competing, interpretations.
163. **The final block reads rather than constructs the boundary contrast.** An
    exact three-factor vector-Shapley decomposition assigns 99.0--99.1% of the
    Layer-11 PC3--PC4 state contrast, 98.2--98.4% of its MLP-write change, and
    95.7--98.5% of its Fisher-logit change to the carried residual. Fresh
    embedding injection and Layer-11 attention each contribute roughly zero to
    one percent. Final-block interaction residuals are only 1--7%.
164. **A low-energy Layer-10 precursor is causally amplified at Layer 11.** In
    the fixed final contrast basis, normalized-state contrast energy reaches
    half its final value only at Layer 11. Nevertheless, patching only two
    contrast coordinates of the Layer-10 changed residual into the original run
    recovers 99.2--99.6% of the final Layer-11 contrast on all three corpora.
    Visibility and downstream causal sufficiency are therefore different.
165. **The residual contrast is progressively refined through late MLPs.**
    Contrast-only residual patches recover 44--52% from Layer 8, 75--81% from
    Layer 9, and about 99% from Layer 10. At Layer 10, direct output-contrast
    attribution is about 71% MLP write, 29% carried/mixed input, and essentially
    zero attention. The local circuit is an MLP-to-MLP residual route, not a QK/OV
    attention route.
166. **Contrast sufficiency is local to the semantic subcircuit.** The best
    Layer-10 contrast patch still leaves 0.606--0.921 of full token-replacement
    KL. It reconstructs the final boundary discriminator coordinates but not the
    other lexical, residual, attention-memory, normalization, and output changes
    caused by replacing a token. A clean subcircuit need not be a sufficient
    whole-behavior explanation.
167. **The complete token edit initially travels in two duplicated carriers,
    not one residual stream.** At the Layer-0 boundary, exact three-carrier
    Shapley allocation assigns 0.479--0.526 of the changed-logit direction to
    the residual sequence, 0.392--0.460 to the repeatedly injected original
    embedding, and 0.056--0.082 to the persistent Layer-0 value bus. By Layer
    10 the residual-only hybrid has Fisher fidelity 0.994--0.997 and receives
    0.985--0.987 of Shapley projection. The suffix progressively compiles the
    repeated embedding code into its carried state.
168. **Carrier credit is a cooperative-game definition, not a unique storage
    percentage.** Early residual and embedding marginals remain positive but
    change by 0.286--0.463 depending on which other carrier is already changed.
    Baseline-Fisher, changed-Fisher, forward-KL, and target-error-KL games all
    preserve the qualitative residual/embedding/value-bus ordering, but their
    allocations can differ by more than the preregistered 0.10. Use Shapley for
    a symmetric interface allocation; use insertion/removal marginals when the
    intervention order itself is the question.
169. **The persistent value bus is a minor whole-edit carrier but can be a major
    local attention value source.** Its whole-model carrier allocation stays
    below 0.15 under all four metrics. Inside late squared-attention writes,
    however, the current layer's value contribution is nearly zero at several
    layers while the Layer-0 bus receives 0.08--0.29 of the local write. Global
    carrier importance and conditional OV source are different questions.
170. **The bilinear MLPs write most of the new residual behavior for a local
    boundary edit.** Exact block-output factorization gives the Layer-0 MLP
    0.989--0.991 of within-residual behavioral Shapley credit. Across all twelve
    layers and three corpora the MLP is always the largest non-carried source.
    Attention peaks at only 0.109 and the block-local fresh embedding injection
    at 0.058, even though future repeated embedding injections collectively
    remain an important early carrier.
171. **State-space magnitude and downstream behavioral credit diverge sharply
    late.** At Layer 11, the MLP supplies 0.618--0.651 of Euclidean residual-
    change projection but only 0.086--0.184 of downstream behavioral Shapley;
    the carried state supplies only 0.344--0.375 of state projection but
    0.795--0.899 of behavioral credit. State reconstruction is appropriate for
    compression; suffix-weighted metrics are required for causal explanation.
172. **An absolute recomposition gate can fail while the algebra is exact.** The
    four block components differ from the native float32 sum by up to 0.00195
    because reassociating large floating-point terms changes rounding, so the
    preregistered `1e-4` absolute state gate failed. Independently propagated
    logits differ by at most `3.5e-5`. The failure should be retained as a
    metric-design warning, not misreported as a missing component.
173. **Final-position edits are a deliberately attention-light control.** An
    exact Q/K/current-value/value-bus decomposition reconstructs every native
    squared-attention write with zero measured error, yet a complete changed
    attention insertion at any single layer causes only 0.1--3.0% of the native
    full-token KL. The MLP can read the changed final embedding directly, so
    this condition cannot support a general claim that attention is unimportant.
174. **Within that small local attention path, query routing usually dominates.**
    Query Shapley exceeds key Shapley at every audited layer and corpus, and
    routing exceeds value-side allocation at at least eight of eleven layers.
    Layer 1 is often current-value dominated; Layers 7--11 frequently use the
    persistent Layer-0 bus. Some local writes are head-concentrated at
    0.53--0.68, but head identity is not globally stable.
175. **Moving the edit into the past makes attention causally material and
    changes its ontology.** At distances 8, 32, and 64, the best single-layer
    attention insertion reaches 0.128--0.348 of full KL, versus 0.004--0.029
    for the final-position reference. Source-key allocation grows, while the
    best route alternates among current-value transport, key routing, and late
    Layer-0-bus reads. QK/OV roles are conditional on source/destination geometry.
176. **No universal single attention layer owns cross-position communication.**
    The best layer ranges from 4 to 11 and fails the cross-corpus depth gate.
    Insertion and removal agree on the largest factor for all earlier-position
    best routes, but not for both final-position controls. Factor stability can
    exceed layer localization.
177. **All attention writes form an exact structural cut for earlier-token
    effects.** Replacing all twelve changed-run attention writes by base-run
    writes makes final logits exactly equal to the base run; transplanting all
    twelve changed writes into the base run exactly reconstructs the changed
    final logits. This follows from attention being the only cross-position
    channel and is a path theorem for this architecture, not evidence that all
    twelve writes are independently meaningful circuits.
178. **Attention messages are state-dependent interfaces, not additive packets.**
    Partial replacement frontiers can be non-monotone, and adding fixed
    changed-minus-base attention deltas becomes strongly destructive at large
    layer sets (Fisher fidelities below -1). Native-write replacement and
    additive-delta patching answer different counterfactuals; only the former
    preserves the discovered all-layer path identity.
179. **The cross-position route is distributed but substantially compressible.**
    Held-out condition-specific search gives four-layer fidelities 0.724--0.941
    and six-layer fidelities 0.810--0.970. Universal four- and six-layer
    sufficiency gates fail, but forward and backward searches share at least two
    layers in every four-set, and every better four-set contains at least two of
    Layers 4--7.
180. **A reusable attention backbone coexists with condition-dependent repair
    layers.** Joint search across all nine corpus/distance conditions selects
    Layers `[6,5,10,8]`: held-out mean fidelity 0.839, minimum 0.704. A universal
    six-set reaches 0.911/0.815 mean/minimum, and an eight-set reaches
    0.975/0.944. Backward search chooses `[5,6,7,8]` at budget four. Layers 5 and
    6 are the invariant core; Layers 7, 8, and 10 provide alternate completion
    routes.
181. **The obvious fixed middle set is not automatically the best universal
    program.** `{4,5,6,8}` falls to fidelity 0.510 on FineWeb at distance 64,
    whereas the jointly selected `{5,6,8,10}` stays above 0.704. Recurrence is a
    discovery prior, not a sufficient subset criterion; held-out downstream
    behavior must choose among recurrent alternatives.
182. **The appropriate explanatory object depends on the use case.** Persistent-
    carrier Shapley describes architectural credit; block-component Shapley
    describes residual construction; exact QK/OV factorization describes local
    routing content; all-attention removal proves path necessity; and held-out
    layer-subset replacement measures interface compression. None is the unique
    answer to “where is the fact or concept,” but together they distinguish
    storage, construction, communication, and behavioral sufficiency.
183. **Attention state magnitude and attention-message behavior can point to
    opposite destinations.** For edits 32 tokens before prediction, 46--71% of
    the selected attention-delta's Euclidean projection lands in the edited
    token's local neighborhood, while only 0.5--1.3% lands at the final token.
    Downstream behavioral Shapley reverses that allocation: the final-token
    message receives 66--101% of the relevant final-logit direction. A large
    local representation update and a small high-gain readout message are two
    parts of one circuit, not competing circuit localizations.
184. **The high-gain final destination is metric-stable but not universally
    dominant in early layers.** Across three edit distances, three corpora,
    baseline- and changed-Fisher geometry, changed-context removal, forward KL,
    and target-error KL, the final-position message carries 0.3--5.2% of state
    projection but 34--101% of behavioral allocation. The locked universal
    50% dominance gates fail at Layer 5: insertion/KL conditions reach 42--47%
    and the most adversarial removal condition reaches 34%; the baseline-Fisher
    state/behavior inversion gate succeeds everywhere.
185. **The prediction-position bottleneck is progressively completed with
    depth.** Layer-10 final-position allocation is about 98--101% under every
    audited metric, corpus, and distance, and is never materially below the
    corresponding Layer-5 allocation. Layers 5--8 construct and route a
    distributed contextual update; by Layer 10 its behaviorally relevant part
    has been compiled into the destination readout.
186. **A message destination is not a storage location.** The final-token query,
    source keys, and values already depend on the preceding computation. Saying
    that the final destination controls the output means that one native
    attention-message component is a behavioral bottleneck under a specified
    intervention game; it does not mean the source token or fact was locally
    stored at the final position all along.
187. **Destination topology is more transferable late than early.** Exact
    source-side allocations remain corpus-dependent, and the largest early
    destination can vary on FineWeb, WikiText, and TinyStories. By Layer 10 the
    same final destination is essentially complete across all three. Universal
    circuit claims should therefore be made about a staged topology--local
    construction followed by final-position readout--not an identical early
    edge or source partition.
188. **The final-message bottleneck is not sparse in native heads.** Frozen
    two-head programs average 0.596 Fisher fidelity and fall to 0.044; three
    heads average 0.708 and fall to 0.248. Five of six heads are needed for
    0.917 mean / 0.797 minimum fidelity. Previously recurrent heads L5H2, L6H5,
    L8H3, and L10H0 are all selected, but their output-weight subspaces lie at
    only the 87th, 67th, and 20th matched-null percentiles across successive
    layer pairs. Functional recurrence does not imply one aligned head basis.
189. **The same message is not a small fixed residual subspace.** A frozen
    layerwise state-PCA basis reaches only 0.386 mean fidelity at rank 16 and
    0.553 at rank 64; a single pooled cross-layer basis is worse. PCA strongly
    beats random and its variance frontier tracks behavior monotonically, so it
    finds real geometry, but it loses to three native input-conditioned heads.
    The heads act as high-dimensional conditional operators, not redundant
    coordinates in one small output plane.
190. **The exact tensor contraction exposes a source-edge program.** Each
    final-destination message can be written as 128 additive source-position
    vectors after retaining native QK weights, current values, persistent value
    memory, head projections, and the output projection. The edited source by
    itself averages only 0.03 fidelity in the first audit and 0.17 on the fresh
    split. The message is constructed from a context-wide edge ensemble rather
    than copied from the edited token.
191. **One-sided importance ranking destroys a signed cancellation web.** On a
    fresh split, keeping sources with the largest positive Fisher cross-score
    plateaus near 0.60 mean fidelity and can remain below -1 at budget 64.
    Ranking by absolute score reaches 0.958 mean at 64, and state magnitude
    reaches 0.986. Positive and negative local evidence have median cancellation
    ratios 1.40--1.52 across corpora; removing the apparently harmful side
    overshoots the target.
192. **Participation ratio is not causal support under cancellation.** Absolute
    Fisher scores have median effective participation only 7.5 positions, yet
    an absolute-score top-16 program can have fidelity -0.35. Participation
    measures concentration of scalar scores, not whether an omitted signed tail
    is required to balance the retained edges.
193. **The correct downstream quadratic object recovers a compact cooperative
    program.** Individual source-edge logit effects are 0.973 additive on
    average. Forward selection under their implicit Fisher Gram reaches 0.898
    mean / 0.753 minimum exact finite fidelity with 16 sources and 0.929 / 0.823
    with 32. It beats one-shot Fisher gain by 0.247 at budget 16, and its linear
    prediction correlates 0.946 with finite recomputation.
194. **Cooperative edges need not be individually helpful.** About 20% of the
    first 16 Gram-selected sources have negative initial target cross-score.
    They become useful after other edges are present because selection optimizes
    residual error, including self-energy and pairwise overlap. This is a direct
    real-model example of a constraint/cancellation web rather than a list of
    independently supportive features.
195. **Local state magnitude is a strong but budget-dependent proxy.** It does
    not beat the cooperative Gram at small budgets--0.856 versus 0.898 mean at
    16--but overtakes it at 32 and 64, reaching 0.988 mean / 0.952 minimum at
    64. For high-fidelity reconstruction of this interface, preserving large
    native tensor edges can be more robust than optimizing a locally additive
    downstream approximation.
196. **The cooperative program is conditional, not a fixed position skeleton.**
    The 16 most frequent absolute-score or Gram-selected source positions cover
    only 38--41% of selections. The reusable object is the selection rule and
    tensor factorization, not one universal list of token addresses.
197. **Attention normalization explains only one part of the web.** Exact
    product Shapley divides every head/source change into routing redistribution
    and value transport. Routing patterns conserve unit mass and therefore have
    exactly balanced positive and negative coefficient mass; values strictly
    before the edit remain unchanged. But routing receives only 0.556 behavioral
    credit on average and ranges from 0.176 to 0.833. Value transport is often
    dominant in Layers 5--6, while routing more often dominates Layers 8--10.
198. **Routing/value roles are context-dependent despite exact algebra.** Their
    isolated effects are nearly additive in 35 of 36 conditions, but one
    WikiText condition falls to 0.764. State and behavioral metrics can choose
    different larger factors, and Layer-10 routing means differ substantially
    across corpora. The stable ontology is a depth-varying handoff between
    changed-value transport and conserved routing, not a universal routing-only
    explanation.
199. **Several exact decompositions fail overly strict runtime tolerances.**
    Head, source, and routing/value sums have maximum relative float32 errors
    `1.47e-5`, `1.53e-5`, and `1.90e-5`, with propagated-logit disagreements
    below `3.2e-5`. The algebraic identities are exact and a float64 head split
    reaches roughly `2e-16`; the preregistered gates remain recorded as failed
    rather than silently relaxed.
200. **A target-free local tensor rule predicts semantic source programs.** For
    19 sentiment, gender, tense, scale, and polarity substitutions, ranking
    exact source-edge changes by residual-space norm reaches 0.967 aggregate
    fidelity with 16 sources and 0.993 with 32. The downstream cooperative
    oracle reaches 0.979 and 0.986. Thus almost all oracle compression can be
    recovered without the unembedding, target logits, or downstream Fisher
    direction once the counterfactual source contractions are available.
201. **The local rule transfers at the individual-example level.** On a fresh
    third context for every lexical pair, state-norm top-32 has mean 0.991,
    fifth percentile 0.976, and minimum 0.878 across 152 layer-example effects.
    Every layer median exceeds 0.990 and every semantic-family minimum exceeds
    0.877. Aggregate Fisher weighting is therefore not hiding a large tail of
    failed examples.
202. **Pattern change is a useful but incomplete shortcut.** Ranking only the
    magnitude of attention-pattern redistribution reaches 0.947 aggregate
    fidelity at 16 and 0.990 at 32, but its fresh per-example minimum is 0.698
    and fails the robust gate. Full source-write norm retains how routing and
    values combine through head output projections and is the safer local rule.
203. **Predictive stability does not imply semantic source addresses.** The
    same-family and different-family cooperative top-16 position Jaccards are
    0.258/0.259 on the first panel and 0.268/0.269 on fresh contexts. Semantic
    families reuse the tensor selection algorithm, not a family-specific fixed
    list of token positions.
204. **Recency is a weak small-budget explanation.** On semantic edits,
    selecting the last 16 sources reaches only 0.437 fidelity versus 0.967 for
    state norm. At budget 32 recency rises to 0.934 because half of the 64-token
    context is retained, but remains below the tensor-local rule. Edge magnitude
    is not merely a proxy for proximity to prediction.
205. **Small semantic effects do not manufacture the tensor-rule result.** A
    fresh newline scale sweep is locally quadratic in target Fisher energy, yet
    state-norm top-32 fidelity increases slightly from 0.976 at `0.125x` to
    0.981 at `1x` and 0.982 at `1.5x`. Shrinking does not repair the worst
    conditions. The semantic/newline gap at small scale is only 0.017, below the
    locked 0.02 geometry gate, so neither pure effect scale nor a large geometry
    difference is supported as the sole explanation.
206. **Diagnostic and predictive programs serve different uses.** The
    downstream Fisher Gram is an oracle explanation of why a small cooperative
    set is sufficient and which anti-aligned edges repair overlap. State-norm
    truncation is a target-free predictive rule suitable for fast circuit
    screening. The former explains interactions; the latter generalizes cheaply
    but does not label semantics. They should not be ranked on one universal
    “interpretability” axis.

The best present ontology is therefore **depth-, metric-, and state-dependent:
clean and shared lexical predicates feed distributed middle sequence codes whose strongest local
causal verbs rotate across contexts but vary smoothly under nearby edits; a
partly reusable attention/position skeleton broadcasts those codes; and a small
stable late output subspace sits above a long behaviorally relevant tail**. The
causally relevant equivalence classes depend on downstream computation, allowed
perturbations, and position. Output-conditioned eigenspaces answer named scalar
questions; basis-invariant local Jacobians describe intrinsic sensitivity;
sequence-boundary and frozen-edge patches identify where contextual information
travels; factorial path interventions identify what those routes transport.
Global spectral modes prioritize dominant operations, while conditioned spectra
recover exceptional interaction forms for already posed output questions. A
late raw-outlier bottleneck coexists with a smaller covariance-compact tensor
program whose individual axes are not generally semantic invariants. A
downstream-Fisher optimization recovers a reproducible six-ish-dimensional
behavioral core, while a context router turns the remaining broad fixed tail
into a sparse `4x8` conditional program. The routed function, execution rank,
and coarse lexical/structural geometry recur; exact expert identities do not.
In the middle layer, reusable sensitivity addresses feed a broad cooperative
native-channel ensemble: attribution is sparse while exact sufficiency is not.
An independent checkpoint preserves the power-law bulk and useful raw outliers
but moves the strongest spike to the middle and loses the four-spike late phase.
Its final layer nevertheless has a cross-corpus raw-four natural bottleneck and
a stable variance-compact tail. Exact-KL optimization shows a genuine downstream
metric mismatch, but fixed rank 68 is still behaviorally insufficient; the
remaining tail is broad or context-routed rather than a hidden small PCA space.
Hard conditional routing yields reproducible relative compression, and top-two
expert unions approximate the full learned union with 36 active directions.
However, the experts, routes, selected projectors, reconstructed writes, and
output distributions are not restart-stable. This is a noncanonical family of
partial approximations, not a recovered conditional circuit.
The power-law fit itself is not a circuit definition, and covariance cleaning
must name its sample size and regularization.
None is “one concept per neuron,” and no single global tensor factorization
substitutes for all five.

The full-token carrier audits add an architectural layer to that ontology. A
token identity begins as a duplicated residual/embedding code, is repeatedly
compiled by bilinear MLP writes, and uses attention as the sole cross-position
message channel. For local edits, attention can be behaviorally tiny; for
earlier edits, Layers 5--6 form a recurrent routing core supplemented by
condition-dependent Layers 7--8/10. Exact QK/OV structure explains what an
individual message contains, while all-layer cuts and held-out subset searches
explain where messages must travel. These are complementary descriptions, not
competing candidates for one canonical “location.”

Inside the late prediction-position bottleneck, neither native-head sparsity nor
a small fixed residual subspace survives held-out tests. The useful compression
appears one level lower: exact head/source tensor edges form a signed cooperative
program. A downstream Fisher Gram accounts for their self-energy and pairwise
overlap and recovers about 90% finite fidelity with 16 adaptive sources. Some
selected sources initially oppose the target, so the program is a cancellation
web rather than a list of individually supportive features. Attention
normalization supplies an exact conserved routing component, while changed-value
transport supplies a second, equally important and depth-dependent component.
Thus the clean abstraction is the tensor algebra plus a cooperative conditional
rule, not one fixed basis or token-address list.

That oracle description now has a predictive companion. Across five semantic
edit families, simply retaining the source contractions with largest local
write change recovers 99% of the final-message behavior at budget 32 on both
training-like and OOD text. Fresh per-example and finite-scale tests preserve
the result. The selected positions themselves do not cluster by semantic
family, so this is a reusable tensor-local execution rule rather than a semantic
address dictionary. The oracle Gram remains useful for explaining signed
cooperation; local state norm is preferable for inexpensive screening.

## The fifteen explanatory levels

| Level | Object | Question it answers | Appropriate test | What it does not establish |
|---|---|---|---|---|
| 1 | Native neuron | Does this trained scalar channel implement a stable discriminator/write role? | Held-out activation selectivity, signed ablation, cross-corpus transfer | That the neuron is the smallest or unique causal unit |
| 2 | Native group/circuit | Do several neurons jointly implement a reusable predicate or transformation? | Frozen group interventions, matched controls, leave-one-out tests | That downstream effects stay localized |
| 3 | Output-conditioned interaction eigenspace | Which input directions algebraically drive one specified feature or output role? | Cleaned quadratic spectrum, held-out prediction, local residual-subspace ablation | That the full downstream effect travels through that feature |
| 4 | Global bilinear operator | What does the layer compute on arbitrary residual directions? | Tensor/Frobenius probes, exact value–tangent–curvature identity | What language actually uses most often |
| 5 | Language-manifold function | Does a replacement match writes on natural states? | Held-out natural write error and full-model KL/CE | Correct responses to edits leaving the manifold |
| 6 | Tangent operator | Does it map a specified distribution of small interventions correctly? | Jacobian-vector products and finite scale sweeps | Global curvature or large interventions |
| 7 | Downstream quotient | Which write errors matter after later layers and the unembedding? | Fisher-weighted downstream effect fidelity | A unique internal basis |
| 8 | Conditional sparse program | Is only a small subset used for each token even if the global vocabulary is broad? | Per-token routing, corpus support, finite routed replacement | A small universal neuron set or lower dense FLOPs |
| 9 | Persistent architectural carrier | Which stable interface—residual, repeated embedding, or value memory—transports a finite change across a chosen boundary? | Exact hybrid games, insertion/removal marginals, metric red teams | A unique storage location or metric-independent percentage |
| 10 | Layer message/interface | Which attention or MLP writes are necessary or sufficient communication packets at chosen boundaries? | Native-write replacement, all-path cuts, held-out subset search | Independent additive packets or unique ownership by those layers |
| 11 | Broadcast skeleton | Do fixed heads, positions, or edge addresses recurrently relay contextual changes? | Frozen-address patches, head/source controls, OOD transfer | A fixed content representation, semantic abstraction, or complete route |
| 12 | Behavioral destination bottleneck | Where does a distributed representation change become a high-gain message for the selected prediction? | Destination-partitioned native-write games under Fisher and exact KL | That the information was stored at that destination or used identically for other outputs |
| 13 | Cooperative tensor-edge program | Can a signed distributed message be shortened only after accounting for self-energy, overlap, and cancellation among exact tensor edges? | Exact source factorization, downstream Fisher Gram selection, finite subset patches | A fixed source list, independent edge meanings, or semantic abstraction |
| 14 | Predictive local edge rule | Can important conditional edges be selected without using the downstream target or unembedding? | Frozen tensor-local rankings, fresh semantic edits, per-example and scale red teams | Why edges cooperate, semantic labels for their positions, or a context-free circuit |
| 15 | Canonical mechanism | Does the same factor/subspace recur across seeds, corpora, and independent fits? | Alignment-free stability, independent restarts, causal transfer | Guaranteed semantic labels or complete model understanding |

These levels should not be collapsed into a single “monosemanticity” score.
A model may score well at one level and poorly at another. For example, a native
neuron can have a clean `-ing` predicate while its causal write is consumed by a
large distributed population. Conversely, a non-native subspace can predict an
intervention reliably without having a privileged neuron axis.

## Output-conditioned spectral modes

For a bilinear product layer with factor matrices `L` and `R`, and a chosen
feature/output coefficient vector `c`, the exact scalar preactivation is

`q_c(x) = cᵀ[(Lx) ⊙ (Rx)] = xᵀ A_c x`,

where `A_c = sym(Lᵀ diag(c) R)`. This turns a broad three-way tensor into a
feature-conditioned symmetric matrix. After whitening by the natural-input
second moment and removing the isotropic trace, its eigenvectors are candidate
interaction directions. They are not native neurons: they are directions in
residual space whose squared coordinates, weighted by signed eigenvalues,
implement the selected feature.

### Important correction: the first intervention was not an eigenmode ablation

The pilot, rank red-team, independent confirmation, and prospective atlas all
used a **feature-coordinate intervention**. They decomposed the scalar value
`q_c(x)` into spectral pieces, then changed the native product vector along
`c / ||c||²` so that the chosen scalar coordinate equaled one piece. Every such
edit follows the same one-dimensional product-space path. It can test whether a
spectral piece reconstructs the scalar coordinate's effect, but it does not
test what happens when the residual eigenvector itself is changed.

This distinction became visible in a precision red-team. With the same weights
evaluated in float32, rank-4 coordinate fidelity rose from 0.296–0.463 in
bfloat16 to 0.975–0.986, and the local effect energy followed scale squared with
a median coefficient of variation of 0.007. The earlier nonmonotonic rank and
scale curves were largely bfloat16 edit quantization, not evidence of nonlinear
downstream curvature.

Accordingly, the historical results below are retained as **semantic and
coordinate-reconstruction evidence**, not residual-eigenmode causal evidence.

The initial five-category pilot and a disjoint rank-resolved red-team give a
more precise version of the unpublished observation that motivated this test:

| Test | Result | Interpretation |
|---|---:|---|
| Semantic top-2 held-out AUROC | 0.958 median versus 0.883 for the full form | Very few modes preserve the selected token-form discrimination |
| Seed-101 significant-only coordinate fidelity | 0.503 median versus 0.000 controls | Null-exceeding modes reconstruct the selected scalar coordinate better than controls in bfloat16 |
| Seed-101 top-2 coordinate fidelity | 0.466 median; spike beats bulk in 5/5 categories | A compact spectral component predicts the feature-coordinate edit under this numerical implementation |
| Rank-4 plateau | Failed: 0.454 median, below the locked 0.50 gate | More high-magnitude modes are not monotonically more causally useful |
| Raw-basis rank-4 coordinate fidelity | 0.313 semantic versus 0.205 control | The coordinate-reconstruction finding is not solely a whitening artifact |
| Cross-seed top-4 category matching | exact permutation `p = 1/120`, 5/5 diagonal matches | The subspace is category-specific across independent SAE fits |

The independent confirmation changed to SAE seed 102, fresh FineWeb rows, and
two OOD corpora:

| Corpus | Leading-mode coordinate fidelity | Complementary bulk | Matched control | Semantic−control |
|---|---:|---:|---:|---:|
| FineWeb | 0.358 | 0.274 | 0.278 | 0.080 |
| WikiText-103 | 0.372 | 0.118 | 0.037 | 0.336 |
| TinyStories | 0.270 | 0.011 | 0.089 | 0.181 |

All five locked confirmation hypotheses passed. The FineWeb
semantic-versus-control difference nevertheless missed the 0.10 per-corpus
criterion, so the positive control result is driven by the two OOD corpora. One
seed-102 newline direction had no mode above its frozen sign-flip threshold and
was counted as zero in the significant-only test rather than omitted.

The strongest cross-fit object is the individual leading direction: squared
overlap for matching categories averages 0.517, all five diagonal entries are
the best match, and the exact five-category permutation test is `p = 1/120`.
That is stronger evidence for a reproducible axis than the earlier SAE
feature-coordinate alignment. The prospective result below shows why
“reproducible axis” and “causal circuit” must nevertheless remain separate.

### True local input-subspace interventions

The corrected test removes a spectral component from the actual layer-1
post-RMSNorm MLP input, recomputes every native bilinear channel, and runs the
complete downstream model. To stay near the language operating point, the edit
is normalized to 5%, 10%, or 25% of the input norm. Top modes are compared with
modes 13–16 using exactly matched input-edit norm.

For 10%-norm, rank-4 edits:

| Basis | Feature change, top/low | Downstream effect energy, top/low | Known-feature mediation, top/low | Prospective mediation |
|---|---:|---:|---:|---:|
| Raw Euclidean | 16.69× | 5.43× | 0.408 / 0.006 | 0.170 |
| Covariance-whitened | 20.02× | 2.18× | 0.410 / -0.001 | 0.222 |

All five known categories exceeded 0.30 mediation fidelity in both bases. The
raw and whitened known medians differ by only 0.001. This is real local causal
evidence: the top eigenspaces selectively move the intended feature and produce
downstream effects partly predicted by that feature's write path.

It is also evidence against calling them self-contained circuits. The best
median mediation fidelity is only about 0.41. This score is
`1 - normalized squared error`, not a literal percentage of effect stored, but
it still shows large predictive error outside the proposed feature path. Other
native product coordinates changed by the same residual edit matter. Dense
tensor coupling is not a nuisance around the circuit; it is part of the
computation.

Two feature interventions were compared:

- **Minimum-norm dual:** change products along `c / ||c||²`. This is the
  smallest Euclidean product edit realizing a requested encoder-coordinate
  change.
- **SAE decoder:** change products along the learned decoder vector, normalized
  to realize the same encoder-coordinate change. This asks what the feature
  means under the SAE's own encode/decode model.

The SAE decoder is better for every known category. At 10% input edits its
median advantage is about 0.20 in the raw basis and 0.23 in the whitened basis.
Thus “intervene on feature `c`” has no basis-free meaning: the encoder defines a
measurement, while a decoder or causal dual defines how that measurement is
written back.

The result is scale-local. Known-category median mediation is 0.47 raw and 0.44
whitened at 5%; about 0.41 for both at 10%; and 0.02 raw versus 0.31 whitened at
25%. Full component removal was rejected as a causal criterion because several
edits were 2–4 times the original input norm.

### Four meanings of “this feature is causal”

| Meaning | Intervention | Use case | Current evidence |
|---|---|---|---|
| Scalar-coordinate cause | Change `cᵀh` along a specified product-space dual | Editing or controlling a known internal measurement | Algebraically clean, but dual-dependent and bfloat16-sensitive |
| Residual-direction cause | Perturb an eigenvector at the actual MLP input and recompute all channels | Testing whether an input discriminator participates in computation | Supported locally for the five known categories |
| Feature-path mediation | Replace the residual edit's product change by the SAE feature decoder alone | Testing whether the effect travels through the named feature | Partial, about 0.41 median fidelity |
| Sufficient circuit | Use only the proposed subspace/path and retain the relevant behavior | Compression or complete mechanistic explanation | Not established |

These meanings are useful for different goals and should not be combined into
one monosemanticity or causality score.

### Selection bias, gating, and multiple routes into one feature

The eight-site local pilot selected the highest-scoring category examples. A
larger preregistered replication used 16 high positives, 16 random positives,
16 hard negatives, and 16 random negatives per eligible category on each of
FineWeb, WikiText-103, and TinyStories.

| Corpus | Known high positive | Known random positive | Known hard negative | Prospective random positive |
|---|---:|---:|---:|---:|
| FineWeb, raw | 0.463 | 0.210 | 0.203 | 0.059 |
| WikiText, raw | 0.328 | 0.229 | 0.267 | -0.005 |
| TinyStories, raw | 0.268 | 0.120 | 0.082 | 0.011 |

Five of six locked hypotheses failed. The known-versus-prospective gap survived
(about 0.20 in both bases), but OOD random-positive mediation and surface-label
specificity did not. Thus the original 0.41 estimate describes a strong-state
regime, not the average token carrying that hand-written label.

Inspection showed that many “hard negatives” were not mechanistically negative:
`ing`, `made`, `written`, `again`, `also`, and `<|endoftext|>` frequently activated
the selected SAE features despite failing narrow suffix/newline rules. Conversely,
surface positives such as `need`, `Lily`, and `Billy` can satisfy a suffix rule
without using the same internal route.

A fresh panel therefore stratified examples by the SAE's exact ReLU plus Top-128
gate:

| Basis | Group | High active | Random active | Near inactive | Random inactive |
|---|---|---:|---:|---:|---:|
| Raw | Known | 0.369 | 0.208 | 0.004 | 0.017 |
| Whitened | Known | 0.367 | 0.114 | 0.005 | -0.001 |
| Raw | Prospective | 0.194 | -0.020 | 0.004 | -0.018 |
| Whitened | Prospective | 0.145 | 0.013 | -0.002 | 0.003 |

Gate state cleanly separates active from inactive uses, but it is not sufficient.
Within random-active known sites, raw-basis mediation was 0.363 for surface-label
positives versus 0.090 for negatives; the whitened figures were 0.257 versus
0.010. A separate activation-matching attempt found positive semantic gaps in
both bases, but failed its preactivation-balance gate, so magnitude remains a
confound for the aggregate comparison. Closely matched exploratory subsets keep
a positive gap for several categories, especially newline, `-ing`, and `-ly`,
but not uniformly for `-ed` and all-caps.

The best current interpretation is **input-side multiplexing**: one SAE feature
can be active in several contexts, while different spectral mode groups describe
different causes or routes into that shared coordinate. “The feature is active”
does not identify which route produced it.

### Frozen companion-feature ensembles

The target SAE decoder alone predicts only 0.297 median downstream fidelity for
known high-active spectral edits. Encoding the before/after native product
states reveals coordinated changes in other SAE features:

| SAE reconstruction | Direct-write fidelity | Downstream fidelity |
|---|---:|---:|
| Named feature only | 0.238 | 0.297 |
| Adaptive top 4 | 0.386 | 0.487 |
| Adaptive top 16 | 0.521 | 0.640 |
| Adaptive top 64 | 0.561 | 0.636 |
| Complete changed SAE code | 0.537 | 0.552 |

The named feature is among the top 16 changed decoded-write contributions on
every site. Yet 90% of summed individual decoded-write energy requires a median
84 changed features. More features are not monotonically better: lower-ranked
SAE changes can reconstruct SAE activation while worsening the causal effect.

Companions are partly stable. Within-category top-16 Jaccard medians range from
0.103 to 0.280, with 4–12 features appearing on at least half of a category's
16 sites; cross-category median overlap is zero. Freezing cores on eight
discovery sites and testing eight held-out sites gives:

| Core size | Frozen category core | Adaptive per-site | Next-category control |
|---:|---:|---:|---:|
| 4 | 0.422 | 0.467 | 0.000 |
| 8 | 0.501 | 0.575 | 0.000 |
| 16 | 0.606 | 0.632 | 0.000 |
| 32 | 0.658 | 0.740 | 0.000 |

The next-category control is intentionally severe—most of its sparse code
changes are zero—so it establishes specificity but is not a norm-matched random
control. The fixed core's stronger test is transfer to a separate random-active
surface-positive panel:

| Core size | Positive fixed | Positive adaptive | Matched-negative fixed |
|---:|---:|---:|---:|
| 8 | 0.345 | 0.453 | 0.045 |
| 16 | 0.417 | 0.529 | 0.059 |
| 32 | 0.514 | 0.525 | 0.062 |

The 16-feature core narrowly missed its 0.50 and 80%-of-adaptive gates; the
32-feature core transfers successfully. Positive-over-negative gaps are positive
for all five categories, but the activation matching was imperfect, so this is
strong ensemble-transfer evidence rather than a clean semantic causal contrast.

This supplies a concrete middle representation for the CPU analogy: an input
mode group selects a category-specific sparse feature ensemble, whose joint
write is reused downstream. Native connectivity remains dense, but the
conditional active program is much shorter than the whole layer.

### What is canonical across SAE seeds?

Feature IDs are not expected to survive independent SAE training. The stronger
candidate is the span of the companion decoders after the native Down projection.
A completely independent seed-102 discovery gives:

| Object on seed-102 validation edits | Direct fidelity | Downstream fidelity |
|---|---:|---:|
| Frozen seed-102 16-feature program | 0.487 | 0.540 |
| Adaptive seed-102 top-16 program | 0.522 | 0.639 |
| Oracle seed-102 core write subspace | 0.581 | 0.760 |
| Oracle corresponding seed-101 write subspace | 0.572 | 0.636 |
| Oracle next-category seed-101 subspace | 0.045 | 0.026 |

The five-by-five seed-101/seed-102 core-subspace matrix has mean diagonal overlap
0.137, exact assignment `p = 1/120`, and all five diagonal entries are their
row's best match. A random pair of 16-dimensional subspaces in 1,152 dimensions
would have expected overlap about 0.014, so the diagonal is meaningful even
though it is far from identity.

Removing the named feature from both cores leaves companion-only diagonal
overlap 0.123, `p = 1/120`, and 5/5 row maxima. The recurrence is therefore not
carried only by the obvious semantic feature. Named-feature-only writes also
match 5/5.

The oracle projection is a **subspace sufficiency** test: it uses the actual edit
write to choose coefficients within a frozen subspace. It is basis-invariant and
useful for canonicalization, but is not an executable sparse program. The frozen
seed-102 code/decoder program is the stricter executable object and reaches
0.540. Conversely, the seed-101 subspace transfers poorly for newline (0.004)
but well for the other four categories (0.671–0.844), so the aggregate median
must not be read as universal category recovery.

This yields two valid definitions:

- **Program identity:** the same sparse coordinates and routing rule execute the
  computation. This is useful for editing and efficient implementation, but is
  SAE-basis dependent.
- **Mechanism identity:** independently learned coordinates span the same
  category-specific causal write subspace. This is useful for scientific
  comparison and survives rotations, but needs an additional encoder/router to
  become executable.

### Donor transfer exposes two engagement regimes

The strongest causal test copies only a category's rank-4 spectral coordinates
from a high-active donor into a random-inactive recipient. The input edit is
capped at 10% of the recipient norm; low spectral modes and another category's
modes receive exactly the same edit norm.

| Patch | Median donor-feature transfer | Target gate on | Downstream effect energy |
|---|---:|---:|---:|
| Category top modes | 0.482 | 95% | 0.00335 |
| Category low modes | 0.048 | 25% | 0.00117 |
| Another category's top modes | -0.005 | 0% | 0.00173 |

The transfer is dose-responsive: median feature transfer is 0.021, 0.125, and
0.482 at 2%, 5%, and 10% input-norm edits. This is evidence for a selective
nonlinear input gate, not merely correlation between a spectral direction and
an SAE coordinate.

The downstream program is less compact than the gate. With native SAE
coefficients, an adaptive 32-feature reconstruction reaches 0.428 fidelity and
the frozen category core reaches 0.281. Refitting coefficients for the exact
residual write separates the possible failures:

| Support/coordinate rule | Residual fidelity | Downstream fidelity |
|---|---:|---:|
| Observed complete SAE code change | 0.259 | 0.450 |
| Same encoder-selected support, refit coefficients | 0.599 | 0.784 |
| Frozen own-category 32 support, refit coefficients | 0.491 | 0.616 |
| Frozen wrong-category 32 support, refit coefficients | 0.071 | 0.090 |
| Per-site correlation-selected 32, refit coefficients | 0.540 | 0.635 |

The full 6,144-vector dictionary spans the 1,152-dimensional write exactly, but
that is a dimensionality sanity check, not semantic evidence. The useful result
is the gap between own- and wrong-category supports. The encoder often selects
the right region of feature space, while its native sparse coefficients are not
faithful coordinates for an off-manifold intervention.

Frozen FineWeb cores also transfer to gate-selected OOD high-active examples:

| Corpus | Frozen core 32 | Adaptive 32 | Frozen refit | Wrong refit |
|---|---:|---:|---:|---:|
| WikiText-103 | 0.457 | 0.532 | 0.604 | 0.052 |
| TinyStories | 0.413 | 0.513 | 0.630 | 0.058 |

A hard control draws the same number of actually changed features from the same
SAE code change and exactly matches the own core's decoded-write norm. The own
core still beats contribution-matched controls 0.457 versus 0.004 on WikiText
and 0.413 versus 0.155 on TinyStories, winning all 5/5 category medians in each
corpus. Thus high-tail specificity is about write direction and organization,
not just activation or effect magnitude.

The result changes in the lower-engagement regime. On random gate-active OOD
sites, frozen-core fidelity falls to 0.171 and 0.127, and its advantage over the
hard matched control is only 0.079 and 0.090. The current ontology is therefore
conditional:

- **Strong engagement:** a stable category input route recruits a transferable,
  category-specific companion support. A compact circuit description is useful.
- **Weak engagement:** the category axis remains present, but the coefficient
  router and companion membership are more context-dependent. A fixed circuit
  list is not an adequate explanation.

“Feature identity,” “support identity,” “subspace identity,” and “executable
coordinates” are four different claims. The experiments now distinguish them
rather than calling all four monosemanticity.

### Layer 17: a compact joint quadratic code, not category eigenspaces

The final bilinear MLP permits an SAE-independent version of the same query. For
an output-token class with normalized unembedding contrast `g_c`, its exact
pre-final-normalization MLP contribution is

`q_c(x) = xᵀ sym(Lᵀ diag(Dᵀg_c) R) x`.

For newline, `-ing`, `-ed`, `-ly`, and all-caps, the top four modes of these
forms retain median next-token AUROC 0.650 on FineWeb, 0.697 on WikiText, and
0.731 on TinyStories. Norm-matched edits along those modes affect the queried
class contrast 8–9 times more than the least-coupled modes. This first result is
real but insufficient: a low-mode comparison only establishes output
sensitivity.

A complete cross-category matrix falsifies the tempting interpretation that
each class owns its leading eigenspace. Rank-4 diagonal assignment has exact
`p = 0.442`, `1.000`, and `0.500` on the three corpora, with only 2/5, 1/5, and
1/4 row maxima. Rank 1 improves the assignment statistic but still gives only
one row maximum per corpus and does not transfer robustly. Direct MLP-write
effects fail similarly, so residual addition and final RMSNorm are not the main
cause of the missing specificity.

The correct object is the **joint family of quadratic forms**:

| Joint-form diagnostic | Result | Meaning |
|---|---:|---|
| Effective rank of the five-form Gram matrix | 4.76 | The class computations are almost linearly independent |
| Median rank-4 input-subspace overlap | 0.154 | Shared input geometry; random expectation is 0.0035 |
| Median normalized commutator norm | 0.059 | No single rotation exactly diagonalizes all five forms |
| Shared 4D basis, median full-form energy | 0.091 | Globally small energy fraction |
| Shared 4D basis, median AUROC | 0.679 / 0.725 / 0.739 | Strong distributional semantic retention on all corpora |
| Shared 32D basis, cross-term energy | 0.728 | Most restricted-form energy is off-diagonal in this frozen basis |

The shared basis is obtained from the leading eigenspace of the sum of the five
normalized squared forms. Every output class then has a different small
quadratic coefficient matrix inside that common basis. This reconciles shared
directions with distinct computations: category identity is stored in the
matrix of pairwise interactions, not in ownership of a direction.

A diagonal-only red-team makes the point operational. At shared dimension 32,
deleting every cross term changes median AUROC from 0.684 to 0.563 on FineWeb,
0.721 to 0.524 on WikiText, and 0.654 to 0.532 on TinyStories. Cross terms carry
0.70–0.81 of restricted-score variance on actual language states. Thus the
off-diagonal structure is not merely global tensor energy that language never
uses.

This supplies a concrete definition of **computational multiplexing**:

- several tasks reuse a compact set of scalar coordinates;
- no coordinate or class-specific subspace identifies the computation alone;
- different quadratic interaction matrices over those coordinates implement
  different readouts;
- the joint code remains short enough to inspect, intervene on, and compare.

It is more structured than arbitrary superposition and less neuron-aligned than
the high-engagement layer-1 companion circuits. Both regimes coexist in the
same model.

### A recoverable joint basis—and where it stops generalizing

The shared layer-17 basis was frozen and evaluated on nine untouched output
predicates: delimiters, quotes, operators, URLs, non-ASCII tokens, word
continuations, punctuation, and colons/semicolons. At dimension 32 it
reconstructs the original five scalar quadratic scores on language states with
correlation 0.96–0.97 and centered `R² ≈ 0.91`. For eligible held-out concepts:

| Corpus | Median score correlation | Median centered R² | Concepts retaining oriented AUROC within 0.05 |
|---|---:|---:|---:|
| FineWeb | 0.820 | 0.650 | 87.5% |
| WikiText-103 | 0.866 | 0.700 | 87.5% |
| TinyStories | 0.655 | 0.429 | 100% |

This occurs while the frozen 32D subspace contains only about 5% of held-out
forms' global Frobenius energy. It is therefore a **distributional quadratic
quotient**: it preserves the part of each scalar computation used on language
states, not the whole operator on arbitrary residual vectors. Individual
concepts remain nonuniform—word continuation transfers especially well, while
some punctuation and non-ASCII scores are much less faithful.

Approximate joint diagonalization tests whether the off-diagonal terms were
merely an inconvenient basis. A weight-only orthogonal rotation inside the
frozen 32D subspace reduces off-diagonal energy:

| Family | Before | After | Reduction | Multi-start matched-axis overlap |
|---|---:|---:|---:|---:|
| Layer-0 embedding queries | 0.386 | 0.059 | 84.8% | 0.998 |
| Layer-17 output queries | 0.682 | 0.157 | 77.0% | 0.979 |

Six independent starts recover all 32 axes as mutual nearest matches for both
families. These are reproducible query-family-relative axes, not one optimizer's
arbitrary rotation.

The functional results differ by depth:

- At layer 0, the diagonalized axes preserve the full restricted computation:
  score correlation is 0.987–0.988 and median current-token AUROC is unchanged
  across all three corpora. For this fixed family of embedding-facing lexical
  questions, a neuron-like basis genuinely exists.
- At layer 17, diagonal score correlation for the original five concepts rises
  to 0.93–0.95, but semantic AUROC still fails on WikiText and TinyStories.
  Small residual cross terms matter disproportionately near semantic decision
  boundaries.
- Applying the same layer-17 rotation to held-out concepts improves numerical
  correlation but does not recover their semantics. Word continuation is a
  particularly clear failure. The full quadratic matrices generalize better
  than the diagonal axes.

The layer-0 comparison also makes readout relativity concrete. Conditioning the
same tensor on embedding-class directions gives current-token AUROC 0.95–0.98;
conditioning it on unembedding directions produces a different form family.
Same-category embedding/unembedding form cosine has median -0.090. There is no
single context-free answer to “what does this layer represent?” without naming
the readout and use case.

This motivates three increasingly strong meanings of a clean coordinate:

1. **Family-diagonal coordinate:** a stable axis that approximately diagonalizes
   a declared set of queries.
2. **Distribution-stable coordinate:** the diagonal approximation remains
   accurate across corpora for those queries.
3. **Open-vocabulary coordinate:** the same diagonal basis works for new queries.

Layer 0 currently passes the first two for embedding-facing token-form queries.
Layer 17 passes the first numerically, partly passes the second, and fails the
third. Its shared subspace plus query-specific interaction matrix is the more
portable explanatory object.

### Finite axes reveal activation and role regimes

The stable axes were next tested by finite coordinate removal in actual
language states. For every intervention, the exact change in 14
output-conditioned scalar forms was compared with (a) the isolated diagonal
role and (b) the complete interaction matrix inside the 32D subspace:

| Layer | Site stratum | Diagonal cosine / fidelity | Full-matrix cosine / fidelity |
|---|---|---:|---:|
| 0 | top activation | 0.936 / 0.847 | 0.972 / 0.932 |
| 0 | random | 0.399 / 0.092 | 0.697 / 0.410 |
| 17 | top activation | 0.790 / 0.472 | 0.974 / 0.925 |
| 17 | random | 0.468 / 0.051 | 0.935 / 0.841 |

The top interventions have roughly 100–1,000 times more effect energy than the
random interventions. An axis therefore approximates a standalone quadratic
verb when it dominates the state; when weak, its effect is mostly conditional
on partner coordinates. This is not arbitrary meaning-switching: the full
interaction model predicts those contextual effects, especially at layer 17.

To quantify meaning-switching without treating a sign reversal as a new role,
the exact effect vector of zeroing each axis was collected over thousands of
contexts. With the 14-predicate semantic vocabulary, four role directions
capture median energy 0.915 at layer 0 and 0.974 at layer 17, and those role
subspaces have 0.93–0.96 OOD overlap. But this view is probe-relative. Three
independent 96D random sketches of the entire residual write give:

| Layer | First-role energy | Four-role energy | Median rank for 90% | Sketch CV |
|---|---:|---:|---:|---:|
| 0 | 0.214 | 0.421 | 48.5 | 0.025 |
| 17 | 0.712 | 0.903 | 4.3 | 0.007 |

Thus early selected axes are clean lexical discriminators under particular
readouts but produce highly context-dependent full writes. Late selected axes
are much closer to reusable output operations. Random ambient bases remain far
worse than the selected subspace, so the late concentration is not generic to
all directions in the layer.

Finally, random rotations of the same selected subspace make within-axis roles
look even cleaner, falsifying coordinate uniqueness. They also duplicate the
same roles. The joint basis has higher effective between-axis diversity and wins
a locked concentration-times-diversity diagnostic:

| Layer | Basis | First-role energy | Effective role diversity | Adjusted score |
|---|---|---:|---:|---:|
| 0 | joint | 0.229 | 8.74 | 0.063 |
| 0 | random rotation | 0.305 | 4.28 | 0.041 |
| 17 | joint | 0.715 | 6.51 | 0.146 |
| 17 | random rotation | 0.785 | 4.16 | 0.102 |

The safest hierarchy is now: **selected subspace → query-specific interaction
form → optional axes chosen to balance simple roles and role diversity**. Calling
the final axes “the model's neurons” would overstate what is identified.

### Downstream composition expands early roles

The full-residual audit left open whether layer-0 write complexity was erased by
later computation. Twelve predeclared axes—four clean lexical, four locally
role-concentrated, and four locally multiplexed—were zeroed at 32 high-effect
final-token sites apiece and propagated through blocks 1–17:

| Axis group | Local first / four-role energy | Local rank90 | Final first / four-role energy | Final rank90 |
|---|---:|---:|---:|---:|
| Lexical | 0.327 / 0.576 | 17.5 | 0.160 / 0.367 | 25.0 |
| Locally concentrated | 0.684 / 0.841 | 7.5 | 0.165 / 0.326 | 25.5 |
| Locally multiplexed | 0.196 / 0.461 | 21.5 | 0.149 / 0.377 | 25.0 |

Later computation does not quotient these edits into a simpler fixed role. It
expands all three groups into similarly high-rank, context-dependent final-state
effects. The 384 edits are behaviorally nonzero (median next-token KL
`1.19e-4`), although their median correct-token effect is only 0.017 logits.

A matched scale sweep rules out the simplest nonlinear-edit objection:

| Removal scale | Local rank90 | Final rank90 | Median KL |
|---:|---:|---:|---:|
| 0.10 | 13.5 | 25.0 | `1.52e-6` |
| 0.25 | 14.0 | 25.0 | `8.72e-6` |
| 1.00 | 16.5 | 25.0 | `1.19e-4` |

Final-effect direction cosine between 10% and 100% removal is 0.939, and the KL
scale exponent is 1.90. The high downstream role rank is therefore consistent
with a context-dependent downstream Jacobian, not just a catastrophic full-axis
ablation. A neuron can be a clean *reader* or discriminator at its input while
its causal consequence remains a broad conditional program.

### Query-conditioned maps: exact algebra, relative sufficiency

For the frozen 32D layer-17 input subspace `U`, every residual readout direction
`y` has an exact small interaction matrix `A(y) = Uᵀ M(y) U`. Because `M(y)` is
linear in `y`, the weights contract once into a 1,152→32×32 query map. Direct
contraction and mapped quadratic scores agree to relative error `5.1e-7`.

Natural-state fidelity depends on the query family:

| Query family | FineWeb | WikiText | TinyStories |
|---|---:|---:|---:|
| Semantic token predicates | 0.856 | 0.903 | 0.899 |
| Frequent-token unembeddings | 0.797 | 0.800 | 0.878 |
| Random Gaussian readouts | 0.886 | 0.904 | 0.921 |
| Natural-write PCA directions | 0.302 | 0.366 | 0.387 |

The Gaussian result survives centered-R² checks. Random directions average over
a broad well-preserved bulk; the highest-variance natural write directions
expose structured residuals missed by the semantic 32D input subspace. A
rank-128 factorization retains 95.8% of query-map energy and loses less than
0.004 median correlation for semantic and frequent-token queries.

### Power-law bulk and validated circuit candidates

The query-map spectrum has a rank-33–512 power-law exponent 0.838 with log-log
R² 0.950. Channel-sign and Down-to-factor pairing nulls retain similar bulks
(exponents 0.794 and 0.835). The power law is generic structured weight
geometry, not a semantic result by itself.

The leading four are stronger relative to all twelve null spectra, but remain
below a naive extrapolation of the observed bulk. They are **top-spectrum
components**, not strict above-power-law outliers under this fit. Likewise,
29/32 trained leading output directions pass a broad token-predicate AUROC
threshold, but sign nulls pass just as many because they retain the learned
Down/unembedding geometry.

Natural-state and finite causal validation supply the missing evidence:

| Corpus | Top-four score corr. | Top-32 corr. | Top/middle variance | Strength↔variance |
|---|---:|---:|---:|---:|
| FineWeb | 0.944 | 0.689 | 24.1× | 0.940 |
| WikiText | 0.951 | 0.711 | 26.1× | 0.922 |
| TinyStories | 0.957 | 0.745 | 22.2× | 0.911 |

Their 32D input matrices are low-rank: four eigenmodes contain 92–97% of their
energy. Norm-matched top-eigenspace edits produce 29–119× the target scalar
effect and 70–137× the final-logit KL of low-eigenspace edits. The top spectrum
is therefore a high-yield search ordering, but only after independent
natural-state and finite causal tests.

### Depth-dependent spectra: bulk, spikes, and natural usage

A standardized replication at layers 0, 8, and 17 changes the scope of that
claim. With the same 64-dimensional state basis:

| Layer | Whole-write R² | Top-four map energy | Top-four score corr. | Four-mode form energy |
|---:|---:|---:|---:|---:|
| 0 | 0.455 | 0.147 | 0.829 | 0.567 |
| 8 | 0.154 | 0.068 | 0.686 | 0.532 |
| 17 | 0.934 | 0.375 | 0.980 | 0.911 |

The result is not merely a 64D basis artifact. At state dimension 512, write R²
is 0.910/0.701/0.980. Full-width raw top-four tensor energy is only
0.031/0.020/0.037, so most total weight energy remains broad at every layer;
the late concentration is strongest on the natural state manifold.

Centered empirical MLP-write covariance provides the least assumption-laden
usage measure:

| Layer | FineWeb top-four / eff. rank | WikiText | TinyStories |
|---:|---:|---:|---:|
| 0 | 0.324 / 21.8 | 0.343 / 19.2 | 0.356 / 20.8 |
| 8 | 0.090 / 226.2 | 0.126 / 140.1 | 0.117 / 142.7 |
| 17 | 0.910 / 3.5 | 0.921 / 3.2 | 0.928 / 2.7 |

The FineWeb-discovered top-four subspace retains 98–99% of the OOD-optimal
top-four variance at layer 17, 90% at layer 0, but only 62–72% at layer 8.
Thus the middle layer is not just higher rank; its dominant output subspace is
more distribution-dependent.

The pure weight spectra most directly test the proposed random-matrix picture.
All full-width bulks fit power laws well (`R²=0.945–0.966`) and both null
families retain them. The fitted exponents are 0.200, 0.201, and 0.178 for
layers 0, 8, and 17. Only layer 17 has four individually positive deviations
above the extrapolated bulk; their mean log excess is 0.182 versus null medians
near -0.07 and exceeds every tested null replicate. Layer 8's mean excess is
-0.264. This supports **power-law bulk plus clean spikes as one late-layer
regime**, not a universal tensor-layer law.

Prospective validation of those four layer-17 full-width spikes is mostly but
not completely positive. Their subspace nearly equals both the earlier
state-restricted query subspace and empirical natural-write subspace. Exact
input forms predict finite effects at correlation 1.000, and their top modes
massively beat low modes in scalar effect and final-logit KL. But all four are
coarsely labeled `comma_period` by the restricted output vocabulary, one misses
AUROC 0.70, and two miss 50% top-four input-form energy. The spikes identify
strong operations; their human ontology and complete upstream conditions remain
partially unresolved.

The two reported power-law results are not contradictory. The earlier exponent
0.838 and negative top-four excess describe a **32D state-restricted query
map**. The exponents near 0.2 and positive layer-17 spikes describe the **entire
raw bilinear tensor as a symmetric-interaction-to-write operator**. Spectrum,
outlier status, and circuit identity are conditional on the operator being
decomposed.

### Query-wise and per-example regimes

Global output rank does not reveal whether each individual queried operation is
simple. Sixteen FineWeb-discovered natural-write queries were therefore
converted into exact interaction forms inside a common 512D state basis. Raw
form-energy rank90 is broad at every depth—about 204–210 modes for the leading
natural queries. Executed score fidelity gives a very different ordering:

| Layer | First rank reaching corr. 0.90 and R² 0.80 | Rank-16 corr./R² | Full-512 corr./R² |
|---:|---:|---:|---:|
| 0 | 64 | 0.833 / 0.584 | 0.984 / 0.960 |
| 8 | 256–512 | 0.562 / 0.298 | 0.919 / 0.824 |
| 17 | 4–16 | 0.953 / 0.907 | 0.983 / 0.964 |

The ranges reflect corpus variation. Layer 8's middle-spectrum, tensor-top, and
random queries do not reach the joint gate even at rank 512. Layer 17's random
queries do reach it in four modes, which shows that its compactness is a generic
property of the naturally exercised write bottleneck rather than independent
evidence for semantic axes.

Per-token adaptive selection then separates two notions of sparsity:

| Layer | Absolute-mass K90 | Effective support | Fixed / adaptive rank-16 corr. | Effective occupied modes / union |
|---:|---:|---:|---:|---:|
| 0 | 164 | 68 | 0.848 / 0.979 | 93 / 342 |
| 8 | 193 | 95 | 0.649 / 0.931 | 100 / 398 |
| 17 | 149 | 11 | 0.970 / 0.991 | 41 / 355 |

These are FineWeb values; the patterns replicate OOD. Layer 8 is not sparse by
absolute contribution mass. It is sparse enough for signed-score prediction if
an oracle chooses a different top-16 set for each token. Its cancellation ratio
is about 0.11, compared with 0.21 at layer 17, so much of the dense tail
cancels. The next mechanistic object is therefore a **context-predicted mode
router**, not a universal list of middle-layer neurons or eigenmodes. Until such
a router predicts selections without observing the contributions and survives
finite interventions, adaptive sparsity remains a diagnostic rather than an
executable circuit.

Three frozen router families test that next object:

| Router, budget 16 | FineWeb corr. | WikiText | TinyStories | Typical support recall |
|---|---:|---:|---:|---:|
| 64 context clusters | 0.727 | 0.686 | 0.697 | 0.312 |
| 128D quadratic support estimate | 0.847 | 0.829 | 0.822 | 0.500 |
| 128D, 8-neighbor retrieval | 0.759 | 0.761 | 0.749 | 0.312–0.375 |
| 2.14M-parameter learned 128D | 0.775 | 0.762 | 0.753 | 0.312 |
| Oracle realized top 16 | 0.931 | 0.927 | 0.925 | 1.000 |

The cluster router's gains saturate from 64 to 256 clusters. The quadratic
router is the strongest and improves monotonically with state dimension, but
its fully reduced-context score reaches only 0.69–0.71 even at 128D. Nearest
neighbors fail despite storing the full discovery contribution table. Thus the
oracle result currently shows **post-hoc compressibility**, not a recovered
contextual program. A successful next router must exploit structure beyond
coarse regions, Euclidean context similarity, or leading-coordinate quadratic
projections, and must be judged against both the oracle and fixed-mode baselines.

A final quadratic dimension sweep locates the information scale rather than
leaving “high-dimensional” qualitative:

| Router state width | Exact routed corr. Fine/Wiki/Tiny | Fully reduced corr. | Support recall |
|---:|---:|---:|---:|
| 128 | 0.847 / 0.829 / 0.822 | 0.713 / 0.711 / 0.689 | 0.500 |
| 192 | 0.877 / 0.862 / 0.858 | 0.790 / 0.787 / 0.766 | 0.562 |
| 256 | 0.896 / 0.886 / 0.884 | 0.839 / 0.838 / 0.825 | 0.625 |
| 384 | 0.919 / 0.911 / 0.910 | 0.899 / 0.892 / 0.888 | 0.750 |
| 512 | 0.931 / 0.927 / 0.925 | same | 1.000 |

The 512D endpoint equals oracle by construction. The significant result is that
half-width routing is near-oracle while 128D routing and a much larger learned
network fail. This separates **payload sparsity**—16 modes suffice once selected—
from **address complexity**—the selection depends on a broad state quotient.
For reverse engineering, a sparse circuit listing without its address
computation is incomplete.

### State, semantic, and behavioral quotient dimensions differ

An uncentered SVD of natural layer-17 inputs gives a stronger generic basis than
the original semantic form basis. A 64D union of 32 semantic and 32 state
directions performs best on every scalar family:

| Basis | Semantic corr. | Natural-write PCA corr. | Top-four corr. | Whole-write R² |
|---|---:|---:|---:|---:|
| Semantic 32 | 0.856–0.903 | 0.301–0.387 | 0.944–0.957 | 0.669–0.682 |
| State SVD 32 | 0.877–0.897 | 0.562–0.579 | 0.961–0.965 | 0.903–0.911 |
| Hybrid 64 | 0.956–0.969 | 0.714–0.736 | ≈0.996 | 0.961–0.964 |

Inserting the projected input into the native final MLP exposes a long
behavioral tail. Hybrid-64 produces median KL 0.040–0.046 despite write cosine
0.991. On disjoint panels:

| Hybrid dimension | FineWeb KL | WikiText KL | TinyStories KL |
|---:|---:|---:|---:|
| 256 | 0.0201 | 0.0241 | 0.0218 |
| 512 | 0.0109 | 0.0120 | 0.0107 |
| 768 | 0.0050 | 0.0056 | 0.0050 |
| 1,024 | 0.0013 | 0.0015 | 0.0013 |

This layer is compact for its strongest interpretable operations and broad for
exact next-token behavior. “How many dimensions?” is metric-dependent: roughly
32 for scalar score interpretation, 64–128 for calibrated tangents of leading
operations, about 768 for KL below 0.01, and about 1,024 for near-exact
replacement.

### Executable tensor programs and their frontier

Projecting into an input basis is not an efficiency result if all 4,608 native
channels are still evaluated. The Hybrid-64 weights were therefore contracted
exactly into an output×64×64 tensor and executed directly. The contraction
matches the projected native MLP to relative error `7.4e-7`.

Global tensor SVD is a poor ordering for natural execution: rank-128 adds
0.0069–0.0087 KL to Hybrid-64. Natural-write PCA at identical rank and compute
reduces that to 0.0012–0.0015 and transfers to both OOD corpora. Rank-64 adds
only 0.0031–0.0033. This cleanly separates **global tensor energy** from
**natural output importance**.

Crossing input width and natural output rank gives the actual algorithmic
frontier:

| Input×output rank | Counted speedup | FineWeb KL | WikiText KL | TinyStories KL |
|---:|---:|---:|---:|---:|
| 64×128 | 21.2× | 0.0446 | 0.0472 | 0.0410 |
| 128×64 | 12.4× | 0.0363 | 0.0414 | 0.0343 |
| 128×128 | 6.6× | 0.0333 | 0.0377 | 0.0320 |
| 256×128 | 1.8× | 0.0274 | 0.0322 | 0.0273 |

The frontier is monotone in both input width and output rank, but all locked
high-fidelity gates fail. These are useful approximate programs, not drop-in
near-exact replacements. Counts include projection, monomials, quadratic
features, and output decoding, but not memory traffic or measured kernel time.

### The tangent interpreter restores omitted cross terms

For an edit `δ` inside the selected subspace, the exact scalar effect is
`Δq = 2 δᵀ M(y) x + δᵀ M(y) δ`. The observational 32D interpreter replaces
`x` with its projection and omits interactions between the edited coordinate
and outside context.

That omission explains why one top component had observational correlation
0.945 but finite-effect correlation only 0.290. Adding state directions in
Hybrid-64 raises it to 0.991; the full formula is exact to `1.1e-6`.

Across sixteen components and disjoint corpora, Hybrid-64 median correlations
are 0.973–0.997 on high-score sites and 0.986–0.994 on random sites; 14–15 of 16
components exceed 0.90 in every corpus. Hybrid-128 lowers median relative error
to 0.046–0.075 except WikiText random sites at 0.104. Directional prediction is
more compact than perfect calibration.

The current hierarchy is:

1. **Weight spectrum:** prioritizes candidate operations.
2. **Query-conditioned score matrix:** explains observational scalar scores.
3. **Context-conditioned tangent:** predicts local interventions.
4. **Large behavioral quotient:** preserves the complete output distribution.

None can replace the others without changing the question.

### What the power law does and does not mean

The whitened semantic spectra have a median fitted exponent near 0.692 with
`R² ≈ 0.968`. Sign-flipping the feature coefficients gives essentially the same
bulk (`α ≈ 0.691`, `R² ≈ 0.968`); matched controls do too. Therefore:

- a power-law-like bulk is a property of these structured weights and cleaning
  procedure, but is not by itself a semantic discovery criterion;
- an outlier is only a circuit candidate until it predicts an independently
  defined behavior, recurs across fits, and passes a finite intervention;
- the useful decomposition is conditional on a readout/feature direction. The
  whole layer can remain high-rank even when each queried circuit is low-rank;
- eigenvalue magnitude is not a causal ordering. Adding modes sometimes hurts,
  so mode selection must use held-out downstream effect, not explained energy.

### Prospective held-out-category failure

To measure the base-rate problem, a follow-up froze nine token categories that
were not used in the five-category pilot. It selected three candidates solely
by squared overlap of their leading whitened eigenvectors across SAE seeds,
before running any causal evaluation. This is a prospective test of the rule
“stable spectral outliers identify circuits.”

| Frozen candidate | Cross-seed overlap | FineWeb full/top-1 AUROC | Top-1 coordinate fidelity | Complementary-bulk fidelity | Matched control |
|---|---:|---:|---:|---:|---:|
| word continuation | 0.724 | 0.799 / 0.608 | -0.207 | 0.417 | 0.024 |
| non-ASCII | 0.570 | 0.702 / 0.885 | -0.144 | 0.395 | 0.579 |
| colon/semicolon | 0.390 | 0.918 / 0.995 | 0.160 | 0.235 | -0.049 |

Only the semantic-prediction hypothesis passed (median top-1 AUROC 0.830).
None of the three leading modes reached the locked coordinate-fidelity criterion or beat its
complement. The semantic median causal fidelity was -0.144 versus 0.024 for
controls, and the significant-only test also failed.

Across all nine categories, the diagonal assignment was globally nonrandom
(`p = 1/36,288`), but only four of nine rows had their diagonal as the best
match; the preregistered identity criterion therefore failed. This is evidence
for shared, recurrent interaction geometry, not nine cleanly separated circuit
axes.

The positive and negative experiments together establish a hierarchy:

1. A power-law bulk is descriptive weight structure.
2. A null-exceeding outlier is an exceptional weight direction.
3. Cross-fit recurrence makes it a reproducible direction.
4. Held-out AUROC makes it a semantic predictor.
5. A local residual-subspace intervention establishes causal participation.
6. Feature-path mediation measures how much of that effect follows the proposed
   named feature.
7. Sufficiency and behavioral replacement are still stronger claims.

The first four do not imply the fifth. For prospective discovery, the spectral
analysis is currently a candidate generator, not a standalone circuit detector.

## Exact bilinear local structure

For a self-bilinear layer `f(x) = B(x,x) + b`, the response to an edit is exactly

`f(x + αv) - f(x) = α J_f(x)v + α² B(v,v)`.

This yields three separable requirements:

- **Value:** match `f(x)` on language states.
- **Tangent:** match `J_f(x)v` for the intervention distribution of interest.
- **Curvature:** match `B(v,v)` away from the operating point.

The direct CP model is better at Gaussian tangents than the language model
(0.507 versus 0.358 layer-write fidelity), while the language model is much
better at natural tangents (0.805 versus 0.554) and natural values (0.860 versus
0.559). Neither dominates. This is why a mixed operator can outperform both.

## What the validated blend means

The frozen replacement

`f_mix = 0.8 f_language + 0.2 f_global`

improved every preregistered metric on disjoint data:

| Corpus | KL, language → mix | Gaussian causal fidelity | Natural causal fidelity |
|---|---:|---:|---:|
| FineWeb | 0.0144 → 0.0132 | 0.173 → 0.185 | 0.628 → 0.644 |
| WikiText | 0.0189 → 0.0178 | 0.203 → 0.223 | 0.633 → 0.649 |
| TinyStories | 0.0100 → 0.0090 | 0.099 → 0.130 | 0.617 → 0.623 |

This validates complementarity between a language-manifold approximation and a
global-operator approximation. It does **not** mean the literal rank-4096 blend
is minimal or canonical.

Three CP-2048 recompression objectives reached at least 0.9993 local value
agreement with the blend, yet all lost much of its downstream causal gain. That
is direct evidence that tiny errors in downstream-sensitive directions matter
more than large errors in insensitive directions.

## Sparse-factor and subspace tests

### Inherited factor selection

Selecting 512 of the 2,048 direct-versus-language term corrections gave:

- 46% of the FineWeb behavioral improvement under downstream-gradient ranking;
- 85% of the natural-intervention improvement;
- no Gaussian-intervention improvement;
- modestly better aggregate results than a random 512-term control.

The selected identities were not stable. Across two disjoint FineWeb samples,
score Spearman correlation was 0.041 and top-512 Jaccard was 0.178. Random
top-512 Jaccard at this width is about 0.143, so this is weak shared signal, not
literal randomness, but it is far below the preregistered 0.5 stability bar.
Only 16 rows were used per estimate, so sample noise remains a live alternative.

### Output subspaces

Ordinary PCA of the correction write is basis-invariant with respect to CP term
permutation and rescaling. It still failed as a compact causal account:

| Output rank | Value variance explained | Hybrid variance explained | FineWeb KL, hybrid | Gaussian fidelity | Natural fidelity |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.0144 | 0.173 | 0.628 |
| 32 | 0.600 | 0.430 | 0.0144 | 0.158 | 0.620 |
| 128 | 0.797 | 0.636 | 0.0140 | 0.162 | 0.625 |
| 256 | 0.894 | 0.771 | 0.0137 | 0.176 | 0.632 |
| 512 | 0.963 | 0.901 | 0.0134 | 0.174 | 0.640 |
| 1,152 | 1.000 | 1.000 | 0.0132 | 0.187 | 0.643 |

The leading-variance directions can be behaviorally irrelevant or actively
harmful. A useful subspace must be weighted by the downstream metric, not by
activation variance alone.

## Current hypotheses and status

| Hypothesis | Status | Evidence |
|---|---|---|
| Language behavior occupies a simpler region than the global tensor | Supported | Natural-value fidelity can be high while isotropic fidelity is low |
| Sparse conditional use exists | Supported | Routed CP uses 128/2,048 terms per token, although all terms are used over the corpus |
| A small universal native/CP subset explains the layer | Falsified at tested budgets | About half the native channels are needed; 512 correction terms miss Gaussian causal gain |
| The direct global tensor alone is the best causal explanation | Falsified | It has poor natural behavior and downstream causal fidelity |
| Manifold and global descriptions are complementary | Supported on disjoint data | The frozen 80/20 blend improves all three corpora and both intervention types |
| Ordinary local MSE is enough to recompress the blend | Falsified for tested optimizers | Near-perfect local agreement still loses downstream effects |
| The important inherited CP factors are stable mechanism units | Not supported | Low rank correlation and low top-set overlap across samples |
| A low-rank output PCA captures the relevant correction | Falsified at ranks 128–256 | High variance explained but low recovered behavioral/causal gain |
| A rank-256 downstream-weighted linear quotient captures it | Not supported | It recovers 65% of behavior, 67% of natural causal gain, and none of the Gaussian gain |
| Intervention-specific rank-256 quotients capture it | Not supported | Gaussian and natural specialists both fail their in-distribution gates |
| A nonlinear, input-conditioned, or higher-rank quotient captures it | Open | Fixed rank-256 linear explanations are now an inadequate class |
| Output-conditioned semantic forms contain compact spectral discriminators | Supported for five token-form directions | Rank 1–2 preserves semantics across seeds/corpora |
| A power-law bulk or large eigenvalue identifies semantics by itself | Falsified in this pilot | Sign-flip nulls and matched controls have nearly identical bulk fits and also show outliers |
| The leading interaction direction is reproducible across independent feature fits | Supported for the tested categories | Exact cross-seed rank-1 matching gives `p = 1/120` and 5/5 diagonal matches |
| Bfloat16 scalar-coordinate rank curves measure eigenmode causality | Falsified | Float32 raises rank-4 fidelity to 0.975–0.986 and restores smooth local scaling |
| Keeping more spectral modes monotonically improves bfloat16 coordinate reconstruction | Falsified | Rank curves are nonmonotonic under quantized product edits |
| Spectral recurrence plus semantic prediction prospectively identifies the cleanest local paths | Not supported for the first held-out panel | Known modes have 0.408–0.410 local mediation; prospective modes have 0.170–0.222 |
| Shared recurrent spectral geometry exists beyond the original five categories | Supported descriptively | Nine-category assignment is globally nonrandom, but only 4/9 diagonal matches are row-best |
| Natural-value greedy ranking improves coordinate reconstruction | Supported at rank 4 | Disjoint bfloat16 fidelity rises from 0.296 to 0.463, but float32 shows both are already near one |
| Fisher weighting improves over natural-value ranking | Not supported | The two rank-4 orderings tie at 0.463 bfloat16 and 0.986 float32 |
| Known rank-4 spectral modes are locally feature-selective and causally active | Supported | 16.7–20.0× feature selectivity and 2.2–5.4× effect energy versus norm-matched low modes |
| Known spectral modes form self-contained feature circuits | Not supported | SAE decoder mediates only about 0.41 of the downstream effect |
| SAE decoder and minimum-norm dual are interchangeable causal definitions | Falsified | SAE decoder is consistently better by roughly 0.20–0.23 locally |
| High-scoring local mediation estimates average category behavior | Falsified | Stratified random-positive mediation falls to 0.12–0.23 and most locked replication gates fail |
| SAE gate state alone identifies the semantic route | Falsified | Active surface positives mediate much better than other active uses of the same feature |
| Inactive features use the same route | Falsified locally | Near- and random-inactive mediation is approximately zero |
| A small reusable companion ensemble improves the named feature path | Supported on held-out FineWeb sites | Frozen 8/16-feature cores reach 0.501/0.606 on held-out high-active sites |
| The 16-feature core fully transfers to random active semantic sites | Just below locked gates | 0.417 versus 0.529 adaptive; retention 78.8% versus an 80% gate |
| A 32-feature fixed core transfers to random active semantic sites | Supported descriptively | 0.514 fidelity, near the adaptive 0.525, with matched negatives at 0.062 |
| The complete SAE code change is the best causal explanation | Falsified | Adaptive top-16 reaches 0.640 versus 0.552 for all changed SAE features |
| Companion feature IDs are canonical across SAE seeds | Not the right invariance | Exact IDs change; compare decoded-write subspaces instead |
| Category-specific companion write subspaces recur across SAE seeds | Supported for five categories | 5/5 category assignment, exact `p = 1/120`; companion-only result also 5/5 |
| An independently discovered seed-102 sparse program remains causal | Supported on held-out high-active sites | Frozen 16-feature program reaches 0.540 downstream fidelity |
| Seed-101 subspace transfer is uniform across categories | Falsified | Four categories reach 0.671–0.844, but newline is 0.004 |
| A corresponding cross-seed subspace is more relevant than another category's subspace | Supported | 0.636 versus 0.026 median oracle-projection fidelity |
| Category spectral coordinates can transfer a feature into inactive recipients | Supported | 0.482 median donor-feature transfer, 95% gate activation; matched low/wrong modes are near zero |
| A fixed 32-feature program fully mediates donor transfer | Falsified | Native frozen-core fidelity is 0.281; adaptive 32 is 0.428 and complete observed code is 0.450 |
| SAE encoder coefficients are faithful coordinates for the intervention | Falsified | Reusing its support with refit coefficients improves downstream fidelity from 0.450 to 0.784 |
| Frozen category support is directionally specific OOD at matched activity | Supported for high-active sites | Own core beats activity/write-norm-matched controls in 5/5 categories on both OOD corpora |
| The same frozen support describes all gate-active uses | Falsified at the locked random-active gate | OOD random-active fidelity is 0.127–0.171 and matched-control gaps are only 0.079–0.090 |
| Layer-17 leading output modes are class-specific circuits | Falsified | Cross-category diagonal assignments fail for ranks 1–8, both direct and end-to-end |
| Layer-17 output classes reuse a compact shared input geometry | Supported | Rank-4 overlap is 44× random; a shared 4D quadratic code retains 0.679–0.739 median AUROC |
| A single neuron rotation diagonalizes the output code | Falsified exactly for these forms | Nonzero pairwise commutators; diagonal-only language-state AUROC loses 0.12–0.20 at dimension 32 |
| Pairwise interactions in the shared code are behaviorally active | Supported descriptively | Cross terms carry 70–81% of score variance and restore semantic prediction across three corpora |
| A shared layer-17 quotient generalizes beyond the five discovery concepts | Supported distributionally, nonuniformly | Held-out median score correlation 0.66–0.87; 88–100% retain oriented AUROC, despite only 5% global form energy |
| Layer-0 lexical form families admit a stable neuron-like rotation | Supported | Joint diagonalization leaves 5.9% off-diagonal energy, 0.987–0.988 score correlation, and unchanged AUROC |
| Layer-17 original concepts admit stable approximate joint axes | Supported numerically | 15.7% off-diagonal energy, 0.93–0.95 score correlation, 0.979 cross-start axis overlap |
| Layer-17 diagonal axes generalize semantics to new concepts | Falsified | Held-out diagonal AUROC remains far below the full quadratic quotient despite improved score correlation |
| A layer has one readout-independent semantic form family | Falsified at layer 0 | Embedding- versus unembedding-conditioned same-category form cosine has median -0.090 |
| Stable joint axes execute context-free diagonal roles | Falsified | Diagonal finite-effect fidelity collapses at ordinary sites; full 32D interactions restore it, especially at layer 17 |
| Four semantic role directions describe the full residual write | Falsified at layer 0 | Four directions capture 0.915 under semantic probes but only 0.421 in random residual sketches |
| Selected layer-17 axes have low-dimensional contextual residual roles | Supported | First/four-role energy 0.712/0.903, rank90 about 4.3, stable across sketches and OOD corpora |
| The recovered coordinate basis is uniquely clean | Falsified | Random rotations have lower within-axis rank but substantially duplicate first-role directions |
| Joint axes balance role cleanliness and between-axis diversity better than random rotations | Supported by a new diagnostic | Adjusted score 0.063 vs 0.041 at layer 0 and 0.146 vs 0.102 at layer 17 |
| Later layers collapse clean layer-0 predicates into simpler fixed output roles | Falsified on the selected panel | Median local rank90 16.5 rises to final rank90 25; every predeclared group broadens |
| Downstream broadening is caused only by full coordinate removal | Falsified | At 10% removal, final rank90 remains 25, effect directions align 0.939 with full removal, and KL scales with exponent 1.90 |
| A compact linear query map can answer arbitrary layer-17 scalar readouts | Supported algebraically and distributionally | Exact inside 32D; semantic/frequent-token correlation 0.80–0.90 OOD |
| Power-law bulk alone identifies learned circuits | Falsified again | Sign and pairing nulls retain exponent and fit; broad semantic AUROC also survives sign nulls |
| Leading query-map components prioritize real natural operations | Supported for top four | 22–26× middle-spectrum variance, 0.944–0.957 OOD score correlation, low-rank forms, large specific finite effects |
| The semantic 32D basis is the best late-layer quotient | Falsified | State-SVD-32 is stronger on write reconstruction and top operations; Hybrid-64 is stronger still |
| Hybrid-64 is a high-fidelity final-MLP replacement | Falsified | Write R² ≈0.96 and cosine 0.991 coexist with KL 0.040–0.046 and CE loss |
| The final MLP is behaviorally tiny in input dimension | Falsified at tested gates | KL <0.01 appears around 768/1,152 dimensions; near-exact output needs about 1,024 |
| A 32D observational quotient predicts all finite edits | Falsified component-wise | One top component falls to 0.290 because outside-context cross terms are omitted |
| A 64D context-conditioned tangent generalizes | Supported for leading components | Median correlation 0.973–0.997 high-score and 0.986–0.994 random over three corpora |
| Contracted quadratic programs can eliminate native-channel scoring | Supported | Exact Hybrid-64 contraction error 7.4e-7; factored programs execute from basis coordinates directly |
| Global tensor SVD is the best efficient ordering | Falsified | Natural-write factors reduce extra Hybrid-64 KL by 70–95% at matched rank and compute |
| A tested program is both near-exact and highly accelerated | Not supported | 21× point has KL 0.041–0.047; KL 0.027–0.032 point is only 1.8× by counted multiplies |
| Every tensor layer has a few dominant clean spectral operations | Falsified | Under identical audits layer 8 is broad, weakly concentrated, and has no strict full-width positive outlier |
| Depth-dependent natural-write regimes transfer across corpora | Supported | Effective ranks are 19–22 at layer 0, 140–226 at layer 8, and 2.7–3.5 at layer 17 |
| A power-law-like full-width bulk identifies learned circuits | Falsified | All layers and coordination nulls retain high-R² bulks |
| Layer-17 full-width tensor spikes exceed coordination nulls | Supported | Four positive deviations; mean log excess 0.182 beats every sign/pairing null |
| Full-width late spikes are the same subspace as natural/query operations | Supported | Mean canonical cosine 0.930 with natural writes and 0.979 with the state-64 query map |
| Every full-width late spike is a simple low-rank semantic circuit | Falsified at the locked gates | One misses AUROC 0.70 and two miss 50% four-mode form energy despite enormous causal specificity |
| Gaussian covariance cleaning is a faithful substitute for empirical usage | Not supported | Full-width covariance cosine is 0.674/0.811/0.778 across layers 0/8/17 |
| A globally broad middle layer is a library of fixed low-rank queried operations | Falsified | Leading natural queries need 256–512 modes for held-out score fidelity at layer 8 versus 4–16 at layer 17 |
| Late low-rank query scores imply semantic axes | Falsified as a standalone inference | Random layer-17 readouts also reach the observational gate in four modes |
| Layer-8 query computation is sparse by absolute contribution mass | Falsified | Median per-token K90 is about 193 modes and effective support about 95 |
| Layer-8 query scores admit oracle conditional sparsity | Supported observationally | Adaptive top-16 correlation is 0.925–0.931 versus 0.604–0.649 for fixed modes |
| A small fixed mode identity set explains layer-8 multiplexing | Not supported | Adaptive top-16 slots use 378–398 modes in union, effective occupancy about 95–103 |
| Coarse context clusters predict layer-8 oracle routes | Not supported | 64 clusters reach 0.686–0.727 correlation and 0.312 support recall versus oracle correlation about 0.93 |
| A 64D weight-derived quadratic router recovers the adaptive program | Not supported | Exact selected modes reach only 0.760–0.782 correlation; approximate scores reach 0.548–0.579 |
| A 128D quadratic router nearly closes the oracle gap | Not supported | Correlation 0.822–0.847 remains 0.08–0.10 below oracle despite 50% support recall |
| Nearest-context retrieval recovers the missing route information | Falsified for the tested grid | Preregistered 128D/8-NN reaches 0.749–0.761; exploratory 64D retrieval peaks below 0.79 |
| Router failures are caused by OOD distribution shift | Falsified for these routers | Their mediocre fidelity is stable across FineWeb, WikiText, and TinyStories |
| A large nonlinear 128D router closes the oracle gap | Falsified for the tested GELU router | 2.14M parameters reach only 0.753–0.775 and overpredict FineWeb performance from internal validation |
| Layer-8 routing requires all 512 state dimensions | Falsified | 192D first passes correlation 0.85; 256D is within 0.05 of oracle on all corpora |
| Layer-8 routing has a tiny address code | Not supported | 128D misses predictive/near-oracle gates; approximately half of the tested 512D quotient is needed |
| Sparse payload implies sparse end-to-end computation | Falsified as an inference | Sixteen modes suffice only after a 192–256D routing computation identifies them |
| Hierarchical routing makes a 64D coarse address near-equivalent to flat 256D routing | Not supported | Shortlist-64 misses fidelity gates; shortlist-128 costs 43.8% of candidate projections but remains below flat score correlation |
| A hierarchical-128 route predicts finite direct query effects | Supported at the locked aggregate gate | Direct-effect correlation is 0.858 on FineWeb and 0.853 on WikiText |
| The hierarchical route is causally near-equivalent to flat-256 OOD | Not supported | Its WikiText median Fisher fidelity is 0.640 versus 0.798 for flat-256 |
| A local query-score match guarantees downstream equivalence | Falsified | Fixed-16 has high direct correlation but only 0.302–0.448 downstream Fisher fidelity |
| Layer-8 routed effects require inseparable Layer-9 direct/attention interactions | Falsified over 1%–20% edits | Summed hybrid logit effects retain at least 0.998 median Fisher fidelity; immediate interaction energy is below 0.001 |
| Neither Layer-9 path is sufficient by itself | Falsified | The direct residual/recomputed-MLP hybrid retains 0.90–0.94 fidelity; attention alone is near zero |
| Hierarchical routing fails mainly through the attention-mediated path | Falsified | WikiText direct path match is 0.739 while attention path match is 0.930 at 5% |
| Flat-256 improves both direct and attention path matching | Supported | It equals or exceeds hierarchical route-to-oracle fidelity for both paths and corpora |
| Ordinary 16-sample Fisher sketches are reliable on every site | Falsified | Peaked output distributions make 3–19% of site sketches degenerate across the two panels |
| Stratified downstream-Fisher mode impacts are stable | Supported | Exact top-token plus sampled-tail estimates reach 0.995–0.996 correlation and 0.938 top-16 recall |
| Local query modes and downstream-sensitive modes are the same circuit | Falsified | Median support overlap is 0.312 and each wins strongly on its own objective |
| Top individual Fisher modes form a strong joint intervention | Falsified | Exact joint energy is only 0.20–0.23 of summed individual energies because their effects cancel |
| The small joint Fisher-set effect is caused by downstream nonlinearity | Falsified | The signed sum predicts the exact joint effect at 0.956–0.971 fidelity |
| The full signed Fisher Gram can select a constructive causal program | Supported | Joint-aware top-16 exact KL is 0.0058–0.0060 and held-out coherence is 2.79–5.03 |
| A stable causal program requires stable native mode identities | Falsified as a necessity | WikiText support recall is 0.406 while effect cosine is 0.954 and objective efficiency 0.904 |
| The joint Fisher result is a 5%-scale tangent artifact | Falsified over the tested range | It beats the local route by 3.21–7.19× KL from 1% through 20% with effect cosine at least 0.920 |
| The joint Fisher program is already a semantic or executable router | Not established | It is per-example, suffix-gradient-derived, and overlaps the local-query set by only 6–9% |
| A global frozen suffix-Fisher metric predicts the conditional program | Falsified for the tested quotient | Full-rank effect fidelity is only 0.425–0.531 and learned metrics do not consistently beat geometric controls |
| Nearby Layer-8 states have reusable suffix metrics across corpora | Falsified for the tested KNN router | Best common K is 16, with fidelity 0.606 FineWeb but 0.154 WikiText |
| Attention-clamped tensor propagation is sufficient everywhere | Not supported | It recovers 79–84% of magnitude in one panel but effect fidelity is panel-dependent and can fall to 0.540 |
| Full analytic tensor propagation recovers the local causal program | Supported | Effect fidelity is 0.994/0.983; 5%-finite tangent fidelity is 0.984/0.986 |
| The state-conditioned eigenmode-removal Fisher Gram requires high rank | Falsified for that perturbation family | Rank 4 gives 0.982–0.997 finite-effect fidelity and rank 16 is effectively exact |
| Eigenmode-removal Gram compactness is specific to natural readouts | Falsified | Random readouts on random sites are equally or more rank-4 compact; rank 16 is exact throughout |
| The basis-invariant unit-input Layer-8 causal operator is rank 4–16 | Falsified | Rank-4 energy is 0.455–0.512 and rank-16 is 0.690–0.746; roughly rank 64 is needed for 0.88–0.92 |
| Leading Layer-8 local causal axes are stable global input verbs | Falsified at rank 4 | Input-subspace overlap is about 0.010 across sites, near the random 4/512 baseline |
| Leading Layer-8 local causal axes have stable token/output meanings | Falsified | Fisher subspace overlap is 0.002–0.005 and top-token Jaccard is 0.000–0.016; unweighted logit overlap remains only 0.165–0.174 |
| The decoded local axes are finite-causally real | Supported | Fidelity is 1.000 at 1% and 0.995–0.997 at 5%; KL is about 48–50× matched random directions |
| Conditioning on the exact native prediction recovers a stable internal Layer-8 verb | Falsified | Fisher overlap improves by 0.057–0.077, but logit advantage is only 0.004–0.005 and input advantage 0.001–0.003 |
| Named one-token semantic edits produce reusable Layer-8 state or sensitivity directions | Falsified on five families | Same-pair state cosine is only 0.046–0.074; fixed-position and high-effect controls do not rescue family recurrence |
| A local Layer-8 causal chart is stable under nearby semantic edits | Supported | Rank-4 input/logit/Fisher overlap remains 0.985–0.997 despite absent global recurrence |
| The final-position Layer-8 MLP is the main semantic carrier | Falsified | Its write patch reaches 0.107–0.134 fidelity and 2–3% of full KL |
| The Layer-8 residual sequence is the main semantic carrier | Supported | Whole-sequence fidelity is 0.934–0.956; signed Shapley projection is 0.917–0.934 |
| Semantic residual effects require all seventeen affected positions equally | Falsified | Oracle top four reach 0.860–0.907 and top eight 0.955–0.972; final/source/intermediate roles are unequal |
| Positionwise semantic effects are strongly nonlinear downstream | Falsified at the tested scale | The signed sum of individual-position effects reaches 0.996 fidelity to the whole residual patch |

## Learned downstream quotient and specialist red-team

The learned rank-256 map is applied to the direct-minus-language correction and
is trained through all later layers against the exact blend. It improves over
hybrid PCA in aggregate, showing that the downstream metric matters, but it is
not sufficient:

| Model | FineWeb KL | Gaussian causal fidelity | Natural causal fidelity |
|---|---:|---:|---:|
| Language baseline | 0.0144 | 0.173 | 0.628 |
| Hybrid PCA rank 256 | 0.0137 | 0.176 | 0.632 |
| Learned quotient rank 256 | 0.0136 | 0.171 | 0.638 |
| Exact blend | 0.0132 | 0.187 | 0.643 |

The learned quotient recovers 65% of the behavioral improvement and 67% of the
natural-intervention improvement, but its Gaussian score is below baseline.
Only WikiText improves on behavior and both intervention metrics.

Fine-tuning separate rank-256 Gaussian and natural specialists does not rescue
the result. The Gaussian specialist remains at 0.171 Gaussian fidelity; the
natural specialist reaches 0.636 natural fidelity, below the universal model's
0.638. This makes simple objective averaging an unlikely explanation of the
failure. It does not rule out better optimization, nonlinear routing, or a rank
between 256 and 1,152.

## Which object to use for which goal

### Explain a natural token prediction

Start with native neuron/group selectivity, attention routes, and natural
activation patches. Then validate a proposed circuit with finite interventions.
The language-manifold object is primary; global isotropic fidelity is a robustness
check rather than the optimization target.

### Predict a model edit or counterfactual

Use the tangent operator for the actual edit distribution and score it after the
remaining network. Gaussian edits, donor-state edits, activation replacement,
and parameter edits are different distributions and need not share a ranking.

### Compress inference

Count actual operations, not descriptive rank. The output-subspace projections
and exact blend still evaluate both CP parents. Routed CP is sparse in selected
features but currently performs dense scoring. A deployment claim requires an
index or router whose own cost is included.

### Discover reusable abstractions

Require cross-context and cross-corpus transfer, plus causal specificity. Stable
groups or subspaces are stronger candidates than individual factors from one
non-unique decomposition. Semantic predictability alone is not enough.

### Reverse-engineer the whole model

Use a breadth-first map of clean predicates and important boundaries, but allow
the intermediate representation to change. The likely useful objects are a mix
of native predicates, attention routes, low-dimensional state changes, and
downstream-weighted subspaces. Dense connectivity by itself is not evidence of
confusion; failure of compact predictive intervention models is the stronger
signal.

## Red-team boundaries

- All CP refinements share an initialization ancestry. Factor-level conclusions
  are not independent decomposition recovery.
- The blend coefficient was selected exploratorily, then validated on disjoint
  rows. It still needs independent model checkpoints and layers.
- Causal fidelity is intervention-distribution specific. Gaussian directions are
  useful stress tests but are not natural language states.
- Rank curves show that the tested algorithms did not find a compact account;
  they do not prove that no nonlinear, input-conditioned, or alternative-basis
  account exists.
- The 16-row factor-stability audit is deliberately cheap and noisy. A larger
  audit is required before concluding that all individual importance is unstable.
- Current results concern selected layers of one trained checkpoint. They
  establish useful methods and falsify several shortcuts, not a universal
  theory of LLMs.
- The five spectral categories were selected because earlier SAE analysis found
  strong token-form features. This validates compactness for known discriminators but
  does not measure the base rate of discoverable circuits.
- The feature direction is SAE-derived, so the spectral method currently
  localizes and canonicalizes a proposed output role; it has not yet discovered
  that role from weights alone.
- Whitening uses an ill-conditioned natural-input covariance. Raw-basis results
  remain positive, but any individual whitened eigenvector should be checked for
  stability under shrinkage and alternate corpora.
- Eight causal sites per category give qualitative evidence but wide
  uncertainty for category-specific effect sizes.
- The prospective atlas uses 12 sites per category and tests scalar feature
  coordinates through minimum-norm native-product edits. Other interventions—
  editing the residual direction itself or following the SAE decoder—may answer
  different causal questions and must not be silently conflated with this one.
- The true input-subspace result now has stratified FineWeb and two-corpus OOD
  tests. They show a high-engagement circuit and a weaker context-dependent
  regime; the high-tail result must not be reported as average prevalence.
- Local interventions are intentionally not full ablations. They identify a
  tangent-scale mechanism; they do not show that the subspace remains meaningful
  after large removal or that it is sufficient by itself.
- Surface token predicates are noisy semantic labels. Some failures are genuine
  route failures; others are false-positive/false-negative labels. Gate-,
  context-, and activation-matched analyses must be reported separately.
- Companion cores are defined in one non-identifiable SAE basis. Their causal
  transfer within that basis is real. Seed-102 re-discovery now supports a
  canonical decoded-write subspace, but a native tensor-basis replication is
  still needed.
- Fixed-core validation now includes independent FineWeb panels, donor transfer,
  and gate-selected WikiText/TinyStories panels with hard magnitude-matched
  controls. It still concerns one layer and one model checkpoint.
- The layer-17 joint basis was selected from the same five output forms it
  summarizes. Corpus transfer tests state dependence, not held-out concept
  discovery. New output concepts and another checkpoint are still required.
- Noncommutation rules out exact simultaneous diagonalization, but the reported
  commutator magnitude is not a universal threshold for interpretability.
- Cross-term necessity is established in the frozen aggregate-energy basis.
  Approximate joint diagonalization removes most of that energy and fully
  rescues layer 0, but it does not rescue open-vocabulary layer-17 semantics.
- The recovered joint axes are stable across six optimization starts and now
  have large-corpus semantic and residual-role audits. Their meanings remain
  readout- and context-relative; stable optimization is not coordinate uniqueness.
- Random residual sketches preserve geometry only approximately. Agreement
  across three sketches is strong, but the reported rank90 values are lower
  bounds on complexity outside the 96D sketch and do not measure downstream
  survival through the rest of the transformer.
- The concentration-times-diversity product is a diagnostic, not a uniquely
  justified objective. It prevents a clear duplication failure, but other
  simplicity/diversity tradeoffs may select different axes.
- Downstream role propagation currently uses high-local-effect sites at the
  final sequence position. It does not measure average prevalence, earlier
  source-position effects transmitted through attention, or whether the same
  rank persists in logit space under a task-conditioned metric.
- The query-map top-four excess is comparative to the tested nulls, not a
  positive deviation above the fitted power-law extrapolation. Calling them
  strict RMT outliers would overstate the evidence. The later full-width
  layer-17 operator is a different object and does contain four positive
  deviations; this distinction must be preserved.
- Full-width spectral spikes do not imply that the whole late MLP is low-rank.
  They contain only 3.7% of raw tensor-map energy, while natural states amplify
  their output subspace to over 90% of centered write variance.
- Uncentered activation weighting is dominated by residual-state anisotropy and
  can manufacture near-rank-one spectra. Centered Gaussian moments are better
  defined but still miss empirical covariance, especially at layers 0 and 17.
- The restricted 14-predicate output vocabulary labels three full-width spikes
  strongly and the fourth weakly as `comma_period`; those labels are coarse and
  do not explain the heterogeneous positive and negative natural contexts.
- Query-mode adaptive truncation is oracle-selected after observing each
  realized contribution. It demonstrates a compressible signed score, not a
  learned router, causal sufficiency, or cheap executable program.
- Absolute-mass K90, participation support, signed-score reconstruction, and
  fixed spectral rank answer different sparsity questions. The layer-8 result
  changes from dense to sparse depending on that choice, so reports must name
  the metric rather than say only “the computation is sparse.”
- The quadratic router still scores all 512 candidate modes before selecting
  16, and the reported `exact routed` metric evaluates selected projections in
  the full 512D state. It isolates support predictability, not wall-clock
  sparsity. Its `approximate routed` score is the stricter reduced-context test.
- The KNN router is deliberately a high-memory predictability control, not a
  plausible mechanistic decomposition. Its failure rules out simple local
  reuse in the tested coordinates but not supervised nonlinear routing.
- The supervised router is an information probe with 2.14M parameters and
  dense 8,192-way logits. Its failure is evidence against that architecture;
  success would not have established description-length or compute savings.
- The 512D quadratic-router endpoint is oracle by construction. The meaningful
  claims concern the frozen 192–384D prefixes and their OOD curves. State-SVD
  prefix dimension is one coordinate-dependent address-complexity measure, not
  a universal intrinsic dimension.
- Token-predicate AUROC for output singular vectors is not an independent null:
  sign-flipped maps retain learned Down/unembedding geometry and produce equally
  many threshold-passing directions.
- The 64D tangent generalization is strongest directionally. Relative-error
  calibration is weaker on small random-site effects and one WikiText cell
  narrowly misses the 0.10 gate.
- Input projection dimension does not equal execution cost. The initial
  behavioral sweeps still evaluate all 4,608 native products and must not be
  cited as speedups; only the later contracted-program experiments remove them.
- The later contracted-program experiments remove native-channel scoring, but
  report algorithmic multiply counts rather than wall-clock kernels. Natural
  output factors are selected on FineWeb and transfer OOD; they still need an
  independently optimized implementation and hardware benchmark.

## Next decisive experiments

The next experiment should extend and operationalize the canonical ensemble:

1. Identify semantics for stable companion features from large-corpus examples
   and weight-derived input forms. Ask whether they are coherent substeps—such
   as boundary, morphology, address, or output-role features—or arbitrary helpers.
2. Learn a router that predicts coefficients inside the cross-seed canonical
   write subspace without observing the actual edit. This closes the gap between
   oracle mechanism identity and an executable program.
3. Predict the high-versus-weak engagement regime from native context without
   consulting the SAE gate, and validate it on a fresh corpus.
4. Condition directly on unembedding/logit directions in layer 17, then repeat
   the validated local-route/companion-core procedure at layers 0, 2, and 16.
5. Split the validated layer-0 downstream expansion into MLP→MLP and
   MLP→attention→MLP paths, and repeat for earlier source positions. Then use a
   logit/Fisher metric to distinguish task-relevant expansion from final-state
   directions that the unembedding ignores.
6. Optimize axes jointly for within-axis role concentration, between-axis role
   diversity, semantic transfer, and reconstruction. Validate on held-out
   concepts rather than selecting the tradeoff post hoc.
7. The layer-17 query map, natural-write factorization, pulled-back Fisher
   metric, fixed quotient, cross-term audit, context router, granularity sweep,
   seed replication, and semantic atlas are now complete. A hard `4x8` routed
   program beats fixed rank 32 and all shuffled/position/fixed-expert controls.
   Its performance and coarse lexical/structural profiles recur, while named
   experts and route partitions do not. Next test early-stopped exact-KL/CE
   objectives, replicate conditional compactness at a middle layer using the
   validated forward-mode suffix metric, and implement fused hard-routing
   kernels to measure wall-clock rather than multiply counts alone.
8. The middle-layer replication is complete and falsifies universal spectral
   compactness. Hierarchical routing now provides a modest observational Pareto
   improvement, and finite validation shows that direct query fidelity can
   coexist with weaker downstream equivalence, especially OOD. Next replace
   the local score oracle with a downstream-Fisher-aware target. Diagonal
   Fisher ranking identifies real individual effects but fails as a joint
   program because those effects cancel. Global and KNN suffix metrics fail,
   but exact forward-mode propagation through the tensor architecture recovers
   the program without backpropagation. The query-mode removal Gram is rank
   4–16 even for random readouts, while the corrected basis-invariant unit-input
   operator needs roughly rank 64 for 90% energy. Its leading axes are highly
   causal locally but do not recur across contexts in input, logit, or Fisher
   coordinates. Exact-prediction conditioning recovers only Fisher/readout
   recurrence, not input verbs. Natural semantic counterfactuals also fail to
   produce global edit axes, but reveal smooth local tangent charts and a
   near-linear residual-sequence broadcast: the full residual carrier dominates,
   while four oracle positions recover most of it. Exact QK1×QK2×V tracing is
   now complete. A frozen layer-4 edge-address skeleton transfers OOD but only
   partially recovers per-example oracle routing. Exact downstream factorial
   interventions show that the route is overwhelmingly current-value content,
   with a smaller cooperative inherited-bus path and secondary QK changes.
   The transported current-value code does recur by lexical pair and semantic
   family, but routing does not create that abstraction. Direct layer-4
   embedding injection is causally negligible despite its large coefficient;
   the code arrives in the residual from layers 0--3. The exact telescope and
   matched cross-family null show that MLP3→MLP2 dominance is a generic
   token-change transport hierarchy, not a semantic signature. The Layer-3
   native-channel, input-form, polynomial, pair-sparsity, and natural-semantic
   decompositions are now complete. They separate a shared lexical backbone
   from compact family-specific coordinates and a broad increasing-scale
   program. The explicit quotient, rank sweep, unseen-word interventions, and
   hierarchical aggregation audit are now complete: the “backbone” is a
   sentiment/scale/polarity cluster, shared/residual coordinates interact, and
   scale changes regime with lexical pair and corpus weighting. Weight-
   canonical output-conditioned spectra recover low-rank learned coordination
   missed by the global power-law rule and survive fair random-subspace and two-
   null controls. Individual-axis causality and the layers 0/3/8/17 replication
   are now complete. A matched natural-write label null overturns the semantic-
   spike reading: family labels add no spectral concentration, and Layer 17 is
   a shared natural-write bottleneck whose restricted output-4 patch remains
   behaviorally partial. The unrestricted late-bottleneck and residual-tail
   ontology is now complete: the raw outliers dominate variance, while a
   basis-invariant covariance-compact rank-eight tail program carries additional
   causal work and requires cross terms. Downstream-Fisher optimization finds a
   restart-stable core; fixed rank plateaus, but context routing recovers a
   sparse conditional program with a reproducible function class and
   noncanonical expert names. The raw/null/natural/cleaned part has now been
   independently replicated on a 12-layer checkpoint: the bulk and useful
   outliers recur, but the four-spike late phase and depth ordering do not.
   The Layer-11 fixed-tail and conditional-router replications are now complete:
   stable low-rank variance geometry coexists with broad behavioral support,
   and routers provide reproducible compression without canonical experts or
   functions. Do not import the four-expert ontology from the 18-layer model.
   Separately, learn the smallest upstream-state router that closes the frozen-
   to-oracle edge gap without observing counterfactual outputs. In parallel,
   compress the layerwise JVP computation rather than replace it with a global
   metric.
9. The independent full-width bulk/spike/natural-write/cleaning replication,
   fixed Layer-11 tail search, conditional top-k frontier, and restart audit are
   complete. They falsify the original depth
   ordering as a scaling law, identify covariance sample size and ridge as
   first-class choices, and show that variance-compact does not imply
   behaviorally compact. They also separate reproducible lossy compression from
   a canonical mechanism. Next run the restricted address test at Layer 5 using
   a large stable covariance estimate, then ask whether a weight-conditioned
   object there recurs more strongly than these learned router decompositions.
10. The unrestricted Layer-17 bottleneck/tail ontology is complete. Next replace
    the broad structural-token vocabulary with unsupervised token/context
    clusters on a fresh corpus, while treating the output spaces—not their
    current PCA coordinates—as the primary objects.
11. Split layer-8 routed modes by direct MLP→MLP and
    MLP→attention→MLP propagation. The first split is complete: the direct path
    dominates and the two paths are nearly additive. Next localize the direct
    mismatch across layers 9–17 with downstream Fisher sketches rather than
    introducing increasingly elaborate attention routers.

The earlier compression question remains separate: rank-512 versus a routed
mixture of rank-128 downstream quotients should still be compared at matched
parameter and execution cost. Success there would yield an efficient global
replacement; success in the spectral atlas yields an interpretable query system.
Neither result logically implies the other.

## Result files

- `artifacts/bilin18_tensor_decomposition/TANGENT_ONTOLOGY_REPORT.md`
- `artifacts/bilin18_tensor_decomposition/VALUE_TANGENT_BLEND_VALIDATION.md`
- `artifacts/bilin18_tensor_decomposition/BLEND_RECOMPRESSION_REPORT.md`
- `artifacts/bilin18_tensor_decomposition/BLEND_RANK_CURVE_REPORT.md`
- `artifacts/bilin18_tensor_decomposition/DOWNSTREAM_RANK_SELECTION_REPORT.md`
- `artifacts/bilin18_tensor_decomposition/RANK_STABILITY_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/CORRECTION_SUBSPACE_REPORT.md`
- `artifacts/bilin18_tensor_decomposition/DOWNSTREAM_CORRECTION_QUOTIENT_REPORT.md`
- `artifacts/bilin18_tensor_decomposition/CORRECTION_QUOTIENT_SPECIALIST_REPORT.md`
- `artifacts/bilin18_tensor_decomposition/SPECTRAL_SPIKE_PILOT.md`
- `artifacts/bilin18_tensor_decomposition/SPECTRAL_SPIKE_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/SPECTRAL_SPIKE_CONFIRMATION.md`
- `artifacts/bilin18_tensor_decomposition/SPECTRAL_ATLAS.md`
- `artifacts/bilin18_tensor_decomposition/SPECTRAL_DOWNSTREAM_RANKING.md`
- `artifacts/bilin18_tensor_decomposition/SPECTRAL_PRECISION_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/SPECTRAL_INPUT_INTERVENTIONS.md`
- `artifacts/bilin18_tensor_decomposition/SPECTRAL_LOCAL_INPUT_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/SPECTRAL_STRATIFIED_REPLICATION.md`
- `artifacts/bilin18_tensor_decomposition/SPECTRAL_GATE_VALIDATION.md`
- `artifacts/bilin18_tensor_decomposition/SPECTRAL_MATCHED_SEMANTICS.md`
- `artifacts/bilin18_tensor_decomposition/SPECTRAL_COMPANION_PATHS.md`
- `artifacts/bilin18_tensor_decomposition/SPECTRAL_FIXED_COMPANIONS.md`
- `artifacts/bilin18_tensor_decomposition/SPECTRAL_COMPANION_TRANSFER.md`
- `artifacts/bilin18_tensor_decomposition/SPECTRAL_CROSS_SEED_COMPANIONS.md`
- `artifacts/bilin18_tensor_decomposition/JOINT_AXIS_ATLAS.md`
- `artifacts/bilin18_tensor_decomposition/JOINT_AXIS_INTERVENTIONS.md`
- `artifacts/bilin18_tensor_decomposition/JOINT_AXIS_CONTEXTUAL_ROLES.md`
- `artifacts/bilin18_tensor_decomposition/JOINT_AXIS_RESIDUAL_ROLES.md`
- `artifacts/bilin18_tensor_decomposition/JOINT_AXIS_BASIS_CONTROLS.md`
- `artifacts/bilin18_tensor_decomposition/JOINT_AXIS_BASIS_DIVERSITY.md`
- `artifacts/bilin18_tensor_decomposition/LAYER0_DOWNSTREAM_ROLES.md`
- `artifacts/bilin18_tensor_decomposition/LAYER0_DOWNSTREAM_SCALE_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/QUERY_CONDITIONED_INTERPRETER.md`
- `artifacts/bilin18_tensor_decomposition/QUERY_MAP_SPECTRAL_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/QUERY_MAP_COMPONENT_VALIDATION.md`
- `artifacts/bilin18_tensor_decomposition/INPUT_BASIS_TRADEOFFS.md`
- `artifacts/bilin18_tensor_decomposition/FINAL_MLP_BASIS_REPLACEMENT.md`
- `artifacts/bilin18_tensor_decomposition/FINAL_MLP_BASIS_DIMENSION_SWEEP.md`
- `artifacts/bilin18_tensor_decomposition/FINAL_MLP_LARGE_DIMENSION_SWEEP.md`
- `artifacts/bilin18_tensor_decomposition/QUERY_MAP_OUTLIER_CIRCUITS.md`
- `artifacts/bilin18_tensor_decomposition/QUERY_MAP_TANGENT_INTERPRETER.md`
- `artifacts/bilin18_tensor_decomposition/QUERY_MAP_TANGENT_GENERALIZATION.md`
- `artifacts/bilin18_tensor_decomposition/HYBRID64_EFFICIENT_FACTORIZATION.md`
- `artifacts/bilin18_tensor_decomposition/HYBRID64_NATURAL_FACTORIZATION.md`
- `artifacts/bilin18_tensor_decomposition/EFFICIENT_FACTORIZATION_PARETO.md`
- `artifacts/bilin18_tensor_decomposition/DEPTH_QUERY_MAP_REPLICATION.md`
- `artifacts/bilin18_tensor_decomposition/DEPTH_SPECTRAL_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/DEPTH_NATURAL_SPECTRUM.md`
- `artifacts/bilin18_tensor_decomposition/DEPTH_NATURAL_SPECTRUM_OOD.md`
- `artifacts/bilin18_tensor_decomposition/DEPTH_POWERLAW_NULLS.md`
- `artifacts/bilin18_tensor_decomposition/FULLWIDTH_OUTLIER_VALIDATION.md`
- `artifacts/bilin18_tensor_decomposition/DEPTH_QUERYWISE_COMPACTNESS.md`
- `artifacts/bilin18_tensor_decomposition/DEPTH_QUERYWISE_TRUNCATION.md`
- `artifacts/bilin18_tensor_decomposition/QUERY_MODE_SPARSITY.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_MODE_ROUTER.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_QUADRATIC_ROUTER.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_KNN_ROUTER.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_LEARNED_ROUTER.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_ROUTER_DIMENSION_SWEEP.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_HIERARCHICAL_ROUTER.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_ROUTER_CAUSAL_VALIDATION.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_PATH_DECOMPOSITION.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_PATH_SCALE_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_DOWNSTREAM_FISHER_ORACLE.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_STRATIFIED_FISHER_ORACLE.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_FISHER_CANCELLATION_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_JOINT_FISHER_PROGRAM.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_JOINT_FISHER_SCALE_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_FROZEN_FISHER_QUOTIENT.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_FROZEN_FISHER_CONTROLS.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_KNN_FISHER_METRIC.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_DIRECT_TENSOR_JACOBIAN.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_FULL_TENSOR_JACOBIAN.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_FULL_TENSOR_GRAM_RANK.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_GRAM_RANDOM_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_BEHAVIORAL_BOTTLENECK.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_LOCAL_AXIS_VALIDATION.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_EXACT_PREDICTION_RECURRENCE.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_SEMANTIC_COUNTERFACTUALS.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_SEMANTIC_COUNTERFACTUALS_FIXED.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_SEMANTIC_PATH_DECOMPOSITION.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_SEMANTIC_POSITION_PATHS.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_SEMANTIC_ATTENTION_ATLAS.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_SEMANTIC_ATTENTION_CAUSAL.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_SEMANTIC_ATTENTION_RANK_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_SEMANTIC_ATTENTION_FROZEN_ROUTES.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_SEMANTIC_ATTENTION_FACTOR_CAUSALITY.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_SEMANTIC_ATTENTION_VALUE_BUS_CAUSALITY.md`
- `artifacts/bilin18_tensor_decomposition/LAYER4_CURRENT_VALUE_CODE_RECURRENCE.md`
- `artifacts/bilin18_tensor_decomposition/LAYER4_EMBEDDING_RESIDUAL_CAUSALITY.md`
- `artifacts/bilin18_tensor_decomposition/LAYER4_RESIDUAL_TELESCOPE.md`
- `artifacts/bilin18_tensor_decomposition/LAYER4_RESIDUAL_TELESCOPE_NULL.md`
- `artifacts/bilin18_tensor_decomposition/LAYER3_NATIVE_CHANNEL_CIRCUITS.md`
- `artifacts/bilin18_tensor_decomposition/LAYER3_FROZEN_CHANNEL_TRANSFER.md`
- `artifacts/bilin18_tensor_decomposition/LAYER3_FROZEN_CHANNEL_LOPO.md`
- `artifacts/bilin18_tensor_decomposition/LAYER3_POWERLAW_SPIKES.md`
- `artifacts/bilin18_tensor_decomposition/LAYER3_POWERLAW_SENSITIVITY.md`
- `artifacts/bilin18_tensor_decomposition/LAYER3_FAMILY_INTERACTION_MODES.md`
- `artifacts/bilin18_tensor_decomposition/LAYER3_FAMILY_OUTPUT_RANK_SWEEP.md`
- `artifacts/bilin18_tensor_decomposition/LAYER3_NATIVE_BANK_OUTPUT_SVD.md`
- `artifacts/bilin18_tensor_decomposition/LAYER3_NATIVE_BANK_BASIS_SPECIFICITY.md`
- `artifacts/bilin18_tensor_decomposition/LAYER3_NATIVE_BANK_INPUT_FORMS.md`
- `artifacts/bilin18_tensor_decomposition/LAYER3_INPUT_FORM_CAUSAL_NULLS.md`
- `artifacts/bilin18_tensor_decomposition/LAYER3_INPUT_FORM_K90.md`
- `artifacts/bilin18_tensor_decomposition/LAYER3_INPUT_POLYNOMIAL_SPLIT.md`
- `artifacts/bilin18_tensor_decomposition/LAYER3_QUADRATIC_INTERACTION_SPARSITY.md`
- `artifacts/bilin18_tensor_decomposition/LAYER3_INPUT_MODE_SEMANTICS.md`
- `artifacts/bilin18_tensor_decomposition/LAYER3_UNSEEN_LEXICAL_TRANSFER.md`
- `artifacts/bilin18_tensor_decomposition/LAYER3_SEMANTIC_SUBSPACE_RANK.md`
- `artifacts/bilin18_tensor_decomposition/LAYER3_SHARED_BACKBONE_QUOTIENT.md`
- `artifacts/bilin18_tensor_decomposition/LAYER3_SHARED_QUOTIENT_RANK_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/LAYER3_UNSEEN_CAUSAL_QUOTIENT.md`
- `artifacts/bilin18_tensor_decomposition/LAYER3_UNSEEN_CAUSAL_QUOTIENT_COVERAGE.md`
- `artifacts/bilin18_tensor_decomposition/LAYER3_LEXICAL_REGIME_HIERARCHY.md`
- `artifacts/bilin18_tensor_decomposition/LAYER3_CONDITIONED_SPECTRAL_SPIKES.md`
- `artifacts/bilin18_tensor_decomposition/LAYER3_CONDITIONED_SPIKE_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/LAYER3_CONDITIONED_AXIS_CAUSALITY.md`
- `artifacts/bilin18_tensor_decomposition/DEPTH_CONDITIONED_SEMANTIC_SPIKES.md`
- `artifacts/bilin18_tensor_decomposition/DEPTH_FAMILY_COLLAPSE_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/DEPTH_SEMANTIC_LABEL_NULLS.md`
- `artifacts/bilin18_tensor_decomposition/LAYER17_BOTTLENECK_TAIL_ONTOLOGY.md`
- `artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_CIRCUIT_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_CLEANING_SWEEP.md`
- `artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_CLEANING_NULL_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_TENSOR_CANONICALIZATION.md`
- `artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_FISHER_BASIS.md`
- `artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_FISHER_FINITE_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_CONTINUOUS_FISHER.md`
- `artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_CONTINUOUS_FISHER_BASIN_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_CONTINUOUS_RANK_FRONTIER.md`
- `artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_DUAL_METRIC_UNION.md`
- `artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_CROSS_TERM_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_ROUTED_EXACT_PROGRAM.md`
- `artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_ROUTED_PROGRAM_CONTROLS.md`
- `artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_ROUTED_GRANULARITY_SWEEP.md`
- `artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_ROUTED_SEED_STABILITY.md`
- `artifacts/bilin18_tensor_decomposition/LAYER17_TAIL_ROUTED_LOCAL_FIELD.md`
- `artifacts/bilin18_tensor_decomposition/LAYER17_ROUTED_SEMANTIC_ATLAS.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_CONDITIONAL_SUBSPACE_QUANTIZATION.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_CONDITIONAL_QUANTIZATION_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_MULTIMETRIC_QUANTIZATION.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_DIRECT_PROTOTYPE_CAUSAL_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_PROTOTYPE_OUTPUT_CONSISTENCY.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_RESTRICTED_TENSOR_PROGRAM.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_RESTRICTED_TENSOR_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_NATIVE_CHANNEL_PROGRAM.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_CHANNEL_SHAPLEY_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/LAYER8_CHANNEL_CALIBRATION_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_CHECKPOINT_SPECTRUM_REPLICATION.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_NATURAL_OUTLIER_VALIDATION.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_CLEANING_FOURTH_MOMENT_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_CLEANING_STABILITY_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_LAYER11_BOTTLENECK_TAIL_REPLICATION.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_LAYER11_BEHAVIOR_SUBSPACE.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_LAYER11_BEHAVIOR_OPTIMIZER_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_LAYER11_CONDITIONAL_OUTPUT_ROUTER.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_LAYER11_ROUTER_TOPK_FRONTIER.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_LAYER11_ROUTER_SEED_STABILITY.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_LAYER11_ROUTER_FUNCTIONAL_EQUIVALENCE.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_BOUNDARY_CARRIER_DEPTH_AUDIT.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_BOUNDARY_CARRIER_METRIC_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_RESIDUAL_COMPONENT_DEPTH_AUDIT.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_ATTENTION_QKOV_DEPTH_AUDIT.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_ATTENTION_SOURCE_DISTANCE_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_MULTILAYER_ATTENTION_ROUTING_AUDIT.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_ATTENTION_LAYER_SUBSET_SEARCH.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_UNIVERSAL_ATTENTION_BACKBONE_AUDIT.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_ATTENTION_RELAY_TOPOLOGY_AUDIT.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_ATTENTION_DESTINATION_METRIC_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_FINAL_DESTINATION_HEAD_PROGRAM.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_FINAL_DESTINATION_SUBSPACE_AUDIT.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_FINAL_DESTINATION_SOURCE_PROGRAM.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_SOURCE_CANCELLATION_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_SOURCE_FISHER_GRAM_PROGRAM.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_SOURCE_ROUTING_VALUE_AUDIT.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_SEMANTIC_SOURCE_PROGRAM_TRANSFER.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_SEMANTIC_SOURCE_INDIVIDUAL_REDTEAM.md`
- `artifacts/bilin18_tensor_decomposition/BILIN12_NEWLINE_SOURCE_SCALE_REDTEAM.md`
