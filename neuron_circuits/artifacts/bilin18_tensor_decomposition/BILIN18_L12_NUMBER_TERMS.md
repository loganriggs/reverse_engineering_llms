## H890 — L12 number–context interaction carries most of the centered flip effect, but finite curvature matters

Decomposed L12's bilinear output exactly along each frozen number direction and intervened on the resulting output changes separately. With the coordinate centered at the discovery singular/plural midpoint, the number–context cross term alone reproduces most of the full flip's mean agreement-margin reduction. However, a separate native-displacement expansion has a large quadratic remainder. Attribution depends on the coordinate origin and on whether one decomposes the function or the finite intervention. The result supports a number–context interaction, not a uniquely linear semantic unit.

### Exact decompositions and causal intervention

Let b be the unit number axis, u the normalized MLP input, and a the coordinate origin. Set t=u·b−a and w=u−t*b. For F(u)=Down[(Left u)*(Right u)]+bias:

F(u)=F(w)+t*C(w)+t²*Q,

where C(w)=Down[(Left w)*(Right b)+(Left b)*(Right w)] and Q=Down[(Left b)*(Right b)]. Changing t to t' while fixing w gives the exact output change (t'−t)*C(w)+(t'²−t²)*Q.

Test both a=0 (w orthogonal to b) and a equal to the discovery number-midpoint expressed in absolute coordinates (w has a fixed nonzero b-coordinate). The full coordinate intervention is identical under both origins. Centered w must not be described as orthogonal context.

Also decompose the same flip around each native input: ΔF=δ*J(u)b+δ²*Q, with δ=t'−t. This local-linear/local-quadratic split is invariant to the arbitrary coordinate origin. It is different from the coordinate-cross/coordinate-square split.

For each frozen axis, apply full change, coordinate-cross change only, coordinate-square change only, local-linear change only, local-quadratic change only, or identity, at L12. All later computation stays live. Use the same120 eligible task/copy endpoints as H889. This is posthoc mechanistic decomposition of the newly localized L12 effect, not new lexical confirmation. Output-component interventions need not correspond to a realizable single input-coordinate change.

### Agreement margin reductions

|Cohort|Axis|Full flip|Centered cross term|Centered square term|Native local-linear term|Native local-quadratic term|
|---|---|---:|---:|---:|---:|---:|
|irregular|A|2.039|2.012|0.127|1.141|1.622|
|irregular|B|1.400|1.419|0.024|0.619|0.957|
|new_regular|A|1.883|1.751|0.191|1.207|1.888|
|new_regular|B|1.126|0.994|0.162|0.715|0.831|

The centered cross term gives0.994–2.012 margin reduction compared with full1.126–2.040. The centered square term is smaller,0.024–0.191. These are separate causal output interventions; their final-margin effects need not sum because downstream computation is nonlinear. At the MLP output vector, the decomposition itself is exact.

The origin control matters. Under the absolute origin, cross effects range1.033–1.705 and square effects−0.011–0.516. Full-flip effects remain unchanged. Moving the coordinate origin reallocates contributions between cross and square terms without changing the computation, so neither term alone is a coordinate-independent ontology.

The native-displacement quadratic term has large separate effects,0.831–1.888. Therefore the large direct coordinate flip is not faithfully described by a first-order approximation alone. This does not contradict H879's accurate centered response approximation for much smaller L8H3 headwrites: the interventions, sizes and response summaries differ.

### Endpoint polarity exposes cancellation

Pooling regular and irregular task examples, the centered cross term reduces the correct-number margin for both singular and plural endpoints: axis A2.523/1.223, axis B1.060/1.324. In contrast, the native local-linear term increases the singular margin (drops−1.604/−1.033) while reducing the plural margin (drops3.957/2.373). The local-quadratic term largely does the reverse: singular drops3.916/2.185 and plural drops−0.388/−0.405. These complementary effects are hidden by a single mean. The full flip reduces both endpoint margins.

This makes the midpoint-centered interaction a useful description of number polarity in these examples, while preserving the warning that attribution is origin-dependent and downstream interactions prevent unique mediation percentages. Both axes, both cohorts and copying controls remain in the saved analysis.

### Interpretation and next step

The candidate computation is a scalar number-related coordinate modulating a context-dependent output vector C(w), with additional quadratic structure. The exact algebra follows from a bilinear layer; the empirical contribution is that the frozen, semantically validated direction's cross-term intervention transfers most of the measured native agreement-margin effect. It does not establish that C(w) itself is low-dimensional, stable, semantic, or shared across arbitrary tasks.

Next assess whether the cross-term output vector is reusable across nouns and templates, using discovery-only estimates and explicit native-effect fidelity on transfer examples. Compare a mean interaction vector, a small discovery-derived subspace and recipient-specific vectors, with polarity-separated evaluations. That tests whether this is a compact computational operation rather than a scalar gate on a large uncontrolled context-dependent vector. The copying lexical-coverage learning curve remains open; the agreement decomposition does not resolve the general featurizer problem.

### Verification

3,000 full-model forwards. Native and identity logits replay exactly. Bilinear coordinate recomposition relative error 1.99e-15; equality of coordinate and local-displacement flip decompositions 1.87e-15; double formula versus native float module relative error 8.24e-07. H889 full-flip margin replay error 0. Origin-invariant full/local controls replay within 0. All hashes unchanged and outputs/KL finite. No reserved test evaluated; natural-document pool unchanged at26,112. Broad research goal active.
