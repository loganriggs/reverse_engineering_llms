# An exact function-level starting point for first-layer attention

For a first-layer, **unnormalized causal bilinear-attention** module on
discrete token embeddings, a data-free canonicalization can begin with its
complete observable token interaction tensor. Let $e_a$ be the embedding
of query token $a$, and let head $h$ have two QK forms $A_h,B_h$ and
an OV map $C_h$. Include any fixed attention-score scaling in these
matrices. Define

\[
T_{abo}=\sum_h(e_a^\top A_h e_b)(e_a^\top B_h e_b)
                     (e_b^\top C_h)_o.
\]

The attention contribution at causal position $i$ is exactly
$y_i=\sum_{j\le i}T_{t_i,t_j,:}$. This tensor depends on the **summed
function**, not on how it is split between heads or factored into two QK
branches. It is an exact first-stage object for the weight-only route;
choosing a distribution or a legibility prior comes afterward. RoPE and
position-dependent terms require a tensor indexed by query/key positions,
or a fixed-position version of this argument. Layer norm, residuals, and
later layers are outside this simple formula.

## What is identifiable

If one can separately observe the attention contribution on all one- and
two-token sequences, $T$ is uniquely determined. A one-token sequence
$[a]$ gives $T_{aa,:}$. The last-position output of $[b,a]$ gives
$T_{ab,:}+T_{aa,:}$, so subtraction gives every $T_{ab,:}$. This is
an algebraic identification statement, **not** a claim that ordinary
language data contains every token pair or that the complete model exposes
this internal contribution at its final logits. The weights give $T$
directly without gathering such a dataset.

After flattening $T$ into a query-by-(key,output) matrix, its rank is
the smallest number of separable query factors in that **specified
partition**. SVD gives the best rank-$r$ approximation under the uniform
Euclidean metric and canonical singular values. Singular vectors are fixed
only up to signs and degeneracies; SVD does not name semantic variables.
For this two-QK formula, each query occurs quadratically, so the query
factor space lies in the symmetric-square embedding space of dimension at
most $d(d+1)/2$, where $d$ is the embedding width. Independent heads
do not multiply that *global* bound. A tighter bound can depend on the
actual query-projection widths and shared subspaces. For the 546M model,
the embedding width makes the global bound too large to promise useful
compression; rank has to be measured, preferably by matrix-free methods.

The exact tensor cannot identify physical head allocation. Scaling one
QK branch by $c\ne0$ and the other by $1/c$ leaves $T$ unchanged.
Splitting one head's OV map between two heads with identical QK forms also
leaves $T$ unchanged. Weight inspection and internal interventions can
distinguish those implementations; external function fitting cannot. This
agrees with the separate [shared-computation identifiability red
team](SHARED_COMPUTATION_IDENTIFIABILITY.md).

## CPU check

`exact_attention_observable.py` constructs two random heads with a
31-token vocabulary, width-four embeddings, and three output coordinates.
For seeds 0–4, the exact query-vs-(key,output) rank is **10**, equal to
the symmetric-square bound and well below vocabulary size 31. Direct
causal sequence computation, table lookup, and reconstruction from all
length-one/two experiments agree to floating-point precision. The rank-9
Frobenius error equals the tenth singular value, as predicted by SVD.
Changing QK branch scale or splitting a head gives the same complete
tensor to within about $10^{-15}$. A deliberately changed token-pair
cell has exactly zero error on all other cells and nonzero withheld error:
a dataset measure with zero support there cannot determine its value.
Run `exact_attention_observable.py --seed 0` through `--seed 4` on CPU;
the five JSON files under `full_width_results/exact_attention_observable_s*.json`
retain the numerical checks.

At rank three, we also compute the **exact best weighted SVD cut** for
three query laws: uniform; a common-heavy law putting 95% mass on five of
31 query IDs; and their half-mixture. Key/output cells remain uniformly
weighted conditional on the query, so this is a controlled measure test,
not a natural-language co-occurrence model. Mean relative squared errors
over the five seeds are:

| Rank-three fit law | Uniform test | Common-heavy test | Rare-query test |
|---|---:|---:|---:|
| Uniform | **.0975** | .1472 | **.0983** |
| Half-mixture | .1121 | .0706 | .1257 |
| Common-heavy | .1691 | **.0373** | .1973 |

The fit law changes which errors a fixed rank budget preserves. This
simple example reproduces the *form* of the fidelity tradeoff, not the
546M model's actual values or a semantic component. It also shows why
“data-free” must be separated from “uniform-metric”: $T$ comes exactly
from weights, while a rank-constrained approximation still needs a metric.

## Consequence for the next experiment

There are two separate decisions. First choose **which function and
metric to preserve**: an exact weight-derived $T$, a corpus-weighted
approximation, or an intervention-weighted approximation. Then choose
**how to describe the retained function**: sparse token responses,
output effects, causal signatures, or reusable groups. No sparse rotation
can repair an omitted pair cell or certify a physical head. The decisive
comparison is at matched function error and intervention dose, using
held-out whole behaviors. The existing 546M rare-key comparison shows why
the metric choice matters; this construction shows precisely what the
data-free reference can and cannot determine.
