# Repeated inputs, multiplication kernels, and top-layer width

These are direct algebraic derivations useful for this project; no claim of priority or novelty in the mathematics literature is made. The numerical checks are in `multiplication_geometry.py` and `results/multiplication_geometry/results.json`.

## A precise width characterization for a fixed hidden representation

Fix real-valued hidden features h(x) in R^m and consider an unrestricted bilinear output layer with q product units:

`p(x) = Σᵣ (cᵣᵀh(x))(dᵣᵀh(x)) = h(x)ᵀ Q h(x)`.

Scalar output weights can be absorbed into either factor. Only the symmetric part of Q matters. For a **specified symmetric matrix Q as a quadratic form on unrestricted hidden coordinates**, the smallest q is

`q_min(Q) = max(n₊(Q), n₋(Q))`,

where n₊ and n₋ count strictly positive and strictly negative eigenvalues.

Proof of lower bound: each symmetrized outer product `(cdᵀ+dcᵀ)/2` has at most one positive and at most one negative eigenvalue. Positive and negative inertia counts are subadditive under sums, hence each count of Q is at most q. One way to see subadditivity is to intersect the nonpositive subspaces of the summands: the intersection has codimension at most the sum of their positive inertia counts, and the sum is nonpositive there. The analogous argument applies to negative counts.

Proof of upper bound: diagonalize Q. Pair one positive square u² with one negative square −v² and write `u²−v²=(u+v)(u−v)`. Each leftover positive or negative square needs one product. This constructs exactly max(n₊,n₋) products. The zero form needs zero units. Numerical tests reconstruct 80 random symmetric forms of dimensions 1–8, with maximum relative error 2.33e-15.

This distinguishes rank from width even before input constraints: rank-r positive semidefinite Q needs r products; a balanced indefinite rank-r Q needs approximately r/2. This statement assumes signed unrestricted real factors and a fixed Q; positivity constraints or restrictions on factors change it.

## Repeated inputs introduce another equivalence

Define the linear multiplication map `μ_h(Q)(x)=h(x)ᵀQh(x)`. Actual hidden states need not fill R^m. If their components obey polynomial identities, μ_h has a nonzero kernel. The same output function has a family of symmetric lifts `Q + ker μ_h`.

The functional minimum top width, for this fixed h and unrestricted real top factors, is therefore

`q_min(p | h) = min_{Q : μ_h(Q)=p} max(n₊(Q), n₋(Q))`.

This is a characterization, not an efficient optimization algorithm. Optimizing over the fiber can be difficult. It is also not a claim about the minimum hidden width, end-to-end learned architectures, ordinary CP rank, or training convergence.

For a complete degree-two monomial hidden dictionary in d inputs, m=d(d+1)/2. Every quartic monomial is the product of two quadratic monomials, so μ_h is surjective onto the homogeneous quartics, of dimension binomial(d+3,4). Thus

`dim ker μ_h = m(m+1)/2 − binomial(d+3,4)`.

Verified exactly by integer-valued multiplication matrices for d=2,…,6: kernel dimensions 1, 6, 20, 50, 105. This is caused by repeated input variables and polynomial multiplication, not parameter tying. A tree of tensor occurrences does not remove these identities after the input leaves are identified.

## A concrete same-function, different-width lift

Let `h=(x²,xy,y²)` and `p=x⁴+y⁴`. The diagonal lift `Q₀=diag(1,0,1)` has inertia (2,0), requiring two products if that matrix itself must be represented.

But the hidden coordinates satisfy `h₁h₃−h₂²=0`. Adding twice this zero polynomial gives the alternative lift

```
Q₁ = [1  0  1]
     [0 -2  0]
     [1  0  1]
```

Its inertia is (1,1), and the identical function needs just one product:

`x⁴+y⁴ = (x²+√2xy+y²)(x²−√2xy+y²)`.

Both matrices have rank two. Their fixed-lift product requirements differ, but the polynomial's true minimum for this h is one. The cancellation in the expanded alternative lift is an exact polynomial identity, not evidence of removable native neuron contributions. The script checks the two forms and the one-product expression on 1,000 inputs.

This example does not establish two intrinsically different minimal complexities for the same function. It shows why judging a function by one chosen lift can overestimate its minimum and misidentify its internal computation. The identical output tensor also cannot reveal which factorization the teacher executed.

## Consequences for the empirical designs

The oracle hidden-product map has functional dimension bounded by both its number of products and the ambient quartic dimension. For the frozen E1 teacher dictionaries, numerical coefficient-map ranks at relative threshold 1e-10 are:

| Setting | Tied hidden products | Untied left-right products | Full quartic space |
|---|---:|---:|---:|
| d=4, h=6 | 21 of 21 | 35 of 36 | 35 |
| d=8, h=8 | 36 of 36 | 64 of 64 | 330 |

All five teachers of each type have these ranks. These are observed ranks of the frozen dictionaries, not a general-rank theorem for arbitrary rank-two quadratic features.

At d=4, the untied oracle hidden dictionary spans the entire quartic space. Successful fitting through that dictionary cannot establish recovery of the teacher's special features: generic alternatives may suffice. At d=8, the oracle hidden spaces are much smaller than the full quartic space, making the gap between oracle-feature fitting and learned features a more informative target.

For E3, graph recovery must specify what is quotiented out: permutations/scalings are insufficient because polynomial identities and execution reuse preserve the same function too. A useful next test is whether learned hidden spans predict targets with a newly fitted linear readout, compared with both oracle and random hidden spans. That tests learning a useful function subspace while avoiding an unjustified claim to have recovered the unique teacher graph.
