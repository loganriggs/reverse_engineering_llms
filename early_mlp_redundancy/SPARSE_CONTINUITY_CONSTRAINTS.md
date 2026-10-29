# When can sparse selection preserve amplitudes and continuity?

M113 gives a narrow impossibility result, explicit counterexamples to overbroad versions of it, and measurements of its assumptions in our learned encoders. It does not claim that all continuous sparse featurizers must lose useful amplitudes.

## A precise constraint for the current hard-block architecture

Let a=(a_1,...,a_G) range over all signed block coordinates. Let q(a) select the k largest block norms, with0<k<G. Decode the unmodified selected coordinates as

`f(a) = b + sum_{g in q(a)} D_g a_g`.

**Claim:** if this map is continuous everywhere in the full code space, every D_g must be zero. The only globally continuous decoded map of this exact form is the constant bias. This statement is about fixed linear block decoders and unchanged hard-selected coordinates; it does not cover shrinkage, nonlinear decoders, restricted coordinate domains, or alternative selection rules.

**Proof.** Choose any two groups g,h, keep k−1other groups at strictly larger norms, and all remaining groups at strictly smaller norms. At the final selected slot, let g and h tie with vectors u,v of the same positive norm. Approaching the tie from either side gives the same common contribution, plus D_g u on one side and D_h v on the other. Continuity requires D_g u=D_h v. The full signed domain also permits replacing u by−u while holding v and the other blocks fixed, so−D_g u=D_h v. Therefore D_g u=D_h v=0. Since u and v can have arbitrary directions, both decoders are zero. Repeat for every pair. The tie-breaking rule at the boundary cannot repair unequal one-sided limits.

For an affine encoder a=E x+c with full row rank, E is onto the full code space and has a continuous right inverse. A discontinuity in f therefore gives a discontinuity in f(E x+c) on the unrestricted input space. Standardization by a positive scalar does not change this argument. A rank-deficient encoder or a restricted input manifold may avoid the relevant code configurations.

## Why shared directions are not enough

Two scalar groups with the same decoder direction still jump on the signed path a=(1+t,−1+t): hard Top1 switches the decoded output between opposite signs. The toy verifies a nonzero limiting gap of2.

On the restricted nonnegative domain, however, the same decoder with Top1 produces max(a_1,a_2), which is continuous and preserves the selected coefficient. This is a counterexample to any claim that amplitude-preserving sparse selection is inherently discontinuous without specifying the coordinate domain. It also illustrates why shared output directions and coordinated coefficients must be considered together.

For overlapping groups {0,1} and {0,2}, atom0 is shared and decoded once. Along the toy path a=(2,1+t,1−t), the shared contribution remains exactly2, but switching the exclusive identity-decoded atoms leaves a limiting jump ofsqrt(2). Shared membership protects the shared contribution; it does not automatically match the entering and leaving contributions.

These toys are function-level examples, not evidence that native MLP channels are redundant. Nor does equality at one boundary establish global interchangeability of two components.

## Do the trained encoders satisfy the rank assumption?

All four M112 encoders have numerical row rank128, confirmed with independent NumPy and Torch singular-value calculations. The smallest singular values are not close to the fixed relative1e-8rank tolerance:

| Seed | Trained map | Minimum encoder singular value | Maximum | Rank |
|---|---|---:|---:|---:|
| 71 | hard | 0.849 | 1.912 | 128 |
| 71 | continuous | 0.868 | 1.918 | 128 |
| 72 | hard | 0.886 | 1.956 | 128 |
| 72 | continuous | 0.892 | 1.960 | 128 |

Constructed code-space ties were mapped back through explicit encoder right inverses. All requested coordinates reconstruct within1e-10. Hard-decoder gaps converge to their predicted finite limits (about1.38–1.45in standardized decoder units), while the continuous-map gaps vanish. For continuous-trained states, the hard-gate result is a hypothetical hard decoder on those weights, not a discontinuity of their actual continuous map.

The witness inputs live in unrestricted standardized-write space. We have not shown that they are outputs of the native MLP or occur on real text. Full encoder rank alone cannot settle that question.

## Does native MLP computation locally restrict those coordinates?

We evaluated the Jacobian of raw encoder coordinates composed with native MLP1 at16fixed, evenly spaced natural-training states, for each of the four encoders. We restrict input perturbations to the constant-RMS sphere tangent, removing the radial direction introduced by normalization.

Writing native product computation as F(z)=D[(Lz)*(Rz)] and the featurizer encoder as E, the measured derivative is

`T(z) = (E / scale) J_F(z) [I - z z^T / ||z||^2]`.

All64matrices have numerical row rank128. Their smallest singular values range1.160–2.479. Independent autograd JVPs through the native polynomial and normalization agree to relative error below1.18e-15; NumPy SVD checks agree on the first and last state for each encoder. The radial-null checks pass. The native checkpoint hash matches the training receipt.

Thus, at these states, the normalized native MLP can locally vary all128raw encoder coordinates; a simple local rank deficit does not explain away the code-space issue. This is a differential result. It does not locate a reachable nearby gate boundary, bound the distance to one, or establish that arbitrary tangent directions are generated by text. A low-dimensional distribution of naturally occurring states can coexist with full differential rank off that distribution.

## What the theory changes in the research direction

We should not expect raw hard TopK plus a freely varying signed linear code and nontrivial linear decoder to become globally continuous merely by aligning decoder directions. A successful amplitude-preserving alternative needs to change an assumption: coordinate-domain restrictions, coordinated duplicate contributions, a different selection/decoding rule, or continuity only on the relevant data manifold.

Continuity of the reconstructed write is also stronger than equality of downstream effects. Two distinct decoded limits may be approximately equivalent for a particular downstream reader and context. That must be tested as causal equivalence, with declared output-fidelity criteria; it is not literal write equality or native channel redundancy.

The M112 results do not make shrinkage mandatory, and this theorem does not justify forcing nonnegative features without testing whether that prior fits the model. It narrows the design question and identifies which assumptions need empirical support. No new training or architecture sweep was performed.

## Next bounded discriminator

The remaining gap is between an affine path in write space and a path through the native bilinear computation. Use fixed pairs of actual normalized MLP inputs, interpolate on the input sphere, recompute native writes, and compare native versus extracted behavior at analytically located selection crossings. This avoids assuming that straight write-space chords are attainable by the native MLP, while still acknowledging that intermediate states need not correspond to text.

There is an exact tractable route: along normalized linear input interpolation, each raw encoder coordinate is a quadratic numerator divided by a common quadratic normalization factor. Equality of two squared block norms reduces to a quartic equation. Verify this algebra in toys and against direct native evaluations, then test a fixed set of paths without searching for high-impact crossings. Stop after one path study, rather than reopening the closed shrinkage-training sweep.

Artifacts: check_sparse_continuity_constraints.py, measure_native_code_tangent.py, validation/sparse_continuity_constraints.json, native_code_tangent_protocol.json and native_code_tangent.json. All checks are terminal; no native language-model forward intervention or new fit occurred.
