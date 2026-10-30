# A depth-three obstruction to component recovery

The shallow component-identification theorem does not automatically extend layer by layer. Here is an exact counterexample with two hidden features per layer and a complete product-task span. It preserves every output function while changing internal feature directions and their ablation effects. This is an elementary algebraic construction; no literature novelty claim is made.

## Two networks with identical outputs

Fix the first bilinear layer on four inputs:

    z1 = x1*x2,    z2 = x3*x4.

Its four input factors are independent. Let the second bilinear layer be

    u = (z1-z2)(z1+z2) = z1²-z2²,
    v = (2z1)z2.

The final three tasks are u², uv, v². These form the complete three-dimensional unordered-product span, and each is a degree-eight polynomial in x. The product head can be implemented by bilinear gates followed by a linear readout, giving three bilinear stages under our depth convention.

For any real t, define

    g = u+t*v = z1²+2t*z1*z2-z2².

This remains a valid real bilinear feature:

    g = (z1+(t+sqrt(t²+1))*z2)(z1+(t-sqrt(t²+1))*z2).

Replace (u,v) with (g,v), and change the task readouts to

    u² = g² - 2t*g*v + t²*v²,
    uv = g*v - t*v²,
    v² = v².

For t!=0 the first feature is not a scaled or permuted original feature. Nevertheless, the identities hold for every x, not merely a finite dataset or one input distribution. Any number of additional task heads in the same product span remains indistinguishable by the corresponding readout transformation. No optimizer, sample count or output measurement precision resolves this ambiguity.

Rational checks use r=1,2,3,1/2, set t=(r-1/r)/2, and factor g=(z1+r*z2)(z1-z2/r). This avoids approximate irrational coefficients. The t=0 case is a positive control; all three nonzero cases retain exact task equality.

## Why the shallow theorem does not apply

The depth-two theorem required the 2h factors of h quadratic features to be independent in their input space. At this second layer, four factors live in a two-dimensional space. They cannot be independent. Every nondegenerate indefinite binary quadratic is a product of two independent real linear forms; indefiniteness is an open condition. Consequently, sufficiently small invertible mixtures of (u,v) remain valid bilinear features. A freely transformed product readout absorbs these changes.

This is a full-function internal basis ambiguity. It is different from the earlier sampled-value GL(h) ambiguity, which might not extend to legal features away from observed inputs. Here all mixtures in a neighborhood remain legal globally. It also differs from physical caching-versus-recomputation ambiguity: the hidden functions themselves change.

## Exactly two additional local ambiguities beyond scaling

With the first layer fixed, the two second-layer products use eight factor parameters. Three product-task readouts use nine parameters, for17 total. Expanding the three outputs in binary quartic monomials gives15 coefficient outputs.

The coefficient Jacobian at the displayed network has exact rank11. The script saves a nonzero11-by11 minor with determinant -4096; rational elimination confirms the full rank. Independently evaluated-output central differences at steps one and two reproduce every derivative exactly, since the output has degree at most two in each individual parameter.

There is also a local upper bound11: the two hidden quadratics span a two-dimensional subspace U of the three-dimensional binary-quadratic space. Choosing U costs dim Gr(2,3)=2 dimensions. Each of the three output tasks belongs to the three-dimensional space Sym²(U), costing nine further dimensions. Thus the output family has at most11 local dimensions. The nonzero minor attains the bound, so the constant-rank theorem gives a six-dimensional local fixed-function fiber.

Four of these dimensions are familiar scales: reciprocal scaling of the two factors within each feature, and feature scaling compensated by readouts. Four explicit independent tangent vectors are verified in the kernel. Two further dimensions remain, corresponding to mixing hidden feature directions beyond diagonal scaling. Discrete factor swaps and permutations do not change the local dimension count.

This is a specified width-two counterexample. It does not prove that every wider/deeper network has this same ambiguity or that all layers are unidentifiable. In particular, the h64 width-scaling panel remains a depth-two experiment under different assumptions.

## Identical predictions, different interventions

Zero the first second-layer feature while keeping the second fixed. In the original network, the task outputs become

    (0, 0, v²).

In the alternative network, applying the corresponding coordinate deletion produces

    (t²*v², -t*v², v²).

The ablations differ for t!=0, despite identical unablated functions. Output-only supervision cannot choose which coordinate deletion describes the generating network's mechanism. An internal measurement or intervention tied to a specified physical feature can distinguish them; an arbitrary basis-dependent ablation preference is an additional prior, not information already supplied by the output tasks. An intervention transported through the entire basis change would be a different operation from deleting one coordinate.

## Implication for deeper experiments

When extending to depth3, report hidden-span agreement separately from component agreement and intervention agreement. Penalizing a trained model for not recovering a unique set of axes is inappropriate in a family where the axes are not identifiable. More output tasks cannot fix this particular ambiguity once the complete product span is already observed.

A useful deeper recovery experiment must either choose a family with justified component identifiability, use an equivalence class that includes legal hidden mixtures, or add clearly labeled internal supervision/interventions. The existing deep transfer failures are not retroactively explained by this example: it supplies a warning about the recovery target, not evidence that these exact mixtures caused those failures. The current h64 training/evaluation remains the priority; this result refines the subsequent depth3 protocol.

Artifacts: `check_depth3_mixing.py` and `results/depth3_mixing/checks.json`. Four exact function/intervention examples, rank/minor certificate and explicit scale-gauge checks pass.
