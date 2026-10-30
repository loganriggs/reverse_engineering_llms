# Which internal intervention resolves the depth-three ambiguity?

This extends DEPTH3_INTERNAL_MIXING.md for its specific fixed-first-layer, width-two, three-task family. A single specified ablation can identify the feature axes, but only if its measured output tasks are informative. The statement does not apply automatically to unknown graphs or larger widths.

## Setup

Let z1=x1*x2,z2=x3*x4, u=z1²-z2² and v=2z1*z2. The original three outputs are (u²,uv,v²). Alternative second-layer features are two real bilinear functions of z, with arbitrary linear readouts over their three products. They must reproduce all three original output functions exactly. The additional experiment zeros the first second-layer feature, without altering the readouts or the second feature. In the generating network the ablated output is (0,0,v²).

The ablation refers to a specified physical coordinate, not a basis-covariant operation. Its observations are additional supervision; they cannot be reconstructed from the unablated functions alone.

## Why every legal alternative lies in the same feature plane

The three original task polynomials are independent, so any matching two-feature competitor has the same three-dimensional product span S=span(u²,uv,v²). Each alternative feature g=a*z1²+b*z1*z2+c*z2² must satisfy g² in S. Comparing coefficients yields

    a²=c²,     b(a+c)=0.

If c=-a, then g is in span(u,v). Otherwise c=a and b=0, so g=a(z1²+z2²). For a!=0 that is a definite binary quadratic and cannot be a product of two real linear forms. Therefore every nonzero legal alternative belongs to span(u,v). The two features must be independent to span all three tasks.

Write the alternative features as g1=a*u+b*v and g2=c*u+d*v, with Delta=ad-bc!=0. The unique corresponding output readouts implement u=(d*g1-b*g2)/Delta and v=(-c*g1+a*g2)/Delta.

## Two ablated output measurements identify the axes

After setting g1=0, the mixed and second squared outputs are

    output_uv = -ab*g2²/Delta²,
    output_v² = a²*g2²/Delta².

Matching the original ablation requires output_uv=0 and output_v²=v². The second equation forces a!=0, so the first forces b=0. Then Delta=ad and g2²/d²=v². Comparing coefficients of u² forces c=0. Thus g1=a*u and g2=d*v: both original axes are recovered up to scaling. The named intervention also fixes their ordering. Observing u² and v² after ablation also identifies the axes exactly, but has weaker first-order information, as shown below. Factor swaps and reciprocal factor scales remain unobservable.

This is exact global identification within the stated competitor family, conditioned on exact unablated functions and these two exact ablated functions. It supplies neither a finite-data/noise bound nor a training-convergence guarantee.

## Exact local rank certificates

There are17 factor/readout parameters and four independent scaling freedoms. The unablated coefficient Jacobian has rank11. Appending coefficients of the observed ablated tasks gives:

| Additional measured tasks after zeroing feature1 | Jacobian rank | First-order ambiguities beyond four scales |
|---|---:|---:|
| None | 11 | 2 |
| u² only | 11 | 2 |
| uv only | 12 | 1 |
| v² only | 12 | 1 |
| u² and uv | 12 | 1 |
| u² and v² | 12 | 1 |
| uv and v² | 13 | 0 |
| All three | 13 | 0 |

`certify_depth3_interventions.py` certifies each rank over the rationals and saves a nonzero maximal minor. Independent integer forward evaluations verify every Jacobian column by exact central differences at steps one and two. The four scale symmetries persist under the ablation and give a local upper bound13; the informative pairs attain it. All eight measurement choices pass. Rank-deficient entries describe first-order information, not every possible higher-order identification constraint.

## A readout can identify through a weak, higher-order signal

Consider the previously ambiguous shear g1=u+t*v,g2=v with the compensating unablated readout. Its ablated task discrepancies are

    (t²*v², -t*v², 0).

For the u² output alone, the squared discrepancy is proportional to t⁴, so its Jacobian adds no rank at t=0. Nevertheless exact equality forces t=0 within this shear family: higher-order information exists despite zero first-order information. The other mixing direction remains unconstrained, so this single task does not identify the full network.

For the mixed output, the squared discrepancy is proportional to t² and gives ordinary first-order information. Normalize both half-squared losses by E[v⁴]>0. In the one-dimensional constrained family these losses and Euclidean gradient flows are

    L_square(t)=t⁴/2,     dt/dtau=-2t³,
    t(tau)=t0 / sqrt(1+4*t0²*tau);

    L_mixed(t)=t²/2,      dt/dtau=-t,
    t(tau)=t0*exp(-tau).

The first has algebraic decay and the second exponential decay. These are exact scalar flows along the specified compensated shear family, not the full17-parameter training dynamics: unconstrained joint training can leave this family and uses a different induced metric. This shows why choosing an intervention readout for local information matters, even when exact zero-error constraints seem sufficient in one direction. No network RLCT is inferred from this example.

## Consequence for the next deeper test

The observation-only and observation-plus-ablation conditions can now be compared with explicit recovery targets. Use the same architecture, teachers and initializations; measure unablated fit, ablated fit, feature-plane agreement and ordered component agreement separately. The ablation condition intentionally supplies additional information, so success would not establish that a better optimizer recovered information already present in ordinary labels. It would test whether the additional causal supervision is usable. Wider networks require a separate identification analysis.

The current d128/h64 run remains the main scaling experiment. This exact small-family result prepares a defensible depth-three supervision experiment rather than another unrestricted optimizer search.
