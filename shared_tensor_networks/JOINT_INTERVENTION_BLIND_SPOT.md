# An exact blind spot for single-feature causal validation

This CPU-only constructed negative control extends the depth-three ambiguity work toward mechanism-discrimination experiments. It is not a learned-model discovery result or a novel general theorem.

## Matched circuits

For input x in R^4 define a=x1*x2 and b=x3*x4. Four physical first-layer features are h=(a,a,b,b). Two second-layer bilinear gates followed by a linear readout implement

F_k(h) = ((h1+h2)/2)*((h3+h4)/2) + k*(h1-h2)*(h3-h4).

Every k gives the identical ordinary function a*b, identical first-layer activations, identical top-gate activations, equal architecture width and equal parameter-slot count. Only the readout coefficient of the second top gate differs. The second gate is zero on all natural inputs. White-box access to the weights exposes k; the indistinguishability claim concerns the specified observations/interventions, not an impossibility for an analyst with unrestricted weight access.

If an arbitrary value replaces just one of h1,h2,h3,h4, at least one of the two differences remains zero. Thus **all single-coordinate replacement interventions**, on every input and at every replacement magnitude, give identical outputs for all k. This includes deletion, rescaling, noise and donor activation patches when each acts on only one of these physical features. More data of this type cannot identify k. This is a statement about a specified layer/interface, not every possible intervention elsewhere in the network; directly setting the dormant top gate can distinguish k.

Deleting h1 and h3 together instead yields

F_k(0,a,0,b) = a*b/4 + k*a*b.

For a*b!=0, a single exact joint measurement identifies k within this candidate family. The mixed derivative d²F/(dh1 dh3)=1/4+k also distinguishes it; first derivatives evaluated on the natural hidden manifold do not. More generally, ambiguity is not about the amount of observational data but its intervention order and interface.

## Interpreting the construction carefully

This is a dormant conditional pathway at the top-gate level. Expanding its product yields four cancelling terms, but calling it active cancellation throughout the network would be misleading: the factored gate itself is exactly zero on natural inputs and after any single-feature intervention. Mechanism descriptions depend on the declared computational granularity. This construction is deliberately a stress test, not evidence that our trained networks contain this pathway.

Within this family, knowing k predicts all joint interventions exactly. Outside this family, fitting this one coefficient does not certify a complete mechanism. A method must report its hypothesis class and abstain when its evidence cannot distinguish members. The central benchmark outcome is held-out intervention prediction, not merely choosing a mechanism name.

## Bounded exact assay

check_joint_intervention_blind_spot.py uses Fraction arithmetic to:

1. Verify the factored network against an independently expanded quadratic under all16 deletion masks and separately specified replacement cases.
2. Verify ordinary outputs and every single-coordinate replacement on the finite fixtures are independent of k; the algebra above supplies the all-input statement.
3. Enumerate eight candidate readout coefficients. Single-coordinate probes leave all eight viable. One cross-pair deletion recovers the coefficient exactly on a nonzero fixture.
4. Predict unseen two-coordinate donor replacements on disjoint inputs after that measurement.
5. Compute the exact random two-coordinate-deletion discovery probability: four of the six pairs cross the two duplicate groups, while two do not. A predictor using the candidate mean is scored before and after the informative observation; this is a noiseless, known-family positive control, not a practical learned interpreter.

The candidate family and replacement fixtures are fixed in source. The method sees intervention outputs and candidate predictions, not a target coefficient. Teacher coefficients are used only to generate observations and score predictions. All failures/ambiguities are retained. The zero a*b case is included as a deliberately uninformative acquisition control.

Next, any broad mechanism benchmark should retain this assay as an adversarial negative control, introduce noisy measurements and richer unknown candidate circuits, and hold out entire intervention combinations. Passing this simple assay is necessary implementation evidence, not sufficient evidence of useful real-model interpretation. Native-model research remains paused; the existing h128 training protocol is unchanged.
