## H787 — Sign pairing separates substantial even drift and exposes extrapolation beyond native edits

Previous turn was progress: H786 showed the L1/L4 nonlinear-error ordering in random directions, weakening task-specific interpretations. H787 tests opposite tangent rotations, preserves the exact previous positive-direction controls, and refreshes twenty hypotheses below.

For a normalized MLP input u and a unit tangent t, form v_plus=c*u+s*||u||*t and v_minus=c*u-s*||u||*t, where c=cos(theta), s=sin(theta). Both preserve ||u|| and share the radial displacement (c-1)*u. If Q is the MLP's homogeneous quadratic part, then:

    odd module change = [Q(v_plus)-Q(v_minus)]/2
    even module change = [Q(v_plus)+Q(v_minus)]/2 - Q(u)

The odd term equals c*s times the bilinear cross term between u and ||u||t. The even term equals (c^2-1)*Q(u)+s^2*Q(||u||t). The output bias cancels. The even term therefore includes both radial and tangent-quadratic contributions: this experiment does not attribute all even effects to radial drift alone.

We intervene separately with plus, minus, odd and even module-output changes, using the same true, within-family shuffled and three random directions as H786. Geometry is checked at chord/input radii .05/.15/.30; causal readouts use .30. The paired odd behavioral response is (margin_plus-margin_minus)/2; the paired even response is (margin_plus+margin_minus)/2 minus the native margin. Neither is assumed equal to the response from directly injecting the corresponding module component.

### Results on competence-passing tasks

For true directions at L1, the paired even answer-margin RMS is at least as large as the paired odd RMS in all four competence-passing families:

| Family | L1 odd RMS | L1 even RMS | L4 odd RMS | L4 even RMS |
|---|---:|---:|---:|---:|
| Agreement | .2799 | .3135 | .0674 | .0348 |
| Capital recall | .1808 | .2296 | .2111 | .0848 |
| Entity binding | .2472 | .2657 | .1846 | .0558 |
| Copying | .4170 | .5943 | .1667 | .0834 |

These are response magnitudes, not causal shares. Odd and even components can correlate or cancel.

The positive-direction L1 entity-binding intervention previously had mean donor-directed effect +.2092. Its sign-sensitive component has mean only +.0116; the paired even component accounts algebraically for the remaining +.1976. This demonstrates that the positive-only statistic was mostly not sensitive to the intended direction's sign. L1 copying retains a negative donor-directed odd mean (-.2252); L4 copying has a positive odd mean (+.0531). These are synthetic, site-local perturbations, not native donor swaps or whole-task necessity tests.

True-minus-shuffled donor-directed odd contrasts do not show a clear positive advantage in any of the four competent families/layers on this panel. L1 copying's contrast is negative: -.2195, with an item-cluster bootstrap interval [-.3624,-.0989]. L4 copying exceeds pooled random by +.0589, interval [.0162,.0999], but its true-minus-shuffled interval includes zero. L1 entity binding exceeds random because random odd response is negative, not because true odd response is large. The bootstrap uses eight items per family, groups reciprocal edits/seeds within item, and is descriptive without multiplicity correction; these are exploratory leads, not confirmed discoveries.

Directly injecting the odd module component does not reproduce the paired odd behavioral response. The L1 RMS discrepancy is .2384 for agreement and .5106 for copying; corresponding L4 discrepancies are .0220 and .0569. This confirms that downstream responses depend on the accompanying even component/background. Module-level algebra alone cannot justify additive behavioral attribution.

### Important follow-up diagnostic: native donor distances

After inspecting the parity results, we measured the unit-normalized chord distance between the actual receiver and donor MLP inputs. This diagnostic was not predeclared as a test outcome. At the final position, every one of the 64 L1 task pairs has native chord distance below .30; 63 of 64 L4 pairs do too. For competent families, the medians are:

| Family | L1 native distance | L4 native distance |
|---|---:|---:|
| Agreement | .0526 | .1873 |
| Capital recall | .0460 | .1488 |
| Entity binding | .0309 | .0694 |
| Copying | .1005 | .2382 |

Thus the matched .30 radius frequently extrapolates far beyond the native donor displacement, especially at L1. This does not invalidate the fixed-size geometric comparison, but sharply limits its interpretation as a natural semantic intervention. It also prevents concluding that the sign reversals reflect the native computation. H784's native source-position swaps remain a different experiment and are not refuted by this final-position diagnostic.

### Verification and scope

7,936 model forwards, 5,760 geometry cases. Exact quadratic parity formulas match to 3.02e-14 relative L2. Norm/radius checks remain at numerical precision. Recomputed positive patches match H785 to 4.77e-7; positive margin effects match H786 to 4.77e-6. Identity logits differ by at most 2.17e-5 across batches. Source/model/input hashes unchanged and effects finite. The original reserved test remains untouched; natural-document pool remains 26,112. All eight families are retained, but only agreement, capital recall, entity binding and copying pass native competence. Shared prompts, reciprocal edits, and seeds do not constitute independent replications.

### Current priorities

The best-supported account of the recent fixed-radius results is a combination of general quadratic layer geometry, sign-independent finite responses, and downstream background dependence. Natural task structure remains possible, but these stress tests have not isolated it. The most urgent unresolved alternative is extrapolation beyond native edit scales.

Next experiment: use a fraction of each pair's actual donor angle (e.g. .25/.5/1.0), with positive/negative true, shuffled and random directions at the same pair-specific radius. Include the actual native donor MLP output swap as the endpoint reference and audit any residual norm difference separately. Test whether the odd true-direction effect recovers donor-aligned behavior and exceeds shuffled/random controls within the natural scale. Freeze the design before outcomes. If it does, test held-out lexical/template transfer; if it does not, redirect attention toward site/path localization rather than fitting a new manifold to these final-position states.

### Twenty live hypotheses and the tests that distinguish them

These are competing or complementary hypotheses, not twenty established findings. Priorities reflect current evidence rather than numerical posterior probabilities.

1. **Generic quadratic layer geometry explains much of the L1/L4 contrast.** Supported by H786 random controls. Test sign-paired responses and separate module curvature from downstream gain.
2. **A common radial component of sphere rotations causes part of the apparent semantic effect.** Plausible from the intervention algebra. Compare opposite directions and explicitly isolate radial and tangent-quadratic contributions.
3. **Some task information lies in shared family-level directions.** True and shuffled directions often behave similarly. Freeze directions from discovery items and test new lexicons, templates and item identities.
4. **Item-specific directions carry additional binding information.** Not established by pooled geometry. Test true-minus-shuffled signed behavioral contrasts and fresh name/value permutations.
5. **Representations encode context-conditioned transformations more reliably than fixed activation vectors.** Consistent with background-dependent interventions. Fit compact local operators and compare cross-context finite-response prediction with fixed subspaces.
6. **A small reusable dictionary plus overlapping gates represents those operators.** Still unverified. Compare shared atoms against independent dictionaries at matched capacity, and ablate shared atoms across held-out combinations.
7. **Hierarchical gates capture inclusion relations.** Architecture remains a hypothesis, not inferred from low reconstruction loss. Require implication consistency, specificity and transfer on known nested synthetic structure and real semantic contrasts.
8. **A computational DAG captures reusable bilinear subcomputations.** Plausible for this architecture, but graph recovery remains open. Require interventions on a shared node to affect its distinct consumers and transfer to unseen combinations.
9. **Smooth nonlinear manifolds outperform linear groups.** Not established. Test local chart consistency, held-out interpolation, tangent transport and causal fidelity at matched reconstruction/parameter budgets.
10. **The best activation basis is not the best causal basis.** Supported by reconstruction/KL versus causal-transfer failures. Compare the same compression budgets under activation, KL, CE and finite-intervention objectives.
11. **Most model behavior depends on a small number of locally sensitive output directions.** Context-sensitive sensitivity evidence supports testing this. Evaluate local response spectra and how their span changes across contexts.
12. **Sensitivity changes are largely scalar gain changes rather than rotations.** Prior probe-gain results motivate this. Compare scalar gain, fixed anisotropy and context-dependent orientation on disjoint probes.
13. **RMS normalization contributes important apparent feature interactions.** H784 measured substantial scale-change terms. Separate normalized-input geometry from raw-residual interventions and test finite causal effects of each.
14. **Attention supplies context while token-local MLPs transform copied values.** Consistent with source-position graph-cut controls; broad minimality is unproven. Test path-specific routing/value perturbations on new formats and lexicons.
15. **Persistent value-bus contributions matter despite small isolated norms.** Norms alone cannot exclude conditional roles. Test bus/local combinations with sign and background controls.
16. **Early MLPs carry lexical identity; later ones carry context-dependent refinement.** A possible account of L1/L4 differences. Cross lexical identity with task context while matching displacement and source position.
17. **Cancellation across sites is essential, making additive attribution misleading.** Supported by joint-restoration failures. Test selected site interactions and predict joint effects from a model fit only to single-site responses.
18. **Apparent feature failure reflects vocabulary/category distribution shift.** The previous tools-to-fruits and colors-to-animals changes confound several shifts. Cross within-category versus between-category generalization independently.
19. **Off-native intervention states create effects absent from natural computation.** Still viable for sphere rotations and hybrid swaps. Compare native donor interpolation, norm-preserving paths, and natural-context perturbations at matched distances.
20. **Some apparent task findings reflect weak competence or answer-set artifacts.** Demonstrated as a concern by failed full-vocabulary gates. Require native competence, diverse answer alternatives and task-specific outcome checks before semantic claims.


Priority order: first resolve hypotheses 1/2/19 using native-scale sign controls; then distinguish 3/4/5 with frozen cross-item transfer; only then use the resulting causal targets to compare 6/7/8/9 at matched capacity. Hypotheses 10/11/12/13/17/20 remain evaluation constraints and competing explanations, rather than separate excuses to train more architectures. No manifold, universal semantic dictionary, or fundamental unit of computation is established yet.
