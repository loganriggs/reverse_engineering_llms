## H827 — Identical early-layer edits have context-dependent local responses

H826's angle distortions could arise from a common anisotropic map or recipient-dependent sensitivity, since previous recipients received different edit directions. H827 crosses identical frozen directions across new recipients to distinguish those explanations.

### Frozen crossing design

For each H824 selected block, choose the first frozen donor pair admitting four unused recipient documents with the exact same token, absolute position and active block. Exclude every H825 native endpoint document and previously selected recipient documents. Hash-sort candidates before any new output evaluation. Residual0 block134 cannot supply four eligible new documents and remains a coverage failure. Residual0 block104 was already absent from H824. Thirteen groups remain: five residual0 blocks and eight residual1 blocks, four unique recipient documents each (52 total).

Use the SAME saved H825 near-block edit vector at all four recipients in a group. Also cross the corresponding frozen random-subspace direction at its matched norm. Neither direction is rescaled by recipient state. Evaluate signed multipliers -.25,-.10,+.10,+.25 and a zero edit. Estimate central finite-difference logit derivatives at each magnitude. Recipients are new to these causal evaluations, although their unedited early activations were captured in H824. Directions themselves are reused, not fresh discoveries.

For each group, define a COMMON output probability reference as the mean native distribution of its four recipients. Weighted-center logit derivatives and multiply by the square root of this common reference. All six recipient-pair comparisons therefore use the same metric. Uniformly centered logit derivatives provide an independent metric check. Configuration mentions recipient-specific weighting as diagnostic only, but this implementation evaluates only common-reference and uniform metrics; no recipient-specific result is claimed.

### Main result: sensitivity differs across contexts

| Site | Direction | Recipient pairs | Median response cosine, common metric | Uniform metric | Negative pairs, common metric |
|---|---|---:|---:|---:|---:|
| residual0 | block |30|.266|.084|8|
| residual0 | random |30|.039|.008|13|
| residual1 | block |48|.058|.067|21|
| residual1 | random |48|.093|.010|19|

These cosines compare responses to IDENTICAL input vectors, not merely nearby donor states. Low and sometimes negative alignment persists under a common output metric and uniform centering. A single fixed linear output map cannot reproduce these directional differences across the selected recipients. This supports context-dependent local downstream sensitivity in this panel.

The result does not locate the context dependence: it could arise in later attention/MLPs, residual normalization, or the final softcap/output path. It does not prove that a particular BSF block is itself performing a context-dependent operation. Local linearity within each context is fully compatible with a context-dependent derivative across contexts.

Block directions are not consistently more portable than random controls: residual0 has a higher descriptive median, but residual1 does not under the common metric. There are only5/8 groups and four recipients each; the30/48 pairwise comparisons are dependent. No formal independent-pair significance claim is made.

### Dose and implementation checks

Central derivatives are highly consistent between .10 and .25 multipliers. Across104 direction/recipient cases, the median relative derivative change is .242% under uniform centering and .192% under the common metric; maxima are1.90% and1.83%, respectively. Every case is below2%. Median derivative cosines by site/mode exceed .999995. Small-dose even/odd response ratios are approximately2–3%, so finite nonlinear contributions are present, but cannot by themselves explain the much larger cross-context changes in response direction. These checks apply only along the tested vectors and dose range.

Maximum applied patch norm is9.90% of its recipient residual norm at the larger magnitude. Therefore the descriptor “small dose” is relative to the frozen donor edit, not a claim that every edit is infinitesimal in activation space. Stability of central derivatives provides the stronger local-regime evidence.

520 whole-model forwards:52 native plus468 edited (416 nonzero and52 zero). Native prefix replay relative error1.44e-6; zero-edit logits match native exactly; all hashes unchanged and outputs finite. All52 recipient documents are mutually unique and disjoint from prior H825 native endpoint documents. The saved numerical noise ratio is extremely large because its replay denominator is exactlyzero and clamped at1e-30; it is not a meaningful noise estimate or evidence of unlimited precision. No reserved-test evaluations or new documents; natural-document pool26,112.

### A contrasting block and next experiment

Group-level block response medians vary considerably. Residual0 block56 is an exception, with common-metric median cosine .997 across its four recipients; the other residual0 group medians are .341,.024,-.086,.142. Residual1 group medians range roughly-.080 to.334. The exception was identified after outcomes and must not be promoted as a discovered invariant unit without independent confirmation. Earlier lexical work associates block56 with byte fragments, suggesting a potentially narrow tokenization-driven operation rather than broad semantics.

Next localize the context dependence using the saved fixed directions and frozen groups: compare immediate next-block responses, later residual responses, and final logits, with identical-direction replay at each stage where feasible. Include the portable block56 as a prespecified contrast while retaining all groups. Distinguish changes introduced by RMS normalization, attention interactions, bilinear MLP terms and final output saturation. A useful computational unit may need a direction PLUS a recipient-state condition or operator, rather than a context-free activation vector. This is a hypothesis motivated by the crossing result, not yet an identified mechanism. Preserve the broader alternatives and diverse-task requirement before assigning a feature ontology.
