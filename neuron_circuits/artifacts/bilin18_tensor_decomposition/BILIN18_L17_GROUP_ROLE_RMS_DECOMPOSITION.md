

## H418 — Decompose the angle effect into bilinear and input-RMS terms

Reuse H413 documents, H416 mappings and fixed H410 head6 factor groups. For source removal direction−h, let s²=mean(p²)+native_float32_epsilon, x=p/s and alpha=mean(p·h)/s². The group derivative is g=−2B_group(x,h)/s + 2alpha B_group(x,x). Call these fixed-scale f and radial r. Native Down weights remain fixed, biases have zero derivative, and the recipient native output Fisher metric remains unchanged. Input derivatives and factor arithmetic use double precision; the baseline output state/probabilities retain the native float32 forward path.

Expand2<g_core,g_added> into four terms:2<f_core,f_added>,2<f_core,r_added>,2<r_core,f_added>,2<r_core,r_added>. Their sum is the full cross term. Within each condition divide every term by the same full native-response Fisher energy; angle-minus-unmatched differences then sum exactly to the observed normalized angle contrast. Removing radial terms defines a fixed-input-scale derivative counterfactual while retaining the original denominator and output metric. These terms are an accounting decomposition, not a unique causal mediation fraction or semantic interaction parameter.

Predeclare one joint4 max-absolute95% band from4000 paired document/mapping bootstrap draws for the two angle effects and two radial-associated fractions (all three terms involving r divided by total angle effect), at donor and recipient norms. Require both angle effects lower>.1; radial-associated fractions lower>.8 for radial dominance, or upper<.2 for fixed–fixed dominance. Fractions may lie outside[0,1] due signed cancellation and are not clipped. Numerical derivative/cross closure must be<1e-10 and pooled prior float ratios replay within1e-5.

|Primary statistic|Estimate|Joint95% interval|
|---|---:|---|
|angle_effect_donor_norm|0.271423|[0.10454526615516058, 0.4382998949056264]|
|angle_effect_recipient_norm|0.272302|[0.10542426206149486, 0.43917889081196065]|
|radial_fraction_donor_norm|-0.894092|[-1.060969785847058, -0.7272151570965921]|
|radial_fraction_recipient_norm|-1.390889|[-1.5577665407616839, -1.224011912011218]|

Gates: {'decomposition_identity': True, 'pooled_replay': True, 'angle_effect10': True, 'radial_accounts80': False, 'fixed_accounts80': True}.

|Condition|Fixed–fixed|Fixed–radial|Radial–fixed|Radial–radial|Full cross/full energy|
|---|---:|---:|---:|---:|---:|
|actual|-0.487056|-0.057627|0.208314|0.168801|-0.167567|
|raw|-0.423862|-0.107051|0.180920|0.117400|-0.232593|
|norm_only|-0.512378|-0.103059|0.209760|0.195356|-0.210321|
|angle_only|0.090237|-0.103989|0.003891|0.048690|0.038830|
|both|0.138665|-0.143177|-0.000182|0.066674|0.061980|

Audits: {'group_derivative_decomposition_max_relative': 2.4815821516325705e-15, 'cross_term_closure_full_energy_normalized': 8.32733519700832e-14, 'pooled_role_float_replay_max_absolute': 1.0854409676630539e-08}. Independent array checks: {'condition_terms_replay': True, 'angle_contribution_and_fraction_replay': True, 'all_rows_finite': True}. This is diagnostic reuse, not fresh confirmation. All signed terms and denominators are saved per site/draw.

Outcome: decomposition identity, pooled replay, angle-effect and fixed–fixed-dominance gates pass; radial-dominance fails. Input-RMS corrections oppose rather than produce the head6 angle effect in this accounting. At donor norms, fixed–fixed contributes+0.514099 while the three radial-associated terms together contribute−0.242677, leaving total+0.271423. At recipient norms, the corresponding contributions are+0.651043 and−0.378741, leaving+0.272302.

Radial-associated fractions of the total change are−0.894092 with joint95% interval[−1.060970,−0.727215] and−1.390889,[−1.557767,−1.224012]. Negative fractions indicate an offsetting contribution; they are not negative probabilities or an additive semantic mediation measure. The fixed–fixed term consequently exceeds the observed net shift. The two total-effect intervals are[0.104545,0.438300] and[0.105424,0.439179], both clearing the frozen+.1 gate in the same joint4 band. These intervals are wider than H417's because this family also covers the contribution fractions.

The fixed-scale counterfactual itself changes sign: core/add cross divided by the original full-response energy is−0.423862 raw and+0.090237 angle-only; norm-only is−0.512378 and both is+0.138665. Thus input-RMS radial terms are not necessary to obtain the normalized cross-term reversal. They reduce its magnitude. The output Fisher metric still includes native final RMS, unembedding, cap and probabilities; this does not establish metric-independent factor interaction or a global property of the weights.

For the original actual source, fixed–fixed is−0.487056 and radial-associated terms sum to+0.319489, yielding−0.167567. This highlights an unresolved distinction: actual and fully matched donor inputs have the same source norm and residual projection, yet their fixed-scale group alignment differs. A scalar angle is therefore not a complete description of the source; the orientation of its remaining component matters. Raw shuffling can retain negative alignment through a different balance of directional components, so that result alone does not make conditional source/recipient structure irrelevant.

Numerical checks are strong with double derivative arithmetic: group decomposition discrepancy2.48e-15 relative to full derivative norm, cross closure8.33e-14 relative to full Fisher energy, and maximum pooled replay difference from H416/H4171.09e-8. This independently shows that H417's three float32 ratio corner cases do not drive the approximately27-point pooled angle effect. No prior failed audit is changed. No source vectors, factor rankings or readout coefficients were fitted in H418.

Hypothesis update: #6 (input normalization alone creates the role/control pattern) is weakened for this specific angle effect. #12 (orientation-dependent fixed-scale bilinear/output alignment) is now leading. The precise remaining alternatives are a positive response from the recipient-aligned component itself, interference between recipient-aligned and orthogonal components, or altered weighting of the orthogonal response. These alternatives have an exact low-dimensional decomposition and need no new factor-selection search. #19 semantic labels remain unsupported.

Next: decompose each head6 source into its component along the recipient residual projected into the head output subspace and the orthogonal component. In the fixed-scale bilinear map, the core/add cross numerator splits into parallel–parallel, parallel–orthogonal (both orders), and orthogonal–orthogonal terms. Reconstruct the H418 fixed–fixed term and angle effects from those pieces. Include actual sources as well as the same frozen donor shuffles: actual and fully matched sources share parallel amplitude/norm, so their difference must be carried by mixed and/or orthogonal terms. Freeze the component-fraction comparisons before outcomes, retain signed terms and the original denominator, and avoid interpreting a quadratic accounting identity as a recovered global feature. This directly tests which directional relationship produces the group role.

No fresh documents were consumed. The next fresh panel must exclude8896 completed-panel documents. The broad research goal remains open, and no experiment processes remain running.
