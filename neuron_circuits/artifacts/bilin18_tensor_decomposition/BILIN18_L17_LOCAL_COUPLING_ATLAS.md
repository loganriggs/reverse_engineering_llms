

## H403 — All-pair local curvature-to-sensitivity atlas

All36 L17 head pairs, actual and six full-Gram-preserving rotations, reused H400512-document panel. No finite-amplitude responses enter the derivative statistics. The primary quantity compares actual versus rotated K/J, where K is pooled mixed-Taylor squared energy and J is pooled summed first-order source squared energy, with each site scaled by the same intact-final-state squared norm. K/J is not a global invariant under arbitrary reparameterizations; input source norms and all joint geometry are explicitly matched by the control.

For each pair, mixed coefficient M=2B(h,k)/s0²−2gamma*M0−2alpha_h*m1_k−2alpha_k*m1_h. Keep squared energies of M, of m1_h and m1_k summed, of the fixed-input-scale bilinear term F=2B(h,k)/s0², of normalization correction N=M−F, and cross energy2<F,N>. The identity ||M||²=||F||²+||N||²+2<F,N> is checked from saved arrays. This decomposition is not an additive causal attribution of behavioral performance. A low K/J can arise from higher J even when absolute K is higher.

Audit the mixed coefficient with central native float64 mixed finite differences at step.001, four fixed sites per pair/condition, retaining float32 RMS epsilon. Native bilinear coefficients used in the main statistics are float32 and energy assembly is float64. Input rotations preserve full source geometry to the previously enforced1e-6 tolerance. No output KL or semantic feature label is measured.

Frozen gates: {'finite_difference_audit': True, 'universal_suppression': False, 'twofold_suppressed_candidate': True, 'twofold_enhanced_candidate': False}. Universal suppression requires all36 upper K/J actual/rotated ratio bounds<1. Twofold candidate rules require a joint upper bound<.5 or lower bound>2. Select up to two most extreme pairs in each direction, for future fresh confirmation. Four thousand paired document/rotation-draw resamples give separate joint36 bands for curvature energy, first-order energy and their ratio; no joint108 claim is made. The panel and rotations are reused, so this is a broader discovery screen.

|Pair|Curvature ratio|First-order ratio|Normalized coupling ratio (joint95%)|
|---|---:|---:|---:|
|[0, 1]|1.68908|5.48428|0.30799 [0.12304,0.77095]|
|[0, 2]|1.29005|2.33292|0.55298 [0.22091,1.38421]|
|[0, 3]|1.20536|1.78506|0.67525 [0.26975,1.69028]|
|[0, 4]|1.40959|1.81051|0.77856 [0.31103,1.94888]|
|[0, 5]|1.04820|1.40371|0.74673 [0.29831,1.86922]|
|[0, 6]|1.29476|10.27652|0.12599 [0.05033,0.31538]|
|[0, 7]|1.16180|1.66038|0.69972 [0.27953,1.75154]|
|[0, 8]|1.84948|3.51583|0.52604 [0.21015,1.31679]|
|[1, 2]|2.82241|5.14595|0.54847 [0.21911,1.37293]|
|[1, 3]|2.76048|6.26636|0.44052 [0.17598,1.10272]|
|[1, 4]|1.46006|6.09713|0.23947 [0.09566,0.59943]|
|[1, 5]|1.84175|6.27604|0.29346 [0.11723,0.73458]|
|[1, 6]|4.52199|9.90338|0.45661 [0.18241,1.14298]|
|[1, 7]|2.36647|6.37609|0.37115 [0.14827,0.92905]|
|[1, 8]|5.24657|6.71206|0.78166 [0.31227,1.95665]|
|[2, 3]|2.34070|2.77653|0.84303 [0.33678,2.11027]|
|[2, 4]|2.22490|2.72879|0.81534 [0.32572,2.04096]|
|[2, 5]|1.58155|2.63278|0.60071 [0.23998,1.50371]|
|[2, 6]|2.54034|9.63476|0.26366 [0.10533,0.66000]|
|[2, 7]|2.02833|2.76845|0.73266 [0.29269,1.83399]|
|[2, 8]|4.52314|3.66430|1.23438 [0.49312,3.08989]|
|[3, 4]|1.63341|2.75597|0.59268 [0.23677,1.48360]|
|[3, 5]|3.03938|2.34703|1.29499 [0.51734,3.24162]|
|[3, 6]|3.56855|10.88308|0.32790 [0.13099,0.82079]|
|[3, 7]|1.65966|2.95045|0.56251 [0.22472,1.40808]|
|[3, 8]|5.67326|5.86752|0.96689 [0.38626,2.42032]|
|[4, 5]|1.33948|2.26798|0.59061 [0.23594,1.47840]|
|[4, 6]|1.87976|10.74166|0.17500 [0.06991,0.43805]|
|[4, 7]|1.72276|2.72444|0.63234 [0.25261,1.58286]|
|[4, 8]|2.90104|5.20134|0.55775 [0.22281,1.39615]|
|[5, 6]|1.44377|10.93661|0.13201 [0.05274,0.33045]|
|[5, 7]|1.23704|2.16920|0.57027 [0.22782,1.42751]|
|[5, 8]|4.73013|5.85894|0.80733 [0.32252,2.02092]|
|[6, 7]|1.97141|10.98335|0.17949 [0.07170,0.44930]|
|[6, 8]|3.62844|10.99786|0.32992 [0.13180,0.82586]|
|[7, 8]|3.77688|6.42920|0.58746 [0.23468,1.47052]|

Selected discovery candidates: [{'kind': 'suppressed', 'pair': [0, 6], 'coupling_ratio': 0.12599155478361596, 'joint95_interval': [0.05033228125108315, 0.31538153014773784]}, {'kind': 'suppressed', 'pair': [5, 6], 'coupling_ratio': 0.13201244652041233, 'joint95_interval': [0.05273756323049218, 0.33045300102581326]}]. Audits: {'max_relative_native_double_mixed_finite_difference_error': 0.00013226184423670154, 'finite_difference_step': 0.001, 'audit_sites_per_pair_condition': 4, 'geometry': [{'seed': 20270724, 'norm_scaled_gram_max_error': 3.810464967172756e-08, 'norm_scaled_residual_dot_max_error': 2.5115461557783615e-08}, {'seed': 20270725, 'norm_scaled_gram_max_error': 3.116312570687793e-08, 'norm_scaled_residual_dot_max_error': 2.2323191496826936e-08}, {'seed': 20270726, 'norm_scaled_gram_max_error': 4.0908587531560625e-08, 'norm_scaled_residual_dot_max_error': 2.6880952128976435e-08}, {'seed': 20270727, 'norm_scaled_gram_max_error': 3.020253555803863e-08, 'norm_scaled_residual_dot_max_error': 2.1853360069183966e-08}, {'seed': 20270728, 'norm_scaled_gram_max_error': 3.6941856236840525e-08, 'norm_scaled_residual_dot_max_error': 2.4686722268322785e-08}, {'seed': 20270729, 'norm_scaled_gram_max_error': 3.396755517584083e-08, 'norm_scaled_residual_dot_max_error': 2.339745315961153e-08}]}. Independent saved-array checks: {'means_max_absolute_error': 8.673617379884035e-19, 'energy_decomposition_max_relative_closure_error': 8.841821499319742e-15, 'H402_local_energy_replay': [{'pair': [1, 8], 'mixed_energy_max_relative_error': 4.4385063185724076e-16, 'first_order_energy_max_absolute_error': 0.0}, {'pair': [6, 8], 'mixed_energy_max_relative_error': 4.2209953347924122e-16, 'first_order_energy_max_absolute_error': 0.0}]}. Complete site metrics and bootstrap contrasts are retained.

Outcome: native finite-difference audit passes (maximum relative discrepancy0.0001323 at step.001). Universal normalized-coupling suppression fails. The twofold-suppressed-candidate gate passes; the enhanced-candidate gate fails. Actual first-order energy is significantly larger than rotated first-order energy for all36 pairs under its joint36 band. Actual mixed-curvature energy has point ratios above1 for all36 pairs and significant excess for15. Lower normalized coupling therefore does not mean smaller absolute mixed curvature.

Actual/rotated curvature-to-sensitivity point ratios range0.1260–1.2950. Thirty-four of36 point estimates are below1; eleven have joint upper bounds below1, four below0.5, and none have joint lower bounds above1. The frozen candidate rule selects heads0/6 with ratio0.1260,[0.0503,0.3154], and heads5/6 with ratio0.1320,[0.0527,0.3305]. These are discovery candidates requiring independent confirmation. The two earlier pairs remain informative but not both significant under the broader band: heads1/8 ratio0.7817,[0.3123,1.9567]; heads6/8 ratio0.3299,[0.1318,0.8259]. This does not invalidate the narrower H402 intervals, which covered a different family of contrasts.

For heads0/6, actual/rotated absolute mixed-curvature energy ratio is1.2948 and first-order energy ratio10.2765. For heads5/6 the ratios are1.4438 and10.9366. The mixed-curvature intervals for these two pairs include equality; their first-order energy intervals are[8.6724,12.1774] and[9.2294,12.9595]. Thus the selected low K/J ratios are driven chiefly by much larger first-order sensitivity, not demonstrated suppression of absolute curvature.

A supplementary algebraic diagnostic reconstructs each head's first-order energy from the complete pair-sum system. All reconstructed energies are positive; pair-sum replay differs by at most2.98e-8 in absolute squared-energy units. In actual heads0/6, head6 contributes98.793% of the pair's common-scaled first-order energy; for heads5/6,99.717%. Corresponding rotated fractions are91.101% and97.859%. This exposes a serious interpretation limit: a low ratio to summed first-order energy can be dominated by one source. It does not establish balanced independence between two equally responsive sources. The diagnostic does not change the preregistered metric or candidate selection.

Normalization cancellation also differs across pairs. For heads0/6, full mixed energy divided by the sum of fixed-scale and correction energies is0.555 actual versus0.536 rotated; for heads5/6,0.742 versus0.635. Their low normalized coupling is not accompanied by stronger cancellation on this descriptive measure. For heads6/8 the same ratio is0.315 actual versus0.629 rotated, showing cancellation can matter elsewhere. These decomposition ratios are not independently confirmed causal attributions.

The simultaneous normalized-coupling band is wide (multiplicative factor2.503). A descriptive bootstrap audit finds heads5/8 produce the largest centered deviation in2,252 of4,000 resamples, heads3/8 in467, heads3/6 in214 and heads2/7 in201. Those pairs remain included; no exclusions or narrowed post-hoc band are used. This limits universal conclusions, while the predeclared selected-pair results survive the full band.

Hypothesis update: enhanced first-order sensitivity along natural source orientations is a broad within-model pattern; relative mixed-curvature suppression is supported for a subset, not universally. Selected low ratios may reflect unequal source sensitivity rather than pair-specific decoupling. Head-associated local operators remain useful quantitative descriptions, but the current evidence is insufficient to call the head pair an independent or semantic computational unit.

Next: confirm the fixed heads0/6 and5/6 K/J contrasts on fresh documents, preserving the twofold criterion and all geometry controls. Before seeing fresh outcomes, add a diagnostic that normalizes each source direction by its own first-order response magnitude, then evaluates mixed-curvature energy. This asks whether apparent suppression survives balancing individual responsiveness, rather than allowing head6 to dominate a sum. Such normalization changes the intervention scale and must be labeled a distinct diagnostic, not a replacement for the geometry-matched primary comparison. Retain absolute curvature, each individual source energy, and full energy/cancellation terms. Any next fresh panel must exclude all6,848 complete-document hashes from completed panels.
