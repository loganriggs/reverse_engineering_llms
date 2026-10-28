

## H413 — Fresh confirmation of selected source-dependent group roles

Confirm three contrasts selected after H412, with the same frozen H410 core64, added64 and union128. Use512 new complete documents, independently rehashed against8384 prior completed documents: zero overlap,8896 total. Same model/corpus, not cross-domain or training holdout. Heads4,6,8 at unit MLP-path source edits; direct residual fixed. No factor selection or readout refit.

Before fresh outcomes, freeze one joint3 max-absolute95% band from4000 paired document bootstraps and thresholds: head6 core/add Fisher cross divided by full energy upper<−.05; head8 cross/full lower>.05; head4 additions-only minus core-only finite native KL recovery lower>.1. These confirm selected hypotheses and do not repair H412's failed joint27 gates. All other table metrics are descriptive.

The local metric includes final RMS with native float32 epsilon, unembedding, capped logits and softmax centering, evaluated in double arithmetic. Core, additions and remainder give three energies and three cross terms that sum to the full Fisher energy. A direct full-derivative audit checks the grouped tangent calculation at16 fixed sites per source. Finite predictions and baseline-preserving clamps use exact native normalization and probabilities. Cross terms describe output-metric alignment, not a nonlinear factor-interaction parameter or semantic label.

|Primary statistic|Estimate|Joint95% interval|
|---|---:|---|
|head6_core_add_cross_over_full|-0.167567|[-0.21424140988180673, -0.12089208635131471]|
|head8_core_add_cross_over_full|0.115869|[0.06919444444409059, 0.16254376797458264]|
|head4_additions_minus_core_finite_recovery|0.433501|[0.38682649438875716, 0.48017581791924924]|

Gates: {'document_disjointness': True, 'energy_identity': True, 'head6_cancellation05': True, 'head8_alignment05': True, 'head4_additions_over_core10': True}.

|Head|Core64 recovery|Added64 recovery|Union128 recovery|Added64 clamp remaining|Native effect KL|
|---|---:|---:|---:|---:|---:|
|4|0.010408|0.443910|0.467006|0.559491|0.00035387|
|6|0.691036|-0.076470|0.836618|1.065995|0.00231042|
|8|0.770503|0.172166|0.826338|0.825357|0.00034239|

Audits: {'Fisher_energy_identity_max_relative': 6.206717349985412e-15, 'direct_clamp_state_max_absolute': 5.7220458984375e-06, 'direct_full_tangent_relative_squared_error': 7.710128090281387e-29}. Independent arrays: {'mean_metrics_replay': True, 'three_part_energy_max_relative': 9.12752551468443e-15, 'primary_replay': True, 'largest_document_share_of_effect_per_head': [0.03286872173660906, 0.03671163351908223, 0.013645517066415163]}. Documents: {'independently_rehashed_excluded_documents': 8384, 'fresh_unique_documents': 512, 'overlap': 0, 'completed_unique_total': 8896, 'all_windows_match_source': True}. No influential documents were excluded or ratios clipped.

Outcome: all five gates pass, including all three scientific contrasts. On512 fresh documents, head6 core/add cross/full energy is−0.167567 with joint95% interval[−0.214241,−0.120892], clearing the−.05 cancellation threshold. Head8 is+0.115869,[0.069194,0.162544], clearing+.05. Head4 additions-only minus core-only finite recovery is0.433501,[0.386826,0.480176], clearing+.1. This confirms the H412-selected role pattern on independent documents; H412's broad-family failures remain unchanged.

The same fixed factor groups therefore have different output-response relationships depending on source. For head6, core64 finite recovery is0.691036, added64 alone is−0.076470, and union128 is0.836618. The added response is not a useful standalone head6 predictor by this point metric, yet it improves the combination. For head8, recoveries are0.770503,0.172166 and0.826338, with positive local alignment. For head4, the added group supplies most of the union's predictive value:0.010408 core,0.443910 additions,0.467006 union. Standalone/union recovery levels are descriptive; the head4 difference and two cross-term signs/magnitudes are the prespecified confirmed contrasts.

A negative additions-only recovery means its prediction has larger target KL error than the intact/no-edit predictor, not a negative causal effect. Similarly, a negative local cross term denotes destructive alignment in the native output Fisher metric. It is not an interaction coefficient in the factor architecture: factor outputs add before downstream normalization, and the readout metric determines which vector errors count. The result supports source-conditioned roles for a group, not a single intrinsic semantic name for that group.

Numerical audit: three-part energy closure6.21e-15 relative; direct full-tangent versus sum-of-components squared Fisher error7.71e-29 at16 sites per source; direct clamp normalized-state discrepancy5.72e-6. Independent source-document rehashes exclude all8384 previous documents. Largest single-document effect shares are3.287% for head4,3.671% for head6 and1.365% for head8; all documents remain included. This panel does not repeat the large head3 concentration seen in the prior exploratory family, although it does not test head3 at all.

Hypothesis narrowing: #5 (compensating combinations) and #11/#14 (source-dependent reuse) now have selected fresh confirmation at the group-response level. #10 (one-panel artifact) is weakened for these exact contrasts. The leading unresolved alternatives are #12 (fixed output/subspace geometry accounts for the role) versus #13/#20 (specific source activations and recipient-conditioned operators determine it). #19 (semantic features) remains untested; no linguistic interpretation follows from these signs alone.

Next high-information control: reuse this panel and frozen groups, replacing each actual source direction with six draws inside that head's fixed output subspace, while matching source norm and its dot product/angle with the recipient residual. Use the existing frozen per-position source moments for one family and isotropic in-subspace directions for another. The moment-based draw is transformed by geometry matching, so do not claim exact post-transformation covariance matching. Audit norm, residual projection and head-subspace membership. Compare the same three role statistics against actual inputs with paired document/draw uncertainty and predeclared sign-preservation/equivalence thresholds. If the roles survive these replacements, head output geometry plus recipient state can explain them without the particular source activation; if they disappear, the source content/alignment becomes necessary within this test. This is a counterfactual diagnostic, not a claim that random draws are natural attention outputs.

The existing helper angle_matched_payload in run_bilin18_l8_execution_alignment.py implements the within-head norm/residual-angle constraint; H398's run_bilin18_l17_all_pair_interaction_screen.py demonstrates use of the frozen bilin18_l17_source_moment_controls_moments.npz and head output bases. Review and audit these before reuse. No new factor ranking is needed for this control.

The broad feature/computation-unit goal remains open. There are now8896 distinct completed-panel documents; every future fresh panel must exclude all8896. No experiment processes remain running.
