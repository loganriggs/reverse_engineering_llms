

## H410 — Add factors selected to correct a fixed core response

Reuse H400 discovery and H406 evaluation panels; actual head6, no fresh documents. Freeze Fisher64 core from H408. On discovery, rank omitted factors j by mean phi_j D_jᵀF(g−g_core). Select64 additions in one step and freeze before evaluation. Native Down vectors and coefficients are unchanged. This residual-correlation criterion omits candidate self-energy and interactions among the64 additions; it is not exact greedy squared-error minimization.

Compare the resulting128-factor set with Fisher128 (equivalent to choosing the next64 by original full-response attribution), contrast128, individual-energy128, and six independent random64 additions to the same core. Retain core64 as a diagnostic anchor, not an equal-budget competitor. Evaluate exact local Fisher retained/omitted/cross energies on both panels and finite native-output KL recovery and baseline-preserving clamping on evaluation, amplitudes1 and4. Direct residual is fixed. Panels are disjoint from selection but previously used in research.

Frozen primary gates at unit amplitude: residual128 recovery lower>.9, improvement over Fisher128 lower>.05, over individual-energy128 lower>.02, and over mean core+random64 lower>.05. One joint4 max-absolute95% band from4000 paired evaluation-document bootstraps, also resampling the six random addition sets. Scores summed over omitted factors must replay omitted Fisher energy within1e-8; native effect must replay H408 within1e-10.

|Primary statistic|Estimate|Joint95% interval|
|---|---:|---|
|residual128_unit_native_recovery|0.816157|[0.790663371569407, 0.8416504643607928]|
|residual128_minus_fisher128_recovery|0.113715|[0.0882216690131409, 0.1392087618045267]|
|residual128_minus_energy128_recovery|0.024656|[-0.000837097577214823, 0.050149995214171]|
|residual128_minus_core_random128_recovery|0.152729|[0.1272358428178749, 0.1782229356092607]|

Gates: {'score_identity': True, 'native_replay': True, 'recovery90': False, 'over_fisher05': True, 'over_energy02': False, 'over_random05': True}.

|Set|Discovery tangent recovery|Evaluation tangent recovery|Finite unit recovery|Finite4 recovery|Unit clamp remaining|
|---|---:|---:|---:|---:|---:|
|core64|0.668427|0.673223|0.655703|0.611807|0.337099|
|residual128|0.826938|0.823266|0.816157|0.782969|0.184830|
|fisher128|0.713949|0.717847|0.702442|0.685760|0.292557|
|contrast128|0.766680|0.774172|0.767165|0.761178|0.233388|
|individual_energy128|0.789744|0.802009|0.791500|0.775576|0.207808|
|core+random64 mean|0.675700|0.680807|0.663428|0.619016|0.329451|

Audits: {'residual_score_energy_relative_closure': 1.566650388666933e-16, 'serialization_recovery': 'Original run completed and saved all rows; JSON write failed on numpy bool. Recomputed audits from saved arrays; direct clamp audit rerun on original first four sites with original batch size8.', 'discovery_energy_identity_relative_error': 4.2901928025419874e-14, 'evaluation_energy_identity_relative_error': 3.262952747621803e-14, 'direct_clamp_state_absolute_error': 2.384185791015625e-06, 'native_effect_replay_max_absolute': 0.0, 'core64_finite_replay_max_absolute': 0.0, 'core64_tangent_replay_max_absolute': 0.0}. Independent saved-array checks: {'means_replay': True, 'unique128_factor_sets': True, 'frozen_core_retained_where_required': True, 'residual_score_sort_replay': True, 'H408_controls_unchanged': True, 'rescue_additions_with_negative_original_signed_attribution': 9, 'rescue_additions_overlap_fisher_next64': 23}. Nonprimary tangent, amplitude4 and clamp comparisons are descriptive. No threshold changes, readout fits, semantic labels or permanent pruning claims.

Outcome: both numerical gates pass. The residual128 set recovers0.816157 of the unit finite native-output effect, joint95% interval[0.790663,0.841650], so the90% gate fails. It beats Fisher128 by0.113715,[0.088222,0.139209], passing the5-point superiority gate; it beats the mean core+random64 control by0.152729,[0.127236,0.178223], also passing. Its advantage over individual-energy128 is0.024656,[−0.000837,0.050150], which fails the2-point gate and does not establish superiority to that simple baseline. Do not describe this as a generally superior sparse-selection algorithm.

Finite unit recoveries are core64 0.655703, residual128 0.816157, Fisher128 0.702442, contrast128 0.767165, individual-energy128 0.791500, and core+random64 mean0.663428. At4× amplitude the corresponding recoveries are0.611807,0.782969,0.685760,0.761178,0.775576,0.619016. The larger-amplitude comparisons are descriptive. Residual128 unit clamp remaining fraction is0.184830, versus Fisher1280.292557 and energy1280.207808; no confirmatory clamp threshold was specified in H410, so the point below20% is not a passed gate.

The local accounting supports the proposed correction. Discovery tangent recovery increases from0.668427 at core64 to0.826938 for residual128, versus0.713949 for Fisher128. Evaluation gives0.673223,0.823266 and0.717847 respectively. Retained–omitted cross/full energy changes from−0.211442 at the evaluation core to+0.033725 with residual additions, while Fisher128 is−0.319070. These exact-energy summaries are descriptive and show that this selection removes the strong negative cross term in the tested decomposition. They do not establish a unique grouping or prove cancellation is the sole source of remaining error.

Of64 targeted additions,9 had negative original full-response signed attribution, and23 overlap the next64 in the original Fisher order. Thus residual selection does not simply append the next-highest full-response contributors. All128 factors retain their native output vectors; no readout compensation was fitted. The set is a frozen input-dependent response predictor/activation patch, not a permanently pruned model or a semantic feature list.

Update to the twenty-hypothesis tournament: #5 (compensating factor combinations) now has an intervention/prediction rescue beyond algebraic attribution accounting. #3 (sparse support) receives limited support at128 factors for roughly82% of this head6 output effect, while #4 (distributed remaining support) stays viable. #12 (alignment) and #16 (output geometry) remain important, and the failure to beat individual energy conclusively means a sophisticated selection story is not required. #13 (context dependence) remains unresolved at broader source/domain scales; no evidence was added for semantic labels (#19).

Next most informative step: freeze these exact sets and confirm H410's original head6 gates on a genuinely fresh document panel, excluding all7872 completed documents. On the same fresh panel, also evaluate the fixed128-factor sets for each of the other eight head-source interventions at unit amplitude. Predeclare those transfer analyses and their simultaneous uncertainty before collecting outcomes. This tests whether the compensating group is specific to the discovery source or supports a reusable response component across sources, rather than repeatedly tuning this one head. Keep head6 confirmation distinct from the across-source transfer family. Preserve the failed90% and energy-superiority gates; do not lower them based on these results.

Execution note: all forward arrays were saved before a NumPy-bool JSON serialization failure. The serialization was fixed without altering the experiment or thresholds. Summary bootstraps and audits were reconstructed from saved arrays; the direct clamp identity was independently rerun at the original first four sites. Initial failure and recovery logs remain in work/. Native effects and core64 finite/tangent rows replay previous experiments exactly.

The broad research goal remains open. No fresh documents were consumed. No experiment processes remain running.
