# Evidence guide

Read README first. The source reports retain their historical wording and hypotheses; they are
not all endorsed as current conclusions. Local indices below work without the original checkout.

## The central MLP1 question

- [Historical findings with corrections](evidence/prior/EXCERPTS_FINDINGS.md): token-table CE
  ceilings and the instrument check. A constant-attention MLP0 is token-deterministic by architecture;
  showing an exact lookup there is a control, not discovery of the source of context dependence.
- [Historical timeline excerpts](evidence/prior/EXCERPTS_RESULTS.md): article localization, layer1
  synthesis, and the later correction from loosely described “redundancy” to distributed cooperation.
  The older universal claims from a few tested circuits must not be generalized literally.
- [Methodological corrections](evidence/prior/EXCERPTS_LESSONS.md): coverage, scorer/intervention
  mismatch, and individual-versus-joint comparisons.
- [MLP1 reading MLP0](evidence/prior/mlp1_reads_mlp0_results.json): article cluster experiment,
  including random/artifact controls. The broad “independent” label is narrower than universal independence.
- [MLP1 article localization](evidence/prior/article_mlp1_localization_results.json): the
  sufficient-versus-necessary distinction for native channel clusters.
- [Contextual residue attribution](evidence/prior/mlp1_residue_attribution_results.json): exact
  fitting/evaluation filenames, hybrid replacement, head rankings, and changing-model limitation.
- [Native gate null](evidence/prior/MLP1_GLOBAL_GATE_FINDINGS_2026-08-28.md): repeatable supports
  did not satisfy the full causal admission test.
- [Split-probe null](evidence/prior/MLP1_SPLIT_PROBE_FINDINGS.md): concentrated response energy
  did not yield a repeatable local physical subspace under independent probes.
- [Coefficient tensor geometry](evidence/prior/MLP1_IMPLICIT_FOLDED_TENSOR_V2_FINDINGS.md): full
  multilinear rank, finite-cost limits of dense Tucker, and explicit limits on causal interpretation.

## Later prior-work checks (do not stop at the early synthesis)

These JSON files contain prediction gates, intervention definitions, source hashes and next steps:

- [rung489](evidence/prior/mlp1_native_state_dominance_rung489_results.json): common reader classification null.
- [rung490](evidence/prior/mlp1_branch_resolved_response_rung490_results.json): branch-resolved response analysis.
- [rung491](evidence/prior/mlp1_live_state_source_decomposition_rung491_results.json): named source decomposition passes its gates.
- [rung492](evidence/prior/mlp1_attention1_reader_path_intervention_rung492_results.json): actual attention1 dependence passes,
  but the selective reader and intervention transfer claims fail. This limits how far rung491 can be interpreted.
- [MLP0 approximation → MLP1 interchange](evidence/prior/MLP0_C512_MLP1_INTERCHANGE_V3_FINDINGS.md):
  comparison of computation under an altered upstream interface.

## What this session adds

- [H1009 input-to-output prediction](evidence/current/BILIN18_EARLY_MLP_PREDICTION.md):
  explicit computational prediction rather than compressibility. Input rank, data volume, and
  regularization remain confounded; the rich quadratic student overfits. Its proposed downstream
  replacement sweep was superseded by other work, so this packet does not claim that sweep ran.
- [Layers0–3 compositional study](evidence/current/BILIN18_LAYERS0_3_COMPOSITIONAL_ONTOLOGY.md)
  and [replication](evidence/current/BILIN18_LAYERS0_3_COMPOSITIONAL_ONTOLOGY_REPLICATION.md):
  distributed native-neuron intervention curves and downstream Q/K sensitivity. State-specific
  gradient oracles are not discovered semantic features or autonomous predictors.
- [Weight-versus-data scaling](evidence/current/BILIN18_MLP_WEIGHT_VS_DATA_SCALING_REDTEAM.md):
  more data helps MLP1 write geometry, but low-rank write sufficiency remains limited.
- [Head scaffold stability](evidence/current/BILIN18_LAYER1_HEAD_RANKING_STABILITY.md): multiple
  overlapping downstream head sets can explain similar sensitivity. This is suffix-path ambiguity,
  not duplicate MLP1 neurons.
- [Original layers0–2 plan](evidence/current/LAYER_0_2_INTERPRETATION_PLAN.md): motivation and
  exact source-pair decomposition. It is a plan, not evidence that every listed stage was completed.

## Neighbor controls

[MLP0 dossier](evidence/prior/MLP0_CURRENT_UNDERSTANDING.md) and
[MLP2 dossier](evidence/prior/MLP2_CURRENT_UNDERSTANDING.md) are copied intact. The MLP2
native-channel-selection and learned-coordinate comparison is especially relevant to cancellation.
Additional historical scripts/results are indexed below. Scripts preserve original source bytes;
external imports/data are intentionally not presented as runnable from this packet alone.

## Full copied-source index

See PROVENANCE.json for original absolute paths, SHA256 hashes and the source repository commit.
The following list is generated from that provenance; vendored source is included for attribution.

- [vendor/jacclust/tt_model.py](vendor/jacclust/tt_model.py)
- [vendor/basis_aligned/polynomial_causal/bilin18_observed_model_facade.py](vendor/basis_aligned/polynomial_causal/bilin18_observed_model_facade.py)
- [evidence/current/BILIN18_EARLY_MLP_PREDICTION.md](evidence/current/BILIN18_EARLY_MLP_PREDICTION.md)
- [evidence/current/bilin18_early_mlp_prediction.json](evidence/current/bilin18_early_mlp_prediction.json)
- [evidence/current/scripts/run_bilin18_early_mlp_prediction.py](evidence/current/scripts/run_bilin18_early_mlp_prediction.py)
- [evidence/current/BILIN18_LAYERS0_3_COMPOSITIONAL_ONTOLOGY.md](evidence/current/BILIN18_LAYERS0_3_COMPOSITIONAL_ONTOLOGY.md)
- [evidence/current/bilin18_layers0_3_compositional_ontology.json](evidence/current/bilin18_layers0_3_compositional_ontology.json)
- [evidence/current/bilin18_layers0_3_compositional_ontology.png](evidence/current/bilin18_layers0_3_compositional_ontology.png)
- [evidence/current/scripts/run_bilin18_layers0_3_compositional_ontology.py](evidence/current/scripts/run_bilin18_layers0_3_compositional_ontology.py)
- [evidence/current/BILIN18_LAYERS0_3_COMPOSITIONAL_ONTOLOGY_REPLICATION.md](evidence/current/BILIN18_LAYERS0_3_COMPOSITIONAL_ONTOLOGY_REPLICATION.md)
- [evidence/current/bilin18_layers0_3_compositional_ontology_replication.json](evidence/current/bilin18_layers0_3_compositional_ontology_replication.json)
- [evidence/current/bilin18_layers0_3_compositional_ontology_replication.png](evidence/current/bilin18_layers0_3_compositional_ontology_replication.png)
- [evidence/current/BILIN18_MLP_WEIGHT_VS_DATA_SCALING_REDTEAM.md](evidence/current/BILIN18_MLP_WEIGHT_VS_DATA_SCALING_REDTEAM.md)
- [evidence/current/bilin18_mlp_weight_vs_data_scaling_redteam.json](evidence/current/bilin18_mlp_weight_vs_data_scaling_redteam.json)
- [evidence/current/BILIN18_LAYER1_HEAD_RANKING_STABILITY.md](evidence/current/BILIN18_LAYER1_HEAD_RANKING_STABILITY.md)
- [evidence/current/bilin18_layer1_head_ranking_stability.json](evidence/current/bilin18_layer1_head_ranking_stability.json)
- [evidence/current/bilin18_layer1_head_ranking_stability.png](evidence/current/bilin18_layer1_head_ranking_stability.png)
- [evidence/current/scripts/run_bilin18_layer1_head_ranking_stability.py](evidence/current/scripts/run_bilin18_layer1_head_ranking_stability.py)
- [evidence/current/LAYER_0_2_INTERPRETATION_PLAN.md](evidence/current/LAYER_0_2_INTERPRETATION_PLAN.md)
- [evidence/prior/mlp1_reads_mlp0_results.json](evidence/prior/mlp1_reads_mlp0_results.json)
- [evidence/prior/mlp1_reads_mlp0.py](evidence/prior/mlp1_reads_mlp0.py)
- [evidence/prior/article_mlp1_localization_results.json](evidence/prior/article_mlp1_localization_results.json)
- [evidence/prior/article_mlp1_localization.py](evidence/prior/article_mlp1_localization.py)
- [evidence/prior/article_mlp0_localization_results.json](evidence/prior/article_mlp0_localization_results.json)
- [evidence/prior/article_mlp0_localization.py](evidence/prior/article_mlp0_localization.py)
- [evidence/prior/mlp1_residue_attribution_results.json](evidence/prior/mlp1_residue_attribution_results.json)
- [evidence/prior/mlp1_residue_attribution.py](evidence/prior/mlp1_residue_attribution.py)
- [evidence/prior/front_token_vs_context_results.json](evidence/prior/front_token_vs_context_results.json)
- [evidence/prior/front_token_vs_context.py](evidence/prior/front_token_vs_context.py)
- [evidence/prior/mlp1_residual_ladder_results.json](evidence/prior/mlp1_residual_ladder_results.json)
- [evidence/prior/mlp1_residual_ladder.py](evidence/prior/mlp1_residual_ladder.py)
- [evidence/prior/mlp1_residual_ctx_results.json](evidence/prior/mlp1_residual_ctx_results.json)
- [evidence/prior/mlp1_residual_ctx.py](evidence/prior/mlp1_residual_ctx.py)
- [evidence/prior/mlp1_scale_results.json](evidence/prior/mlp1_scale_results.json)
- [evidence/prior/mlp1_scale.py](evidence/prior/mlp1_scale.py)
- [evidence/prior/mlp1_native_state_dominance_rung489_results.json](evidence/prior/mlp1_native_state_dominance_rung489_results.json)
- [evidence/prior/mlp1_native_state_dominance_rung489.py](evidence/prior/mlp1_native_state_dominance_rung489.py)
- [evidence/prior/mlp1_branch_resolved_response_rung490_results.json](evidence/prior/mlp1_branch_resolved_response_rung490_results.json)
- [evidence/prior/mlp1_branch_resolved_response_rung490.py](evidence/prior/mlp1_branch_resolved_response_rung490.py)
- [evidence/prior/mlp1_live_state_source_decomposition_rung491_results.json](evidence/prior/mlp1_live_state_source_decomposition_rung491_results.json)
- [evidence/prior/mlp1_live_state_source_decomposition_rung491.py](evidence/prior/mlp1_live_state_source_decomposition_rung491.py)
- [evidence/prior/mlp1_attention1_reader_path_intervention_rung492_results.json](evidence/prior/mlp1_attention1_reader_path_intervention_rung492_results.json)
- [evidence/prior/mlp1_attention1_reader_path_intervention_rung492.py](evidence/prior/mlp1_attention1_reader_path_intervention_rung492.py)
- [evidence/prior/MLP1_GLOBAL_GATE_FINDINGS_2026-08-28.md](evidence/prior/MLP1_GLOBAL_GATE_FINDINGS_2026-08-28.md)
- [evidence/prior/MLP1_SPLIT_PROBE_FINDINGS.md](evidence/prior/MLP1_SPLIT_PROBE_FINDINGS.md)
- [evidence/prior/MLP1_IMPLICIT_FOLDED_TENSOR_V2_FINDINGS.md](evidence/prior/MLP1_IMPLICIT_FOLDED_TENSOR_V2_FINDINGS.md)
- [evidence/prior/MLP2_IMPLICIT_FOLDED_TENSOR_V1_FINDINGS.md](evidence/prior/MLP2_IMPLICIT_FOLDED_TENSOR_V1_FINDINGS.md)
- [evidence/prior/MLP2_CMR_V1_VALIDATION_FINDINGS.md](evidence/prior/MLP2_CMR_V1_VALIDATION_FINDINGS.md)
- [evidence/prior/MLP2_RANK512_REFIT_V2_RECOVERY_FINDINGS.md](evidence/prior/MLP2_RANK512_REFIT_V2_RECOVERY_FINDINGS.md)
- [evidence/prior/MLP0_C512_MLP1_INTERCHANGE_V3_FINDINGS.md](evidence/prior/MLP0_C512_MLP1_INTERCHANGE_V3_FINDINGS.md)
- [evidence/prior/MLP0_CURRENT_UNDERSTANDING.md](evidence/prior/MLP0_CURRENT_UNDERSTANDING.md)
- [evidence/prior/MLP2_CURRENT_UNDERSTANDING.md](evidence/prior/MLP2_CURRENT_UNDERSTANDING.md)
- [evidence/prior/EXCERPTS_FINDINGS.md](evidence/prior/EXCERPTS_FINDINGS.md)
- [evidence/prior/EXCERPTS_RESULTS.md](evidence/prior/EXCERPTS_RESULTS.md)
- [evidence/prior/EXCERPTS_LESSONS.md](evidence/prior/EXCERPTS_LESSONS.md)


## Follow-up sources

Read [HISTORY_AUDIT.md](HISTORY_AUDIT.md) for the amended chronology and implications.

- [evidence/prior/front_table_compose.py](evidence/prior/front_table_compose.py)
- [evidence/prior/front_table_compose_results.json](evidence/prior/front_table_compose_results.json)
- [evidence/prior/front_table_compose4.py](evidence/prior/front_table_compose4.py)
- [evidence/prior/front_table_compose4_results.json](evidence/prior/front_table_compose4_results.json)
- [evidence/prior/mlp0_c512_mlp2_compensation_v2_results.json](evidence/prior/mlp0_c512_mlp2_compensation_v2_results.json)
- [evidence/prior/early_mlp_state_complete_compiler_v21_final_authority.json](evidence/prior/early_mlp_state_complete_compiler_v21_final_authority.json)
- [evidence/prior/MLP0_C512_MLP2_COMPENSATION_V2_FINDINGS.md](evidence/prior/MLP0_C512_MLP2_COMPENSATION_V2_FINDINGS.md)
- [evidence/prior/EARLY_MLP_STATE_COMPLETE_COMPILER_V21_FINAL_FINDINGS.md](evidence/prior/EARLY_MLP_STATE_COMPLETE_COMPILER_V21_FINAL_FINDINGS.md)

## Parent audit and native pilot

See [PARENT_CONTRACTS.md](PARENT_CONTRACTS.md) and [NATIVE_MLP2_RESPONSE_PILOT.md](NATIVE_MLP2_RESPONSE_PILOT.md).

- [evidence/prior/evaluate_mlp0_c512_mlp1_interchange_v1.py](evidence/prior/evaluate_mlp0_c512_mlp1_interchange_v1.py)
- [evidence/prior/evaluate_mlp0_c512_mlp2_compensation_v1.py](evidence/prior/evaluate_mlp0_c512_mlp2_compensation_v1.py)
- [evidence/prior/mlp0_c512_mlp2_compensation.py](evidence/prior/mlp0_c512_mlp2_compensation.py)
- [evidence/prior/early_mlp_state_complete_compiler_v21_final.py](evidence/prior/early_mlp_state_complete_compiler_v21_final.py)

## Native token-transfer follow-up

[EMLP-P2](MLP2_TOKEN_TRANSFER.md) reports the crossed noun/context panel and its local/future MLP2 response factorial.

## Direct MLP0→MLP1 response

[EMLP-P3](MLP1_RESPONSE.md) uses the same opened panel and controls to test MLP1 response.
