

## H411 — Fresh confirmation and cross-head transfer of fixed factor groups

Freeze the exact H410 native factor sets. Use512 fresh complete documents, independently rehashed against all7872 previous completed documents; zero overlap,8384 total. This is the same cached corpus/model, not a model-training holdout or a different text domain. Evaluate each of nine head-source MLP-path edits at unit amplitude, plus head6 at4× amplitude. No factor reselection, coefficient fitting or readout refitting. Direct residual remains fixed.

Score retained-response native KL recovery and baseline-preserving clamp remaining KL through native final RMS, unembedding, cap and softmax. Each ratio divides pooled KL by that source case's pooled native effect; source effect sizes differ and normalized recoveries do not compare absolute causal importance. Core64 is an anchor; all other sets have128 factors.

Preserve H410 head6 gates: recovery lower>.9, advantage over Fisher128 lower>.05, over individual-energy128 lower>.02, over mean core+random64 lower>.05. Four thousand paired document/random-order bootstrap draws yield a joint4 band for head6. Separately, a joint16 band covers residual128 recovery and residual128−Fisher128 recovery for each of the other eight heads. Transfer gates require all eight recovery lower bounds>.8 and all eight advantage lower bounds>.05. These are two separate confidence families, not one global20-statistic band. Amplitude4 and clamping remain descriptive.

|Head6 primary statistic|Estimate|Joint95% interval|
|---|---:|---|
|head6_residual128_unit_recovery|0.811105|[0.7889533804764334, 0.8332556524503733]|
|head6_residual128_minus_fisher128|0.110332|[0.0881807424324896, 0.13248301440642948]|
|head6_residual128_minus_energy128|0.029631|[0.0074799994782135115, 0.05178227145215339]|
|head6_residual128_minus_core_random128|0.151797|[0.12964616454030167, 0.17394843651424155]|

Gates: {'document_disjointness': True, 'recovery90': False, 'over_fisher05': True, 'over_energy02': False, 'over_random05': True, 'all_other_heads80': False, 'all_other_heads_over_fisher05': False}.

|Head|Native effect KL|Residual128 recovery|Fisher128 recovery|Energy128 recovery|Residual clamp fraction|
|---|---:|---:|---:|---:|---:|
|0|0.00025098|0.489253|0.442442|0.458033|0.509226|
|1|0.00123541|0.677307|0.607549|0.661610|0.319706|
|2|0.00068443|0.546523|0.472760|0.488442|0.451801|
|3|0.00022596|0.665927|0.480367|0.373119|0.332806|
|4|0.00048113|0.508205|0.061889|0.576365|0.500399|
|5|0.00002713|0.406987|0.380129|0.365545|0.593178|
|6|0.00195622|0.811105|0.700773|0.781473|0.188264|
|7|0.00005984|0.607940|0.570606|0.569931|0.392071|
|8|0.00033332|0.842506|0.828501|0.845152|0.156629|

|Other head|Residual128 recovery joint16 interval|Residual−Fisher improvement joint16 interval|
|---|---|---|
|0|[0.40375243585283865, 0.5747531525208899]|[-0.03868988607174355, 0.13231083059630772]|
|1|[0.5918065383216228, 0.7628072549896739]|[-0.01574287017018132, 0.15525784649786994]|
|2|[0.4610227361862063, 0.6320234528542575]|[-0.01173696052284133, 0.15926375614520993]|
|3|[0.5804270071488344, 0.7514277238168856]|[0.10006001940321448, 0.27106073607126574]|
|4|[0.4227042706134317, 0.5937049872814829]|[0.360814790463018, 0.5318155071310693]|
|5|[0.3214870050919751, 0.49248772176002636]|[-0.0586415146799079, 0.11235920198814335]|
|7|[0.5224399747564555, 0.6934406914245067]|[-0.04816602968169874, 0.12283468698635251]|
|8|[0.7570056062905045, 0.9280063229585557]|[-0.0714949078946542, 0.09950580877339706]|

Audits: {'direct_clamp_state_absolute_error': 7.62939453125e-06}. Independent arrays: {'mean_metrics_replay': True, 'transfer_statistics_replay': True, 'all_site_KLs_finite': True}. Complete-document audit: {'independently_rehashed_excluded_documents': 7872, 'fresh_unique_documents': 512, 'overlap': 0, 'completed_unique_total': 8384, 'all_windows_match_source': True}. All per-site scores, bootstrap samples, manifest and exact source hashes are retained.

Outcome: fresh head6 confirmation reproduces the main H410 advantage. Residual128 unit recovery is0.811105,[0.788953,0.833256]. Its advantage over Fisher128 is0.110332,[0.088181,0.132483], and over mean core+random64 is0.151797,[0.129646,0.173948]; both frozen5-point superiority gates pass. The90% recovery gate fails. Advantage over individual-energy128 is0.029631,[0.007480,0.051782]: this interval is positive, but its lower bound does not clear the predeclared2-point threshold, so that gate still fails. Report the positive comparison without silently relaxing the gate.

Head6 native unit effect KL is0.00195622. At4× amplitude, residual128 recovery is0.789565 versus Fisher1280.685149 and energy1280.764176; these stress comparisons are descriptive. Unit clamp remaining fraction is0.188264, with no new clamp-threshold confirmation claimed. Exact groups, primary thresholds, and native vectors remained frozen from H410.

Transfer is heterogeneous. Residual128 recovery for the eight other heads ranges0.406987–0.842506. Head8 has the highest point recovery,0.842506,[0.757006,0.928006] in the joint16 band. Head5 has0.406987,[0.321487,0.492488]. Thus the group is not uniquely head6-specific, but it does not transfer at uniformly high fidelity. Both all-eight transfer gates fail. Six other heads (0,1,2,3,4,7) have simultaneous recovery upper bounds below.8, and head5 is below.5; head8's interval crosses.8.

Residual additions improve over Fisher128 at every other head in the point estimates, but simultaneous intervals clear the predeclared5-point advantage only for head3 (gain0.185560,[0.100060,0.271061]) and head4 (gain0.446315,[0.360815,0.531816]). The other six intervals include zero. This is a family-controlled transfer pattern, not evidence of a uniform improvement. For head4, individual-energy128 has descriptively higher recovery0.576365 than residual1280.508205; the raw-energy comparator remains important.

Absolute effects differ substantially: head5 unit KL is0.00002713, head7 0.00005984, head6 0.00195622, and head8 0.00033332. Recovery fractions should not be read as shares of total model computation. These are MLP-path source interventions in one layer/model and one text distribution, not complete head ablations, semantic labels or a globally compressed model.

Interpretation: a fixed native group can transfer across documents and partly across sources, but its usefulness has a structured source dependence. Compensation-aware selection corrects a particular attribution failure without establishing an elementary semantic unit. The striking head3/head4 rescue suggests the added64 factors may supply different roles for different sources: compensating the core for head6, while carrying a larger direct share elsewhere. That last explanation is a hypothesis, not established by the current KL tables.

Next highest-information diagnostic: separate the fixed Fisher64 core and the fixed64 residual-selected additions across all nine sources, without selecting any new factors. Measure each component's native Fisher energy, its mutual cross term, and the omitted remainder; pair this with finite retained prediction/clamping for core-only, additions-only and their union. This can distinguish a source-dependent compensator from a generally useful standalone component and directly test the head3/head4 interpretation. Predeclare simultaneous bands across all nine sources. Then, if these group roles prove structured, construct the broader source-by-independently-selected-group transfer matrix; do not jump from pooled fidelity to semantic names.

The broad feature/computation-unit goal remains open. There are now8384 distinct completed-panel documents; every future fresh panel must exclude all8384. No experiment processes remain running.

### Twenty hypotheses revisited

These are qualitative priorities, not posterior probabilities; several mechanisms can coexist. Numbers retain the earlier H404 tournament mapping.

|#|Explanation|Current standing|Next discriminating information|
|---|---|---|---|
|1|Learned sources align with high-sensitivity local MLP directions.|Supported by geometry-controlled earlier confirmations.|Relate group transfer to derivative-direction overlap.|
|2|Low coupling is solely head6 dominating a summed denominator.|Weakened by separate-source normalization.|Check other source pairs without denominator pooling.|
|3|A small native-factor group carries much of a source response.|Supported in a restricted sense; high-fidelity90% claim remains unestablished.|Fit equal-budget groups independently per source and freeze them.|
|4|The remaining native-output response is broadly distributed.|Still viable; current selection failures do not prove irreducibility.|Measure held-out support curves with an explicit reconstruction objective.|
|5|Compensating factor combinations matter more than individual scores.|Supported by H409 accounting and H410 targeted rescue.|Test whether compensation patterns recur across source-selected groups.|
|6|RMS normalization alone creates the grouping pattern.|Unresolved as a component, insufficient as a universal account.|Compare fixed-scale bilinear versus normalization contributions within the same groups.|
|7|Input norms and pairwise angles explain all learned-source effects.|Contradicted by earlier full-Gram rotations.|Repeat the source/group transfer matrix under matched rotations.|
|8|Generic high-dimensional geometry explains ordinary additivity.|Supported by earlier actual/control comparisons.|Separate generic additivity from learned group specificity.|
|9|A few near-zero denominators create the local coupling result.|Weakened by independent denominator/influence audits.|Keep source effect magnitudes visible alongside normalized fidelity.|
|10|The selected-group advantage is only a discovery-panel artifact.|Addressed directly by this fresh confirmation, within one corpus.|Cross-domain confirmation remains open.|
|11|Head6 has special support that does not generalize uniformly.|A leading scope explanation; transfer must be assessed per source.|Build an equal-budget source-by-selected-group transfer matrix.|
|12|Output-direction alignment, rather than unique factor identities, explains useful support.|Still viable; energy ranking remains competitive.|Compare native group membership overlap with output-vector span overlap.|
|13|Different contexts select different useful groups.|Unresolved by fixed-group pooled fidelity.|Train context partitions only on discovery, then test incremental transfer value.|
|14|One compact operator transfers across sources/documents.|Partial prediction transfer is insufficient to establish this.|Test shared versus source-specific groups at matched total parameter budgets.|
|15|Compact single-source formulas reflect architecture, not discovered features.|Supported by exact rational response identities.|Require shared parameters across contexts rather than counting local coefficients.|
|16|Native output geometry determines which residual errors matter.|Supported; replacing raw scores alone was insufficient.|Preserve native KL tests when assessing any new group objective.|
|17|Local behavior remains accurate at large edits.|Not universal; larger-edit gates have failed.|Retain amplitude stress tests rather than inferring finite behavior from derivatives.|
|18|Weak pairwise coupling extends to many simultaneous sources.|Untested by these single-source groups.|Test group reuse and interaction errors under fixed multi-source edits.|
|19|The recovered groups correspond to semantic features.|Still untested.|Only after stable causal reuse, test controlled linguistic contrasts against matched controls.|
|20|The useful unit is a context-conditioned response/curvature operator.|Supported as a local predictive description, not yet an identified reusable primitive.|Test source-group transfer, compact shared parameterization and basis stability.|
