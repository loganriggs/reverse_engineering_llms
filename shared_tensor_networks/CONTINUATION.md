# Toy research continuation — active, 2026-10-29

Goal: complete the toy experiments, scale them as justified, and derive/test interesting theoretical results. The prior native-MLP research remains paused. This project is separate and must not be redirected automatically into the old featurizer loop.

Previous goal turn: progress (gauge controls). Current turn: progress (24 audited readout-recovery fits and restricted scalar-identification argument). Goal remains active.

## Current: two-feature readout recovery COMPLETE

run_readout_recovery.py, audit_readout_recovery.py, summarize_readout_recovery.py: d4,h2, exactGaussianquarticcoefficient loss; three teacher seeds215001–3, twoinit216001–2; one/two/threeindependentreadouts orbasehead+2knowncomponentablations. SciPyTRFanalyticJac,max200evals,fixedtolerances1e-12,norestarts. Fourwholeheldoutreadouts excluded from representation fitting, thenfrozenproductfeatures32/128labelOLS and exactpopulationprojectionfloor. All24audited: directforwardtorchJac vs coefficientderivative4.94e-16,independentproductmap/moments5.55e-17,81nodequadraturecomponent/subspace,transfernormalequations/errors. Allcheckpointshashesrecorded.

Exactfunction/componentrecovery counts one4/6,two3/6,three5/6,interventions5/6. Failuresretained(includingbudgetstops);notmonotonebenefitormanyindependentteachers. Interventionarmprivilegedknownnodecorrespondence,19paramsvs25threeheads. Everynear-exactfit recoveredfeatures andtransferrednearlyexactly. READOUT_RECOVERY_RESULTS.md includesallfailedfittransfererrors,notjustmedians.

Important correction: TWO_FEATURE_SCALAR_IDENTIFICATION.md derives generic scalaridentification for2bilinearfeatures with4independentfactors,nonzerodiagonalreadoutcoefficients,nonzerodiscriminant. Overcomplex,scalarquarticfactorsintotwodistinctirreduciblerank4quadratics;uniquefactorizationidentifiespencil;onlyrank2pencillinesareoriginalfeatures. ExactSymPycheck_two_feature_identification.py determinant/ranks/nonorthogonaltransform passes;auditorverifiesteacherassumptions. Restrictedargument,notnoveltyclaimorstability/globaloptimizationtheorem. Earlierd3h3ambiguityviolatesassumptions.

Nextscaleinformationcontroltod6h3,1/3/6independentheadspluslabeledablations,wholeheldoutheads. Sixfullproductspanheads haveexistingidentifiabilitytheorem;twofeaturefactorizationproofdoesnotautomaticallyextend. Do notclaim scalarinformationinsufficiencyfromoptimizerfailure. No livejobs;nativeMLPprojectpaused.

## Current: function-preserving gauge control COMPLETE

run_bilinear_gauge_balance.py / audit_bilinear_gauge_balance.py / summarize_bilinear_gauge_balance.py completed 54 fits: archived18 fixtures times original/initial/every balancing, fixedGN damping.001 and100Jacobian cap. All original trajectories reproduce priorGN exactly. Balance A/B,AR/BR,C/D reciprocal row norms; verify every raw/balanced pair by independent outer products and polynomial coefficients. Rawprebalance steps auditednormal equations,Armijo,sampledautograd,allstateGaussianmoments. One line-search failure retained/reproduced: untied211003_tied_raw_every iteration98 MSE1.9970e-5, all12trialsrejected. No retry.

Initial balancing 9wins6toleranceties3losses; every8wins6ties4losses vs original across18. Tiedraw median2.09e-5 ->initial5.49e-8; corrected4.38e-4 ->4.27e-7; notuniversal. Initial nonzeroJacobiancondition improves12/18 (repeated fixtures,notindependentdraws). Function-preserving metric intervention,not coordinate-invariantGN. Report BILINEAR_GAUGE_BALANCE_RESULTS.md contains norm-minimization proof and limits. No trainingjobs live. Close bounded optimizer geometry panels; next E3 multiple-readout/intervention supervision plus held-outtransfer to address hidden-recovery failure, not more rescue sweeps. NativeMLPproject remains paused.

## Current: block Gauss–Newton complete

run_bilinear_block_gauss_newton.py, audit_bilinear_block_gauss_newton.py, summarize_bilinear_block_gauss_newton.py completed. 18 archived teacher/student/metric fixtures times full/forward/reverse, damping .001, maximum100 full-Jacobian calls. Each block iteration computes full Jacobian/Gram but solves restricted system; no optimized block kernel or matched wall-time claim. All18full trajectories reproduce priorGNexactly. All54 finite, no failedline searches; all states coefficient-moment audited, restricted solves/Armijo/unchanged coordinates checked, sampledJacobiansautogradverified. AuditmaxJ7.31e-16,linear2.54e-10,moment1.45e-14.

Untiedstudent full/forward/reverse near-exact6/5/6 of6, medianJcalls17/44.5/40.5. Across18pairs forward4wins5tolerance-ties9losses; reverse4wins6ties8losses. Some corrected-tied fits improve, no general blockadvantage. BILINEAR_BLOCK_GAUSS_NEWTON_RESULTS.md includes local quadratic-model dominance proof and nonlinear limitations. Close bounded order panel. Next controlled gauge balancing or E3 multiple-readout/intervention recovery; do not optimize indefinitely to rescue tiedfits. No trainingjobs live. NativeMLPproject stayspaused.

## Current: five-seed SVHN confirmation COMPLETE

All 30 fits and independent audits complete. summarize_svhn_confirmation.py requires all audits and verifies result/checkpoint hashes before aggregating. Results: standard accuracy narrow stack/tree/wide stack 82.07/83.20/83.05%; extra data 87.26/88.96/89.74%. Extra-data wide stack wins accuracy over tree in 5/5 seeds, also 5/5 at matched 8010 updates; tree wins cross-entropy in 5/5 (.4089 vs .4874). Parameter budgets 201024/792896/796080. All five seeds on one split; not dataset-level replication. Equal-width stack/tree initial functions match, wider stack differs. Development seed excluded. No official test used.

Course correction: do not generalize the earlier single-seed reversal into an intrinsic untying advantage. Parameter allocation and evaluation metric materially change the conclusion. Close the fixed image panel; no further seed or hyperparameter rescue. Next B priority is block curvature-aware optimization and gauge balancing on archived exact-population bilinear tasks, with work accounting and paired initial functions. Recovery remains a separate E3 target requiring multiple readouts/interventions and held-out transfer. Do not equate better fitting with recovered hidden computation. No new training launched this turn. Earlier LIVE entries below are historical and superseded by this terminal status.

## Current hidden-recovery assessment complete; final confirmation fits LIVE

Previous goal turn: progress (exactrankcertificate). Current turn: progress. assess_bilinear_hidden_recovery.py assessedall36GNendpoints withoutretraining. Eight near-exactfunctionfits(MSE<1e-12) allfail fullteacherbranchsubspace recovery evenallowingGLbasis changes andbranchswap; worstangles42.7–89.5degrees. Projectingstudentfeatures into matchedteacherbranchspaces andkeepingreadoutfixed changesoutputMSE.00722–1.8133inselected8. All36andinitialbaselines retained; coefficientGram/quadrature projections/patchlosses agree; end-to-endGLinvariancechecked. Teacher-informed evaluation,nottrainingsupervision,notallpossiblecomputationalequivalences. BILINEAR_HIDDEN_RECOVERY_RESULTS.md andfigure.

check_hidden_nonidentifiability.py symbolicallyproves equal18parameterdepth2networks computingsame(x²+y²)² withh=(x²,y²) vs h=(x²-y²,2xy). Eachspanrank2,joint3; notlinearbasisequivalent. ActualNetworkforwardchecks pass. Ablationsdiffer butnotclaimedcanonicallymatched. Illustrativesymmetrictarget,notgeneric theorem. This shows outputfitcannotuniversallyidentify intermediates; futureE3 needs additionalreadouts/interventions andheld-outtransfer,explicitlycounting supervision.

LIVE SVHNsession74676,parent181303; latestseed212005_extra_tree training child183993,28fitsclosedandaudited. Revalidateactualsession/process; doNOTduplicate. Finishall30andrun summarize_svhn_confirmation.py beforeaggregateclaim. Fixedrecipeunchanged. Onlyclosedartifactscommitted;mutablelivefilesexcludedfrommanifest. Noothertrainingjobs. Overallgoalactive.

## Current exact rank certificate complete; five-seed panel LIVE

Previous goal turn: progress (auditedGNfits,tangentdiagnostics). Current turn: progress. certify_bilinear_tangent_rank.py certifiesexactrank15at all6paired archivedinitializations for d3h3q3depth2model. Fifteenintegerprobes,dyadicrationalweights,prime2147483647(trialdivisionverified),nonzero15x15minors. Independentexactcentraldifferences h1/h2verifyeveryJacobianentry becauseindividualparameterdegree<=2. Results/bilinear_tangent_certificate matrices/pivots/inputhashes and BILINEAR_TANGENT_CERTIFICATE.md retained. No floatrankthreshold innewcertificate.

TheorysectionG: localcoefficientsubmersions,regularfixed-functionfiberdim24tied/42untied,fulloutputtangentspaceboth. Under specifiedrealizableGaussianregression atcertifiedsharedoutput andsmoothpositivepriorcompactregularpatch,localRLCT15/2,multiplicity1both. DoesNOTcertifyglobalRLCT,globalrepresentability,trainedteacherrecoveryorRMSNormclassification. DiscretepermutationsdoNOTaddlocaldimensions;scalinggaugesincluded. Elementaryregular-coordinateproof,notnewgeneralSLTclaim.

LIVE SVHNsession74676,parent181303. Latestseed212003_extra_wide_stack training child183249;17fitsclosed/audited. Revalidateactualsession/process;doNOTduplicateorredownload. Continue fixed30-fit panel and independentaudits; all30required for summarize_svhn_confirmation.py. Commitonlyclosedcases,exclude mutableliveartifacts frommanifest. No othertrainingjobs launched. Overallgoalactive,remainingconfirmation/curvature/circuitrecovery worknotcomplete.

## Current Gauss–Newton control complete; confirmation remains LIVE

Previous goal turn: progress (launched confirmation and checkedmetricdamping). Current turn: progress. run_bilinear_gauss_newton.py completed36fits on exactly archived actualbilinear tasks, fixed damping.001/.1,rawI/correctedM,Armijobacktracking,max100steps. All36 improvepaired fixedGD baseline but notcompute-matched. Untiedstudents atlambda.001 reachMSE<1e-12 all6targets in7–71steps; tiedstudents retainmetric/init-sensitive errors. No failedlinesearch. Auditor53894terminalpassed: everytrajectoryloss coefficient-moment replay, linearsystem/Armijochecks, initial/mid/finaltorch.funcjacrev match; maxJerror7.31e-16,linearres2.55e-10,momenterror1.45e-14. BILINEAR_GAUSS_NEWTON_RESULTS.md. No circuit-recovery claim or global convergence.

assess_gauss_newton_tangents.py: all12initialteacher/studentfixtures numericallyrank15 andpairedoutputtangentprojectorsmatch; undampedlinearizedresidualfit succeeds. Rankfloating-point threshold1e-10,notexactsymboliccertificate. Fullquarticspace locally available toboth; metric/conditioning/nonlineartrajectory explanations remain. Avoid describing this tinycase as simply missing first-order output directions.

LIVE SVHN session74676,parent181303. Current configurationseed212002_extra_wide_stack,phasetraining,childPID182423;11closedfits independentlyaudited. Revalidateactualsession/process,doNOTduplicate. All30fit/audit panel mustfinish before aggregateconclusion. Fixedseed/data/capacityrecipeunchanged. Publishclosedcasesonly; liveartifacts excludedfromcommittedmanifest. Next finishconfirmationpairedsummary; then decide separatelybudgeted curvature/circuitrecovery experiment without dampingrescue sweeps. Overallgoalactive.

## Current five-seed data/capacity confirmation is LIVE

Previous goal turn: progress (actual bilinear alternating-gradient panel completed). Current turn: progress. Implemented run_svhn_confirmation.py, audit_svhn_confirmation.py, sequential run_svhn_confirmation_panel.py and all30-required summarize_svhn_confirmation.py. Five fresh seeds212001–212005; depth3standard/extra; width32stack, width32tree, width120stack chosen by parameter count (796080vs792896tree). Same fixed optimizer/data recipe,8010-update checkpoints,full30epochs. Wider-stack CPU/CUDAcheck passed relativeforward1.11e-6,gradient1.22e-6. All initial training-probe output/gradient statistics independently replayed in audit, and equalwidthinitialfunctions checked. No seed selection or pooling development seed.

Live session74676; parent PID181303. panel_status.json and closed per-case logs in results/svhn_confirmation; verify actual session/process before trusting status or restarting. Latest observed configuration: seed212001_extra_tree; phase training; 181555 childPID. 4 cases audited so far. First fresh standardseed212001 stack/tree/widestack79.48/83.02/83.66%; illustrates seed variation, NOT completedpanel evidence. Stop only after all30audits and pairedsummary; no recipe changes based on interim ranking. If subprocess fails, preservefailureandinspect; do not duplicate a livejob. GPU reserved for this sequentialpanel; no othertraining launched.

While GPU runs, check_metric_gauss_newton.py derived/verified occurrence-metric damping equivalence and linear-coordinate covariance on actual-network Jacobians:9checks,maxrelativeerror<9.26e-12. Isotropic distinct-coordinate damping changes local prediction steps1–13% infixtures. SectionF ofSHARING_COMPLEXITY_AND_FLOW.md statesproof andscope. No curvatureoptimizer training has run. This is a control for the next B experiment, not a convergence claim.

Next: finish panel/audits, run summarize_svhn_confirmation.py, publish completedresults and pairedseed comparisons. Then choose a bounded curvature-aware training comparison or circuit-recovery extension based on evidence. Overall goal remains active. Committed manifest excludes mutable live-run artifacts until terminal.

## Latest actual-bilinear alternating-gradient panel complete

Current goal turn: progress. run_bilinear_alternation.py and _long.py completed108fits each (sessions5718,33104terminal). Depth2,d3,h3,q3; tied/untied teachers;3paired teacher/init seeds; tiedraw/tiedcorrected/untiedraw students; joint/forward/reverse layer blocks;lr.01/.003,per-blocktime3/30. Independent A/B features, untied occurrences copied for same initialfunctions, exact Gaussian population objective, frozenRMSscales. No RMSNorm/clipping/momentum. All216finite final losses independently audited with coefficient moments; analyticgradients and3sweepupdates agreeautograd at~1e-15. Short prefixes deterministically reproduced. No general convergence/recovery claim.

BILINEAR_ALTERNATION_RESULTS.md: naive layer alternation has small effects for many cases and large order-sensitive effects on one difficult raw-tied case. Atmatchedfullgradientevaluations (jointtime30 vs alternatingtime10), forward/reverse beatjoint0/36pairs each. Alternatingimplementation computes3fullgradients/sweep vsjoint1; not an optimal specializedblockimplementation or exactFLOPclock. Stop this bounded naive-alternation panel; do not repeatedly extend budgets to findawinner.

check_alternating_projection.py proves/checks metriccorrected/projectedSGD equivalence survives block schedules selecting all shared occurrences;9cases×20sweeps,maxparametererror6.67e-16. Reciprocalrowgauge gradientidentity and exact finiteGDnormdrift residual<1.8e-15. Theory note appended proof. All producer/auditor/checker sessions terminal, no live jobs. Next: prespecified five-seed SVHN data/capacity confirmation and/or a conditioning-motivated damped Gauss-Newton comparison with explicit work controls; learned circuit recovery remains open. Overall goal active.

## Latest completed dynamics and image controls

Current goal turn: progress. All depth4fits/audits complete (producer36633terminal): standard stack/tree84.52/80.40%; extra matched8010updates83.74/86.80%; extra30epochs87.36/89.00%. SVHN_GPU_RESULTS_DEPTH4.md records one-seed evidence.

Long-standard depth3stack passed in session47654, but tree launch crashed SIGSEGV before status. Failure retained in runtime_failure.json. Same scripts retried core0 in session50094: training+audit terminalsuccess. At70260updates standard stack/tree83.22/82.88% versusextra87.34/89.46%; standardtrain96.32/99.26%, validationCE.9667/1.6498. SVHN_LONG_STANDARD_RESULTS.md. No optimization conclusions from nativecrash; rootcauseunknown. Full-update control does not explain extra-data gains; distribution/capacity/seed confounds remain.

User-requested alternating dynamics:36exact-population quartic gradient-step fits plus18exact frozen-block-flow fits completed and audited. Identical archived symmetric four-factor initialization,3teachers,rates/dwells.03/.01/.003,per-blocktime30. Cyclic updates split factors but no strict recovery in54fits. Median MSE at.03 simultaneous.405501,cyclicgradient.404284,exactblockflow.404335; effects shrink withstep/dwell. Thus finite switching itself breaks symmetry, not solely Euler error, but no practical recovery demonstrated. Five-sweep autograd/expm audits and final coefficient-moment replay pass. DOP853simultaneous reference agreesacross2tolerances<1e-8. ALTERNATING_DYNAMICS_RESULTS.md; sources run_/audit_alternating_quartic[_flow].py. Producer7970/37280terminal; auditors passed;43935terminalsuccess. No further training live.

Next meaningful work: five-fresh-seed depth3data/capacity confirmation perSVHN_EXPERIMENT_PLAN.md, plus actual bilinear layerwise alternation with independent initialization and matched work/step-size panels. Do not extend this symmetric-product toy indefinitely to rescue a desired outcome. Broad theory/circuit-recovery/scaling goal remains active.

## Current continuation: full-update control queued, depth4 completing

Previous status-answer turn was a verified wait on live session36633. Current turn made progress: implemented run_svhn_long_standard.py and audit_svhn_long_standard.py, restoring audited standard GPU model+Adam states from8010updates and continuing to70260updates (263epochs+39batches). Source reviewed against original runner; compile check passed. Depth3stack/tree long controls queued in live session47654, which waits for all four depth4 audits, then runs/audits sequentially. Do not duplicate. Revalidate36633 and47654; depth4extra tree finished training and audit was running at latest check. Pending results remain unpublished until verified.

User asks all B results and extensions including alternating gradient flow. SHARING_COMPLEXITY_AND_FLOW.md section E now records cyclic-gradient/continuous-flow/ALS distinctions, second-order block-order expansion, and exact reciprocal-gauge norm-balance invariant. Next targeted dynamics study is tiny exact-population quartics with paired initial functions, simultaneous versus cyclic/reverse/random blocks, step-size and work controls, then informative image extensions. Do not treat matrix-factorization convergence theorems as applying unchanged to RMSNorm bilinear networks. Existing long-image control continues independently.

## Latest depth-3 comparison and normalization theory

All four depth3 GPU fits and independent full-training/validation audits passed. Standard stack/tree final84.30/83.34%; extra stack/tree matched8010updates84.28/86.18%, final30epochs87.34/89.46%. SVHN_GPU_RESULTS_DEPTH3.md reports one-seed results. Extra data reverses the standard-data ranking here; tree has792896parameters vs201024stack, so no isolated sharing benefit or universal claim.

LIVE sequential producer session36633 (parent Python PID177521): runs depth3 then depth4, standard/extra × stack/tree, each followed by independent audit. All depth3 terminalsuccess; depth4 began and is live. Logs results/svhn_gpu/{standard,extra}_d{3,4}_{stack,tree}_{run,audit}.log. Revalidate session/process before resuming; do not duplicate. Existing GPU runner recovers at complete epochs; 8010-update snapshots saved separately. Next finish/audit depth4, summarize with summarize_svhn_gpu.py --depth 4, then replicate seeds and assess longer-update standard-data control as needed.

Theory: SHARING_COMPLEXITY_AND_FLOW.md sections C/D derive RMS Jacobian radial/tangential eigenvalues and prove zero-epsilon intermediate-normalization cancellation for bias-free bilinear DAGs with nonzero intermediates. check_rms_geometry.py:12float64 differential/eigenspace checks pass; descriptive local condition measurements on512validation examples/model. check_rms_collapse.py:6depth2–4 stack/tree output+gradient equivalence fixtures pass. All5000validation examples in each of4trained depth2 models show0predictionchanges between every-node/root-only finite-epsilon evaluation; relative logit differences3.26e-8–2.36e-7. Explicit scalar small-input counterexample prevents claiming exact finite-epsilon equivalence. These are elementary results, not new rank theory or evidence of causal optimization advantage. Sessions39853,23155,45872 terminalsuccess. Theory and completed depth3 artifacts publishable; live depth4 artifacts must not be snapshotted into a supposedly final manifest.

## Latest GPU SVHN depth-2 comparison

Current goal turn: progress. Previous answer was a status-only recap (no progress); this turn implemented and completed four GPU fits with run_svhn_gpu.py: standard/extra × stack/equal-width tree, depth2,width32,RMSNorm,float32,TF32off. Fixed standard split/normalization; extra uses599388train. Standard30epochs=8010updates, extra30epochs=70260; extra8010-update checkpoint retained. Final validation stack/tree standard84.54/84.54%, extra87.24/88.50%; extra matched-update84.24/85.70%. One seed, development validation only, extra distribution differs, no causal sample-size or universal architecture claim. Standardstack producer13763 and sequentialremaining producer39058 terminalsuccess. Audits standardstack57894,standardtree17138,extrastack83353 passed. Extra tree audit78058 terminalpassed. All four full train/validation replays passed; maximum relative NumPy logit error2.587e-6. SVHN_GPU_RESULTS.md and results/svhn_gpu/summary.json report verified outcomes. No GPU training left live. Next extend same paired controls to depths3/4, then replication. Overall theory/circuit-recovery/scaling goal remains active, not complete.

## Latest allocation control complete; all SVHN data verified

Currentgoalturn: progress. NONUNIFORM_BUDGET_RESULTS.md:3CPUfits withleafwidth16/8/4,upperwidth32,nearstackparameterbudgets;30epochsstandardSVHNsameprotocol. Val83.56/78.26/77.98%, versusuniformbudget82.76/79.96/65.78%. Depth4gain12.20points yet6.60belowstack. One seed, differentinitialfunctions; notcausalproofthatsharingexplainsremaininggap. Nonuniformclass exactuniformreduction6cases,storage/countchecks,finite-differencegradcheck,NumPyforward pass. Independentfulltrain/val/logit auditsall3pass. CPU/CUDAfloat32gradientcheckmax6.258e-6. Allproducer/auditorhandles82664/57825/82358/96038/18724/46814terminal. No livejob/GPUtraining.

SVHNextradownload51470terminalsuccess; referenceMD5a93ce644f1a588dc4d68dda5feec44a7 passed,531131NCHWimages. prepareextra28298terminalsuccess;fullarraybyteequalitychecks andSHA256saved. Metadata copiedresults/svhn_data; largearchives/npyoutside repo in/home/loganriggs/.cache/shared_tensor_networks/svhn. Bothstandard+extra ready; pool599388train plusfixed5000standardval. DoNOTredownload. Nextimplement pairedstandard-onlyGPUandstandard+extraGPUdepth2stage (sameprecision,TF32off), checkpointat8010updates forcomputecontrol plusfull30epochs; thenextenddepth3/4asplanned. RecheckGPUavailability, avoidcollidingwitholdproject. CurrentGPUreadiness tests passed; noGPUtrainingsweephasrun. Goalactive,allocationdetourclosed.

## Latest standard SVHN complete; extra download remains live

Currentgoalturn: progress. SVHN_STANDARD_RESULTS.md:10CPUfits,68,257train/5,000val,30epochs,RMSNorm,onefreshseed; all independentNumPytrain/val/logit audits andhistogrampreprocessingchecks passed. Producers46368/92202/93276and auditors19499/27896/21323terminal. Stackvaldepth2/3/4=84.86/83.94/84.58%; equalwidthtree84.92/83.30/80.16%; uniformbudgettree82.76/79.96/65.78%;MLP82.84%. Single-seeddevelopment, noofficialtest. Treefits trainbetter butgeneralizesworse deep; uniformbudgetdepth4underfits. Plan nonuniformleafwidth16/8/4 withupperwidth32 toseparateparameterbudget fromoutputbottleneck (notimplemented/trainedyet).

RTX5080briefCPU/CUDAcheck22949terminalpassed,TF32off,float32maxrelativegrad2.913e-6,float64~4e-15. NoGPUtraining yet; read-onlyGPUcheck showedonlyourCPUprocesscontexts,nocompetingcompute then; recheckbeforeuse. Extra-data GPUruns needstandard-onlyGPUcontrol, because initial agreement isnotAdamtrajectoryequivalence.

SVHNextra downloadstilllive handle51470,PID174791,/tmp/svhn_mirror.log, externalcacheextra_32x32.download (~819MiBobserved, notcomplete/verified). DoNOTduplicate; revalidatehandle. Whencomplete runprepare_svhn_cache.py --splits extra, verifychecksum andcopyprovenance. Standardarraysalreadyverified. Next nonuniformbudgetcontrol and/or pairedstandard/extraGPUdepth2stage, thenextenddepth3/4; matchoptimizerupdatecheckpoint aswellasepochs. Overallgoalactive.

## Latest image replication complete; SVHN download live

Currentgoalturn: progress. FASHION_REPLICATION_RESULTS.md:30freshseedfullpoolfits(190001/2/3),55k/5k,30epochs,10configs; all independentlyaudited fortrain/validation/logits. Producers14557/4762/44260and auditors2650/56959/7696terminal. Meanaccuracytree-minusstack+.47/.63/.58points depth2/3/4 withmoreparams; meanCEworseall3depths. Budgettree-depth4 variable/weaker. NoGPU/testevaluation. Originalseednotpooled.

SVHN: officialhoststillrefused; UCSDmirror trainfile passedreferenceMD5 andSHA256. Train73257NCHWuint8 cached+memorymappedarrays byte-equalitychecked byprepare_svhn_cache.py; provenancecopiedresults/svhn_data. Extra531131file downloading, NOTyetverified. Livehandle51470,PID174791 atlaunch,/tmp/svhn_mirror.log, externalcache/home/loganriggs/.cache/shared_tensor_networks/svhn/extra_32x32.download. Revalidatehandle/process beforeanything; doNOTduplicate download ortrustpartialfile. Itmaytake severalminutes. NoSVHNtrainingyet. NextimplementSVHN_EXPERIMENT_PLAN.md standard68257train/5000val fullpooltraining; extraregime599388 afterchecksum. Keepnormalizefromstandardtrainfixed, reportmatched-updatecontrols before sampleefficiencyclaims. GenericCPU/pinnedcores used; no nativecrash inreplication, rootcauseunresolved. Goalactive.

## Latest ALS optimizer control complete — bounded detour closed

Currentgoalturn: progress. REUSE_ALS_RESULTS.md:54ALSfits, samearchivedfourfactorinitializations/data asSGD;100sweeps400exactLSblocks. Independent21,600blocknormal-equation/monotonicity checks and54populationreplays pass(maxnormalres4.534e-15). ALSimproves46/54pairedpopulationerrors. GenericnoiselessN512median~3.4e-30 vsSGD.0467,recovery3/3. Notcomputematched,nouniversalwin; noisyN32canoverfit.

Two1factorteacherN128noiselessplateaus remain:positive trainMSE.0181/.184, tinygradients,3explicitgaugezeros+9positivetransverseHessianeigenvalues(min.01165/.03281). Explicitnearzerolossfeasiblepoints. Numericalnongloballocalminimumcandidates, notrigorousintervalcertificates. Conditionalquotientsecondderivativetheorem documented. No livejob/GPU. Product-familyoptimizerdetour nowclosed; nextreturnbroaderimage seedreplication/largerdata pipeline. Rigorouslocalminimumcertificate remains explicit theoryfollowup, not an excuse for endless rescues. Goalactive/generalcircuitrecovery stillopen.

## Latest finite-data/noise reuse study complete

Currentgoalturn: progress. REUSE_SAMPLE_NOISE_RESULTS.md:162factorizedfits+54OLSbaselines, fresh3seeds, N32/128/512,noise0/.1, fixedquartic3D. Allfinite andaudited; exactmomentauditmaxrel1.732e-15, trainingobjectives+54normal-equationchecks pass. Producer90763terminal,no livejob/GPU. OLSnoiseless recovery numericalprecisionN32already; factorizedoptimizationpoorfor2/4factorfamilies. Modest1factormatchedsharingnoiseN512median.000421 vsOLS.000624, notbroadadvantage. Correct2/4factor strictrecovery0/3allN.

Exactgeneralizedeigenvalue empirical->population bounds verified81noiselessresiduals andindependentwhiteningSVD. AmplificationN32~944–5888,N512~4–6. ConditionalOLSnoise-risktrace formula derived/evaluated. Identifiability isnotstablelearning. Nextboundedcontrol: fullyuntied4factor alternatingexactLS on same data/init vsSGD; repeatedsharedfactors arenotlinear subproblems. Then concludeoptimizerconfound orretainfailure, notendlesslrsweep. Otherimage/SVHN/generalcircuitgoalsremainopen. Goalactive.

## Latest factor recovery and intervention assessment complete

Currentgoalturn: progress. FIXED_DEGREE_RECOVERY_RESULTS.md assessesall81fits with24permutation occurrence matching, invariant directions/amplitude andinitialbaselines. Generic4factor independentstudents recover3/3:worstangles2.304/2.837/.765degrees, amplitude.9826/1.0212/1.0011. Restrictedgrammarrecovery, notphysicalDAG. All12canonicalfactor-deletion errors6.41e-5–.004564; shiftedcovarianceerrors.003118/.0001097/.00003979. Independentquadratureaudit36comparisonsmaxrel1.257e-14. No newtraining/livejob/GPU.

Failures retained: correct2factorindependentstudent strictrecovery0/3; onefactor symmetric seed170001 MSE.940894 despite nearzero median fromother2. Do not implyuniversal success. Exact repeatedfactor perturbation fixture gives functionMSE=(3/35)delta^4 while directionangle~delta, proving ill-conditioning atfactorcollisions. Checks passed. Next meaningfulstep: finite-data/noise sampleefficiency with exactpopulation and factorrecoverymetrics, optionally separationcontrol. Broaderimagescaling/SVHN/generalcircuitrecovery stillopen. Goalactive.

## Latest fixed-degree reuse experiment complete

Current goal turn: progress. FIXED_DEGREE_REUSE_RESULTS.md:81fits of quartic3Dproducts, teacher/studentgroups1/2/4,3seeds,symmetric/.05perturbed/independentinitialization. ExactGaussianpopulationobjective via125Gauss-Hermitenodes;1000correctedSGDsteps. Independentcoefficientmomentauditall81passes; coefficient-gradientaudit81maxrel1.742e-15; exactrationalteacherfactor distinctness checks pass. Symmetrictrajectoryparameterdifference2.665e-15 acrossstudentgroups. Fourfactorteacher/fourfactorstudent medianMSEsym.4055,perturbed.3937,independent.000453. Notconvergedgeneralcomparison; independentinit matchesRMS notfunction. Restrictedproductfactorcomplexity1/2/4 certified byunique factorization, notgeneralminimalcircuitrank.

Producer13526terminal, auditscomplete, no livejob/GPU. Next: evaluate factorrecovery up to scaling/permutation beforeclaimingrecovery; perturbation-size control can test symmetry escape. Then finite-data/noise sample-efficiency with exactevaluation and wider scaling. Overallgoal active; image/SVHN and generalcircuitrecovery remainopen.

## Latest completed controls and exact reuse family

Current goal turn: progress. FASHION_FLOW_RATE_RESULTS.md:18new calibratedSGDfits, all audited, no failure. Rate chosen from initialtrainingprobe to equalize instantaneouslossdecrease, notoutputspeed;3disjointtrainingprobes assess transfer without tuning. Correctedmedianaccuracydepth2/3/4 now77.50/77.42/76.88%, versus73.70/67.02/60.58% atbase.01. Freenormalizedmedians78.64/76.20/74.56%. Finite-budget3seedcontrol, no robust architecture ranking. Producer21914/auditor70954terminal. No livejob/GPU.

REUSE_COMPLEXITY_AND_FLATNESS.md adds exact scalar repeated-squaring minimalmultiplicationD/formula2^D-1certificate, closed-formgradientflow and specifiedGaussianlikelihood localRLCT=1/2^(D+1). Six derivative/gradient/flow checks passed. Classical elementary results, notnovelty; changingdegree isexplicitconfound. NextA: fixed-degree multivariate repeatedvsdistinctfactor teachers with known sharing and matchedfunctionscale/metrics. Nextscaling: seed/sample-size controls, SVHN verifiedsource outstanding. Avoid unbounded one-epoch SGD tuning. Goalactive.

## Latest theory-to-image experiment: gradient-flow control complete

Current goal turn: progress. FASHION_FLOW_RESULTS.md:9cases×4arms=36fits,55ktraining,215steps,depth2/3/4,threefreshseeds,width16,float64SGDlr.01. Initial functions equal; correctedtied vsprojectedtree maximum recorded parametererror3.272e-13. Initial disagreement fractionsdepth2.363–.479,depth3.738–.776,depth4.838–.920. IndependentNumPy audit final36validationmetrics passes; finite-difference initial loss derivatives pass. Producer85540 and auditor57282terminal; no livejob. NoGPU/testevaluation. Sources/models/results saved. GenericCPU+pinnedcore0 run didnotcrash, not a definitive runtimefix.

Ordinarytied/free medianaccuracy near eachother atdepth3/4; corrected/projected substantiallyslower atsamebaselr. This is a finite-budget metric/control result, notconvergence/generalization evidence. Next bounded B control: independentlyfixed learning-rate/function-speed controls and multipleprobes; A certifiedreuse-complexity teacher experiments remainqueued. Scaling/seedreplication andSVHNloader provenance remainoutstanding. Goalactive.

## Explicit theory directions A and B — active

User requested reuse-aware rank/complexity and sharing gradient-flow proofs. SHARING_COMPLEXITY_AND_FLOW.md now defines grammar-dependent circuit/formula complexity and resource profiles, proves q^(2^L) small-DAG/large-formula/large-flattening separation, and derives tangent-kernel PSD decomposition and multiplicity bounds. These are classical/elementary results applied here, not novelty claims. check_sharing_complexity_flow.py passed exact rational moment/coefficient/rank fixtures and4actual-network Jacobian/kernel checks with/withoutRMSNorm. E2 existing20trajectories verified correctedSGD/projectedSGD equivalence. A did not previously have an implemented minimal-circuit measure.

Next theory experiments explicitly queued in note: A controlled reusable-polynomial teachers with degree/reuse controls and certified bounds; B image-probe disagreement gradients and ordinary/corrected/projected/freeSGD from equal functions, multipledepths/seeds. No large new sweep running. Do not let image accuracy replace these theoretical questions; both contribute to broadergoal alongside scaling. Exact minimum general circuit complexity and full circuit recovery remain unsolved here.

## Latest completed work: full-pool Fashion-MNIST

Current goal turn: progress. All10 full-pool classifiers complete at30epochs,55k training/5k validation; independent NumPy replay of train/validation metrics and saved logits passes. Producer34085 crashed SIGILL, resume43211 crashed SIGILL, generic26747 crashed SIGSEGV after budget-tree epoch16; pinned generic resume18656 completed. Final auditor32389 terminal passed. No active experiment. Sources/checkpoints/curves/crash traces saved; report FASHION_FULL_RESULTS.md and figures/fashion_full.png. Runtime root cause unresolved; genericCPU alone did not fix it. CPU affinity0 plus genericCPU completed, but that is not proof of a fix. NoGPU, officialtest unused.

Depth2/3/4 final stack accuracies88.50/88.90/89.04%; equalwidthtree89.12/89.16/89.06%; budgettree89.32/87.90/87.00%. Equalwidthtree trainaccuracy97–97.5%, stacks95.3–95.4%, little validation gain. Late validationCE mostly worsens while trainingCE improves. One seed only: no significance or universal ranking claim. Pilot vsfull changes bothdata andupdatecount, not an isolated samplecomplexity measurement.

Next: nested data-size/seed replication with explicit epoch and update budgets, preserving officialtest, and SVHN standard versusstandard+extra. svhn_data.py loader prepared with torchvision-reference checksums and layout/label assertions, but officialhost HTTP andHTTPS connections refused; noSVHN data downloaded or training started. Obtain a checksum-matching trusted mirror or restored officialsource; avoid arbitrary unverified data. Existing Fashiondata allows independent progress. Preserve synthetic frozen-hidden-feature transfer as deferred work. Overallgoal incomplete.

## Latest user direction: real-image occurrence-tree comparison

User requested Fashion-MNIST, then potentially SVHN/ImageNet, depths2–4 with RMSNorm as a controlled factor. Implemented vision_circuits.py and eight float64 correctness conditions (independent occurrence storage, function/input-gradient equality, summed occurrence gradients), all passed. Fashion-MNIST publisher checksums verified; cache outside repository.

Pilot: 2048 training / 5000 validation from official training split; official test unused. One seed, five epochs, 19 planned fits, shared stack / equal-width untied tree / narrower near-budget tree, normalization on/off, ReLU MLP baseline. Producer session76751 exited139 (segmentation fault); results/fashion_pilot/results.json now explicitly interrupted. Saved rows preserved. No producer remains live. Cause undiagnosed; independent replay audit not yet run. Do not present pilot as complete or sample-complexity evidence. Next: diagnose crash, independently audit saved rows, resume missing configurations without overwriting completed evidence, then freeze multi-seed/sample-size study. No GPU started.

The synthetic frozen-hidden-feature transfer experiment below is deferred in favor of this explicit user steering.

## Latest completed representative scaling sweep

Current goal turn: progress. SCALING_RESULTS.md:72fits complete, no nonfinite failures. Producer47326 and auditor28762 terminal. Independent replay all72passes(maxrelative3.16e-15); design audit and summary/figure complete.70/72source metrics and470/576learnedtransfer task metrics pass10%batchCVgate. NoGPU. Near-budget shared advantage only consistently paired in d16h8depth2 at128labels(3/3); no broad scaling replication. Fullstate/proposal/data archives retained.

New exact diagnosis: Wick105pairing Gram for shared depth2teachers avoids input-monomial expansion. Independent quadrature passes. Exact source-output projection floors: d16h8 .391/.413/.442; d32h16 .739/.806/.779. Twelve source outputs under-span36/136hidden products. This is NOT a lower bound for all learned hidden representations or necessarily approximate source predictions.

Next bounded experiment: transfer directly from frozen students' hidden feature dictionaries vs their12source outputs, on same32/128adaptation labels, with independent feature RMS and fixedpenalties; include true teacher hidden dictionary oracle. No retraining necessary. Preserve heldouttask isolation and use efficient dualridge for wide dictionaries. Only afterward consider source-task-count changes. Overall goal incomplete; publishing current batch authorized.

## Latest prefix controls — depth5 detour closed

Current goal turn: progress. DEPTH5_PREFIX_RESULTS.md:9audited fits copying1/3/4teacher trunk layers while retaining random readouts and independently recalibrating output energy. Prefix1 population errors1.94/9.45/2.02 despite falling training loss; prefix3 .669/.328/.125; prefix4 .00218/.00327/.00389 (all3 residual precision gates fail). Independent replay and exact prefix/calibration/hash audit pass. Alljobs terminal, noGPU.

Do not turn teacher-copy success into an unprivileged learning or computation-discovery claim. Fixed training proposal may cover modified student errors differently. This bounds the current depth5 detour: coverage, calibration, smallerLRs, localbasin and prefixes are tested. Return to representative feasible depth2/3 sharing/transfer scaling instead of more privileged rescues.

Next: implement and finalize SCALING_GRID_DRAFT.json (not launched/frozen):d16h8/d32h16,depth2/3,bothteacherregimes,3teacherreplicates,3studentarchitectures,width/budgetcontrols,12source+4whole-taskholdouts,expected72fits. Resolve independent calibration, teacher/source-only proposal isolation, exact budgets and residual-focused evaluation. Include oracle source-output span because12tasks do not span larger hidden dictionaries. Save fullstates/replay; report precisionfailures. Main goal remains incomplete.

## Latest depth5 basin controls

Current goal turn: progress. DEPTH5_BASIN_RESULTS.md records9 audited fits on frozen depth5 importance datasets. Random-start lr1e-4/1e-5 still yields relativeerror~1; teacher+1%RMSnoise positive controls give.000258/.000672/.001686. Two absolute precision gates fail and remain flagged. NumPy evaluation/objective replay, perturbation replay and input hashes pass. All producers/auditors terminal; no GPU/live training jobs.

Norm traces: random readout shrinks markedly, trunk norms barely change. Do not describe simultaneous all-parameter shrinkage as observed mechanism. Initial random empirical alignment~1e-6. Exact power-function pairing formula demonstrates that input-degree32 can make moderate directional overlap nearly function-orthogonal; independently checked against exact circle quadrature. This is illustrative, not a theorem about these general networks.

Next bounded step: initialization/curriculum control manipulating early feature alignment, with teacher access labeled privileged. Avoid unbounded LR sweeps. Determine whether deep learning can be stabilized for sharing/task-transfer comparisons, or document a depth limit and complete representative feasible scaling. Independent-teacher, budget controls and whole-task transfer remain outstanding in the deep regime. Overall goal remains incomplete.

## Latest coverage-training stage

Current goal turn: progress. COVERAGE_AND_COLLAPSE_RESULTS.md:12paired depth3/5 fits(3fresh teachers, uniform vs exact weighted importance sampling),1024examples600Adamsteps. Independent output calibration and fresh final proposals; full states/data saved. Producers and audits terminal, no jobs live. NumPy replay/weighted gradient audit passes. Depth3 importance improves all3pairs; depth5 predictions lose alignment and collapse nearzero. Four of six depth5 absolute-energy precision gates fail and remain reported. Cross-fitted scalar correction does not recover depth5 signal.

The simultaneous parameter-scaling path has degree2^(depth+1)-3 (=61atdepth5), verified alongside its loss derivative. Inputdegree is32; do not conflate them. This gives a candidate flatness mechanism, not measured causal explanation of Adam trajectories. Next bounded experiment: reuse exact frozen depth5 datasets/calibrated states, compare smaller learning rates and teacher-near initialization positive control, record norm/alignment/gradient traces and fresh evaluations. Then broaden to independent teachers, budget controls, task holdouts and representative scaling. Full goal remains incomplete; no GPU work.

## Latest directional evaluation stage

Current goal turn: progress. See DIRECTIONAL_SAMPLING_RESULTS.md. Rank-one angular central Gaussian mixture with20%uniform floor implemented and validated against exact moments, independent full covariance densities and an exact d2 depth5 circle oracle. Refined d16 depth4/5 proposals(64pilot starts,200steps,stretches32/128,12x32768fresh draws) meet predeclared batchCV<10% gate; depth5 errorCV4.28%, median errorESS570. All jobs terminal, aggregate and density/circle audits pass. No training/GPU jobs active.

Important new finding: earlier4096-point uniform calibration underestimates teacher energy by8.7x(depth4) and234x(depth5) for these fixed models. Next bounded step: independent importance-weighted calibration, then paired uniform-vs-weighted-importance training to distinguish rare-direction coverage from optimizer failure, with fresh per-model evaluation proposals. Save full states for replay. Then representative multi-seed sharing/budget/task-holdout scaling. Methodological evidence uses the same fixed teachers; no claim of deep learned sharing advantage yet. Full goal remains incomplete.

## Latest evaluation/scaling feasibility stage

Current goal turn: progress. HOMOGENEOUS_EVALUATION_AND_DEPTH.md records exact radial integration and a variance lower bound for naive Gaussian evaluation. Six independent-factor quartic comparisons passed; observed batch SD reduction2–4x. DeepMultiTask at d16/h8/depth2–5 passes NumPy forward, degree, matched-function and cached/recomputed gradient checks. Sixteen100-step CPU smoke fits completed with finite training; aggregate audit passed. All processes terminal (producer session90759 exited0). No GPU work.

Critical finding: at depth5(degree32),8192sphere samples yield error effective sample sizes1–6. Do not treat smoke MSEs as reliable learning results or launch a large conventional sweep. Next: implement and validate exact directional importance sampling with independent pilot/proposal/evaluation data and a uniform mixture floor; first against exact quartic moments, then deep repeated evaluation. Calibrated teacher energy also needs scrutiny. Then freeze representative scaling and whole-task transfer grid. Main goal remains incomplete.

## Latest completed stage

Five-teacher confirmation completed:80fits,4560transfer rows, all independently audited. See CONFIRMATION_RESULTS.md and results/multitask_confirmation. Component confirmation:10trained h4 models,24/40components above.95, mean cosine.7845, independent audit passed. All producer/auditor handles terminal; no live jobs. No GPU/native-MLP work started. Current goal turn is progress. Prior heartbeat removed an obsolete reminder; it did not restart experiments.

Shared h10 wins all5 paired shared-teacher blocks against shared h4, separate h4/h1 and polynomial at32/128labels. Independent-teacher transfer medians remain worse than zero prediction. Next stage: validate Monte Carlo population evaluation against exact quartic metrics, build depth2–5 smoke cases, then freeze representative input/width/depth scaling. Preserve whole-task holdouts and all failure/negative conditions. Full goal not complete.

## Evidence current through this batch

- Main report MULTITASK_AND_GRAPH_RESULTS.md; TASK_SPANS_AND_GRAPH_IDENTIFICATION.md; GENERIC_BILINEAR_COMPONENT_IDENTIFIABILITY.md; GENERIC_FIBERS_AND_LOCAL_RLCT.md.
- results/multitask_transfer: 72 fits, 4176 held-out task records, 3 independent teachers x 2 optimizer seeds per teacher regime. Inputs d8, hidden h4, 12 source/4 held-out heads, source N64/256, 3000steps, fixed transfer penalties0/.001 with exactly8/32/128 adaptation labels and no held-out validation. Independent direct-forward/QR coefficient and Gaussian-metric audit passes.
- results/instrumented_graph_recovery: 8 known graph templates x20 seeds =160teachers,1280 probe-subset records; all-three probes identify160/160. Independent literal DAG interpreter and compatible-set audit pass. Known operations/target slots/noiseless access; not blind recovery of trained neural graphs.
- results/component_recovery: 12 trained shared students assessed with gauge-aware matching; independent quadratic-form metric audit passes. At N256, mean matched cosine .777 versus .238initial;13/24 correlated components exceed .95. Not full recovery. Three teacher factor matrices satisfy independence and task-head rank10.
- results/generic_rank: integer coefficient Jacobian3960x184;8 independent exact gauge vectors; rank176 certificate, independently checked 176-minor determinant714573 modulo1000003; autograd agreement1.59e-16. Generic fiber dimension8. Under the explicitly specified local realizable Gaussian likelihood/prior, regular quotient gives local RLCT88,multiplicity1; not an empirical or global singular-model RLCT estimate.
- Earlier readout/longer-training/singular-path/quotient studies remain valid and closed. Source snapshots and hashes preserved. Do not rerun existing output directories.

## Findings that change the next action

At sourceN256 on shared teachers, shared students beat both separate controls in all3 teacher blocks for adaptationk32/128, including the fixed OOD covariance shift. Not consistent against polynomial baseline (paired2/3wins at32,1/3at128), not uniform at8labels orsourceN64. This is exploratory3-teacher evidence, not confirmation. Wrong sharing assumption has a proven rank-tail source-error floor; independent tasks transfer poorly even using true source-output features.

The exact component theorem: with2h independent teacher linear factors and full task-product span h(h+1)/2, an exact h-feature bilinear competitor must match teacher quadratic functions up to scaling/permutation. Proof uses square-root subspace membership plus rank-two axes. Does not identify cached execution, apply to overcomplete competitors, or give approximate/finite-sample guarantees. Counterexamples cover incomplete task span and dependent factors; no literature novelty claim.

## Completed confirmation protocol (historical)

Confirm the transfer signal on five fresh teachers with separately varied optimizer seeds. Suggested disjoint seed ranges: teachers90000–90004,source inputs91000–91004,optimizer seeds92000–92001,adaptation93000–93004. Freeze protocol before outputs. Preserve sourceN256,12training/4held-out tasks,3000steps, same two teacher regimes, adaptation8/32/128, fixed penalties0/.001. Primary comparisons k32/128 specified from development. Do not introduce held-out validation labels or test selection.

Add a wider shared trunk h10 (820 parameters) near the separate h4 model's888 parameters, alongside sharedh4(184),separateh4(888),separateh1(204). Main h4 width pair can start at identical functions; wide shared initialization is not automatically matched and needs recorded scale diagnostics. The h4 identifiability theorem does not extend to overcomplete h10 at d8. This strengthens budget controls but still changes architecture/class.

Compare teacher-block paired means; components/tasks/optimizer seeds are not independent teacher replications. Preserve all failure/negative conditions. If a robust signal survives, define representative depth/width/input scaling and inspect GPU availability without interrupting other work. A source-side meta-validation scheme, if later introduced, must use additional separate meta-task heads; preserve fixed-penalty primary results.

## Remaining scope — not complete

- E1/multi-task learning: fresh confirmation with stronger budget controls is complete; broader sample/task-count behavior and the role of useful component recovery remain.
- E2: existing exact metric projection, optimization-budget controls, and singular-feature conditioning results are complete for their stated settings. Tiny-network spectrum/scaling comparisons may accompany depth experiments; do not reopen old sweeps without a new discriminating question.
- E3: known-family instrumented graph recovery and generic exact feature identifiability are now established under explicit assumptions. Learned models show only partial feature recovery. Need confirmation/robustness and operational transfer beyond this small family, not an unsupported universal graph claim.
- Theory: generic fiber dimension and a local regular-quotient RLCT are derived with an exact rank witness; scalar singular examples remain in MATHEMATICS.md. Degenerate/global RLCTs and robust approximate-identifiability bounds are not solved or claimed.
- E4: representative d16/32,h8/16 plus near-budget widths21–46,depth2/3 sharing/transfer sweep is complete; depth4/5 feasibility and optimization limits are documented. Frozen-hidden-feature transfer diagnosis and final synthesis remain. No general d256/h512 scaling success is claimed.
- Final synthesis must distinguish parameter tying, input identification, physical execution reuse, functional component recovery, and generalization. Preserve negative results and resource comparisons. The full goal is not achieved.

Runtime: /home/loganriggs/Coding/neuron_circuits/.venv/bin/python with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1. Output directories refuse overwrite. Publication authorized. Repository /home/loganriggs/Coding/reverse_engineering_llms, subdirectory shared_tensor_networks. Native-MLP project remains paused.
