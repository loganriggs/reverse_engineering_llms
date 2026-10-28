# Active next step: reconcile shared-dictionary implementation for joint comparison

## Current M33 result

Context-dependent native response directions persist under small symmetric interventions. Of 627 active direction/context cases, 626 pass the fixed two-scale agreement threshold; median relative scale discrepancy is 0.37%. At amplitude .01, centered cross-context change/reference energy is .122–.314 and 55–86% of that change cannot be explained by scalar gain. Full-removal curvature is therefore insufficient to explain the aggregate context dependence. Close wrapper/amplitude diagnostics and return to joint scalar/block/shared comparison. See [report](NATIVE_LOCAL_SENSITIVITY.md). Earlier next steps below are historical.

## Current M32 result

Identical MLP1 interventions have context-dependent relative-logit effects in the native model. Removing the common log-probability shift leaves 57–91% of the response-change energy; 60–91% of that centered change is orthogonal to the reference response. These are descriptive aggregates on familiar tasks, not semantic-unit validation or a universal failure of context transfer. See [report](NATIVE_CONTEXT_RESPONSE_GEOMETRY.md). Next: two fixed symmetric amplitudes on the same directions, then return to joint representation comparison. Earlier next steps below are historical.

## Current M31 result

M31 recomputes sameM18group on targetcontextnativewrite, same720rows/20models. Fullpassagejoint/control ratios improve4/4 butsampleonly2/4; fullsamplewinsstay2/4, bothblockslightlyworse(.74/.81%). Tenthcountsunchanged; scalar72sampleeffectivelytie. Blocksamplemediancosines.994/.997 withnormratios1.018/1.001: wholesalevectormismatchinsufficient, notproofsemanticinvariance. 6802forwards3600responses; allintactmetrics/referenceoutputs/sampleddecodes/geometry audited. Noops31/35/30retained. Stopconstructionvariants. Next native-onlyfullresponsevector replay of fixedM30delta triples, commonreferenceprobabilitymetric, centered/constant decomposition andlogit/logprobidentity; no fitting/prompts/search. See [report](CONTEXT_ADAPTIVE_COMPONENTS.md). Earlier next steps are historical.

## Current M30 result

M30 pairedwrappers hold exacttasktokens andtaskspandelta fixed, all240M28taskrows×3contexts/all20models. Joint/KL-only logprobwins full4/4reference4/4passage2/4sample; tenth4/4,3/4,3/4. Bothfullblock sampleadvantages reverse slightly; nativebinary-correct subset and9/10familyomissions retain reversals, butstrictallcontexttop1subsetschangeconclusions. Contextsensitivitynotuniversalloss. 6820forwards3600records1200reference replays60samples660summaries audited;31noops/contextretained. Next samegroups/input/candidates butreencodeM18componentinnewcontext vsfixedvector; geometry/energy/zeros explicit, no fitting/search. Prefixfailures/semanticunits notresolved. See [report](PAIRED_WRAPPER_CROSSOVER.md). Earlier next steps are historical.

## Current M29 result

M29 fixes evaluationinputs/candidates/groupindices/regions and changesM11→M18interventiongenerator. JointbeatsKL-only logprob5→6/8full,6→7/8tenth; margin4→5/8full,4→4/8tenth. Fewwinnerchanges: basis mismatchcontributesbutinsufficient. Equalfamilymargin8/8fullM18; context-onlystratumlogprob5/8full3/8tenth remainsmixed. All880intactbaselines replay;24noopstatuschanges,96→94noops; nativeenergiesreported. 4584forwards3520responses120samples560summaries audited. Refreshed20hypotheses. Next pairedwrappers onallM28task-regionrows, fixsource/taskspandelta, all20models/evalwrappers; no fitting or outcomeselection. Prefixfailuresoutside thattargetedprobe; nosemanticunitclaim. See [report](BASIS_TRANSFER_COMPARISON.md) and [hypothesis review](CAUSAL_TRANSFER_HYPOTHESES.md). Earlier next steps are historical.

## Current M28 result

M28 same-input unsupervised-group probe: jointresponseobjective beatsmatchedKL-only onall4model/all8region logprob aggregates atbothnorms; gainsweakerthan supervisedgroups (full1.1–14.9% versus33.9–51.5%). Familywins25/40full24/40tenth; margin3/8full4/8tenth. 480scheduled/400distinct directions,59noops retained,158exactinputs; exacttoken exclusion acrossaliases/regions. 4526forwards2400records40samples660summaries audited. Partial directionaltransfer, notsemanticunits. Next sameevaluationinputs/groups/regions withM18 versusM11basis, same20candidates; compare withinbasis, reportchangednativeenergies/noops. Reuseoutputmetrics; no newtraining/search. See [report](UNSEEN_DIRECTION_TRANSFER.md). Earlier next steps are historical.

## Current M27 result

M27 matched full-data KL-only controls: adding responseobjective improvestrainingR8/8; native margin4/8bothnorms,logprob5/8full6/8tenth. Thus traininggain survives objectivecontrol buttransferuneven. Samegradients/data/rawnoms/normalization; no newfitting. Allnativeeffects/inventories audited. Runtimeexit139audit/output retries preserved; thread1 run initialscoresreplay. Next same-input unseen-M18-group probe withdeterministic groups excludedfromallprevioussupervisiononinput, nozero/effectrejection; allfamilies/candidates/norms retained. Stopoptimizersweeps; distinguish directioncoverage frominput/basistransfer beforefurthertraining. See [report](AGGREGATE_KL_OBJECTIVE_CONTROL.md). Earlier next steps are historical.

## Current M26 result

M26 evaluates all16prespecified candidates+4initials: aggregate full/tenth native logprob error improves7/8 and8/8 versusinitial; margin5/8both. Effects mixedbyregion/metric; natural-onlycontrol alsoimproves. All704 nativeeffects replay;4572nativeforwards/3520responses,24192outputforwards/23040rows audited. Exact40tensor candidate replay required rawupdate+normalization rather than rounded realizeddelta addition. No semanticunit/newtaskclaim. Next full-data KL-only direction control atboth existingnorms, fromsavedM24gradients, to isolate causalobjective contribution; no search. See [report](FINITE_CANDIDATE_TRANSFER.md). Earlier next steps are historical.

## Current M25 result

M25 finite aggregate updates lower training response error41–52%, complete schedule objective22–28%, and both intact task/natural K inall4 starts. Families improve39/40; fixed supports retain most gains. Full minibatch steps harm bothscalar objectives but help bothblock; tenth steps improveall4. Comparison confounds objective coverage/averaging/preconditioning, not equalcompute. Exit139 firstattempt retained; unchanged retry8433forwards/auditpassed. Close localoptimizerdiagnosis; next evaluate all16live candidates plus4initials on unchanged native704directions and full outputinventory. No winnerselection/extra tuning/semanticunitclaim. See [report](FINITE_GRADIENT_STEPS.md). Earlier next steps are historical.

## Current M24 result

M24 measures all480 scheduled training targets and natural document multiplicities across four unchanged starts. Aggregate descent locally improves39/40 family response losses despite10–24/45 negative family-pair alignments. Task K versus weighted R cosines are positive inall4; natural K is almost orthogonal. This weakens immediate aggregate conflict as the whole explanation, not later interference or finite-step failure. Full gradient/aggregation audit and eight parameter finite differences pass. Next a fixed finite-displacement comparison of aggregate versus first-minibatch directions, matched norms with1/10 local control, actual/fixed supports; no optimizer grid. See [report](FAMILY_GRADIENT_COMPATIBILITY.md). Earlier next steps are historical.

## Current M23 result

M23 completes eight fixed beta1=0 fits and audited native/training-target evaluations. Removing momentum fails to rescue retention: response error worsens versus matched control in all four starts; aggregate native margin/logprob improve1/8 each, equal-family margin5/8, joint-withheld margin3/8. Joint output KL worsensall4. Local M22 explanation is not a sufficient training fix. Close optimizer tuning; next measure family/objective gradient compatibility across all four starts, fixed training targets, before selecting one intervention. No semantic-unit/new-task claim. See [report](NOMOMENTUM_RESPONSE_COMPARISON.md). Older next-step entries below are historical.

[M22](RESPONSE_ADAM_DIAGNOSIS.md) replays five fixed Adam counterfactual steps, preserving all reference tensors. At steps80–83, carried first moment reproduces most/more-than-all sentinel damage; current-only matched-norm updates do little harm or help. Step79 still shows finite-step/support effects. This is a one-case local diagnosis, not a training fix. Next freeze eight beta1=0 fits (both intact/response objectives, fourstarts), retaining all other settings and evaluating full training retention/native benchmark/output fidelity against audited M20beta1=.9 references. No optimizer grid or sentinel tuning.

# Completed step: distinguish current-gradient and optimizer-history interference

[M21](RESPONSE_TRAJECTORY_DIAGNOSIS.md) exactly reproduces all final tensors of block71 treatment and tracks its largest-error training endpoint. Error .01395→1.24825: net smooth change+1.29089, support-change effect−.05659, first-order prediction+1.21566. The five largest increases have unchanged support. This points to continuous update interference for this post-selected case, not a general gate-failure claim. Next fixed steps79–83 diagnose current-gradient versus Adam-history contributions without changing the trained model or tuning against the sentinel.

# Completed step: diagnose failed response-optimization trajectory

[M20](RESPONSE_SUPERVISION_RESULTS.md) executes the fixed8fits and unchanged704-direction native benchmark. Treatment worsens final training response error for all4starts, with40–79% concentrated in one endpoint per start; paired removal of that endpoint still leaves all4worse than control. Native margin transfer improves4/8 aggregates (equal-family3/8), logprob response2/8; ordinary joint-eval KL worsensall4. This is an optimization/retention failure before a clean generalization test. Next bounded exact replay of block71 treatment with support/update instrumentation; no coefficient/learning-rate sweep or semantic-unit claim.

# Completed step: execute response-supervised comparison

[M19](RESPONSE_SUPERVISION_TARGETS.md) freezes the direct native-response objective and collects480 scheduled train-only perturbation targets (57inactive retained). Four training-only weights3.08–11.40 equalize initial mean KL/response contributions. Full cached-target audit and8model/region gradient checks (16finite differences, maxrelative.482%) pass. No response-supervised fitting yet. Next execute8fixedfits, preserving the old704 M18/M11-based native evaluation directions rather than recomputing them from the newer starts. Family loss concentration and different active-target counts are explicit limitations.

# Completed setup: explicit native-response supervision

[M18](CONTEXT_TRAINING_NATIVE_FIDELITY.md) executes eight paired120-update fits plus initial baselines and704 frozen native perturbations. Expanded contexts cut lexical-evaluation KL54–61% on training templates, but withheld-template gains are uneven and joint-eval top1 falls in allfour arm comparisons. Native margin-response fidelity improves3/8 model/region aggregates (equal-family4/8; joint-withheld subset6/8), logprob-response4/8. Better ordinary prediction does not uniformly imply better causal fidelity. Next freeze direct native-response supervision versus intact-output continuation, using training-only targets and unchanged exclusions; no matching/gain search.

# Completed step: paired context-training and native intervention validation

[M17](CONTEXT_EXPANSION.md) freezes six wrappers over80 familiar pairs (four training templates, two excluded from training) and screens960 native endpoints.87/96 cohorts pass;33/40 training cohorts provide246 eligible context pairs across allten families. Past tense has only one passing training template; failures retained. Exact source-token separation and all token boundaries audited. No featurizer fitting yet. Next freeze a family-balanced original-vs-expanded context comparison and native-vs-reconstructed fixed-perturbation validation at both prefix/task sites, preserving whole-task exclusions and reporting all three nontraining lexical/template quadrants.

# Completed dataset step: broader context coverage

[M16](CONTEXT_REGION_ROLES.md) separates prefix/task interventions across all groups and controls. Actual-block task effects have .757–.918 times whole squared margin-effect energy; interactions .004–.016. Task-only geometric prediction errors .840/.747 remain partial (equal-family .978/.866), close to whole-position .854/.726. The large-gain failure is task-local under exercise and prefix-local under example, so ignoring prefix tokens is no general repair. Close localization; next expand context coverage across allowed task families, preserve lexical/task exclusions and whole-template evaluation, and require causal validation on native as well as reconstructed computation before unit claims.

# Completed step: localize context-dependent causal effects

[M15](CONTEXT_ROLE_TRANSFER.md) tests two fixed preambles with frozen models, group matches and gains. Eleven of twelve native family/context cohorts pass; plural nouns under the exercise preamble fail and remain reported. Actual-block geometric prediction errors are .854/.726 versus same-family original .515/.551 (zero predictor=1). Equal-family exercise error is 1.044, so correspondence is partial. Functional/random errors are dominated by a fragile gain of 93.84 on a near-silent training group, not independent evidence against the ontology. Intact replacement also loses fidelity. Next: one bounded prefix-only/task-only intervention decomposition, with all groups/controls and no gain refitting, to identify which causal targets could improve the featurizer.

# Completed step: fixed context-generalization challenge

[M14](GROUP_ROLE_ENERGY_BASELINE.md) completesenergy/meancontrols. Stopregressor/matching/thresholdsearchonoldpanel. Preparechangedcontexts forallowedtaskoperations,freezeM11weights/M13geometryassignments/trainingamplitudes, andpredeclare nativecompetencegatesbeforeablationoutcomes. Reportallfailedcohorts;nooutcome-drivenreplacement. Contexttransferisnotnewtaskorlexicaldiscovery;originalexcludedfamiliesunused. Testsurvivingcausalcorrespondenceonindependentcontexts beforestrongerunitclaims. Earlierplansarehistorical.

# Completed step: removed-energy baseline for cached causal roles

[M13](GROUP_CAUSAL_ROLES.md) finds some crossseedprediction undergeometricmatching, but concentratedeffects and weakgroupwisecorrespondence. Usecachedgroupenergy/margins tofit a simple training-onlyenergy predictor, scoreevaluationwithoutrefit, comparefixedgeometry/functionmatches. No newnativeforwards/groupselection/seedsearch. This tests a genericdisruptionconfound before semanticselectivityclaims. Keepweightsand14pairevalfixed; independentfuturecontexttestsneededforstrongerclaims. Originaltaskexclusionsuntouched. Earlierplansarehistorical.

# Completed step: causal role stability versus geometric block stability

[M12](BLOCK_STRUCTURE_AUDIT.md) findsconditionalmultidimensionality butweakindividualblockreproducibility. Freeze a bounded decoded-group ablation panel onmixedmodels, othercodes/gatesheldfixed. Comparetask-margin/outputKL effects againstintactreconstruction, removedenergy/cost andscalar/shuffledgroupcontrols. Fit rolematching/candidateselectiononlyontrainingtasks; evaluateexisting14source-separatedpairswithoutselection. Fulltaskexclusionsreserved. Geometry/function/semanticclaimsremainseparate; no moremixtureorwidthtuning. Earlierplansarehistorical.

# Completed structural step: validate candidate block structure in frozen models

[M11](DIVERSE_CAUSAL_SUPERVISION.md) completes8fixedfits and source-separatedtransfer/forgettingevaluation. Freeze finalmodels. Audit conditionalblockdimensionality, occupancy, seedsubspace/supportstability and selectivecausalroles withscalar/shuffledgroupcontrols. Handle sign/permutation/withinblockrotation. Choosecandidatesonlyfromtrainingdata; do notrepeatedlyselectontiny14pairevaluation. Stable/multidimensional≠semanticmanifold. Originalwhole-taskexclusionsreserved; stopmixture/width/gate tuning. Earlierplansarehistorical.

# Completed step: execute frozen mixed-supervision comparison

[M10](DIVERSE_SUPERVISION_SPLIT.md) completes metadata splitting and independent audit. Execute exact `validation/diverse_supervision_protocol.json`: fourM8causalinitialstates×natural-only/mixed arms,120Adam.0003updates, frozenfamilybalancedtask/naturaldocument schedules, noselection. Evaluate14pairs/6families,66trainingpairs separately andnaturaldevelopmentdocs32–63. Fourfamiliesaretraining-only; no lexicaltransferestimate. Comparefrozenbaseline andbotharms,retainallfailures,originalexcludedfamiliesuntouched. Do notresplitafteroutcomes. Earlierplansarehistorical.

# Completed step: audit splits for diverse causal supervision

[M9](FEATURIZER_TASK_TRANSFER.md) finds average transfer but consistent family-specific failures. Audit lexical source-token groups/context duplicates in the existing10trainingfamilies, then freeze a bounded family-balanced taskKL versus continued natural-textKL comparison fromidenticalcheckpoints. Keep paired endpoints/lexicalgroups together fortransferclaims; flagfamilies lacking multipleindependentgroups. Retain natural-text eval to detectforgetting, and original excludedfamilies. Do not fit selected failures then call them validation. No manifold/newtaskclaim or additionalnatural-textobjective search. Earlier plans are historical.

# Completed step: frozen diverse-task transfer

[M8](CAUSAL_FEATURIZER_OBJECTIVE.md) finds59–63%lower outputKL under explicit downstream training than continuedMSE, with worse rawreconstruction. Freeze all8models. Evaluate existing80-pair/ten-family task panel without fitting, retaining allseeds/families and original exclusions. Measure outputKL/top1, supported task decisions/margins and localerror separately. These tasks were not featurizer training data but were previously examined in thisresearch; do not call them pristine holdouts or discoveredtasks. Stop natural-text objective tuning; test transfer before interpreting groups. Earlier plans are historical.

# Completed step: explicit causal objective versus continued reconstruction

[M7](WRITE_FEATURIZER_PILOT.md) completes six matched-parameter fits and live causal evaluation. Freeze a bounded fine-tuning comparison from the saved scalar/block checkpoints: continued reconstruction versus native-output KL supervision, matched train-document streams and optimizer steps, no evaluation-based selection. Verify differentiable native replay/gradient path first and fix loss scaling on training data only. Preserve active32 budget, evaluate reconstruction/token/context/KL/CE separately, retain both seeds, and keep original task exclusions unused. Do not tune fixed-shared gates or widen models based on this pilot. Earlier plans below are historical.

# Completed step: reconnect the causal benchmark to featurizer fitting

[M6](MATCHED_ANGLE_CONTROL.md) completes the fixed directional controls. Stop perturbation/seed sweeps. Audit existing scalar/block/shared implementations and prior failures (H835 pre-MLP sites; H984/H989 MLP12; H1029 stopped selector). Prepare a bounded native MLP1-write comparison with document-separated fitting, matched active-coordinate costs and causal output evaluation, preserving raw/token/context reconstruction separately. Fixed shared memberships are a baseline; do not resume collapsed learned gates without a new reason. Freeze protocol before training. This is a benchmark and representation comparison, not proof of manifold semantics or new-task discovery. Earlier plans below are historical.

# Completed step: matched-angle directional control at MLP1

[M5](TOKEN_TABLE_CAUSAL.md) shows table replacement has substantially greater downstream effects than norm-only changes. Before attributing this to special contextual directions, match per-position final product norm AND angle from native write using random tangent directions at MLP1, same covered positions/bias/cohort, fixed small seed set and all results. Freeze construction and seeds before outcomes. This compares equal local errors with different orientations, not semantic features. Stop after the fixed control; use outcome to choose causal-featurizer targets. No whole-task holdout or toy-grid expansion. Prior next-step sections are historical.

# Completed step: causal test of cross-document token-mean predictions

[M4](NATURAL_TOKEN_CONTEXT.md) shows much larger contextual variation on natural text and limited MLP2 token-mean prediction. Use the same64 reused training documents and opposite-half token means; replace product writes at covered positions, retain Down bias/native uncovered writes, and test each early layer separately. Include native-product norm-only scaling to table norm, output KL/top1/CE, coverage and local vector error. No refit, new task holdout, or hyperparameter sweep. Freeze this protocol before outcomes. These are table-replacement interventions, not evidence of native-channel removability. Earlier next-step sections are historical.

# Completed step: natural-text replication with audited provenance

[M3](TOKEN_CONTEXT_PARTITION.md) completes the token/context partition and independent audit. Audit existing natural-text data and cached native writes before freezing a finite replication of the same decomposition. Verify model/site and token-prefix alignment, document prior use, and label reused data honestly. Preserve whole-task exclusions. The featurizer deliverable is separate token/context reconstruction diagnostics plus causal fidelity, conditional on broader replication; this observational partition does not establish causal features. No additional toy width/noise/dropout tuning.

# Completed step: token geometry versus contextual variation

[M2](EMPIRICAL_OUTPUT_SPECTRA.md) completes the real-output spectral mapping. Next decompose
observed writes into token-conditioned means and within-token residuals on the same opened panel,
with occurrence and unique-prefix weighting, singleton counts and exact energy accounting.
Do not interpret within-sample means as a validated predictor or context removal as a measured
causal intervention. This separates potential clustering/re-clustering from contextual computation
before choosing a featurizer target. Prior next-step paragraphs below are historical.

## Completed step: map the completed toy grid to real-model signatures

[T4](TOY_DROPOUT_RESULTS.md) completes the fixed capacity/dropout comparison and closes that toy
grid. Read [STRATEGY_SYNTHESIS.md](STRATEGY_SYNTHESIS.md) for the current mechanism account.
Next measure empirical output spectra and contribution-energy bounds for MLP0/1/2 on the same
opened inputs, including deduplicated causal-prefix views. No fit/pruning or claim that a matrix
rank bound yields an executable compact bilinear MLP. Preserve whole-task exclusions and distinguish
this task-panel diagnostic from natural-text generalization. Prior next-step paragraphs below
are historical, not simultaneous experiment queues.

## Completed step: can both toy robustness types be trained?

[T3](TOY_WIDTH_RESULTS.md) completes the fixed width comparison and checks class specialization
and selective channel vulnerability. Its next-test section freezes the small dropout/capacity
comparison: fine targets, input noise.25, penalty.01, widths16/32, channel dropout0/.25, three seeds,
fixed budget. Reuse three existing models and train nine new ones. Stop this toy grid after that
comparison and relate the diagnostic signatures to the real-model evidence, without assuming a
universal tradeoff. Prior plans below are preserved history, not additional simultaneous next runs.

## Previous step: capacity and robustness of learned implementations

The user's extremes question is now tested in [T2](TOY_CAPACITY_RESULTS.md), with a
[pre-outcome protocol](TOY_CAPACITY_PROTOCOL.md). Next hold the fine target task fixed and compare
width8 with16, retaining paired seeds, penalties, noise levels and training budget. This isolates
capacity more directly than changing both target complexity and capacity demand. Label-noise and
held-out-combination tests remain unperformed. The broader taxonomy and real-layer evidence below
remain relevant; do not equate toy results with an explanation of the checkpoint's training.

## Broader research direction: identify computational mechanisms

The user's latest request broadens the question from removal tolerance to how MLPs implement
functions, including clustering, overlap and cancellation. Use [COMPUTATION_STRATEGIES.md](COMPUTATION_STRATEGIES.md)
for the operational taxonomy, twenty hypotheses, first exact-function toy and fixed next sequence.
Do not treat these as mutually exclusive layer categories. The [matched comparison](MATCHED_LAYER_MECHANISMS.md) is complete and rejects a simple layer
dichotomy on this panel. Next add shared-plus-private contributions and controlled downstream
readers to the diagnostic toys before training explanatory models. Corpus checks remain approved
and necessary for generalization; they are not the sole next step.

## Preserved subquestion: removability of MLP1

The user's definition is: can many parts be removed while the model performs the same computation?
MLP1 is zero-based block1. Separate four claims: a clustered signal can be patched; a module can
be approximated by a token table; a direction can steer behavior; a subset can be deleted without
materially changing outputs. Only the last directly addresses removal redundancy.

Read [REMOVAL_REDUNDANCY.md](REMOVAL_REDUNDANCY.md) for the source-level history correction,
fixed literal-deletion experiment, limitations and next comparison. Read the README for loading.

The primary outcome is native-versus-edited output distribution, supplemented by task accuracy,
observed-token CE, worst-case examples and local write changes. A preserved binary decision alone
is insufficient. A distributional tolerance must be explicit before making an approximate-equivalence
claim; this diagnostic reports effects without inventing a passing tolerance after the fact.

The fixed [direction-preserving norm control](DELETION_NORM_CONTROL.md) is complete: at matched
magnitude it causes substantially less KL than literal deletion. Next audit natural-text row
provenance and freeze a finite corpus evaluation of the same masks and controls. Do not optimize
masks or call reused/unknown rows fresh. Preserve per-document tails and position-stratified results.
The task-panel results remain descriptive until this generalization boundary is tested.

The previous [context-response plan](ARCHIVED_CONTEXT_RESPONSE_PLAN.md) is retained as history.
Its proposals and P10's projection/residual test are deferred, not additional active next steps.
The broader goal remains to identify faithful computational units and eventually use diverse
competent-task causal targets to improve featurizers, preserving whole-task evaluation exclusions.
