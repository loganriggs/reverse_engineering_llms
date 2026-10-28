# What the toy and real-model studies currently say about computational strategies

2026-10-28. A compact reading guide after the fixed capacity/noise/dropout grid. This is a synthesis,
not a declaration that the LLM is reverse engineered.

## Name the object before naming the strategy

- **Function geometry:** which inputs get similar outputs? Token clustering, re-clustering and
  context-dependent conjunctions live here.
- **Implementation geometry:** how do product-channel writes combine? Distributed aligned
  contributions, overlap and signed cancellation live here.
- **Behavioral use:** which changes do readers notice? Preservation of a binary answer, output
  distribution and intermediate vector are different claims.

The same clustered function can have duplicated or cancelling implementations. A class-specific
function can be implemented across many channels. These labels are coordinates of a profile,
not exclusive species of MLP.

## Evidence map

| Finding | Supported conclusion | Not established |
|---|---|---|
| Same-function hand-built toys differ under channel deletion | Intact function/clustering does not identify implementation | Which implementation an optimizer or LLM will choose |
| MLP1 quarter removal preserves binary accuracy but changes top predictions | Task-level removal robustness can hide distributional damage | Unchanged computation or96% removable channels |
| Matched MLP0/1/2 contributions are net constructive on the task panel | Simple “MLP1 redundancy versus MLP2 cancellation” is not a general layer distinction | No cancelling pairs, or corpus-wide absence of cancellation |
| Contribution penalty improves toy deletion robustness when rank permits spreading | Training objective and capacity can change implementation at similar clean fit | Universal superiority of distributed coding |
| Input-noise training increases class specificity and selective channel vulnerability | Different robustness pressures can favor different toy implementations | Noise always causes localization, or an unavoidable tradeoff |
| Width32 plus channel dropout improves both input-noise and deletion classification | The previous toy tradeoff can be reduced by objective/capacity | Exact functional fidelity, severe-noise robustness or an optimal frontier |

## Current explanation, scoped to the toys

The relevant variable is not width alone but how width compares with the dimensionality of the
required output function, plus which perturbations training penalizes. Spare capacity can be used
to spread useful contributions. Training against input ambiguity constrains the function between
clean prototypes; it can produce selective detectors. Training against channel loss can distribute
those detectors' consequences again. The regularizer changes the cost of coordinated large writes.
These observations support an objective-dependent implementation account, rather than one best
representation independent of task and metric.

Even here, we have not isolated every mediating cause. Parameter count and initial-function variance
change with width; the toy inputs privilege a simple supervised prototype geometry. No held-out
feature combinations or label corruption were tested. Nuclear-norm bounds constrain possible
factorizations but do not prove why gradient descent found one solution.

## What would connect this to the language model?

First establish whether empirical function dimensionality and contribution coordination distinguish
the early layers on a shared input distribution. Then measure selective consequences with task
competence and whole-task holdouts intact. Finally, compare representations that encode the
surviving mechanism: shared dictionary directions for genuine reuse, signed grouped contributions
where cancellation matters, and conditional coordinates for context-dependent functions.
A good featurizer must predict/intervene faithfully on additional tasks, not merely reconstruct
activation variance or steer one answer. The toy grid is now closed; this mapping is the next job.

Start with [M1 real-layer comparison](MATCHED_LAYER_MECHANISMS.md),
[T2 capacity/noise](TOY_CAPACITY_RESULTS.md), [T3 width/specialization](TOY_WIDTH_RESULTS.md),
and [T4 combined robustness](TOY_DROPOUT_RESULTS.md). The historical re-clustering correction and
literal-removal distinction are in [the removal audit](REMOVAL_REDUNDANCY.md).

## Real-output mapping update

[M2](EMPIRICAL_OUTPUT_SPECTRA.md) measures spectra and native contribution energy on distinct prefixes. MLP1 dimensionality rises after removing repeated-prefix weighting; all three layers remain far above an unconstrained factorization lower bound. This is not proof of executable compression. Next separate token-conditioned geometry from within-token context residuals, keeping observational and causal claims distinct.

## M3: token-conditioned and contextual write geometry

[M3](TOKEN_CONTEXT_PARTITION.md) finds 10.1/13.8/19.0% within-token centered variation for MLP0/1/2 on distinct prefixes of repeated-context tokens. Contextual residual spectra are broader than token-mean spectra, though group counts constrain the latter. This motivates evaluating both components when fitting featurizers; it does not establish a causal decomposition or generalization. Natural-text replication must precede making this a training prior.

## M4: natural text revises the M3 interpretation

[Natural-text replication](NATURAL_TOKEN_CONTEXT.md) raises the repeated-context within-token shares to29.5/40.8/53.8%. A small contextual correction is not a general early-layer description. Cross-document token means reduce covered-position error versus a global mean by55–56%,33–35%,11–13% atMLP0/1/2, with only58–60%coverage. Vocabulary/length/position differences prevent attributing the shift to one cause. Next test causal table-replacement effects with norm controls before turning descriptive variance into a featurizer prior.

## M5: local error and downstream sensitivity rank layers differently

[Cross-document table replacement](TOKEN_TABLE_CAUSAL.md) causes KL.0484/.1835/.1062 forMLP0/1/2, versus norm-only.000137/.001428/.001951. MLP1 is most sensitive despiteMLP2's larger local error. Both direction and downstream weighting belong in causal-featurizer evaluation. Radial controls do not match local-error magnitude, table error mixes context and estimation, and downstream normalization may suppress scaling effects. A fixed matched-angle control is the next discriminator, not a new claim of semantic units.

## M6: magnitude-matched errors differ in causal impact

[Matched-angle controls](MATCHED_ANGLE_CONTROL.md) yield MLP1KL.04561–.04692 versus table.18354, despite equal pointwise norms/angles/squared errors. All64documents show this ordering for all3seeds. Orientation and/or cross-position structure matters; semantic selectivity and pure contextual causation remain unproven. This closes the directional-control series and supplies a concrete causal benchmark for scalar/block/fixed-shared featurizer comparison. Historical pre-MLP andMLP12models are different sites and cannot be silently reused asMLP1writefeaturizers.

## M7: reconstruction-trained group structure does not improve this pilot

[Six nativeMLP1write fits](WRITE_FEATURIZER_PILOT.md) favor scalarTop32 over block4×8 in bothseeds on rawerror andKL; fixedshared unions performworse at loweractualactivecost. Allatomsused; no learnedmembershipcollapse here. Tiny128atom/32trainingdoc/1000stepbudget leavescapacity/optimizationlimits. Next directly testdownstreamKLsupervision against matchedcontinuedreconstruction, preserving source/evaldocuments, before attributingfailure solelytoarchitecture. No manifold ornewtaskdiscoveryclaim.

## M8: objective choice substantially changes causal fidelity

[Matched continued-training comparison](CAUSAL_FEATURIZER_OBJECTIVE.md) reducesKL59–63% with puredownstreamKL versusMSE, bothscalar/block andbothseeds, despiteworsened raw/within-tokenreconstruction. Scalarremainsbetter. This supports downstream weighting as a featurizer target, not semantic/canonical unit recovery. Next frozen transfer toalready-opened diversecompetenttaskpanel, no additionalfit orreservedtaskexposure. Onefloat32/float64 TopKnear-tie is explicitlydocumented in the audit.

## M9: supervision distribution matters as well as objective

[Frozen diverse-task transfer](FEATURIZER_TASK_TRANSFER.md) improves overallKL13–23% under natural-textKLtraining butcapital/numberfamiliesworsenall4comparisons atoverallKL. Capital finalKL/marginsalso worsen; numberfinalKLmixed. Binaryanswerscanremaincorrectdespitedistributionaldamage. Addeddiversecausalsupervision ismotivated, butrequiresgroup-auditedsplits andnaturaltextforgettingcontrols; representationtransferonreuseddevtasks isnotnewtaskdiscovery.

## M10: family count overstates independent lexical coverage

[Strictsource-input split](DIVERSE_SUPERVISION_SPLIT.md) joinscross-familyexamples anddistractors,leaving17components. Fourfamiliescannotbothtrainandevaluatelexically;14evaluationpairscoveronly6families. Thislimitmustsurvivefutureclaims. Frozen120updatemixedtask/naturalKLcomparison testsbroadersupervision againstmatchedcontinuednaturalKL, withnaturalforgettingchecks; nofittingyet.

## M11: broader causal coverage helps both architectures

[Mixedsupervision](DIVERSE_CAUSAL_SUPERVISION.md) cutslexicalevaltaskKL61–77%vsnatural-onlycontinuation, with3–10%naturalKLcostrelativecontrol. MostcostisforegonenaturalimprovementratherthanlargeforgettingfromM8baseline. BlockscompetitiveontaskKLbutstillworsenaturalKL. Tiny14pairevalandfourtraining-onlyfamilieslimitclaims. Freezeweightsandtestactualblockstructure/stability/causalroles; improvedfidelityaloneisnotmanifoldvalidationornewtaskdiscovery.

## M12: stable broader geometry does not give stable blocks

[Structureaudit](BLOCK_STRUCTURE_AUDIT.md) findsactualblocks~3.4conditionaldimensions,butshuffled/arbitrarygroupsalsomultidimensional. Mixedblockcrossseed4Doverlap.116 vsfull128D.690, groupreference.077/ambient.006. Occupancy-adjustedinterpretationneededforJaccard(.212block vs.141ambient; scalar.533vs.525). No verifiedsemanticmanifold. Nextdecoded-groupcausalroles maydistinguishunstablecoordinatesfromreproduciblefunctions,withtraining-onlymatching andfrozenweights.

## M13: geometry predicts some effects, functional matching overfits small signatures

[Decodedgroupablations](GROUP_CAUSAL_ROLES.md) giveactualblock geometrypredictionerror.551vsrandom.907/function.881;scalar/shuffledcontrolsnear1. Butantonym1pairhas60%targeteffectenergy,twogroups58%;groupwisecosinesweak. Withoutantonymgeometry.682vsrandom1.058. Somecausalcorrespondence,nocleansemanticunitdictionary. Nexttraining-onlyremoved-energybaseline oncachedrecords tests genericdisruptionexplanation;noadditionalmodel/groupsearch.

## M14: energy-only explanation is insufficient, stronger unit claims remain open

[Cachedbaselineaudit](GROUP_ROLE_ENERGY_BASELINE.md):seed72actualblockerror sharedenergy.998/pergroupenergy.947/mean.882,vsgeometrycrossseed.551;withoutantonym1.067/.993/1.003vs.682. Baselineevaluationinputs differ(withinseedenergyvscrossseedeffects),sothisrulesoutonlysimplelinearenergy/generalmeanexplanations. Stopoldpanelpredictorsearch;fixednewcontexttransferwithcompetencegates isnextdiscriminator.


## M15: contextual stability remains limited

[M15](CONTEXT_ROLE_TRANSFER.md) tests two fixed preambles with frozen models, group matches and gains. Eleven of twelve native family/context cohorts pass; plural nouns under the exercise preamble fail and remain reported. Actual-block geometric prediction errors are .854/.726 versus same-family original .515/.551 (zero predictor=1). Equal-family exercise error is 1.044, so correspondence is partial. Functional/random errors are dominated by a fragile gain of 93.84 on a near-silent training group, not independent evidence against the ontology. Intact replacement also loses fidelity. Next: one bounded prefix-only/task-only intervention decomposition, with all groups/controls and no gain refitting, to identify which causal targets could improve the featurizer.


## M16: causal supervision must cover context regions

[M16](CONTEXT_REGION_ROLES.md) separates prefix/task interventions across all groups and controls. Actual-block task effects have .757–.918 times whole squared margin-effect energy; interactions .004–.016. Task-only geometric prediction errors .840/.747 remain partial (equal-family .978/.866), close to whole-position .854/.726. The large-gain failure is task-local under exercise and prefix-local under example, so ignoring prefix tokens is no general repair. Close localization; next expand context coverage across allowed task families, preserve lexical/task exclusions and whole-template evaluation, and require causal validation on native as well as reconstructed computation before unit claims.


## M17: context coverage is uneven even for familiar tasks

[M17](CONTEXT_EXPANSION.md) freezes six wrappers over80 familiar pairs (four training templates, two excluded from training) and screens960 native endpoints.87/96 cohorts pass;33/40 training cohorts provide246 eligible context pairs across allten families. Past tense has only one passing training template; failures retained. Exact source-token separation and all token boundaries audited. No featurizer fitting yet. Next freeze a family-balanced original-vs-expanded context comparison and native-vs-reconstructed fixed-perturbation validation at both prefix/task sites, preserving whole-task exclusions and reporting all three nontraining lexical/template quadrants.


## M18: native response fidelity is a distinct training target

[M18](CONTEXT_TRAINING_NATIVE_FIDELITY.md) executes eight paired120-update fits plus initial baselines and704 frozen native perturbations. Expanded contexts cut lexical-evaluation KL54–61% on training templates, but withheld-template gains are uneven and joint-eval top1 falls in allfour arm comparisons. Native margin-response fidelity improves3/8 model/region aggregates (equal-family4/8; joint-withheld subset6/8), logprob-response4/8. Better ordinary prediction does not uniformly imply better causal fidelity. Next freeze direct native-response supervision versus intact-output continuation, using training-only targets and unchanged exclusions; no matching/gain search.


## M19: native causal supervision implemented and differentiated

[M19](RESPONSE_SUPERVISION_TARGETS.md) freezes the direct native-response objective and collects480 scheduled train-only perturbation targets (57inactive retained). Four training-only weights3.08–11.40 equalize initial mean KL/response contributions. Full cached-target audit and8model/region gradient checks (16finite differences, maxrelative.482%) pass. No response-supervised fitting yet. Next execute8fixedfits, preserving the old704 M18/M11-based native evaluation directions rather than recomputing them from the newer starts. Family loss concentration and different active-target counts are explicit limitations.


## M20: verify training-objective retention before transfer claims

[M20](RESPONSE_SUPERVISION_RESULTS.md) executes the fixed8fits and unchanged704-direction native benchmark. Treatment worsens final training response error for all4starts, with40–79% concentrated in one endpoint per start; paired removal of that endpoint still leaves all4worse than control. Native margin transfer improves4/8 aggregates (equal-family3/8), logprob response2/8; ordinary joint-eval KL worsensall4. This is an optimization/retention failure before a clean generalization test. Next bounded exact replay of block71 treatment with support/update instrumentation; no coefficient/learning-rate sweep or semantic-unit claim.


## M21: active-feature identity does not guarantee causal-role stability

[M21](RESPONSE_TRAJECTORY_DIAGNOSIS.md) exactly reproduces all final tensors of block71 treatment and tracks its largest-error training endpoint. Error .01395→1.24825: net smooth change+1.29089, support-change effect−.05659, first-order prediction+1.21566. The five largest increases have unchanged support. This points to continuous update interference for this post-selected case, not a general gate-failure claim. Next fixed steps79–83 diagnose current-gradient versus Adam-history contributions without changing the trained model or tuning against the sentinel.


## M22: separate valid causal targets from optimizer-induced interference

[M22](RESPONSE_ADAM_DIAGNOSIS.md) replays five fixed Adam counterfactual steps, preserving all reference tensors. At steps80–83, carried first moment reproduces most/more-than-all sentinel damage; current-only matched-norm updates do little harm or help. Step79 still shows finite-step/support effects. This is a one-case local diagnosis, not a training fix. Next freeze eight beta1=0 fits (both intact/response objectives, fourstarts), retaining all other settings and evaluating full training retention/native benchmark/output fidelity against audited M20beta1=.9 references. No optimizer grid or sentinel tuning.

## M23: local optimizer causality does not guarantee a global remedy

M23 completes eight fixed beta1=0 fits and audited native/training-target evaluations. Removing momentum fails to rescue retention: response error worsens versus matched control in all four starts; aggregate native margin/logprob improve1/8 each, equal-family margin5/8, joint-withheld margin3/8. Joint output KL worsensall4. Local M22 explanation is not a sufficient training fix. Close optimizer tuning; next measure family/objective gradient compatibility across all four starts, fixed training targets, before selecting one intervention. No semantic-unit/new-task claim.

## M24: pairwise conflict is not aggregate infeasibility

M24 measures all480 scheduled training targets and natural document multiplicities across four unchanged starts. Aggregate descent locally improves39/40 family response losses despite10–24/45 negative family-pair alignments. Task K versus weighted R cosines are positive inall4; natural K is almost orthogonal. This weakens immediate aggregate conflict as the whole explanation, not later interference or finite-step failure. Full gradient/aggregation audit and eight parameter finite differences pass. Next a fixed finite-displacement comparison of aggregate versus first-minibatch directions, matched norms with1/10 local control, actual/fixed supports; no optimizer grid.

## M25: existing coordinates admit a nearby joint improvement

M25 finite aggregate updates lower training response error41–52%, complete schedule objective22–28%, and both intact task/natural K inall4 starts. Families improve39/40; fixed supports retain most gains. Full minibatch steps harm bothscalar objectives but help bothblock; tenth steps improveall4. Comparison confounds objective coverage/averaging/preconditioning, not equalcompute. Exit139 firstattempt retained; unchanged retry8433forwards/auditpassed. Close localoptimizerdiagnosis; next evaluate all16live candidates plus4initials on unchanged native704directions and full outputinventory. No winnerselection/extra tuning/semanticunitclaim.

## M26: training-response gains show partial causal transfer

M26 evaluates all16prespecified candidates+4initials: aggregate full/tenth native logprob error improves7/8 and8/8 versusinitial; margin5/8both. Effects mixedbyregion/metric; natural-onlycontrol alsoimproves. All704 nativeeffects replay;4572nativeforwards/3520responses,24192outputforwards/23040rows audited. Exact40tensor candidate replay required rawupdate+normalization rather than rounded realizeddelta addition. No semanticunit/newtaskclaim. Next full-data KL-only direction control atboth existingnorms, fromsavedM24gradients, to isolate causalobjective contribution; no search.

## M27: causal objective effect survives a matched training control

M27 matched full-data KL-only controls: adding responseobjective improvestrainingR8/8; native margin4/8bothnorms,logprob5/8full6/8tenth. Thus traininggain survives objectivecontrol buttransferuneven. Samegradients/data/rawnoms/normalization; no newfitting. Allnativeeffects/inventories audited. Runtimeexit139audit/output retries preserved; thread1 run initialscoresreplay. Next same-input unseen-M18-group probe withdeterministic groups excludedfromallprevioussupervisiononinput, nozero/effectrejection; allfamilies/candidates/norms retained. Stopoptimizersweeps; distinguish directioncoverage frominput/basistransfer beforefurthertraining.

## M28: partial transfer across discrete groups on the same inputs

M28 same-input unsupervised-group probe: jointresponseobjective beatsmatchedKL-only onall4model/all8region logprob aggregates atbothnorms; gainsweakerthan supervisedgroups (full1.1–14.9% versus33.9–51.5%). Familywins25/40full24/40tenth; margin3/8full4/8tenth. 480scheduled/400distinct directions,59noops retained,158exactinputs; exacttoken exclusion acrossaliases/regions. 4526forwards2400records40samples660summaries audited. Partial directionaltransfer, notsemanticunits. Next sameevaluationinputs/groups/regions withM18 versusM11basis, same20candidates; compare withinbasis, reportchangednativeenergies/noops. Reuseoutputmetrics; no newtraining/search.

## M29: generator sensitivity is not a complete account of transfer

M29 fixes evaluationinputs/candidates/groupindices/regions and changesM11→M18interventiongenerator. JointbeatsKL-only logprob5→6/8full,6→7/8tenth; margin4→5/8full,4→4/8tenth. Fewwinnerchanges: basis mismatchcontributesbutinsufficient. Equalfamilymargin8/8fullM18; context-onlystratumlogprob5/8full3/8tenth remainsmixed. All880intactbaselines replay;24noopstatuschanges,96→94noops; nativeenergiesreported. 4584forwards3520responses120samples560summaries audited. Refreshed20hypotheses. Next pairedwrappers onallM28task-regionrows, fixsource/taskspandelta, all20models/evalwrappers; no fitting or outcomeselection. Prefixfailuresoutside thattargetedprobe; nosemanticunitclaim.

## M30: paired context sensitivity with competence-dependent conclusions

M30 pairedwrappers hold exacttasktokens andtaskspandelta fixed, all240M28taskrows×3contexts/all20models. Joint/KL-only logprobwins full4/4reference4/4passage2/4sample; tenth4/4,3/4,3/4. Bothfullblock sampleadvantages reverse slightly; nativebinary-correct subset and9/10familyomissions retain reversals, butstrictallcontexttop1subsetschangeconclusions. Contextsensitivitynotuniversalloss. 6820forwards3600records1200reference replays60samples660summaries audited;31noops/contextretained. Next samegroups/input/candidates butreencodeM18componentinnewcontext vsfixedvector; geometry/energy/zeros explicit, no fitting/search. Prefixfailures/semanticunits notresolved.

## M31: stable component geometry is not sufficient for causal-fidelity advantage

M31 recomputes sameM18group on targetcontextnativewrite, same720rows/20models. Fullpassagejoint/control ratios improve4/4 butsampleonly2/4; fullsamplewinsstay2/4, bothblockslightlyworse(.74/.81%). Tenthcountsunchanged; scalar72sampleeffectivelytie. Blocksamplemediancosines.994/.997 withnormratios1.018/1.001: wholesalevectormismatchinsufficient, notproofsemanticinvariance. 6802forwards3600responses; allintactmetrics/referenceoutputs/sampleddecodes/geometry audited. Noops31/35/30retained. Stopconstructionvariants. Next native-onlyfullresponsevector replay of fixedM30delta triples, commonreferenceprobabilitymetric, centered/constant decomposition andlogit/logprobidentity; no fitting/prompts/search.


## M32 update

Identical MLP1 interventions have context-dependent relative-logit effects in the native model. Removing the common log-probability shift leaves 57–91% of the response-change energy; 60–91% of that centered change is orthogonal to the reference response. These are descriptive aggregates on familiar tasks, not semantic-unit validation or a universal failure of context transfer. The shift-only and gain-only accounts are insufficient for these aggregates. A legitimate reusable feature can still have context-dependent readers. Separate local sensitivity from finite-removal curvature before fitting conditional representations. See [report](NATIVE_CONTEXT_RESPONSE_GEOMETRY.md).


## M33 update

Context-dependent native response directions persist under small symmetric interventions. Of 627 active direction/context cases, 626 pass the fixed two-scale agreement threshold; median relative scale discrepancy is 0.37%. At amplitude .01, centered cross-context change/reference energy is .122–.314 and 55–86% of that change cannot be explained by scalar gain. Full-removal curvature is therefore insufficient to explain the aggregate context dependence. Conditional native readers remain plausible; no semantic-unit claim. Existing causal supervision already conditions on native context. Close this diagnostic and return to common scalar/block/shared-dictionary training and transfer comparison. See [report](NATIVE_LOCAL_SENSITIVITY.md).


## M34 update

Fixed-overlap union decoding and full dictionary coverage pass all six pilot replays. Shared models were omitted after M7; learned SharedGroups is a distinct larger implementation with earlier membership-concentration failures. A common12-fit scalar/block/fixed-shared comparison is designed, using both native intervention generators for every architecture; training-only calibration and gradient checks precede launch. Group-gate removal and shared-atom removal have different semantics. Equal parameter count does not imply equal active budget; shared means28.26/28.44 versus32. No new learned structures validated.


## M35–M36 update

All12 common-training fits completed and their final states/schedules passed audit. Native-response supervision improves final training response error versus matched intact-KL control in 6/6 starts. These are familiar training targets; held-out native response and ordinary-output evaluation remain pending. Fixed random sharing is a baseline, not discovered reusable structure. Commonlambda7.70262, both native generators for every architecture. Next fixed transfer/output validation, not optimizer search.
