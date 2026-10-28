# How much of MLP1 can be removed while preserving its computation?

A shareable research handoff for the 18-layer bilinear language model, assembled 2026-10-28.
**MLP1 is the second block**, using zero-based indices. MLP0 is the preceding comparison;
MLP2 has a separate native-channel cancellation/pruning problem.

The user clarified that **redundancy means tolerance of removing parts**, not shared decodability,
token-table replaceability, or a useful steering direction. This is the primary question below.

## Latest: initial gradients largely permit joint local improvement

[M24](FAMILY_GRADIENT_COMPATIBILITY.md). M24 measures all480 scheduled training targets and natural document multiplicities across four unchanged starts. Aggregate descent locally improves39/40 family response losses despite10–24/45 negative family-pair alignments. Task K versus weighted R cosines are positive inall4; natural K is almost orthogonal. This weakens immediate aggregate conflict as the whole explanation, not later interference or finite-step failure. Full gradient/aggregation audit and eight parameter finite differences pass. Next a fixed finite-displacement comparison of aggregate versus first-minibatch directions, matched norms with1/10 local control, actual/fixed supports; no optimizer grid.

## Previous: removing momentum does not rescue causal response training

[M23](NOMOMENTUM_RESPONSE_COMPARISON.md). M23 completes eight fixed beta1=0 fits and audited native/training-target evaluations. Removing momentum fails to rescue retention: response error worsens versus matched control in all four starts; aggregate native margin/logprob improve1/8 each, equal-family margin5/8, joint-withheld margin3/8. Joint output KL worsensall4. Local M22 explanation is not a sufficient training fix. Close optimizer tuning; next measure family/objective gradient compatibility across all four starts, fixed training targets, before selecting one intervention. No semantic-unit/new-task claim.

## Previous: carried optimizer momentum explains the local damaging direction

[M22](RESPONSE_ADAM_DIAGNOSIS.md) replays five fixed Adam counterfactual steps, preserving all reference tensors. At steps80–83, carried first moment reproduces most/more-than-all sentinel damage; current-only matched-norm updates do little harm or help. Step79 still shows finite-step/support effects. This is a one-case local diagnosis, not a training fix. Next freeze eight beta1=0 fits (both intact/response objectives, fourstarts), retaining all other settings and evaluating full training retention/native benchmark/output fidelity against audited M20beta1=.9 references. No optimizer grid or sentinel tuning.

## Previous: exact replay finds smooth interference despite stable active features

[M21](RESPONSE_TRAJECTORY_DIAGNOSIS.md) exactly reproduces all final tensors of block71 treatment and tracks its largest-error training endpoint. Error .01395→1.24825: net smooth change+1.29089, support-change effect−.05659, first-order prediction+1.21566. The five largest increases have unchanged support. This points to continuous update interference for this post-selected case, not a general gate-failure claim. Next fixed steps79–83 diagnose current-gradient versus Adam-history contributions without changing the trained model or tuning against the sentinel.

## Previous: direct response supervision failed to retain its training targets

[M20](RESPONSE_SUPERVISION_RESULTS.md) executes the fixed8fits and unchanged704-direction native benchmark. Treatment worsens final training response error for all4starts, with40–79% concentrated in one endpoint per start; paired removal of that endpoint still leaves all4worse than control. Native margin transfer improves4/8 aggregates (equal-family3/8), logprob response2/8; ordinary joint-eval KL worsensall4. This is an optimization/retention failure before a clean generalization test. Next bounded exact replay of block71 treatment with support/update instrumentation; no coefficient/learning-rate sweep or semantic-unit claim.

## Previous: explicit native-response supervision is ready for a fixed test

[M19](RESPONSE_SUPERVISION_TARGETS.md) freezes the direct native-response objective and collects480 scheduled train-only perturbation targets (57inactive retained). Four training-only weights3.08–11.40 equalize initial mean KL/response contributions. Full cached-target audit and8model/region gradient checks (16finite differences, maxrelative.482%) pass. No response-supervised fitting yet. Next execute8fixedfits, preserving the old704 M18/M11-based native evaluation directions rather than recomputing them from the newer starts. Family loss concentration and different active-target counts are explicit limitations.

## Previous: context fitting and native causal fidelity separate

[M18](CONTEXT_TRAINING_NATIVE_FIDELITY.md) executes eight paired120-update fits plus initial baselines and704 frozen native perturbations. Expanded contexts cut lexical-evaluation KL54–61% on training templates, but withheld-template gains are uneven and joint-eval top1 falls in allfour arm comparisons. Native margin-response fidelity improves3/8 model/region aggregates (equal-family4/8; joint-withheld subset6/8), logprob-response4/8. Better ordinary prediction does not uniformly imply better causal fidelity. Next freeze direct native-response supervision versus intact-output continuation, using training-only targets and unchanged exclusions; no matching/gain search.

## Previous: broader context supervision is available, with explicit gaps

[M17](CONTEXT_EXPANSION.md) freezes six wrappers over80 familiar pairs (four training templates, two excluded from training) and screens960 native endpoints.87/96 cohorts pass;33/40 training cohorts provide246 eligible context pairs across allten families. Past tense has only one passing training template; failures retained. Exact source-token separation and all token boundaries audited. No featurizer fitting yet. Next freeze a family-balanced original-vs-expanded context comparison and native-vs-reconstructed fixed-perturbation validation at both prefix/task sites, preserving whole-task exclusions and reporting all three nontraining lexical/template quadrants.

## Previous: task-only interventions do not restore stable roles

[M16](CONTEXT_REGION_ROLES.md) separates prefix/task interventions across all groups and controls. Actual-block task effects have .757–.918 times whole squared margin-effect energy; interactions .004–.016. Task-only geometric prediction errors .840/.747 remain partial (equal-family .978/.866), close to whole-position .854/.726. The large-gain failure is task-local under exercise and prefix-local under example, so ignoring prefix tokens is no general repair. Close localization; next expand context coverage across allowed task families, preserve lexical/task exclusions and whole-template evaluation, and require causal validation on native as well as reconstructed computation before unit claims.

## Previous: context changes expose partial correspondence and fragile calibration

[M15](CONTEXT_ROLE_TRANSFER.md) tests two fixed preambles with frozen models, group matches and gains. Eleven of twelve native family/context cohorts pass; plural nouns under the exercise preamble fail and remain reported. Actual-block geometric prediction errors are .854/.726 versus same-family original .515/.551 (zero predictor=1). Equal-family exercise error is 1.044, so correspondence is partial. Functional/random errors are dominated by a fragile gain of 93.84 on a near-silent training group, not independent evidence against the ontology. Intact replacement also loses fidelity. Next: one bounded prefix-only/task-only intervention decomposition, with all groups/controls and no gain refitting, to identify which causal targets could improve the featurizer.

## Previous: simple energy controls do not explain the strongest role match

[M14](GROUP_ROLE_ENERGY_BASELINE.md) fitscachedtraining-only energy/meanbaselines. Seed72actualblockerrors(sharedenergy/pergroupenergy/mean)=.998/.947/.882 versusfrozengeometry-basedcrossseed.551. Theadvantagesurvivesantonymomission,butdoesnotestablishsemanticunitsorruleoutnonlinear/contextualsensitivity. Stopoldpanelpredictorrefinement;nextfixedcontextchallengewithnativecompetencegatesandfrozenmatches.

## Previous: some causal correspondence survives, but task units remain unverified

[M13](GROUP_CAUSAL_ROLES.md) evaluatesall32decodedgroups withfrozencodes. Actualblockgeometrymatching predicts crossseedevaluationmargin effects better(relativeerror.551)thanrandom(.907)ortraining-effectmatching(.881);shuffledgroups~1.00. Butoneantonympaircontributes60%targetenergy, andpergroupagreementisweak. Concentrationsensitivitypreservessmallergeometryadvantage. Nextcachedremoved-energybaseline testswhetherthis isselectivity orgenericdisruption; no newmodeltuning.

## Previous: block dimensionality is real, block identity remains weak

[M12](BLOCK_STRUCTURE_AUDIT.md) auditsfrozenmodels. Mixedblocks use~3.4of4conditionaldimensions, butarbitrarygroupingsalso lookmultidimensional. Crossseedmatchedblockoverlap.116 versusfull128Ddictionaryoverlap.690; groupingreference.077,ambientnull.006. SupportJaccard.212 versusambientreference.141. Thesearecandidateblocks,notverifiedmanifolds. Nextfixedcausalgroupablation/rolematchingwithscalar/shuffledcontrols,weightsfrozen.

## Previous: diverse supervision improves lexical task transfer

[M11](DIVERSE_CAUSAL_SUPERVISION.md) executes the frozen mixture comparison. Mixednatural/taskKLtraining cuts source-separated taskKL61–77% versus continuednatural-only, inall4comparisons, while naturalKLis3–10%worsethanthatcontrol. Only14evalpairs/6families;4otherfamiliestraining-only. BlocksbecomecompetitivewithscalarontaskKLbutremainworseonnaturaltext. Nextfreezecheckpointsandtestblockstructure/stability/selectivecausalroles, ratherthanassumingbetterfidelitymeansmeaningfulmanifolds.

## Previous: stricter task split before diverse supervision

[M10](DIVERSE_SUPERVISION_SPLIT.md) finds17connected source-input components, including cross-family/distractor overlap. The audited split has66trainingpairs and14evaluationpairs across6families;4families cannot support lexical evaluation and aretraining-only. No changedsource token appears in opposite-split taskprompts. Exact schedules freeze natural-only versusmixed natural/taskKLtraining fromM8causalcheckpoints. Training has NOT run yet; data arepreviouslyexamineddevelopment, notfreshholdouts.

## Previous: diverse-task transfer is real but uneven

[M9](FEATURIZER_TASK_TRANSFER.md) evaluates all8frozen models on80existing development pairs across10families. Downstream training improves overallKL13–23%, smaller than natural-text gains, and taskTop1improves. Capital recall and number successor worsen in all4all-positionKLcomparisons despite100%binaryaccuracy; capital final-position/marginfidelityalso worsens. No newtaskdiscoveryclaim. Next audit lexical/context splits before a bounded diverse causal-supervision comparison, retaining natural-text forgetting checks and originaltaskexclusions.

## Previous: downstream supervision changes what the featurizer preserves

[M8](CAUSAL_FEATURIZER_OBJECTIVE.md) fine-tunes identical scalar/block checkpoints using reconstruction or native-outputKL. KL supervision cuts evaluationKL59–63% while raw/token-context reconstruction gets slightly worse. Scalar remains better than block; neither is near exact. Gradient paths were finite-difference checked, and the audit documents one near-tied TopK boundary. Next freeze models and test diverse-task transfer on the already-opened80-pair development panel, preserving original task exclusions.

## Previous: actual MLP1-write featurizer comparison

[M7](WRITE_FEATURIZER_PILOT.md) fits scalar, disjoint-block and fixed-overlap models on32documents and evaluates32different reused development documents. Scalar outputKL.292–.304, block.424–.434, fixed-overlap.524–.567. Scalar/block both activate32coordinates; shared unions average28.3–28.4. All128atoms are used, but none is close to faithful at this budget. Next compare explicit downstream-KL training against continued reconstruction training from identical saved checkpoints; no architecture/width sweep or semantic-manifold claim.

## Previous: equal local errors cause unequal downstream damage

[M6](MATCHED_ANGLE_CONTROL.md) matches MLP1 token-table replacements' per-position norm, angle and squared error using random tangent directions. Table KL0.18354 versus random0.04561–0.04692; table damage is larger on every document for all3draws. Error orientation/joint structure matters beyond magnitude. The fixed control series is complete; its deliverable is a natural-text causal benchmark for the next featurizer comparison, not discovered semantic units.

## Previous: token-table errors matter downstream

[M5](TOKEN_TABLE_CAUSAL.md) tests opposite-document-half token means on the same64natural-text windows. MLP1 table replacement causes0.184outputKL and+0.192natsCE, versus0.00143KL for a norm-only control. MLP2 has larger local error but less output damage. Directional fidelity matters beyond token means, but finite table-estimation error remains confounded with context dependence. Next compare angular perturbations at matched local error before claiming special causal directions.

## Previous: natural text changes the token/context balance

[M4](NATURAL_TOKEN_CONTEXT.md) repeats the partition on 64 reused natural-text training documents. Within-token variation is 29.5%, 40.8% and 53.8% for MLP0/1/2 among repeated-context tokens, much larger than the short-prompt result. Opposite-document-half token means improve over a global-mean predictor by only 11–13% at MLP2, on 58–60% covered evaluation positions. The small-contextual-correction description is not general. Next test downstream effects of cross-document table replacement with a norm-only control.

## Previous: token geometry versus context

[M3](TOKEN_CONTEXT_PARTITION.md) partitions cached native writes. Among tokens observed in multiple distinct contexts, deduplicated within-token variation accounts for 10.1%, 13.8% and 19.0% of centered MLP0/1/2 write energy. The contextual component spans more directions than the token-conditioned means on this panel. This is descriptive, not a token-table generalization or causal dispensability result. Natural-text replication is the next evidence gap.

## Broader question: which computational mechanisms explain these observations?

The [mechanism framework and first toy model](COMPUTATION_STRATEGIES.md) separate the function
computed, its channel implementation, and downstream sensitivity. A four-cluster token function
can be implemented by duplicated contributions or cancelling ones with identical intact outputs.
The current MLP1/MLP2 contrast also confounds pruning fraction; matched layer measurements come
before assigning different strategy labels. The removal evidence below remains one part of that profile.

The [matched MLP0/1/2 comparison (M1)](MATCHED_LAYER_MECHANISMS.md) now finds net constructive
channel interference in all three layers on the task panel, with decreasing collective alignment
from MLP0 to MLP2. MLP2 has less output KL damage despite greater local relative write error at
25–50% deletion. Thus the earlier redundancy-versus-cancellation dichotomy is not supported as a
general layer classification. Function, implementation geometry and downstream sensitivity stay separate.

A [trained capacity/noise toy screen (T2)](TOY_CAPACITY_RESULTS.md) now tests the user's extremes
question. With spare output-rank capacity, penalizing large contributions improves channel-deletion
robustness at nearly unchanged clean fit. Under rank pressure, benefits depend on the metric; input
noise robustness is a separate tradeoff. This is a controlled toy result, not an LLM training explanation.

The [same-task width comparison (T3)](TOY_WIDTH_RESULTS.md) removes the rank bottleneck and
finds a sharper robustness distinction: noisy-input training produces class-specific channels,
which are selectively vulnerable to single-channel removal. Clean-trained penalized models are
more distributed and tolerate those removals better. These toy outcomes do not establish an
unavoidable tradeoff or identify the LLM's training strategy.

The fixed toy grid is now complete. [T4](TOY_DROPOUT_RESULTS.md) finds that extra capacity plus
channel-dropout training can preserve noisy-input classification while improving deletion robustness;
noisy-vector fidelity still worsens. Read the [compact strategy synthesis](STRATEGY_SYNTHESIS.md)
for what this says—and does not say—about computational mechanisms. Next work maps those signatures
back to the real model rather than expanding toy hyperparameters.

The [real-output spectrum check (M2)](EMPIRICAL_OUTPUT_SPECTRA.md) finds that repeated-prefix
weighting substantially affects apparent MLP1 dimensionality. Distinct-prefix centered r90 values
are48/59/61 for MLP0/1/2. Native contributions sit far above the toy's unconstrained factorization
bound, but that bound does not establish executable compression or removable units. Next separate
token-conditioned geometry from within-token context variation.

## Current removal evidence

Start with [the removal audit and direct deletion test](REMOVAL_REDUNDANCY.md).
The old 46-unit article-cluster result used mean replacement and a narrow average article margin;
it did not test arbitrary large literal deletions or full-output preservation. The historical claim
that layer1 merely repeats layer0 was explicitly retracted: a token table can re-cluster token geometry.

In the new fixed random-partition diagnostic, literally deleting a quarter of MLP1's native products
leaves forced-choice task accuracy unchanged but changes native top predictions at about6–8% of prompt
positions. Deleting half produces mean KL.030–.044, versus.598 for all products. This supports partial
robustness on the opened task panel, not unchanged computation or a corpus-wide pruning guarantee.

A [matched-magnitude control (D2)](DELETION_NORM_CONTROL.md) shows that weakening the native
write while preserving its direction causes4–10 times less KL than literal deletion. Thus deletion
damage is not explained by amplitude alone; similar directions can still differ in behavior.
These controls use the native write and are not pruned models.

The directory contains ten exploratory response/steering studies plus the D1 deletion and D2 norm-control diagnostics,
a pinned standalone model loader, original evidence, and runnable verification scripts. The response
studies below remain relevant to finding computational units, but must not be used as removal evidence.

## Related response and steering studies: not removal evidence

MLP1 is causally active even when its information overlaps earlier layers. It amplifies upstream
changes, including random ones ([P3](MLP1_RESPONSE.md)). Amplification alone is not semantic evidence.

Natural prefix MLP0-write transfer predicts the noun-position MLP1 context change much better than
local-only transfer:83% versus20% reduction in squared target error ([P4](MLP1_MEDIATION.md)).
This supports a cross-position contribution through attention1, without identifying a particular head.

The induced MLP1 writes across positions also carry substantial donor-output recovery: about34%
on their own versus38–40% for the full prefix intervention; freezing them leaves only1–3% recovery
([P5](PREFIX_RESPONSE_FACTORIAL.md)). Matched random-prefix edits do not recover the donor distribution.
This is evidence of a direction-dependent early computation on a small reused authored panel, **not
a verified compact feature basis or broad task understanding**. She/he transfer is notably weak.

Position localization ([P6](RESPONSE_POSITIONS.md)) shows that the changed word's MLP1 response
alone recovers about29% of donor discrepancy, versus34% for all positions and4% for the complement.
This is a source-token response, not a discovered neuron or compact basis. The four later nouns
provide only eight distinct source contexts, so further feature fitting on this panel would be
uninformative. Next work must vary prefixes before the edit and broaden competent-task coverage.

A [training-task screen (P7)](TRAINING_TASK_SCREEN.md) now tests this source response on76
single-token pairs across nine competent task families. Agreement, sentiment, pluralization and
past tense show partial donor recovery; binding and factual recall do not. Only sentiment and
binding have repeated token edits under multiple pre-edit contexts. This is a scoped causal target,
not a universal featurizer objective; whole-task evaluation exclusions remain intact.

A [fixed preamble test (P8)](PREAMBLE_TRANSFER.md) finds that the response direction remains
aligned across new prefixes (mean cosine .87–.98 in passing cohorts), with partial causal transfer.
Magnitude changes matter: an old response can outperform the new one until their norms are matched.
Only five of eight task/preamble cohorts pass native competence, so this is a qualified reuse result.

The [completed direction/norm factorial (P9)](DIRECTION_NORM_FACTORIAL.md) confirms that
agreement's large old-vector advantage comes mainly from magnitude. At fixed norm, the new
context direction performs slightly better. Next is a lexical-pair-held-out direction-sharing
test on agreement and sentiment, with oracle magnitude clearly separated from learned prediction.

A [lexical-pair holdout test (P10)](LEXICAL_DIRECTION_TRANSFER.md) finds that mean directions
estimated from other word pairs recover about29–31% of donor-output discrepancy at oracle native
norm, versus19–20% for the native local response. Their cosine with native responses is only.37–.59:
this is reusable steering evidence, not yet a faithful feature decomposition. Next is the actual
shared projection versus residual at native amplitude, without renormalizing either component.

The earlier [P1](NATIVE_MLP2_RESPONSE_PILOT.md) and [P2](MLP2_TOKEN_TRANSFER.md) studies distinguish
native MLP2 amplification from approximation-specific error attenuation. Read the parent contracts
below before comparing those experiments with historical compressors.

## Read this first

**Follow-up correction:** [the history audit](HISTORY_AUDIT.md) adds completed table-composition,
MLP2 compensation, and compiler results. Read it before planning a new experiment: the older
aligned-repair hypothesis was not established, and the compiler parent differs from this native inspector. The C512 compensation parent
is native apart from its MLP0.Down replacement; see [parent contracts](PARENT_CONTRACTS.md).

| Observation | Evidence | What it establishes / leaves open |
|---|---|---|
| An MLP0 article-related cluster can be changed without substantially changing MLP1's article-related signal. | [Original result](evidence/prior/mlp1_reads_mlp0_results.json); [historical synthesis](evidence/prior/EXCERPTS_RESULTS.md), §847 | Supports parallel encoding for that signal under that intervention. It does not show all of MLP1 ignores MLP0. |
| Small article-carrying clusters can be sufficient under patching without being necessary under removal. | [MLP1 localization](evidence/prior/article_mlp1_localization_results.json), [MLP0 localization](evidence/prior/article_mlp0_localization_results.json) | Readable clusters are handles on a distributed computation; this is not evidence that individual channels are identical. |
| A current-token table recovers 96.01% of MLP1's mean-ablation CE stake on covered positions, versus 90.27% for MLP0 and 76.98% for MLP2. | [Attribution result with protocol](evidence/prior/mlp1_residue_attribution_results.json); [corrected historical account](evidence/prior/EXCERPTS_FINDINGS.md) | Strong token-conditioned replaceability on the measured distribution. Not 96% explained variance, output KL fidelity, or semantic understanding. |
| MLP1 has a large mean-ablation stake: about 7.005 nats; the remaining 3.99% table gap is about 0.280 nats. | Same attribution result | High table recovery does not make the contextual remainder negligible or the whole module unnecessary. |
| Freezing attention0 alone barely changes that table gap; freezing attention1 removes about 19%; freezing both removes it. | Same attribution result | Consistent with distributed/context-dependent routes. Each arm refits a different model and denominator; these percentages are not additive causal shares. |
| Stable native gate selections at budgets 32/128/512 did not pass the registered causal comparison. | [Gate findings](evidence/prior/MLP1_GLOBAL_GATE_FINDINGS_2026-08-28.md) | Stable rankings are insufficient for a faithful sparse circuit. Does not exclude learned alternative coordinates. |
| MLP1's coefficient tensor is full multilinear rank, with 90% coefficient energy requiring 835 output / 937 input directions. | [Weight-only finding](evidence/prior/MLP1_IMPLICIT_FOLDED_TENSOR_V2_FINDINGS.md) | Simple Euclidean Tucker compression is unattractive. Weight complexity is distinct from computation used on data. |
| This session's small input-based predictors explain only 26.5% of MLP1 write variance at linear rank32; output-PCA32 oracle explains 36.8%. | [Early MLP prediction](evidence/current/BILIN18_EARLY_MLP_PREDICTION.md) | Token-table CE success and local vector prediction measure different things, on different panels. These results are not contradictory. |

The prior repo also contains later, more specific source/reader studies. In the included
rung489–492 receipts, the common native-state reader classification fails; source decomposition
at rung491 passes its gates; selective attention1→MLP1 reader intervention at rung492 fails its
selectivity/transfer gates. **Do not turn a source-attribution success into an established removable
semantic circuit.** See [the evidence guide](EVIDENCE_GUIDE.md).

### Secondary puzzle: MLP2 cancellation

The [MLP2 dossier](evidence/prior/MLP2_CURRENT_UNDERSTANDING.md) records that selecting 512 native
channels was worse than deleting the module: local selection added about .265 CE, versus .162
for deletion. A learned 512-product program improved this to about .051 CE in a separate refit
comparison, but still failed its replacement threshold. This is a useful nearby test of
**coordinate choice and cancellation**, not the same phenomenon as MLP1's token-class overlap.

## Run the model inspection

The directory is movable: the runnable entry point does not import either original workspace.
It contains the exact local model implementation and pinned loader in `vendor/`.
Weights are **not** bundled; the pinned checkpoint is about 2.1 GB on disk. Float32 parameters
alone need roughly 2.1 GB; loading and activations require additional memory. Use short prompts
first. GPU is recommended; CPU is supported by the interface but was not separately benchmarked.

Use Python 3.11+ and a PyTorch build appropriate for your hardware:

```bash
cd early_mlp_redundancy
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python inspect_model.py --download --device cuda --output my_smoke.json
```

`--download` permits Hugging Face to download only the pinned model config and weights.
Without that flag, the model must already be cached. The GPT-2 tokenizer uses `tiktoken`;
its first use can fetch tokenizer assets. For offline use, prepopulate its cache as well.
To use an explicit existing checkpoint directory:

```bash
python inspect_model.py --snapshot /path/to/snapshot --device cuda \
  --layers 0 1 2 --text 'The dogs near the house are playing.' \
  --save-activations activations.pt --output inspection.json
```

The CLI:

1. Verifies the checkpoint SHA256 and exact topology.
2. Captures normalized MLP inputs, Left/Right factors, products, writes, and surrounding residual states.
3. Checks its forward pass against the original `GPT.forward`/`Block.forward` path.
4. Verifies exact product reconstruction and the finite bilinear difference identity.
5. Measures individual and joint **zero-write** ablations, recomputing all later layers.
6. Reports output KL, next-token CE, and top1 agreement separately.

Zero-write removal includes the bias. It is **not** the historical mean-ablation protocol.
The default prompt is only an instrumentation check; its removal results are not a corpus-level
redundancy claim. The bundled [validation receipt](validation/smoke_results.json) passed with
exact native logit replay on 13 tokens. See [validation notes](VALIDATION.md).

### Use in a notebook or another script

```python
from inspect_model import load_model, encode, trace
import torch

model, receipt = load_model(download=True)
ids = encode('The dogs are playing.', next(model.parameters()).device)
logits, sites = trace(model, ids)
mlp1 = model.transformer.h[1].mlp
L, R, D = mlp1.Left.weight, mlp1.Right.weight, mlp1.Down.weight
z = sites[1]['normalized_input']  # [batch, token, 1152]
h = sites[1]['products']         # [batch, token, 4608]

# Custom interventions return a WRITE, not a replacement residual state.
edited, _ = trace(model, ids, replace=lambda layer, z, write:
    torch.zeros_like(write) if layer == 1 else write)
```

`trace` is inference-only (`no_grad`). For gradient experiments, adapt it deliberately and verify
native replay again. To test a learned token table, return the table write at covered positions
and the original write elsewhere. Fit the table on separate documents before evaluating it.

## Architecture facts that matter

- 18 blocks; residual width 1152; 9 heads of width128; 4608 bilinear products per MLP.
- For normalized input `z`, `MLP(z) = Down(Left(z) * Right(z)) + Down_bias`.
- Every block mixes its incoming residual with the original normalized token embedding using
  learned `lambdas`, then adds attention and MLP writes with RMSNorm at their inputs.
- Attention reuses the first block's value bus. A residual-only intervention does not erase all
  token information available to later layers.
- Attention uses a product of two QK scores with a causal mask, not softmax attention.
- Final logits use RMSNorm, unembedding, and `30*tanh(logits/30)`.
- Tokenizer vocabulary is 50,257; output vocabulary is padded to 50,304. KL here uses the full
  output distribution, including padded logits, to match native behavior.

Ignoring the embedding reinjection, value bus, normalization, bias, or softcap can manufacture
apparent redundancy or disagreement between implementations.

## A bounded next experiment

First read [HISTORY_AUDIT.md](HISTORY_AUDIT.md). The initial proposal in
[RESEARCH_QUESTION.md](RESEARCH_QUESTION.md) describes one fixed set of token-conditioned replacements
and a 2×2 MLP0/MLP1 intervention factorial, scored on untouched documents and competent behavioral
contrasts. No new featurizer architecture is needed to ask this question. The first goal is to
separate shared predictable content from a context-dependent correction and downstream compensation.

## Contents and provenance

- `evidence/current/`: selected reports, JSON results, figures, and historical scripts from this session.
- `evidence/prior/`: focused earlier results, findings, dossiers, and line-numbered corrections.
- `vendor/`: source copies used by the runnable loader; these are code, not model weights.
- `PROVENANCE.json`: original locations, source hashes, and source repository commit.
- `MANIFEST.json` and `verify_packet.py`: integrity check for all distributed files.
- `validation/`: instrumentation results only; no new research benchmark is claimed.

The old research scripts are archival evidence. They can require external datasets, fitted arrays,
and other source files; they are not standalone reproductions. Original links inside copied reports
may refer to that larger repository. The guide supplies the local entry points and provenance
needed to recover those dependencies. This packet excludes training corpora, activation banks,
fitted compressors, and checkpoint weights. It is a curated handoff, not a complete research backup.
