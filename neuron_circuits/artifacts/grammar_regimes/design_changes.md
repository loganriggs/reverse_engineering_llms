# Disclosed design changes after preregistration

## 2026-10-18: simplify the structured grammar after failed calibration

The original generator defined latent entity traits through modular arithmetic
on three shared name digits and then used additional modular/conditional
relations.  This failed the intended semantic-generalization calibration:

- at 8,192 training facts, depth-4 rho=0 reached 99.4% fixed-template recall
  but only 32% held-out-pair accuracy;
- at 16,384 training facts, depth-4 rho=0 reached 96.8% fixed-template recall
  and 52.5% held-out-pair accuracy;
- direct subject relations generalized, while modular relations remained near
  chance or weakly above it.

These models mostly memorized a rule-generated finite table.  Interpreting
their activations as semantic rule representations would be invalid.

The generator is therefore simplified before the next manifest is created:
the three latent traits equal the three shared name digits; four relations read
role/position-specific traits and four perform conditional routing between
subject and object traits.  Answer tokens remain distinct from name tokens,
and all relations remain exactly balanced.  This preserves role binding,
multi-token parsing, conditionals, template transfer, and arbitrary exceptions
while removing nested modular arithmetic as a prerequisite.

All original manifests, checkpoints, predictions, and failure results remain
under `pilot/` and `load256/`.  New runs use a separate output directory and
manifest hash.
# 2026-10-20: coalition-overlap counting bug

The first channel-coalition analysis multiplied boolean indicator matrices.
NumPy boolean matrix multiplication returned whether any shared channel existed,
not the intersection count, making most overlaps exactly `1/32`. The locked
hypotheses were preserved, indicators were cast to integers before matrix
multiplication, and the analysis was rerun. The corrected result reversed the
initial overlap verdict: mixed coalitions overlap much less than pure coalitions.
A synthetic exact-count regression test was added.

## 2026-10-20: fixed expansion of same-manifest mixed initializations

The first preregistered-style batch of mixed-model initialization seeds 101–103
produced one recall+unseen-template gate pass and two models with high trained-
template recall but poor unseen-template transfer. Before functional alignment,
the replication set is expanded once to the fixed seeds 104–106. All six results
are retained. Gate-passing models support matched-interface inference; failed
runs are reported separately as exploratory evidence about optimization and are
not silently discarded or replaced.
## 2026-10-20: same-manifest cross-initialization functional alignment

- Froze the comparison set before functional alignment: reference seed 0 and
  seeds 101--106, all trained on the same manifest at width 96, depth 4,
  bilinear MLPs, and rho=0.5.
- Froze fixed/transfer fidelity gates at 0.97/0.95. Seeds 0, 101, and 104 pass;
  seeds 102, 103, 105, and 106 remain visible as failed-interface controls.
- Froze the seven alignment hypotheses and thresholds in
  `configs/grammar_mixed_gate_cross_init.json` before computing effects.
- Corrected a pre-analysis implementation mistake: discovery and evaluation
  initially both used template 6. Discovery now uses trained template 0 and
  evaluation uses held-out template 6 on disjoint facts. No alignment result
  had been computed before this correction.

## 2026-10-20: adaptive margin-coordinate alignment red-team

- The locked answer-vector alignment was nearly orthogonal across all
  initializations, including gate-passing models. Before running another
  analysis, froze a narrower shared-coordinate test: project exact channel
  effects onto each fact's correct-answer-minus-mean-alternative margin.
- This is explicitly adaptive, because it was motivated by the first negative
  result. Its thresholds and hypotheses are frozen in
  `configs/grammar_mixed_gate_margin_alignment.json`; it cannot retroactively
  rescue the original hypotheses.

## 2026-10-20: cross-initialization linear conjugacy

- After both native answer-vector and margin-coordinate alignments failed,
  froze a stronger gauge-equivalence test before running it: fit residual-state
  maps on template 0 and test both state prediction and exact Jacobian
  conjugacy on disjoint template-6 prompts.
- Included orthogonal, flexible ridge, and shuffled-pair controls. High state
  prediction without Jacobian conjugacy is not counted as the same local
  computation.
- The first implementation used only the constant final marker at boundary 0,
  yielding a near-zero R2 denominator and meaningless huge negative values.
  Before interpreting the run, residual maps were changed to fit all token
  positions. A same-template held-out-fact segment was also added so template
  shift and code non-equivalence are reported separately. The originally
  locked held-out-template hypotheses and thresholds were not changed.

## 2026-10-20: pure-rule cross-initialization controls

- Reused already-trained same-manifest rho=0 models at seeds 101--103 to ask
  whether the mixed model's cross-run non-alignment is specific to arbitrary
  load or generic to independent training.
- Locked native-effect and residual-conjugacy hypotheses before running. Seed
  102 fails the same 0.97/0.95 interface gate and remains an exploratory
  control; seeds 0, 101, and 103 form the primary matched-interface set.

## 2026-10-20: held-out operation-family tournament

- Froze four alignment competitors before intervention outcomes: scaled native
  permutation, global gate ridge, relation-conditioned gate ridge, and a
  down-projection/write-quotient map followed by the exact target suffix.
- Froze three finite operation families: minimal semantic transitions,
  arbitrary-fact swaps matched on relation and structured answer, and generic
  learned prompt swaps. All alignments use unrelated unlabelled anchor prompts.
- Froze a Fisher-effect norm gate of 0.1 so negligible own-target edits cannot
  make a zero prediction look faithful. Target-model own deltas are reported as
  ceilings and never used to fit a source-to-target map.
- The first execution stopped before fitting the first model pair because the
  relation mask covered one anchor template while activations concatenated
  three. The mask is now tiled in the same template-major order. No
  intervention result existed before this implementation fix.
- The original completed run failed its frozen 64-row effectful-ceiling gate in
  directions targeting the seed-0 pure checkpoint (53--59 semantic rows and as
  few as 7 generic-pair rows). Verdict logic was corrected so a positive claim
  cannot pass when its own family/layer is non-adjudicative. The original output
  is retained and rerun under that logic.
- Before the high-count replication, froze 256 examples per semantic transition,
  4,096 generic pairs, and 1,024 arbitrary swaps while preserving the 0.1 effect
  threshold. This increases power rather than relaxing the causal-effect gate.
- The replication stopped before fitting any pair because only 3,435 disjoint
  eligible generic pairs exist, fewer than the requested 4,096. The fixed count
  was reduced once to 3,000, still implying about 82 effectful rows at the
  original worst-case 7/256 rate. No replication intervention result existed.

## 2026-10-20: within-model mixed address router

- After cross-run arbitrary-swap maps failed, froze a within-checkpoint
  localization test at the final bilinear block. Predictors span raw IDs,
  complete Layer-2 state, pre-attention state, QK pattern, attention write,
  linear/quadratic pre-MLP state, and nearest-neighbor local atlases.
- The target is the exact residual write, not an activation label. Predicted
  writes are executed through final RMSNorm/unembedding. Relation+answer
  prototypes explicitly receive the hidden arbitrary label and are marked as
  leaky actuator diagnostics rather than address mechanisms.
- Selected changed facts and unselected rule facts use separately balanced,
  disjoint discovery/validation/evaluation identities. Template 0 versus 6
  separates fact generalization from template transfer. A shuffled-write
  quadratic control and a 0.1 Fisher-effect norm gate are frozen.
- The first execution stopped during feature construction because uint8 digit
  arrays were passed directly to the one-hot primitive. They are now cast to
  integer indices. No predictor was fit and no address result existed.

## 2026-10-20: mixed address location red-team

- The first address result found a sharp linear-to-quadratic transition only at
  the normalized state immediately before the final bilinear MLP. Before the
  location red-team, froze matched PCA/quadratic budgets at the full prior
  residual sequence, final pre-attention residual, attention write, their
  explicit self+cross expansion, and the post-attention pre-MLP state.
- Froze a pre-MLP rank frontier from 8 to 48, a shuffled-write control, exact
  tensor ceiling, and the original effect-size gate. This analysis is adaptive
  to the first router result and cannot retroactively confirm its hypotheses.

## 2026-10-20: finite final-attention address interventions

- After attention-write features failed but post-attention quadratic features
  improved, froze literal attention deletion, shuffle, global mean, relation
  mean, and leaky answer-prototype replacements. The exact bilinear MLP is
  recomputed after every replacement.
- Separately score decoder-write fidelity with the native post-attention state
  fixed and complete-route fidelity including attention's direct residual write.
  This prevents a direct OV effect from being mistaken for address modulation.

## 2026-10-20: weight-canonical address subspace

- Final-attention deletion retained most arbitrary-fact computation, while a
  learned pre-MLP quadratic chart worked. To distinguish a real compact tensor
  subspace from regression flexibility, froze discovery PCA ranks 8--96 and
  execute projected held-out states through the original bilinear weights.
- Added 20 equal-rank random subspaces, same-template and held-out-template
  evaluation, oracle-residual write R2, and the existing finite effect gate.
  No target write or answer label is used to fit the PCA basis.
## 2026-10-20: lock shared-manifold red team before running

After the weight-canonical PCA intervention succeeded, added a prospective red team that compares the selected-fact PCA subspace with the unselected-fact, pooled, bottom-PC, and foreign-initialization subspaces. The opposite- and foreign-basis tests retain the target status mean so that they test subspace orientation rather than mean replacement. The locked fork is: a large opposite-status deficit supports distinct memorization coordinates, while successful pooled/opposite bases support one shared causal address manifold. Bottom PCs and foreign-model bases are negative controls. No thresholds were changed after observing results.
## 2026-10-20: lock semantic-prototype/residual interventions before running

After rule-only and arbitrary-only PCA bases proved interchangeable, added a finite decomposition of the normalized final-MLP input into an oracle relation-answer discovery prototype and an example residual. Locked tests separately assess final-answer preservation and full eight-logit effect fidelity under prototype deletion, within-cell residual swaps, global residual shuffles, residual scaling, and wrong-answer prototype replacement. This explicitly red-teams the possibility that high answer accuracy from an oracle prototype is being mistaken for reconstruction of the native mechanism.
## 2026-10-20: lock complete-codeword quotient tests before running

The additive prototype/residual ontology failed: the prototype was weak on arbitrary facts, while a complete donor state from the same relation-answer cell reproduced the recipient effect. Before inspecting finer swaps, locked complete valid-codeword interventions across same cell, same answer/different relation, and next answer with same/different relation. Target-answer and donor-answer accuracies are reported separately. This tests whether the final bilinear tensor operates on answer-level equivalence classes without assuming Euclidean semantic axes.
## 2026-10-20: residual-swap coverage bug found by independent red team

The first semantic/residual execution looped over `range(4)` despite eight relations. Consequently, half of the within-cell residual panel was not swapped, and the median causal fidelity was spuriously approximately 1. The subsequent complete-codeword swap used a separate eight-relation implementation and contradicted this result, exposing the bug. Fixed the helper to cover all eight relations, added an assertion that every example is touched, retained all original hypothesis thresholds, and reran the experiment. The corrected artifact contains an explicit correction record.
## 2026-10-20: lock final gate-code dimension before running

The shared 48/96 state manifold and broad mixed native-channel bank leave two possibilities: genuinely high-dimensional gate variation or a compact dense gate code hidden by neuron coordinates. Locked a finite affine-subspace test on the 192 post-product channels across three pure and three mixed initializations. Discovery PCA, variance-times-write-ranked native coordinate subspaces, and matched random dense subspaces are executed through the untouched down projection and suffix. This measures variation around the discovery mean and is explicitly not labeled a sufficient-neuron count.
## 2026-10-20: lock low-rank gate red team before running

The initial gate PCA found unexpectedly high mixed-model fidelity at rank 8--32 but omitted rank zero and had one non-adjudicative pure panel. Before interpreting the mixed positive, locked mean-only, ranks 1/2/4/8, opposite-status, pooled, bottom-PC, top-native-coordinate, and 50 matched random-rank-8 controls on the three high-count mixed panels. The pure-versus-mixed comparison is not reused here.
## 2026-10-20: lock final-gate answer quotient before running

After the rank-8 dense gate code survived mean, bottom, random, coordinate, and cross-status controls, locked a semantic alternative: the seven-dimensional span of eight discovery answer centroids. The test compares this span with unsupervised PCA-8, answer-only and relation-answer oracle prototypes, the exact within-answer residual, and label-shuffled centroid spans. All candidates are executed through native weights. Oracle prototypes are explicitly interpreted as output-controller diagnostics, never retrieval mechanisms.
## 2026-10-20: answer-quotient indexing fix before results

The first execution of the locked answer-quotient test stopped before producing any result because uint8 answer indices were interpreted as an obsolete mask by PyTorch. They are now cast to long integer indices; hypotheses and thresholds are unchanged.
## 2026-10-20: lock permutation null after the single-shuffle cutoff failed

The real answer-centroid span reached approximately 0.96 fidelity, but one predeclared shuffled span exceeded the absolute 0.30 cutoff. Rather than relax that cutoff, locked 100 balanced-label permutations per model/status and require the real span to exceed the null 99th percentile by 0.40 with corrected empirical p at most 0.01. The original failed single-shuffle hypothesis remains in its artifact.
## 2026-10-20: lock exact upstream tensor-pair frontier before running

After validating the seven-dimensional final answer quotient, moved one boundary upstream using the bilinear tensor itself. The normalized final-MLP input is exactly decomposed into embedding plus the attention/MLP residual writes of Blocks 0--2 and final attention, all under the native shared RMS scale. This yields 64 ordered component-pair gate terms. Locked discovery-only pair ranking, affine mean-filled top-k frontiers, self/cross and final-attention group controls, 50 random 16-pair banks, template transfer, coordinate R2, and selected-versus-unselected K90. Exact gate/write/logit recomposition is required before interpreting the frontier.
## 2026-10-20: lock finite Block-2 gate bottleneck before running

The exact final tensor-pair expansion repeatedly ranked the Block-2 MLP residual write as its main interaction hub. To distinguish semantic importance from mere component magnitude and affine bookkeeping, locked literal finite replacements of the Block-2 post-product gate followed by exact recomputation of Block 3 and the output suffix. Compare discovery PCA ranks, native coordinate affine subspaces, random rank-64 subspaces, and an oracle answer-centroid span. Score both the full finite Block-2 effect relative to zeroing the gate and the final seven-dimensional answer coordinates.
## 2026-10-20: lock all-layer finite gate telescope before running

The literal Block-2 test found a dense rank-64 nonsemantic bottleneck feeding the final seven-dimensional answer quotient. Locked the identical finite replacement assay at Layers 0--3: discovery PCA, matched native coordinates, answer-centroid span, random rank-64 subspaces, final answer-coordinate R2, and tested K90. This tests a depthwise contraction and whether selected arbitrary facts become broader at early layers, without comparing incompatible tangent or activation metrics.
## 2026-10-20: lock pure-versus-mixed finite depth comparison before running

The mixed-only telescope found an expand--compute--collapse profile but little selected-versus-unselected separation. To test the earlier global-reorganization result with the same finite metric, locked three pure-rule and three mixed-memory checkpoints sharing the manifest. Compare discovery PCA bottleneck ranks and answer-centroid causal fidelity at every layer, with native exactness, template transfer, and effect-size gates. Paired list positions are replication comparisons, not coordinate alignments.
## 2026-10-20: lock high-count pure/mixed depth replication

The original finite pure/mixed comparison failed its 64-example signal gate because one pure seed-0 layer/template had 48 effectful examples; all mixed cells and the other pure runs were well above threshold. Retained that run and, before new outcomes, increased evaluation examples per relation-answer cell from 12 to 20 while preserving the 0.1 effect threshold, models, methods, and all substantive hypotheses.
## 2026-10-20: high-count panel feasibility adjustment before results

The first replication attempt stopped during panel construction before any intervention result because one selected relation-answer cell had 47 eligible examples versus the requested total of 48. Reduced evaluation count once from 20 to 19 per cell (total request 47), still predicting roughly 76 effectful rows from the original 48/768 rate. Thresholds and substantive hypotheses are unchanged.
## 2026-10-20: second high-count panel feasibility adjustment before results

The 19-per-cell attempt also stopped before interventions because another selected cell had 46 eligible examples versus 47 requested. Reduced evaluation count to 18 (46 total requested). This remains a 50% panel increase over the original run and predicts about 72 effectful examples in the limiting cell. No result existed and no hypothesis or effect threshold changed.
## 2026-10-20: final high-count panel feasibility adjustment before results

The 18-per-cell attempt stopped before interventions on a different selected cell with 45 eligible examples versus 46 requested. Set evaluation count to 17 (45 total requested), a 42% increase over the original panel and an expected limiting effectful count near 68. This is the last feasibility adjustment; no causal outcome existed and all gates remain frozen.
## 2026-10-20: regime-specific high-count panels after feasibility audit

Four prospective attempts to enlarge both regimes stopped during panel construction, before interventions, because progressively different selected arbitrary cells contained only 43--47 jointly correct examples. The original signal failure occurs only in one pure cell; every mixed cell already exceeds 64 effectful rows. The final replication therefore enlarges pure evaluation panels from 12 to 20 per relation-answer cell while retaining mixed panels at 12. The estimator, 0.1 effect threshold, models, and substantive hypotheses remain identical. Panel sizes are disclosed rather than filtering selected facts.
## 2026-10-20: pure trained-template signal-only expansion

The regime-specific replication produced 94 effectful examples for pure seed 0 Layer 3 on the held-out template but only 19 on trained template 0; every mixed cell exceeded 500. The qualitative hypotheses replicated, but the all-cell signal gate therefore remained false. Without changing the threshold or dropping the weak template, enlarged only pure evaluation panels from 20 to 80 per relation-answer cell. This is an adaptive signal-power replication; the earlier result and failed gate remain retained.
## 2026-10-20: pure signal-panel feasibility adjustment before results

The 80-per-cell pure expansion stopped during panel construction before interventions because one pure cell had 103 eligible examples versus 108 requested. Reduced evaluation count to 75 (103 total requested), still predicting about 71 effectful rows in the limiting trained-template cell. No threshold or substantive hypothesis changed.
## 2026-10-20: allocate unused pure validation rows to the signal panel

A read-only feasibility audit found minimum eligible pure cells of 90, 107, and 109 across the three runs. Because the finite comparison does not tune any hyperparameter on its reserved validation split, the final signal replication uses 12 discovery, zero validation, and 78 evaluation examples per pure relation-answer cell; mixed panels remain 20/8/12. Discovery and evaluation identities remain disjoint. This maximizes the limiting pure cell without lowering the effect threshold or filtering outcomes.

## 2026-10-20: lock the Layer-0 raw-token value-route tensor program

The replicated pure/mixed telescope localizes the first representational divergence to Layer 0 but does not say which prompt information constructs it. Layer 0 uniquely permits an exact regrouping of final-position attention values by raw token role. Locked a 14-by-14 bilinear expansion over the final-token boundary plus thirteen canonical source roles, ranked only by discovery residual-write variation. Finite mean-filled pair banks, rule-relevance masks, random pair banks, two templates, and three paired runs test whether mixed memory broadening is specifically carried by otherwise rule-irrelevant entity value routes. This assay intentionally leaves the native QK pattern fixed; its conclusions are restricted to value transport, and a QK red team is required next.

## 2026-10-20: lock the Layer-0 key/value identity red team

The exact role-pair assay found that mixed models, unlike pure-rule models, require variation involving rule-irrelevant digit value routes. Because every source coefficient retained the native attention pattern, this does not distinguish information in QK routing from information in transported values. Locked independent replacement of rule-irrelevant digit keys, values, or both before the Layer-0 final-position gate. Alphabet means remove identity smoothly; cyclic digit substitutions preserve natural projected-vector scale while making identity wrong. The direct attention residual and every other token/layer path remain native, restricting the conclusion to control of the Layer-0 bilinear gate.

## 2026-10-20: lock the finite identity-correction tensor paths

The key/value red team localized the mixed Layer-0 effect to transported digit values. To learn how later blocks use that code, locked a path-integrated decomposition of the actual finite correction from rule-irrelevant value erasure back to the native model. Because only the final token is changed, every causal-prefix state remains identical. Each downstream block is integrated along its realized chord and exactly expanded into identity, QK, value, both bilinear arms, and four bilinear-attention products. The resulting 729 three-block paths are ranked on disjoint mean-erasure examples and tested on held-out templates and cyclic wrong-digit substitutions. Random path banks and signal gates separate a meaningful reusable path program from a merely exact algebraic expansion.

The first execution crashed at process level before producing any result. A one-example diagnostic reproduced the algebra and localized the operational risk to thousands of tiny fused scaled-dot-product-attention calls. Replaced only that kernel inside the analytic tracer with the identical explicit softmax/value contraction; hypotheses, examples, quadrature, and metrics are unchanged. The native model forward remains unchanged.

The explicit-kernel rerun completed every trace but failed during report assembly because the generic nested-metric accessor was also used for the integer K90 field. Replaced that one report-time access with direct integer access and added model-level progress output. No numerical result was serialized or inspected, and no scientific setting changed.

The completed eight-point run failed its exactness gate: median all-path fidelity rounded to one, but rare large cyclic interventions had up to 0.104 state-relative and 0.094 centered-logit error. A targeted convergence audit on the five worst seed-0 selected held-out cases reduced the worst error to 0.0047 at 16 points and 6.7e-6 at 32 points. Locked a full 32-point rerun requiring numerical exactness plus stability of top-path fidelities and bilinear projections relative to the retained eight-point artifact.

## 2026-10-20: cross-scale executable path program

- The first natural-language bridge expands the attention-frozen Layers 7--10
  MLP Jacobian of the 12-layer FineWeb tensor model into all 81 `I/L/R` path
  types. Path ordering is learned only on discovery FineWeb and frozen for
  disjoint FineWeb, WikiText, and TinyStories.
- Before interpreting the result, an exact symmetry red-team was registered.
  `Left` and `Right` factors are exchangeable, so the canonical quotient sums
  them into a full bilinear derivative `B`, leaving 16 invariant `I/B`
  layer-subset paths. This separates learned layer-specific structure from a
  generic low-order residual expansion.

- The first 546M finite run left checkpoint modules in bfloat16. All 16 paths
  still recomposed the float32 analytic JVP below `9e-7`, but the 2%-norm native
  finite edit had negative fidelity even for the complete path sum. This is a
  precision mismatch, not a selection result. The run is retained in this log;
  the script now follows the other causal audits by converting embedding,
  blocks, and unembedding to float32 before rebuilding axes and rerunning every
  locked gate.

- A high-count 546M replication was preregistered after the 32-site pilot. It
  uses 256 disjoint sites per corpus and four times as much axis/path discovery
  data. Thresholds were tightened before execution. All nine gates pass; K90
  rises from 2 in the pilot to 3 in the replication.
- Endpoint-basis red teams repeat the exact program in PCA and Haar-random
  rank-32/rank-64 charts. Random charts remain low-order, but natural charts
  are consistently more compact and use a different top ordering.
- The first dynamic-attention boundary test freezes the four-path ordering from
  the high-count artifact. Complete MLP transport still explains most of the
  native suffix effect, while attention contributes a material transverse
  correction. The gate claiming that attention dominates the remaining MLP
  truncation error fails and is retained.
- The first two-QK autograd preflight stopped before data collection because
  the independent validator received cached inference tensors, which PyTorch
  forbids saving for forward-mode differentiation. The validator now clones
  the cached state and persistent value bus into ordinary tensors. No result,
  path ranking, hypothesis, or threshold existed to revise.
- The completed full-sequence two-QK program matches autograd and finite edits,
  and four `I/A` paths retain 0.997--0.998. Its locked `random p95 + 0.01`
  specificity gate fails, so the retained conclusion is low interaction order,
  not privileged layer identity.
- The exact singleton factorization into exchange-invariant `Q=Q1+Q2` and
  current-value `V` shows QK redirection dominates. Its matched-random
  specificity gate also fails and is retained.
- A prospective native-channel audit decomposed the same local Layers 13--16
  bilinear derivatives into all 4,608 CP channels. The declared 512-channel
  0.8-fidelity and 1,024-channel 0.9-fidelity gates fail in Layers 13--15, and
  the strict ranked-versus-random margin fails. Exactness, symmetry, OOD
  transfer, and the predicted broad early-layer execution all pass. The
  ranking is marginal rather than combinatorial, which is logged as the next
  red-team boundary rather than silently upgraded into a universal claim.
- Before inspecting any spectral result, locked an operator-atlas test on the
  same Layers 13--16 boundary. It compares uncentered SVD coordinates of the
  complete local derivative with ridge-predicted coordinates from the local
  normalized state, uses disjoint FineWeb discovery/validation/evaluation and
  WikiText/TinyStories OOD panels, and compares rank 64 directly with the
  retained 512-native-channel artifact. Oracle projection and state prediction
  are reported separately: the former is coordinate compression, while only
  the latter is an executable representation. Four sign-randomized covariance
  nulls test whether leading spectral outliers exceed preserved-marginal noise.
- Its first execution stopped before state or operator collection because the
  one-level config merge omitted the inherited suffix boundary fields. Added
  the already-fixed `layer=12` and `final_layer=17` explicitly. No result,
  hypothesis, threshold, example range, or analysis setting changed.
- The rerun likewise stopped before collection because the same shallow merge
  omitted the inherited fixed positions. Added the original positions
  `[31,63,95,127]` explicitly; scientific settings remain unchanged.
- The operator-atlas result was broad in Layers 13--15 but sharply spiked at
  Layer 16. Before the next computation, locked two red teams. First, center
  the operator family and compare the frozen mean to determine whether the
  spike is merely a common operator. Second, factor each of the first 64
  operator modes as a 32D-input-to-1,152D-output matrix and derive its state
  read vector directly from `Left`, `Right`, `Down`, and the input chart. This
  produces an exact read–transform–write test with inner ranks 1--32. Oracle
  projection is retained only as an equality check, not used for coefficients.

## 2026-10-20: downstream-relevance operator test

- After the centered Layer-16 result, locked composition of the common and
  contextual local operators with the exact frozen-attention Layer-17 residual
  scale and RMS-normalization Jacobian into a natural output chart. An
  independent autograd JVP is required before causal compression is
  interpreted.
- The first execution stopped before state collection because the reused state
  collector requires an explicit layer list. Added `layers=[16]`; no data,
  outcome, threshold, or hypothesis changed.
- The frozen-attention endpoint result was strongly compact. Before claiming a
  causal output program, locked a dynamic Layer-17 red team using the exact
  full-sequence two-QK JVP, independent end-to-end autograd validation, a
  disjoint 64-site discovery panel, and 32 sites per evaluation corpus. It
  evaluates the same frozen mean and exact-read local factor programs, while a
  direct dynamic endpoint spectrum remains an oracle coordinate bound.
- The first dynamic run stopped in the independent autograd preflight before
  any discovery/evaluation result because the frozen local axes are stored on
  CPU and the validation tangent inherited that device. The primal and tangent
  are now explicitly moved to CUDA. No analytic program, data, threshold, or
  hypothesis changed.
- After the dynamic state-chart program survived, locked a behaviorally
  weighted extension through the final Layer-17 bilinear MLP, final RMS norm,
  unembedding, and tanh logit cap. It reports full-logit, probability-weighted
  centered-logit (local Fisher/KL), true-token cross-entropy derivative, and
  correct-versus-native-best-competitor margin fidelity. Full suffix autograd
  validation is mandatory. This directly tests whether state-space
  compression survives the model's actual prediction objective.
- The 32-site behavioral pilot rejected every high-fidelity inner-rank gate
  despite exact autograd agreement. Before designing a loss-aware basis,
  locked a four-times-larger 128-site-per-corpus replication with disjoint
  FineWeb/WikiText/TinyStories ranges. It reuses the original atlas, programs,
  metrics, thresholds, and hypotheses unchanged.
- The 128-site replication confirmed that the state-optimal atlas is
  insufficient for logits and loss. Before inspecting an alternative, locked
  CE-aware selection from the first 256 frozen local spectral modes. Each mode
  keeps its exact weight-derived state coefficient; only the bank is selected
  using FineWeb reverse-mode CE relevance. Reverse/forward derivative agreement
  is required. Fixed banks are tested on disjoint FineWeb, WikiText, and
  TinyStories against the local-energy prefix and matched random banks. This
  tests metric-aware selection, not a new learned decoder or per-example oracle.
- The 1,024-mode sweep showed a broad behaviorally important low-energy tail.
  Before attributing that breadth to non-neuron coordinates, locked an exact
  decomposition of the same CE directional derivatives into all 4,608 native
  Layer-16 bilinear channels. A static marginal bank is selected on FineWeb
  and compared with random and per-example marginal banks on disjoint
  FineWeb/WikiText/TinyStories. This is a behaviorally weighted native-neuron
  test; it retains the caveat that marginal ranking does not solve the
  combinatorial subset problem.
- The native CE run required 4,096 channels under marginal ranking, while its
  per-example marginal heuristic performed worse than the static bank. This
  demonstrates that the heuristic is not a hindsight ceiling. Corrected that
  wording and, before a breadth claim, locked an exact coordinated greedy
  search using the full discovery contribution Gram matrix. Its frozen ordering
  is compared with the retained marginal ordering on the same evaluation
  panels; discovery and held-out curves are both reported to expose overfit.
- Before generalizing native breadth to executed computations, locked the
  directionality red team. A natural rank-32 Layer-12 boundary basis is
  propagated through the exact dynamic two-QK/MLP JVPs of Layers 13--16. The
  resulting site-specific incoming Layer-16 tangents replace the local PCA
  axes in the native-channel CE decomposition. Manual/autograd tangent
  agreement and direct-write/channel-sum agreement are required. Marginal and
  full-Gram greedy banks use the same disjoint discovery/evaluation panels as
  the local-PCA audit.
- The 128-site natural-tangent run remained broad but showed a large
  discovery/held-out gap. Before interpreting this as context-specific
  multiplexing, locked an eightfold larger 1,024-site FineWeb selection panel.
  Evaluation corpora, axes, budgets, exactness checks, algorithms, and
  thresholds are unchanged. This isolates selection sample size from the
  directionality result.
- The 1,024-site selection replication retained K90=4,096. Before interpreting
  this as density within each computation, locked a per-context audit on the
  same natural upstream tangents. It reports norm and target-projection
  effective support, cancellation mass, fixed-coefficient marginal subset
  fidelity, and a freely reweighted OMP span. The latter is explicitly a
  descriptive coordinate bound—not native execution—because its coefficients
  are allowed to differ from the model's contributions.
- Per-context reweighting compressed each 32D effect to about five vectors but
  supplied a different fitted subset and coefficients for every site. Before
  calling that an intermediate representation, locked a shared descriptive
  dictionary test. One 128-atom MOD dictionary is trained on FineWeb with
  four-sparse OMP codes and selected by disjoint validation. It is frozen for
  held-out FineWeb, WikiText, and TinyStories, and compared with uncentered PCA,
  frozen training prototypes, and 16 random overcomplete dictionaries. Codes
  still see the target, so success would establish a reusable descriptive
  language—not an executable predictor or native circuit.
- The shared dictionary did not clearly beat random overcomplete or prototype
  controls. Before any semantic labeling, locked an exact tensor-network
  canonicalization. For every site, collect the propagated forward tangent
  matrix `J`, native state `x`, reverse loss adjoint on the Layer-16 write `g`,
  and weight-derived effective input adjoint `h=B(x)^T g`, requiring
  `y=J h` exactness. FineWeb-frozen PCA spans derived separately from `J`, `x`,
  `g`, and `h` are compared on held-out/OOD behavioral fidelity and against
  matched random subspaces. This tests which side of the contraction supplies
  a useful shared coordinate system.
- The exact atlas found forward-PCA coordinates substantially better than
  state, raw-adjoint, effective-adjoint, and random spans, but rank 64 remained
  below 0.90. Before concluding that the shared causal interface is broad,
  locked direct optimization of an orthonormal projector inside the exact
  `J P h` contraction. Ranks 32/64/128 start from forward PCA, use fixed Adam
  settings, and select checkpoints only on disjoint FineWeb validation before
  held-out/OOD evaluation. This tests metric mismatch in PCA, not native
  execution sparsity or semantic meaning.
- Projected Adam improved minibatch objectives but every validation-selected
  checkpoint remained the forward-PCA initialization. Before treating that as
  a substantive null, locked a rank-64 optimizer red team with deterministic
  full-training-set gradients, orthogonal projection after each step, four
  smaller SGD learning rates, and the same disjoint validation/evaluation
  panels. This distinguishes optimizer/minibatch instability from genuine
  cross-context subspace breadth.
- The deterministic optimizer improved its full training objective but gave
  essentially no validation or OOD gain. Before calling the resulting breadth
  an inference-time property, locked an endpoint decomposition that addresses
  an external-label confound. The true-token CE adjoint is split exactly into
  the gradient of log-sum-exp (the model's probability-weighted prediction
  term) minus the gradient of the supplied target logit. A model-chosen
  top-1-versus-runner-up margin provides a second target-independent endpoint.
  Forward and component-specific adjoint spectra, frozen subspace fidelity,
  exact recomposition, and repeated-token prototypes are evaluated on the same
  held-out/OOD panels. This separates complexity of inference from complexity
  induced by asking many target-token questions.
- The CE split found that output-covector and Layer-16 write-adjoint ranks are
  already close, while the target-independent margin remains broad. Before
  treating each output-conditioned adjoint as a distinct mechanism, locked a
  shared suffix-pullback test. A single ridge map from final output covectors
  to Layer-16 write adjoints is fit across prediction, target, and model-margin
  endpoints using 576 FineWeb sites, with rank and regularization selected on
  192 disjoint FineWeb sites. It is compared with a shared scaled identity and
  separately fitted component-specific maps, then evaluated both at the write
  and through the exact Layer-16 bilinear/upstream contraction on held-out
  FineWeb, WikiText, and TinyStories. Success would reclassify broad adjoints
  as many output questions served by one stable operator; failure would imply
  materially context-conditioned suffix pullbacks.
- The zero-centered shared suffix regression selected the maximum allowed rank
  and performed worse than a full-rank fitted scaled identity for CE and model
  margins; its negative behavioral fidelities could therefore be a truncation
  artifact rather than evidence for conditional routing. Locked a direct red
  team that fits only a PCA-rank residual around the shared scaled identity,
  extends ranks through 768, repeats validation-only rank/regularization
  selection, and retains component-specific residual ceilings. This is the
  strongest simple static-operator alternative before moving to mixtures or
  state-conditioned maps.
- The identity-residual red team selected rank 768 and improved write-adjoint
  reconstruction, but remained far below behavioral sufficiency and sometimes
  produced negative CE fidelity. Locked the next conditional alternative: a
  shared rank-64 CP tensor maps an output covector and a 64D local-state chart
  to a write-adjoint correction around the scaled identity. It uses 256 output
  coordinates, three training seeds, validation-only checkpoint selection,
  state-shuffle controls, and the same held-out/OOD behavioral contraction.
  This tests a compact local router. Failure would not rule out global-context
  routing through attention, which the saved local state does not explicitly
  expose.
- The local-state CP tensor was seed-stable and state shuffling damaged several
  endpoints, but it did not improve aggregate validation over the static map
  and left margin behavior poor. Rather than add router capacity, locked an
  exact weight-derived backward program for the Layer-17 suffix. The final
  normalization covector is propagated through four path types: final residual
  or final MLP, crossed with direct mixed-state or attention response. All four
  must recompose the saved adjoint and exact Layer-16 behavioral contraction;
  one-, two-, and three-path subsets are selected only on a 64-row FineWeb
  discovery panel and frozen for held-out/OOD evaluation. This tests whether
  the right ontology is a short context-executed program rather than a small
  static matrix or feature basis.
- The first CE-aware execution stopped before any text example or gradient was
  collected because the checkpoint's `FullModel` wrapper is not an
  `nn.Module`. Parameter freezing now iterates over its embedding, block stack,
  and head modules explicitly. No result, selection, threshold, or hypothesis
  changed.
- The centered test showed that Layer 16's spike is principally a simple mean
  operator over broad contextual variation. Before inspecting downstream
  outcomes, locked composition with the exact frozen-attention Layer-17
  residual scale and RMS-normalization Jacobian into a separately discovered
  natural output chart. This compares the mean, exact-read factorized local
  program, and an oracle endpoint spectrum. An independent autograd JVP is
  required before causal compression is interpreted.
- The three-path suffix result makes the final bilinear-MLP backward response
  indispensable, but one path label still invokes all 4,608 native channels.
  Locked an exact channel audit for that path. Channel contributions are
  contracted directly into the existing 32D Layer-12 behavioral tangent using
  the exact Layer-16 direct-write tangent, avoiding a separate downstream
  approximation. Shared and endpoint-specific marginal banks are selected on
  64 FineWeb rows and frozen for held-out FineWeb/WikiText/TinyStories;
  per-context norm support, signed target-projection cancellation, and matched
  random banks are reported. Marginal ranking is explicitly not a global
  subset optimum, so a broad result will motivate symmetry-aware tensor modes
  rather than prove no sparse refactorization exists.
- The final-MLP channel audit produced endpoint-dependent regimes: prediction
  and target-logit effects reach K90 with 32--64 channels, while the model's
  top1-versus-runner-up margin needs 4,096 channels on FineWeb/WikiText and 512
  on TinyStories. Before calling the margin intrinsically dense, locked an
  exact linear decomposition into its top-1 and runner-up logit paths. Each
  individual bank and their frozen union are compared with direct margin
  ranking, and signed projection mass tests whether broadness is created by
  subtracting two cleaner output roles.
- Individual top-1 and runner-up paths are each native-sparse and reuse largely
  overlapping banks, but their cancellation-sensitive margin remains broad.
  Locked a symmetry-aware alternative to native channel counting. Discovery
  PCA defines 64D final-MLP input-state and downstream-adjoint quotients. The
  trained Left/Right/Down weights are contracted exactly into an
  output-coordinate × symmetric-quadratic-input tensor, whose SVD modes are
  executed back through native RMS normalization and the existing 32D
  behavioral tangent. A direct projected-CP implementation is the exactness
  control. This tests coordinate compression of the shared quadratic form; it
  does not claim that PCA axes or SVD modes are semantic.
- The 64D symmetric tensor is exact and internally compact, but its chosen
  input/output quotient loses most native path behavior. Locked a dimension
  scaling red team at 128×128 with modes through 128, preserving all discovery
  and evaluation panels. It must improve every endpoint materially and is
  compared to the saved rank-64 result before any claim that tensor modes
  compress the native computation.
