## H896 — Frozen units have limited and uneven natural-text fidelity beyond component removal

The previous goal turn made progress: H895 located number below the failed leading PC and separated decoding from subspace stability. H896 now evaluates all ten unchanged H894 units on the remaining384 locked natural documents, indices128–511, at one previously fixed token position per document. These documents were used in broader research but not to fit or previously evaluate these units. They are not globally untouched or pretraining-disjoint. The original reserved test remains untouched.

Each unit independently replaces its final-query L12 bilinear cross term with its compiled rank1 approximation. The context-only and quadratic terms stay exact; all downstream computation runs live. Exact replacement and removal of the same cross term are controls. No label-based routing, gating, refitting, best-unit selection, or outcome exclusions. Results concern one local edit per document, not simultaneous edits throughout the model.

### Native-output fidelity

KL is native-to-edited next-token distribution divergence, in nats. A ratio below1 favors replacement over removal in aggregate. The worse fraction counts documents where replacement has larger KL. Top changes are counts out of384. Different units define different components, so their removal controls differ; ranking raw replacement KL alone would be misleading.

|Unit|Replacement mean KL|Removal mean KL|Ratio|Replacement worse fraction|Top changes replacement/removal|
|---|---:|---:|---:|---:|---:|
|irregular_A|0.000297|0.000345|0.861|0.656|2/1|
|irregular_B|0.000161|0.000150|1.071|0.805|4/1|
|new_regular_1_A|0.000260|0.000297|0.876|0.682|2/1|
|new_regular_1_B|0.000202|0.000160|1.264|0.818|4/3|
|new_regular_2_A|0.000177|0.000216|0.820|0.729|1/1|
|new_regular_2_B|0.000177|0.000138|1.277|0.867|3/1|
|old_transfer_A|0.000357|0.000574|0.623|0.432|4/5|
|old_transfer_B|0.000859|0.000640|1.342|0.852|7/6|
|original_A|0.000312|0.000445|0.700|0.578|5/0|
|original_B|0.000096|0.000102|0.940|0.771|3/1|

Six units beat removal on mean KL, four are worse. Nine of ten are worse than removal on most documents. Average benefits are therefore concentrated rather than uniformly distributed. The original A unit's ratio0.700 is similar to0.716 on the earlier128-document panel, but it now changes five top tokens whereas removal changes none. Low average distribution error does not guarantee preservation of the top token, and neither metric alone establishes a mechanism.

Paired document-bootstrap95% intervals for mean replacement-minus-removal KL exclude zero in favor of original A and old-transfer A, and against old-transfer B. The other seven intervals cross zero. These are marginal descriptive intervals across ten comparisons, not simultaneous corrected significance claims or a selection rule. All ten are retained. Old-transfer B was the failed number-recovery fit; its poor fidelity is compatible with that failure, but this comparison does not establish its cause.

Absolute errors remain small, but component-removal errors are also small. This substantially limits the inference from apparent low-disruption replacement. Next-token CE changes are saved separately as descriptive quantities and are not a language-model improvement claim. Natural inputs carry no grammar-task labels here, so this experiment does not measure number-task competence in natural text.

### Updated hypotheses and discriminating experiments

1. A recurring one-dimensional number signal exists: supported by most disjoint discovery fits; test larger balanced lexical splits.
2. Variance ordering hides number behind lexical variation: supported by H895; compare fixed unsupervised extraction rules on independently frozen panels.
3. Number requires a small subspace whose basis rotates: partially consistent, but low full-span overlap limits it; compare principal angles as discovery size grows.
4. A single unique direction exists and small-sample noise explains variation: unresolved; learning curves and bootstrap recovery separate noise from persistent variation.
5. Multiple redundant number directions implement similar effects: possible; displacement-matched interventions and joint ablations can distinguish redundancy.
6. Number direction depends on syntax/template: plausible from A/B differences; cross-template balanced discovery is needed.
7. A shared number direction has context-specific output writes: supported locally; fix the input axis while varying output-fit discovery data.
8. The contextual write is genuinely rank1: not established; held-out output-rank curves must improve causal fidelity, not just reconstruction.
9. Higher output rank is necessary: possible, though H893 rank7 gains were modest; compare ranks at fixed input axis and discovery coverage.
10. Output instability mainly reflects eight-endpoint discovery samples: plausible; fit size curves without changing axis or evaluation panel.
11. Natural text contains contexts absent from grammar discovery: plausible; balanced discovery coverage with prospective evaluation can test it.
12. Weak component influence explains small natural KL: directly supported by removal controls; always report component influence alongside approximation error.
13. A few influential contexts dominate aggregate improvement: supported by average/majority disagreement; characterize influence distributions without fitting an outcome gate.
14. A useful task-blind gate could separate supported contexts: possible but unvalidated; require an independently trained gate and untouched evaluation.
15. Top-token margins amplify small distribution errors: compatible with top changes at low KL; measure native top-two margins prospectively.
16. Number units are specific to agreement rather than copying: supported by synthetic selectivity; extend to other competent number-dependent tasks.
17. Lexical identity contaminates recovered coordinates: plausible; contrast-balanced discovery and new regular/irregular combinations discriminate it.
18. Compact input-output operators are more canonical than latent axes: not established by H894; compare operator stability at matched displacement and origin.
19. Coordinate-origin choices alter the apparent decomposition: established algebraically in H890; retain fixed origins and distinguish gauge-dependent pieces from full interventions.
20. Neither scalar nor block featurizers alone identify computation: consistent with the accumulated separation of decoding, geometry and causal fidelity; test reusable conditional operations against matched scalar/block/shared-dictionary alternatives.

The highest-priority unresolved split is discovery coverage versus intrinsic output rank, with weak natural influence as a required control. Next hold the number axis fixed and vary output-fit sample size, lexical/template coverage, and rank using separated discovery/validation partitions. This prevents input-axis changes from confounding the output question. Freeze resulting rules before any new confirmation panel; do not reuse the384 documents as fresh confirmation after tuning. Keep the failed unit and all current results in the ledger. Manifold validation and the original broader featurizer objective remain incomplete.

### Verification and scope

11,904 full-model forwards completed as planned. Exact replacement gives zero logit difference. Compiled encoder matches explicit projections to relative1.01e−15. All384 document-window hashes verified, document hashes unique, source/model/frozen-unit hashes unchanged, and every recorded KL finite. No reserved test evaluated and no increase to the26,112 natural-document pool. Broad goal remains active.
