## H867 — Shared value storage is consumed by task-dependent readers, with partial reuse

H866 localized five-task dependence across persistent bus slices. H867 now tests every downstream reader layer2–17 and every head index0–8,144 reader locations, plus native and global controls. Capital retrieval is concentrated atL13H0, binding atL14H4, and copying atL8H3. Several readers also contribute to other tasks, so the emerging organization is shared token storage with task-dependent routing/readout and partial reader reuse, not one universal semantic block.

### Intervention and scope

Resume from the boundary after block1 with native residual sequence and embedding anchor. At the selected reader layer only, supply the pair-midpoint value-bus slice, removing the native pair contrast in that128-dimensional slice. Other bus slices stay native. Supply the original native bus at every other layer, while propagating the selected reader's residual effects normally. Thus this isolates direct access to that slice at a particular reader without suppressing downstream consequences through residual/local-value/routing changes.

All76 aligned H843 pairs and all5 task families are retained. Include native replay, complete-bus midpoint and all9 globally erased slices to reproduce H866 controls. Rankings below are exploratory; every reader, including small and negative effects, remains in the saved results.

### Largest reader-local effects

Effect is the fraction of the native paired answer-margin distinction removed, not a fraction of model accuracy or a unique mediation share. Layer and head indices are zero-based.

|Task|Rank on all rows|Reader|Native pair-gap removed|Binary accuracy after edit|
|---|---:|---|---:|---:|
|agreement|1|L8H8|1.4%|100.0%|
|agreement|2|L7H8|1.1%|100.0%|
|agreement|3|L16H8|1.1%|100.0%|
|capital_recall|1|L13H0|48.4%|96.2%|
|capital_recall|2|L14H4|6.8%|96.2%|
|capital_recall|3|L10H3|2.7%|96.2%|
|entity_binding|1|L14H4|37.1%|100.0%|
|entity_binding|2|L13H0|9.2%|100.0%|
|entity_binding|3|L14H6|7.3%|100.0%|
|sentiment|1|L8H1|10.8%|100.0%|
|sentiment|2|L10H1|5.0%|100.0%|
|sentiment|3|L14H0|2.6%|100.0%|
|sequence_copy|1|L8H3|21.2%|100.0%|
|sequence_copy|2|L8H6|6.1%|100.0%|
|sequence_copy|3|L8H4|5.9%|100.0%|

![All reader-local effects](/home/loganriggs/Documents/Codex/2026-10-21/he/outputs/value_reader_erasure.png)

The shared scale shows the larger effects for capital/binding/copy. Agreement's value-bus route is small overall; absence of a large cell here does not imply that agreement lacks localized computation in other pathways.

### What is shared?

L14H4 contributes37.1 percentage points to binding,6.8 to capital retrieval and5.2 to copying. L13H0 contributes48.4 to capital retrieval,9.2 to binding and4.0 to copying. These are candidates for shared reader operations, but effect overlap alone does not show that the same directions or transformation serve every task. Different task vocabularies, source positions, query patterns and output groups remain confounded in the current panel.

H866's bus slice4 copying effect is spread across readers: L8H4, L10H4 and L14H4 all contribute. It is not solely the binding-dominant L14H4 operation. Conversely, capital slice0's global47.9-point effect is largely localized atL13H0's48.4-point marginal effect; slight overshoot is allowed because effects are conditional and downstream computation is nonlinear.

Summed reader-local effects approximately track global single-slice effects for several prominent slices: binding slice4 sums42.0 points versus42.1 globally; sentiment slice1 sums15.7 versus15.8. Capital slice0 sums46.8 versus47.9 and copying slice3 sums24.4 versus26.0. These are empirical comparisons, not proof of additivity or an exhaustive interaction decomposition.

### Prior evidence and split check

L13H0's role agrees with H739's static token-bus map and content substitution results. H778's L5H5 copying decomposition included mixed local values and routing, whereas this experiment isolates persistent bus inputs. These are different interventions; a small L5H5 bus-only effect does not contradict the earlier full-head result.

Selecting the strongest reader from discovery rows givesL8H8 agreement,L13H0 capital,L14H4 binding,L8H1 sentiment,L8H3 copying. Their transfer-row effects are1.2%,37.6%,39.1%,10.3%,24.2%, respectively. The transfer rows were withheld from this ranking, but have been reused throughout research; this is not a fresh confirmation. Bootstrap intervals are descriptive and uncorrected across many correlated tests.

### Research decision and next experiment

Before training another featurizer or declaring a shared computational DAG, separate lexical content from task/query role. FreezeL13H0,L14H4,L8H3 as prospective reader targets. Build a common-source-vocabulary panel crossing copying, binding and associated capital retrieval, with new combinations and task-appropriate answer groups. Keep the same source token contrasts and aligned source positions where possible; vary the query/role. Screen native competence and token alignment before interpreting interventions, retaining exclusions transparently.

At these readers, decompose native source-value contribution versus context-dependent routing using the existing signed-attention machinery. Test whether fixed content directions transfer across task/query combinations while routing controls their use. Include unchanged-source/query controls and held-out lexical combinations. This can distinguish shared storage consumed differently from a reusable reader computation; geometry or head-index sharing alone cannot.

The five-task map guides where a future featurizer belongs. It does not itself validate manifolds, unsupervised task labels, hierarchical gates or reusable computational DAG nodes. Those remain open requirements in the broader research goal.

### Verification

23,560 suffix forwards across155 conditions and152 recipient contexts. Native logit replay error4.48e-05; all native/global-control margins exactly matchH866. Every reader-local arm applied the edited bus at exactly one layer. Bus passthrough error is0, unselected slices remain unchanged before resumption, and the original bus is explicitly supplied to all nonselected layers. All paired answer groups agree, source hashes unchanged, outputs finite. No reserved-test evaluations;26,112-document pool unchanged. Only76pairs with shared templates are available;4096 pair-bootstrap intervals may be optimistic for generalization. Broad goal remains active.
