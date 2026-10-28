## H866 — Value-bus slice profiles differ across tasks, with partial cross-task reuse

The previous turn made progress through H864–865: local relay erasure preserved most decisions, while whole-value-bus erasure strongly reduced binding/capital/copy discrimination. H866 localizes those effects across all nine128-dimensional bus slices, retaining all five tasks and every slice. Capital retrieval concentrates most strongly in slice0, binding in slice4, and copying in slices3–4. This is a useful map for subsequent feature/reader tests, not a semantic feature ontology.

### Reconciliation with earlier evidence

H739 already established a token-local static bus map for an L13H0 reader and tested content substitutions across capital/binding/copy contexts. H778 decomposed an L5H5 reader's head edit into routing, mixed-value and interaction terms. H866 does not rediscover those mechanisms. It measures native contrast dependence on all head-index bus slices across the newer common five-task panel.

A bus slice is reused by that head index in multiple downstream layers. Erasing slice0 is not equivalent to ablating L13H0 alone. Likewise, earlier L5H5 copying evidence involved mixed local/bus values and routing; it does not require global bus slice5 to dominate copying. Context-dependent readers, local values and persistent token storage must remain distinct.

### Experiment and results

At the boundary after block1, retain native residual sequence and embedding anchor. For each of76 aligned pairs, use nativebus, whole-bus pair midpoint, each single-slice midpoint, and each all-but-one-slice midpoint:20 conditions and both recipients. Recompute all downstream states. Single-slice midpoint removes the pair's contrast in that128-dimensional slice at all positions. No head is selected before measuring the full profile.

Numbers below are percentage points of native paired answer-margin distinction removed, not answer error rates and not unique mediation shares. Negative values mean that erasure increases the native distinction.

|Task|0|1|2|3|4|5|6|7|8|Whole bus|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|agreement|-0.0|0.7|0.0|-1.9|-0.0|1.0|0.6|-0.6|3.9|4.3|
|capital_recall|47.9|0.0|2.6|8.5|15.7|2.1|1.3|-0.3|0.1|90.5|
|entity_binding|12.8|6.8|0.8|15.5|42.1|6.6|7.9|3.8|0.0|97.0|
|sentiment|3.5|15.8|1.4|1.3|-0.1|0.2|-0.1|1.0|0.1|25.1|
|sequence_copy|7.9|1.4|3.5|26.0|20.7|2.2|11.2|3.0|0.9|84.9|

![Value-slice contrast effects](/home/loganriggs/Documents/Codex/2026-10-21/he/outputs/value_slice_erasure.png)

The right panel measures how much native distinction is restored by retaining one slice while all others are midpoint-erased, relative to whole-bus erasure. This is a different conditioning state from the left panel.

### Shared storage and nonadditivity

Slice4 contributes42.1 points to binding,20.7 to copying and15.7 to capital retrieval when erased alone. That makes it a candidate for cross-task content reuse; it does not show that the same directions or computation inside the slice serve all three tasks. Different lexical items, readout maps and routing could explain the apparent sharing. A head-index block is an architectural partition, not a discovered manifold.

Binding's single-slice losses sum to96.3 points compared with97.0 for full-bus erasure; conditional effects are similar. Capital single losses sum to77.9 versus90.5 jointly, and copying76.8 versus84.9. Retaining slice0 in an otherwise erased bus restores56.9 points for capital versus47.9 lost when erasing it alone. Copy slice3 similarly restores31.4 versus26.0 lost alone. These observations show context-dependent, nonadditive effects under midpoint interventions; they do not determine a unique redundancy or synergy allocation.

Agreement's entire bus effect is only4.3 points, so slice8's3.9-point effect concerns a small route contribution. Sentiment has a stronger slice1 contribution15.8 points but remains dependent on residual and combined routes, as H865 showed. No representation should be judged against all tasks at this site alone.

### Split check and limitations

Choosing the largest single-slice effect using only discovery rows selects8/0/4/1/4 for agreement/capital/binding/sentiment/copy. Their transfer-row effects are3.4/38.5/43.8/15.5/14.4 points. The copying winner changes when considering the full panel: slice3 has the larger aggregate effect. All split/slice outcomes are saved. This is a held-out-from-ranking diagnostic on extensively reused research data, not fresh confirmation; shared lexical/template structure limits independence.

Next distinguish shared storage from shared computation by restricting the established bus intervention to specific downstream reader layers. Start with all readers for the same frozen slices, without selecting a layer from these new outcomes. Compare isolated reader erasure with global persistent-slice erasure, audit bus restoration for later readers, and use the existing static-map/routing analyses as controls. A shared slice used by task-specific readers supports a different ontology from one shared reader operation. Then test cross-task/lexical combinations before training or labeling another manifold model.

### Verification

3,040 suffix forwards. Native replay error4.48e-05; native/full-bus margins match H865 exactly. Captured bus shape verified as token×9×128, and unselected slices remain bitwise unchanged before resumption. All paired answer groups agree, inputs unchanged and metrics finite. No reserved-test evaluations,26,112-document pool unchanged. Pair-bootstrap intervals are uncorrected descriptive intervals and may be optimistic under shared templates. All9slices, both conditional settings and all5tasks retained. Broad goal remains active.
