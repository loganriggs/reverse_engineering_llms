## H566 — Equal-position query content changes gate strength substantially

Extend H565 with four insertion texts selected by tokenizer length before inference. Four-token group: “(as usual)”, “(in fact)”, “(for once)”. Five-token group: “(as everyone expected)”, “(as people said)”, “(as others expected)”. Within each source/task case and group, source token/position, final token/position and prefix through the source are identical. All candidates remain single tokens. These controls change query content, not the selected source's causal key or the assigned positional rotation.

Use the same fixed L8H3/L14H4 readers and intervention modes. There are 224 native prompts (96 reused H565 anchors plus 128 new insertion prompts), 448 predictions saved before edited-prompt evaluation, and 1,792 interventions. All 14 task/gap cells have 16/16 candidate and full-vocabulary correctness. This panel contains no native accuracy failure to explain.

| Task | Head | Gap length | Mean within-length coefficient range | Mean phase-only predicted change magnitude | Range / predicted change |
|---|---|---:|---:|---:|---:|
|binding|L8H3|4|0.02164|0.04040|53.6%|
|binding|L8H3|5|0.01737|0.05041|34.5%|
|binding|L14H4|4|0.13313|0.46530|28.6%|
|binding|L14H4|5|0.07584|0.57162|13.3%|
|copying|L8H3|4|0.02478|0.08800|28.2%|
|copying|L8H3|5|0.04344|0.08490|51.2%|
|copying|L14H4|4|0.06332|0.07083|89.4%|
|copying|L14H4|5|0.03225|0.08111|39.8%|

Identical phase predictions do not imply identical actual gates. The mean range across the three content variants is 13–89% of the phase-only change magnitude, depending on task/head/length. These range ratios are descriptive comparisons, not additive causal fractions or independent sample estimates. Source keys agree within 3.06e−5, much smaller than the observed gate differences.

The importance of query content is also visible in prediction error: for binding L8H3, the four-token “in fact” insertion makes phase prediction worse than the unchanged-gate baseline (error ratio 1.79), whereas “for once” is well predicted (0.129). At L14H4 binding, “as usual” is poorly predicted (1.81), while the equal-length “in fact” is better predicted (0.557). This is not captured by insertion length alone. All outcomes and per-source interventions remain in JSON; no favorable phrase is selected away.

The emerging object is a conditional identity reader: position modifies a gate whose strength also depends on destination content, with downstream consumers determining its final effect. This is more informative than a distance-only rule, but it does not yet identify a semantic query feature or a complete computation unit. The inserted parentheticals have different meanings; the result is a controlled position comparison, not a pure semantic-variable experiment.

Next H567: separate the two query–key factors of bilinear attention. With source key and position controlled, cross base/edited query factors independently and replay their local source-edge effects. Determine whether one factor supplies a relatively stable match and the other carries contextual adjustment, or whether their interaction is necessary. Keep copying/binding and both heads, report all counterexamples, and do not fit an abstract “role feature” from these labels alone.

Audits: equal-length groups verified; self-margin replay <=1.82e−5; source/manifest/predictions unchanged; finite outputs. H565–H566 total 2,560 interventions and 224 distinct constructed prompts. No new FineWeb documents; total 26,112. Broad goal active and incomplete.
