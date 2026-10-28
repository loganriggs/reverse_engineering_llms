

## H340 — Localizing head6 sensitivity amplification

Exploratory follow-up of saved H339 Grams. Under the common intact-state Fisher metric, total MLP response energy decomposes exactly into compiled squared energy, remaining squared energy and twice their cross term. Compare actual with six-draw/permutation averages of moment and donor controls. Joint document/control bootstrap conditions on the selected panel and frozen models. No new model outcomes.

|Control|Head6 energy ratio [95%]|Other-head energy ratio|Head6 share|Compiled fraction of excess [95%]|Ratio after removing top7 docs|
|---|---:|---:|---:|---:|---:|
|moment|3.90 [2.86,5.19]|0.80|76.3%|119.8% [102.0,138.4]|2.78|
|donor|4.16 [3.13,5.32]|0.83|76.9%|119.6% [105.1,134.1]|2.94|

**Finding.** Head6 grows3.90×/4.16× in absolute sensitivity energy for moment/donor controls; other heads collectively fall to0.80×/0.83×. The compiled squared term contributes about120% of the net excess because increased negative cross terms offset part of it. This is exact signed energy accounting, not a causal percentage or a probability. It localizes the amplification to the compact compiled output path without yet identifying its input causes.

|Control|Top1 document energy share|Top7 share|Energy Kish count|Documents control > actual|Median document ratio|
|---|---:|---:|---:|---:|---:|
|moment|16.1%|48.9%|18.4|122/128|2.54|
|donor|14.6%|49.8%|19.2|123/128|2.50|

Concentration is substantial, but control energy exceeds actual on122/123 of128 documents; removing the seven largest control-energy documents leaves2.78×/2.94× paired amplification. This removal is an exploratory sensitivity analysis, not an unbiased estimate for a different population. Energy Kish count describes concentration and is not statistical effective sample size.

Locked gates: absolute_boost2: PASS, compiled_majority: PASS, tail_robust2: PASS. Both controls must have>=2× amplification with ratio lower>1, compiled excess fraction>=50% with lower>.5, and>=2× amplification after paired top7 removal.

**Audit and limits.** Saved-row hash verified; head6 shares reproduce H339 to1.11e-16. All contributions use the same positive-semidefinite local output metric. No finite task-behavior, semantic interpretation or new fresh-data confirmation. Native weights unchanged.

**Next discriminator.** Decompose head6 compiled response into its16 output modes and identify whether a small set drives the amplification. Compare native, donor and moment-control input interactions in those modes before claiming context-dependent suppression or repair. Broad research goal active.
