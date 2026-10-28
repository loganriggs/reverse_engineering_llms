## H1018 — Third-task transfer fails on new answer vocabulary; the last-position boundary captures little of the computation

Previous turn was progress: H1017showed separate number/valence subspaces with compact joint allocation. This turn selects country-to-language lookup from the existing H882competence bank, rather than assume arbitrary task labels correspond to native capability. First4country/language pairs in formatAprovide8discovery states; familiar pairs in formatBtest template transfer; last4pairs in bothformats test disjoint source countries and output language vocabulary. This historical bank was previously screened; only projected interventions are new, not untouched lexical evidence. No current-example filtering.

Native competence rechecked: both-endpoint binary accuracy100%in every cohort; full-vocabulary intended answer top1is87.5%discovery and100%in all three evaluation cohorts. Language is therefore stronger native capability evidence than H1017's binary-only sentiment label contrast.

Learn language4/8from8discovery contrast gradients at final-position preL11residual. Frozen controls are number4,sentiment4,previous joint4/union8,random4. New three-task joint4/6balances each task gradient matrix by its Frobenius norm. No evaluation gradients. This experiment measures language fidelity only for the new three-task spaces, so it cannot claim preservation of their other tasks.

| Projection | Familiar countries, templateB | New countries/languages,A | New countries/languages,B |
|---|---:|---:|---:|
| Language4 |20.7%|89.8%|91.9%|
| Language8 |9.0%|87.9%|90.4%|
| Prior number4 |99.6%|99.5%|99.6%|
| Prior sentiment4 |100.0%|100.2%|100.2%|
| Prior joint4 |99.8%|99.8%|100.0%|
| Prior union8 |99.7%|99.6%|99.8%|
| Three-task joint4 |95.4%|99.8%|100.5%|
| Three-task joint6 |38.2%|91.8%|94.1%|
| Random4 |99.7%|99.8%|99.7%|

Relative L2errors compare target-margin change from projected edit to full last-position edit. Task-owned language8transfers across wording for known answers but fails lexical/output-vocabulary transfer. Current two-task dictionary contains little useful language lookup effect. Top canonical cosine language4with number4or sentiment4is~0.128; previous joint4~0.111. These do not support atom reuse across the three tasks.

Crucial boundary check: full last-position donor-state patch recovers only17.4%discovery,20.3%template-transfer,10.8%newlexicalA,13.0%newlexicalBof the natural donor-minus-recipient output contrast (aligned projection). Full patch RMSis18.0/20.8/10.9/13.3%of natural donor change. Thus even perfect compression of this boundary would explain only a small channel. Other positions, downstream attention, embedding recurrence and first-layer value bus remain recipient-native. The experiment cannot be reported as reverse engineering language lookup as a whole. Full patch itself differs from true donor effect by80–89%relative error.

Discovery uncentered gradient participation ratios: poolednumber2.26,sentiment1.17,language4.17. Poolednumber rank1energy58.2%does not contradict H1015's~90%within each number separately. Language gradients use different answer contrasts for each pair, whereas number and sentiment use common output labels; the higher dimensionality therefore confounds computational complexity with output-vocabulary diversity. More rank or more supervision may memorize label readouts without identifying a reusable operation.

Leading hypotheses after this third-task test: output-specific read directions explain familiar-answer transfer; task labels alone do not identify a universal task subspace; positional transport/bus state is essential for most language effect; low-rank success for fixed-output number/valence is not automatically scalable; balanced gradient objectives spend a limited coordinate budget unevenly when tasks have different output spectra. This failure should redirect the benchmark boundary before more dictionary or gate tuning.

Audits: native margin replay4.77e-7absolute; unchanged frozen bases; disjoint train/test answer vocab; no evaluation gradients; independentTorch metrics exact; orthogonality1.78e-15;finite. Successful run32prefix sequences,1native sequence,352suffix sequences,8gradientexamples. First attempt completed computations but failed serializing a NumPyBoolean; converted that field to builtinbool and reran unchanged experiment. Total including retry64prefix,2native,704suffix,16gradientexamples. No source/test data mutated, no reservedBSFtest data; natural-document pool26112unchanged; no live process. Broad goal active,H961pending.

Next map the full donor effect across preL11source-position residual, other-position residuals, embedding recurrence and first-layer value bus with explicit finite patch controls. Full boundary transfer must replay donor behavior before using it as a benchmark. Establish which port interactions matter, then define task-conditioned functional features at the relevant source/transport interfaces. Retain number/valence anchors, but do not train an increasingly elaborate BSF on a boundary explaining only~10–20%of the new task. Gates, shared atoms and manifold tests remain downstream of adequate causal coverage.
