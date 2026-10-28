## H758 — Shared groups do not improve the verified causal representation

Frozen H757 learned correlation-initialized groups are compared with three degree-preserving rewires per dictionary, scalar Top8, and full-head references. Every one of the three dictionaries is retained. Each endpoint is reconstructed independently; F(donor)-F(receiver) is added to the native head at both selected positions. The downstream Q2/K2 states are free, fully clamped, or operation-coordinate-clamped, as in H756. No task fitting or causal seed selection occurs.

| Head / representation | Total-effect error | Full-node-dependence error | Operation-dependence error |
|---|---:|---:|---:|
| H3 scalar8 | 47.7% | 36.1% | 44.0% |
| H3 learned shared groups | 55.4% | 55.9% | 57.6% |
| H3 degree-preserving nulls | 62.2% | 55.8% | 60.7% |
| H5 scalar8 | 80.0% | 68.3% | 52.8% |
| H5 learned shared groups | 88.8% | 75.6% | 59.7% |
| H5 degree-preserving nulls | 100.6% | 76.6% | 62.2% |

These are mean seed-level relative RMS errors on120 native-competent narrative contexts, not reconstruction MSE or independent statistical replications. Null means average three rewires per dictionary, with each dictionary equally weighted. Failed list cells remain separately available. Reference effect RMS is0.3572/0.0234/0.0195 logits at H3 and0.3790/0.0390/0.0369 at H5 for total/full-node/operation targets.

Learned groups improve mean total-effect preservation over degree-preserving randomizations, but the operation-dependent advantage is small and inconsistent across dictionaries. At H3, full-node dependence is effectively tied with randomizations. Scalar selection has lower operation-dependence error than learned groups for every dictionary seed at both heads. H5 learned-group total-effect errors remain highly variable:88.2%,65.8%,112.4% across seeds, so grouping does not fix the instability found in H756.

Across all task contexts and both patched positions, learned groups use7.951 unique atoms on average at H3 and7.985 at H5. Simultaneous selected-group overlap occurs4.86%/1.46% of the time; corresponding null rates are1.37%/2.11%. The representation really permits overlap, but overlap alone has not delivered a better causal unit. The earlier natural-data active-count-matched control already showed that a slightly reduced union size is not sufficient to explain the reconstruction gap; there is no new count-matched causal control here.

Interpretation: these memberships encode some data structure, yet that structure is not clearly the verified operation computation. This rejects neither all overlapping-group architectures nor joint dictionary/gate learning: the dictionary and coefficients are frozen, gates optimize an energy proxy, and group sizes/anchors are constrained. It does undermine promoting the present groups to manifold or hierarchy labels. Adding a hierarchy now would introduce assumptions without an established functional benefit.

Audits:23,856 forwards =23,040 representation/head/clamp interventions +720 no-swap controls +96 self-swaps. H756 full and scalar margin references replay exactly. Shared-atom union agrees exactly with binary support; raw full-node clamps exact; compact coordinate error<=7.36e-7; norm error<=1.63e-6; no-swap/self-swap logit error<=1.24e-5. Complement scaling0.909–1.040; all input/source hashes unchanged, outputs finite. Zero original reserved-test evaluations; natural document pool26,112 unchanged.

Next: evaluate these frozen representations on the broader agreement, factual recall, binding, copying and sentiment/nuisance panel, retaining native competence failures and suppressing relative claims when full-head effects are tiny. This checks whether the current ranking is specific to the repeatedly examined operation panel before changing group objectives or adding hierarchy. Continue distinguishing task competence, total causal effects, component-specific effects and activation reconstruction.
