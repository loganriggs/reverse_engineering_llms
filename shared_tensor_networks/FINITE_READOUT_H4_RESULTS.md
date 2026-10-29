# Four-feature fixed-budget scaling

All120fits and independent audits complete. Five fresh teachers225001–225005, three random initializations226001–226003 per condition, and a separate teacher-start diagnostic. Prespecified recipe and endpoint are in FINITE_READOUT_H4_PLAN.md. Source budgets1280/5120, heads1/4/10, label noise.05, fixed200evaluation cap. No restarts, best-seed selection, added teachers, or development-teacher pooling.

Primary endpoint: average Gaussian transfer MSE over four whole held-out heads after128noisy labels/head, then average all three random initializations within each teacher. Five teacher aggregates are the independent units; individual task/initialization records are not extra teacher replications. Teacher starts are privileged diagnostics, not a practical method.

| Source labels | Initialization | Heads | Median teacher transfer MSE (range) | Mean fraction with all component cosines>.99 | Budget stops |
|---|---|---:|---:|---:|---:|
| 1280 | random | 1 | 0.474317 (0.0472375–0.542557) | 0.267 | 5 |
| 1280 | random | 4 | 0.027805 (0.00247558–0.0593095) | 0.333 | 3 |
| 1280 | random | 10 | 0.160603 (0.0220747–0.864886) | 0.333 | 8 |
| 1280 | teacher | 1 | 0.00189047 (0.000749648–0.00238492) | 1.000 | 0 |
| 1280 | teacher | 4 | 0.00114426 (0.000632037–0.00174179) | 1.000 | 0 |
| 1280 | teacher | 10 | 0.0015192 (0.00079908–0.00201151) | 1.000 | 0 |
| 5120 | random | 1 | 0.310219 (0.000944599–0.803101) | 0.333 | 4 |
| 5120 | random | 4 | 0.0332452 (0.000749736–0.10228) | 0.467 | 0 |
| 5120 | random | 10 | 0.000851505 (0.000462128–0.0122857) | 0.867 | 0 |
| 5120 | teacher | 1 | 0.000803213 (0.000547751–0.00098735) | 1.000 | 0 |
| 5120 | teacher | 4 | 0.00062132 (0.000530407–0.00077454) | 1.000 | 0 |
| 5120 | teacher | 10 | 0.000597858 (0.000462128–0.000851505) | 1.000 | 0 |

## Every teacher comparison

| Source labels | Initialization | Teacher | One head | Four heads | Ten heads | Oracle |
|---|---|---|---:|---:|---:|---:|
| 1280 | random | 225001 | 0.488257 | 0.0377707 | 0.508901 | 0.000409625 |
| 1280 | random | 225002 | 0.542557 | 0.0226997 | 0.140299 | 0.000495645 |
| 1280 | random | 225003 | 0.0472375 | 0.027805 | 0.160603 | 0.000712697 |
| 1280 | random | 225004 | 0.474317 | 0.0593095 | 0.864886 | 0.000435449 |
| 1280 | random | 225005 | 0.0529946 | 0.00247558 | 0.0220747 | 0.000667142 |
| 1280 | teacher | 225001 | 0.000749648 | 0.000632037 | 0.000979979 | 0.000409625 |
| 1280 | teacher | 225002 | 0.0012277 | 0.00114426 | 0.0015192 | 0.000495645 |
| 1280 | teacher | 225003 | 0.00189047 | 0.00134796 | 0.00194614 | 0.000712697 |
| 1280 | teacher | 225004 | 0.00238492 | 0.00174179 | 0.00201151 | 0.000435449 |
| 1280 | teacher | 225005 | 0.00189979 | 0.000946555 | 0.00079908 | 0.000667142 |
| 5120 | random | 225001 | 0.547703 | 0.0332452 | 0.000462128 | 0.000409625 |
| 5120 | random | 225002 | 0.310219 | 0.10228 | 0.000597858 | 0.000495645 |
| 5120 | random | 225003 | 0.000944599 | 0.000749736 | 0.000851505 | 0.000712697 |
| 5120 | random | 225004 | 0.803101 | 0.0416547 | 0.0122857 | 0.000435449 |
| 5120 | random | 225005 | 0.00975285 | 0.00539694 | 0.00492277 | 0.000667142 |
| 5120 | teacher | 225001 | 0.000547751 | 0.000530407 | 0.000462128 | 0.000409625 |
| 5120 | teacher | 225002 | 0.000589711 | 0.000549643 | 0.000597858 | 0.000495645 |
| 5120 | teacher | 225003 | 0.000944599 | 0.000749736 | 0.000851505 | 0.000712697 |
| 5120 | teacher | 225004 | 0.00098735 | 0.00062132 | 0.000564004 | 0.000435449 |
| 5120 | teacher | 225005 | 0.000803213 | 0.00077454 | 0.000768787 | 0.000667142 |

## Prespecified paired outcomes

- 1280labels, random initialization: four beats one in5/5teachers; ten beats one in2/5.
- 1280labels, teacher initialization: four beats one in5/5teachers; ten beats one in2/5.
- 5120labels, random initialization: four beats one in5/5teachers; ten beats one in5/5.
- 5120labels, teacher initialization: four beats one in5/5teachers; ten beats one in4/5.

The budget-dependent pattern persists at the larger size. At5120labels, ten heads beat one for5/5teachers and achieve median transferMSE.000852versus.310; all four component cosines exceed.99in13/15random ten-head fits versus5/15one-head and7/15four-head fits. At1280labels, ten heads win only2/5, whereas four heads win5/5. The preferred allocation therefore depends on budget and endpoint.

Teacher-initialized controls meet the component threshold in every condition, with much smaller transfer errors than many random starts. This demonstrates accessible good solutions and leaves basin-finding as an important practical limitation; it does not certify global optima or prove every failure has the same cause.

Scaling context: four hidden bilinear features in eight input dimensions, versus three in six previously. Ten independent heads can span all ten unordered hidden products. Both model size and label budgets increase; this does not isolate the effect of dimension alone. The direct sample-space training implementation passed polynomial-expansion and autograd checks before training. Exact population evaluation is independent polynomial expansion.

These are transfer outcomes at equal source scalar-label budgets, not equal input counts, parameter counts, or computational work. Four held-out128-label tasks add512labels equally to every allocation. TeacherRMSnormalization is a privileged simulation convention. Learning from four output heads need not be uniformly best, and the comparison cannot establish a universally optimal label allocation.

Independent audits replay seeded teacher/input/noise generation, nested budgets, all initializations, direct-forward Jacobians, polynomial population losses,6561-node component matching, polynomial baseline and adaptation normal equations. All unsuccessful and budget-limited fits are retained;32-label adaptation and all source-fit baselines remain in JSON.

Decision: close this fixed confirmation at120fits. Interpret mixed/negative outcomes without altering the recipe. Separate transfer improvement, component recovery and random-initialization robustness before planning any larger-network extension.
