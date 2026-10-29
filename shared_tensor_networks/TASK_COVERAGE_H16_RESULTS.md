# d32/h16: input coverage beats task count at equal total labels

60GPU fits and240transfer fits complete and audited. Five fresh shared teachers, source tasks34/136crossed with512/2048inputs, two random starts plus privileged teacher start,3000full-batch Adam steps. Same source/adaptation noise SD.05, fixed recipe, no restarts. Primary transfer uses512labels for136hidden-product features;128label transfer is retained as an underdetermined secondary diagnostic.

## Primary512label transfer

Unit Gaussian target second moments; MSE1is zero prediction. Random medians average both initializations within each teacher first. All16component criterion is matched absolute Gaussian cosine>.99for every component; this is stricter than recovering eight components at the preceding width.

| Inputs | Tasks | Scalar labels | Initialization | Median teacher transfer MSE | All16components recovered |
|---|---|---:|---|---:|---:|
| 512 | 34 | 17408 | random | 1.67633 | 0/10 |
| 512 | 34 | 17408 | teacher | 0.00295661 | 5/5 |
| 512 | 136 | 69632 | random | 1.53638 | 0/10 |
| 512 | 136 | 69632 | teacher | 0.00208901 | 5/5 |
| 2048 | 34 | 69632 | random | 0.0322085 | 2/10 |
| 2048 | 34 | 69632 | teacher | 0.00188234 | 5/5 |
| 2048 | 136 | 278528 | random | 0.0134433 | 6/10 |
| 2048 | 136 | 278528 | teacher | 0.00179523 | 5/5 |

## Every random-start teacher aggregate

| Teacher |512×34|512×136|2048×34|2048×136|
|---|---:|---:|---:|---:|
| 242101 | 1.75409 | 1.69698 | 0.0446515 | 0.0326314 |
| 242102 | 1.67633 | 1.33241 | 0.0322085 | 0.00788399 |
| 242103 | 1.62862 | 1.53638 | 0.0215004 | 0.0165405 |
| 242104 | 1.60754 | 1.49683 | 0.0299575 | 0.00187624 |
| 242105 | 1.84445 | 1.86185 | 0.0964121 | 0.0134433 |

At equal69632scalar labels,34tasks with2048inputs beats136tasks with512inputs in ALL5teacher blocks. Medians are.0322versus1.536. In this regime, distributing a fixed budget over more input examples is much more useful than allocating it to the complete task-product span with only512inputs.

At fixed inputs,136tasks improves transfer in4/5teachers for512inputs and5/5for2048inputs. But this also supplies four times as many labels and more readout parameters. Full task span therefore helps conditionally, while being insufficient to guarantee useful random-start learning at low input coverage.

Both512input random conditions recover zero of ten full feature sets and have median transfer error worse than zero prediction. All20teacher-start runs recover all16features and transfer well. With2048inputs,136tasks recovers6/10random starts versus2/10for34tasks. This establishes a larger tractable regime, not robust recovery from every start.

## Explicit optimization witnesses

Every low-input random fit has a higher empirical training loss than its corresponding trained teacher-start checkpoint on EXACTLY the same inputs, labels and architecture. The counts across(512,34),(512,136),(2048,34),(2048,136)are10/10,10/10,9/10,5/10using a numerical margin1e-8. The60fit audit independently replays those losses. The checkpoint identities and margins are saved in summary.json.

A feasible lower-loss parameter vector rules out the higher-loss point as a global empirical minimizer. These are explicit numerical witnesses, not interval-certified arithmetic or proof of strict local minima. They do not show optimization explains all population error; poor fits can also generalize badly. Teacher starts are privileged and are not proposed as a practical estimator.

## Controls, resources and limits

Active parameter counts are5648/19520. Batched tensors allocate136readout rows for every case, but unused heads have zero gradients and stay byte-identical to initialization. They are excluded from active counts. Exact float32 initial functions match across input-count arms and task prefixes. The independently verified batched Adam implementation and CPU/CUDA gradient controls from the previous panel are unchanged.

All60fits complete3000steps with no nonfinite losses/gradients dropped. Independent literal forward/data checks, SVD transfer fits, Gaussian quadratic component metrics, exact Wick source/transfer errors, curve coverage, inactive-head and initialization pairing checks pass. Population evaluation is CPUfloat64; training is CUDAfloat32withTF32off. Known Gaussian initial/teacher RMS normalization remains an explicit simulation privilege.

Training each12model teacher batch took approximately6.2–6.6seconds on RTX5080, excluding CPU target construction/final exact analysis. Peak torch-allocated memory was about195MB and peak reserved memory about229MB; these are allocator measurements, not total device memory. The GPU had no external compute jobs before launch. Per-case records repeat their teacher-batch duration and must not be summed as60independent durations.

The source task/input counts, width, seeds and primary adaptation budget differ from h8. This is controlled scaling within this panel, not proof that one variable alone explains differences across panels. More task labels, head parameters, task geometry and optimizer dynamics remain coupled. No generic sample-complexity threshold follows, and the full-span theorem remains sufficient rather than necessary.

## Decision

Close this frozen grid. It supports increasing input coverage along with task diversity when scaling width; task count alone does not repair optimization. The explicit teacher-start solutions demonstrate representability and accessible good solutions under privileged initialization. Further scaling should preserve input-budget controls and memory accounting, not interpret the earlier low median task-diversity result as a universal prescription. The broader proposed scaling envelope remains unfinished.
