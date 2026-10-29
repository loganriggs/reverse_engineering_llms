# Finite noisy labels: diversity versus input coverage

72 fits: three previously used d6/h3teachers, source scalar-label budgets384/1536, one/three/sixheads, normalized label-noise standard deviation0/.05, random or exact-teacher initialization. Each source input carries all selected heads, so n=budget/heads. Input/noise pools and head prefixes are paired within teacher. Same fixedTRFsettings and200function-evaluation cap, no restarts, no validation selection. Teacher starts are privileged basin diagnostics, not a competing practical training method.

Targets use known Gaussian teacher RMS normalization, a privileged simulation convention shared by all arms. The source budget counts scalar labels, not input examples: at384labels the three allocations use384/128/64inputs; at1536they use1536/512/256inputs. Independent Gaussian label noise is added after target normalization. These are finite sample-label experiments, unlike the preceding population coefficient-noise study.

Four whole heads remain held out from representation fitting. Fit linear readouts on frozen hidden products with32or128fresh noisy labels per held-out head; adaptation noise matches the source noise level. At128labels/head this adds512scalar labels equally to every source comparison. True hidden products provide an oracle transfer baseline. A minimum-norm126-monomial quartic regressor provides a source-fit baseline; it has different capacity and is underdetermined in some low-input conditions. All baselines and32-label results are retained in JSON.

## Noisy-label transfer (standard deviation .05)

Primary displayed endpoint: mean exact Gaussian MSE over the four held-out tasks after128adaptation labels/head. Table shows median across three teachers, followed by the full range. Counts of samples/parameters differ by allocation; this is a resource-allocation comparison, not pure causal isolation of head count.

| Source labels | Initialization | Heads | Median held-out MSE (range) | All component cosines>.99 | Budget-limited fits |
|---|---|---:|---:|---:|---:|
| 384 | random | 1 | 0.433827 (0.0809246–0.572946) | 0/3 | 3/3 |
| 384 | random | 3 | 0.00283721 (0.00246853–0.0749616) | 2/3 | 0/3 |
| 384 | random | 6 | 0.00454392 (0.00235643–0.0865021) | 1/3 | 1/3 |
| 384 | teacher | 1 | 0.00530968 (0.00184239–0.00546358) | 2/3 | 0/3 |
| 384 | teacher | 3 | 0.00246853 (0.0017514–0.00283721) | 3/3 | 0/3 |
| 384 | teacher | 6 | 0.00446033 (0.00235643–0.00454391) | 2/3 | 0/3 |
| 1536 | random | 1 | 0.00203386 (0.0010979–0.0462899) | 2/3 | 1/3 |
| 1536 | random | 3 | 0.000440237 (0.000436288–0.0383921) | 2/3 | 0/3 |
| 1536 | random | 6 | 0.000886878 (0.000470697–0.00995858) | 2/3 | 1/3 |
| 1536 | teacher | 1 | 0.0010979 (0.00045569–0.00203385) | 3/3 | 0/3 |
| 1536 | teacher | 3 | 0.000440237 (0.000436288–0.000685544) | 3/3 | 0/3 |
| 1536 | teacher | 6 | 0.00047241 (0.000470697–0.000886878) | 3/3 | 0/3 |

The oracle128-label transfer errors are .0001980/.0003739/.0004108 for teachers218001/2/3, identical across source allocations and initializations. Poorer learned-feature transfer can therefore persist even when the adaptation procedure itself works with the correct features.

## Every paired noisy transfer result

| Source labels | Initialization | Teacher | One head | Three heads | Six heads |
|---|---|---|---:|---:|---:|
| 384 | random | 218001 | 0.0809246 | 0.0749616 | 0.0865021 |
| 384 | random | 218002 | 0.433827 | 0.00246853 | 0.00235643 |
| 384 | random | 218003 | 0.572946 | 0.00283721 | 0.00454392 |
| 384 | teacher | 218001 | 0.00546358 | 0.0017514 | 0.00446033 |
| 384 | teacher | 218002 | 0.00184239 | 0.00246853 | 0.00235643 |
| 384 | teacher | 218003 | 0.00530968 | 0.00283721 | 0.00454391 |
| 1536 | random | 218001 | 0.0010979 | 0.000436288 | 0.00995858 |
| 1536 | random | 218002 | 0.0462899 | 0.000440237 | 0.000470697 |
| 1536 | random | 218003 | 0.00203386 | 0.0383921 | 0.000886878 |
| 1536 | teacher | 218001 | 0.0010979 | 0.000436288 | 0.00047241 |
| 1536 | teacher | 218002 | 0.00045569 | 0.000440237 | 0.000470697 |
| 1536 | teacher | 218003 | 0.00203385 | 0.000685544 | 0.000886878 |

## Noiseless control

| Source labels | Random-start heads | All component cosines>1−1e-6 |
|---|---:|---:|
| 384 | 1 | 0/3 |
| 384 | 3 | 2/3 |
| 384 | 6 | 2/3 |
| 1536 | 1 | 2/3 |
| 1536 | 3 | 3/3 |
| 1536 | 6 | 2/3 |

All noiseless teacher starts retain the exact solution, as expected; this is an implementation/control check, not learning success.

## Interpretation and limits

Three heads beat one on noisy held-out transfer in3/3random-start cases at384labels and2/3at1536. Teacher-initialized comparisons are2/3and3/3respectively. More heads are not uniformly better: six heads sacrifice input coverage and introduce additional readout parameters, and random-start optimization failures remain. The1536-label three-head random-start failure on teacher218003 has transferMSE.0384versus.000685from the teacher start; median-only reporting would obscure it.

The preceding additive-information argument held inputs/noise conventions fixed while adding heads. It does not imply monotonic improvement when a fixed label budget forces fewer distinct inputs. This panel demonstrates why annotation allocation and optimization must be measured separately. Only three teachers and one random initialization per allocation are used; the teachers were used in earlier development experiments. No broad statistical claim or chosen optimal head count follows.

Independent audits replay seeded input/noise generation, source/held-out separation, total budgets, direct-forward Jacobians, polynomial population losses,729-node component comparisons, polynomial baseline normal equations and all transfer solves. All72fits retained; maximum relative Jacobian error1.43e-15 and training-MSE discrepancy3.25e-15. Termination at a local nonzero residual is retained even when the optimizer reports success.

Next: preregister a fresh-teacher confirmation of the fixed-budget comparison, using additional random initializations and teacher-level aggregation. Keep three heads as a candidate rather than selecting it as a universal optimum. Larger-scale or new-task claims should wait for this replication.
