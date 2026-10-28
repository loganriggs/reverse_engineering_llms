# M38 pause handoff: training-derived overlapping groups

Paused at the user request on2026-10-28. No experiment is running. Construction and audit
completed; native transfer, output fidelity and selective reuse tests remain pending.

Four frozen scalar source dictionaries are retained: seeds71/72, each from intact-KL and
joint causal training. Only group memberships change. All encoder/decoder/bias and normalization
tensors are bitwise unchanged. Each condition has32groups of8, Top4 active groups, everyatom
a member of exactly2groups. Two balanced random partitions provide the matched baseline.
Training-only squared-coordinate coactivation cosine affinities give equal total weight to
natural training documents and scheduled training endpoints. Exactly20000 within-partition
swap proposals per source maximize affinity of co-grouped atom pairs, counting shared pairs
once. No evaluation-guided selection or decoder fitting.

## Completed results

On all4096 natural evaluation positions, learned grouping reduces local squared reconstruction
error by6.6–8.4% versus random grouping across allfour sources. The unchanged scalarTop32
baseline still reconstructs better. Learned groups use fewer active atoms than random groups,
so the benefit is not explained by selecting more coordinates; realized budgets are not equal.
All128 atoms remain covered and used; no duplicate groups occurred.

| Source / seed | Scalar relative error | Random | Learned | Random active mean | Learned active mean |
|---|---:|---:|---:|---:|---:|
| intact_KL 71 | 0.4522 | 0.6060 | 0.5654 | 29.07 | 28.54 |
| intact_KL_plus_native_response 71 | 0.4518 | 0.6045 | 0.5648 | 29.09 | 28.48 |
| intact_KL 72 | 0.4485 | 0.6141 | 0.5655 | 29.12 | 28.48 |
| intact_KL_plus_native_response 72 | 0.4543 | 0.6171 | 0.5654 | 29.08 | 28.31 |

Independent audit reconstructs the training affinity matrices, checks all80000 accepted/rejected
swap decisions with set-based edge accounting, and rescores all36 local metrics. Maximum affinity
discrepancy9.99e-16, local metric discrepancy1.24e-8. No claim of semantic manifolds or native
causal fidelity follows from this local reconstruction result.

## Restart here

1. Read validation/coactivation_group_protocol.json, coactivation_groups.json and
   coactivation_groups_audit.json. Twelve candidate states (four scalar originals and eight
   grouped states) are hash-linked in the results; large state/evidence caches remain local.
2. Evaluate all12candidates on the unchanged native response benchmark and ordinary-output
   inventory, retaining both scalar/block source generators and all seeds/source objectives.
3. Test selective shared-atom versus group-gate removal separately: a gate removal retains
   an atom still selected elsewhere, whereas atom removal affects every consumer.
4. Preserve task exclusions and development-data status. This is learned support grouping,
   not new dictionary directions, unsupervised semantic recovery or discovered new tasks.

Scripts: build_coactivation_groups.py and audit_coactivation_groups.py. Construction31770 and
audit24115 both exited0. Do not rerun construction or duplicate experiments. Research goal
is paused; resume only on the user's return. One-time restart reminder requested for12hours.
