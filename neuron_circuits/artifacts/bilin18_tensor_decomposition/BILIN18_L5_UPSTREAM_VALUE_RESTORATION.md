## H782 — Receiver-output restoration supports a distributed, background-dependent value pathway

Previous goal turn was progress: H781 screened donor-output substitutions along an isolated L5H5 local-value readout. H782 performs the complementary restoration without changing the readout definition or choosing only favorable sites.

First pass: run the donor prompt, but restore one attention/MLP module's receiver output at the two copied-value positions, for each of layers0–4. Capture the surviving L5H5 local values. Second pass: start a fresh receiver run and inject only native receiver routing times the surviving local-value difference relative to receiver. The target is the complete donor local-value-word readout from H780; zero, identity and exact receiver-output replacement controls are included.

The difference between target and restored readout measures what this intervention removes from this particular pathway. It is not whole-model task necessity: input embeddings, residual/x0 paths and other module outputs remain donor in the first pass, and the second pass intentionally excludes unrelated downstream effects.

| Restored site | Surviving task readout RMS | Surviving nuisance readout RMS |
|---|---:|---:|
| L0 attention | .2894 | .1421 |
| L0 MLP | .2105 | .1422 |
| L1 attention | .2872 | .1642 |
| L1 MLP | .1154 | .1378 |
| L2 attention | .2798 | .1624 |
| L2 MLP | .1553 | .1486 |
| L3 attention | .3352 | .1439 |
| L3 MLP | .1579 | .1183 |
| L4 attention | .2599 | .1348 |
| L4 MLP | .1653 | .0875 |

Before restoration, target RMS is.2692 for task changes and.1723 for nuisance key renaming. L1 MLP gives the lowest surviving task RMS, consistent with its stronger donor-substitution match in H781. L4 MLP gives the lowest surviving nuisance RMS, also consistent with H781. Substantial effects survive both. RMS changes are not preserved/removed causal percentages: intervention effects are signed, can cancel, and need not remain proportional to the target on each example.

Restoring some attention outputs increases task-effect RMS, especially L3 attention (.3352 versus target.2692). This is compatible with cancellation or changed nonlinear trajectories; it does not make that module unimportant. Full per-cohort results and signed per-example effects are saved.

Supply and removal are not interchangeable. The RMS difference between the H781 supplied margin effect and H782 removed margin effect is.0959 for L1 MLP task edits and.0080 for L4 MLP nuisance edits. Even before the downstream readout, supplied and removed vectors differ: normalized vector gaps are27.7% and2.8% of target vector RMS for those respective cases. Early-site gaps are generally larger. This diagnoses dependence on the receiver/donor computational background; margin gaps additionally include nonlinear downstream responses. It does not isolate one particular nonlinearity or justify assigning module shares.

All ten sites remain in the screen. The same480aligned edits and32nuisance alignment exclusions are used. These reused correlated panels support prioritizing joint tests, not a confirmed universal two-module circuit. In particular, the key/context signal could be formed through several interactions that individual output swaps only partially reproduce or remove.

Audits:10,944 model forwards; native logits, local values, receiver-output restorations, first-pass identity local values, second-pass identity/zero logits and H780 target margins all replay exactly. Model/input/source hashes unchanged; metrics finite. The original reserved test remains untouched; natural document pool remains26,112.

Next: jointly restore prespecified groups: all attention outputs, all MLP outputs, all ten outputs, and combinations of the implicated early/late MLPs. Keep the same isolated readout and compare with single-site results without summing causal fractions. The all-output source-position restoration is an especially useful control for nuisance edits: unchanged source-token embeddings and restored per-layer additions should return those source states to receiver values, providing a stringent compositional audit. Task edits retain different source embeddings/x0 contributions, so that control need not erase their effect. Joint outcomes will distinguish broad distributed dependence from effects that disappear under a small set of restorations.
