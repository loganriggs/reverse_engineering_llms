# Layerwise alternating gradients on depth-two bilinear networks

216fits:108short and108long, with the entire short trajectory prefix reproduced in the long panel. Networks have input dimension3, hidden width3, top width3; tied39parameters versus untied57parameters. Bias-free and unnormalized. Teachers are tied or untied, with3paired teacher/initialization seeds per family. A/B directions are independently initialized; untied student occurrences copy the tied initial hidden parameters, giving identical initial functions without the previous equal-factor product symmetry.

Teacher and initial student Gaussian RMS are normalized to1 with frozen scales. Exact Gaussian population squared error removes sampling noise. This is an optimization study, not a sample-efficiency study. A tied student on an untied teacher is not automatically a certified unrepresentable case.

Blocks: all bottom A/B occurrences, top C/D, then readout w. Forward/reverse alternate these blocks; joint updates all simultaneously. Corrected tied updates divide bottom gradients by2. No momentum, clipping, decay, line search or RMSNorm. Rates .01/.003 and per-block times3/30 were applied to the whole panel.

| Teacher | Student / metric | Step | Joint MSE | Forward MSE | Reverse MSE |
|---|---|---:|---:|---:|---:|
| tied | tied_raw | 0.01 | 0.0138461 | 0.0138956 | 0.0138071 |
| tied | tied_raw | 0.003 | 0.0138622 | 0.0138773 | 0.0138510 |
| tied | tied_corrected | 0.01 | 0.0215213 | 0.0215628 | 0.0214408 |
| tied | tied_corrected | 0.003 | 0.0215504 | 0.0215641 | 0.0215269 |
| tied | untied_raw | 0.01 | 0.0025069 | 0.0024919 | 0.0025240 |
| tied | untied_raw | 0.003 | 0.0025091 | 0.0025049 | 0.0025141 |
| untied | tied_raw | 0.01 | 0.0866621 | 0.0441258 | 0.0769811 |
| untied | tied_raw | 0.003 | 0.0769275 | 0.0865993 | 0.0769533 |
| untied | tied_corrected | 0.01 | 0.0282164 | 0.0278063 | 0.0287603 |
| untied | tied_corrected | 0.003 | 0.0282043 | 0.0280833 | 0.0283649 |
| untied | untied_raw | 0.01 | 0.0041131 | 0.0040894 | 0.0040985 |
| untied | untied_raw | 0.003 | 0.0041094 | 0.0041022 | 0.0041049 |

Table entries are median exact population MSE over3paired teacher/initialization seeds at per-block time30. Medians do not imply robust method rankings. For untied teacher seed211002 with raw tied student and step.01, forward/joint/reverse finish at.0441/.1682/.5616. At.003 they finish at.0900/.1858/.3540. Block order can strongly affect a difficult trajectory; its effect is not a universally beneficial acceleration.

Work control: this implementation computes one full gradient per joint sweep and three per alternating sweep. Comparing joint time30 against alternating time10 matches3000versus3000evaluations at step.01 and10000versus9999at.003. Forward beats joint in0/36pairs; reverse beats joint in0/36. This is a full-gradient-evaluation budget, not measured hardware FLOPs or an optimal specialized block-gradient implementation. Equal per-block time comparisons give alternation roughly3times the gradient work.

Independent validation: all216final losses replay through monomial expansion and exact Gaussian moments. Initial/final analytic gradients agree with autograd; three-sweep block updates agree with an independent autograd implementation. All108long prefixes exactly reproduce short endpoints. No nonfinite fits. Error magnitudes, checkpoints, source snapshots and all individual outcomes are saved.

Theory extension: multiplicity-corrected tied updates and occurrence-projected updates remain equivalent under any common block schedule that selects all occurrences of each shared parameter. Nine20-sweep fixtures agree within6.67e-16. Rowwise reciprocal-scaling gradient identities and finite-step norm-balance drift checks also pass. See SHARING_COMPLEXITY_AND_FLOW.md and results/alternating_projection/checks.json.

Conclusion: naive layerwise alternation changes some trajectories, but has not earned its extra gradient cost in this panel. This is neither a global-convergence result nor a rejection of larger dwell times, adaptive block methods, specialized implementations, or damped curvature-aware methods. Next prioritize an optimizer with a concrete conditioning rationale and independent work controls, alongside the already planned image data/capacity replication. No circuit-recovery claim follows from these function errors.
