# M12: multidimensional blocks are not yet stable computational units

2026-10-28. Frozen M11models only. Audit all32groups in all8models on the same64natural-text windows; training/evaluation document halves reported separately. No native-model forwards, new fitting, task evaluation reads or candidate selection.

## What is measured

For each group of4decoder atoms, condition on at least one active coordinate and center the group's observed coefficients. Measure covariance in the DECODER OUTPUT metric: QR of the group's decoder converts coefficient covariance into an orthonormal output basis. Participation rank is (sum eigenvalues)^2/sum squared eigenvalues, between1and4 for nonzero covariance. r90 counts directions needed for90%of that conditional variation. It does not measure topological manifold dimension, semantics or necessity.

Compare actual trained block memberships with fixed shuffled groups of4atoms. Scalar models also receive arbitrary contiguous/shuffled groupings; these are descriptive controls, not newly gated models. Actual featurizer codes stay unchanged. Decoder column/subspace and coefficient effects are handled together; a coordinate rotation inside a block cannot by itself change the covariance spectrum.

Across seed71/72, compute mean squared principal cosine of every pair of4Ddecoder subspaces. Hungarian one-to-one matching maximizes total overlap, without looking at evaluation activations. Score train/evaluation activation-support Jaccard afterward. Also measure overlap of the full128Ddecoder subspaces. A fixed signed ambient permutation is a geometric null reference; shuffled atom groupings are a separate control. Only one instance of each reference is used, not a population significance test.

## Mixed-trained models: dimensionality does not establish meaningful grouping

| Model | Median conditional participation rank | Mean group occupancy | Conditional fraction with all4coordinates active |
|---|---:|---:|---:|
| scalar71 / contiguous | 3.634 | 0.691 | 0.0039 |
| scalar71 / shuffled | 3.584 | 0.689 | 0.0043 |
| block71 / contiguous | 3.434 | 0.250 | 1.0000 |
| block71 / shuffled | 3.327 | 0.685 | 0.0042 |
| scalar72 / contiguous | 3.636 | 0.692 | 0.0043 |
| scalar72 / shuffled | 3.624 | 0.689 | 0.0045 |
| block72 / contiguous | 3.374 | 0.250 | 1.0000 |
| block72 / shuffled | 3.466 | 0.661 | 0.0109 |

All32groups are occupied in every model, and median r90=4 for the actual block groups. Blocks are not simply collapsed rank-one features on this sample. But arbitrary scalar groups and shuffled groups also have substantial dimensionality. Moreover, block coactivation of all4coordinates is imposed by its gate: its100%conditional all-four rate is not independent evidence of a naturally discovered concept. Shuffled/scalar groups have much higher any-coordinate occupancy, so raw Jaccard cannot be compared across those groupings without that caveat.

## Seed stability: larger subspace more reproducible than its blocks

| Model / objective | Full128Doverlap | Matched4Dblock overlap | Evaluation support Jaccard |
|---|---:|---:|---:|
| scalar / natural_only_KL | 0.739 | 0.067 | 0.534 |
| scalar / mixed_natural_task_KL | 0.746 | 0.068 | 0.533 |
| block / natural_only_KL | 0.680 | 0.114 | 0.210 |
| block / mixed_natural_task_KL | 0.690 | 0.116 | 0.212 |

Mixed block matching overlap.116 exceeds shuffled atom-group reference.077 and ambient-scrambled.006, so there is some learned grouping structure beyond those references. It is still far from identical subspaces(overlap1). The full block dictionary overlap.690 is much larger: broad output geometry is more reproducible than its partition into4Dblocks. These are principal-angle scores, not fractions of the model's computation reverse-engineered.

Mixed block support Jaccard.212 exceeds the ambient-matched reference.141 modestly. Scalar grouping Jaccard.533 looks larger, but its high occupancy already yields ambient-reference.525; that raw score is mostly uninformative about stable features. For shuffled block groups, higher occupancy similarly inflates support overlap. Interpret controls within their occupancy regime.

Natural-only versus mixed training barely changes these structural scores(block overlap.114→.116), despite M11's substantial task-fidelity change. This descriptive two-seed comparison does not establish improved stability from task supervision.

## Research consequence and next causal discriminator

Do not label these blocks verified manifolds or fundamental units. The evidence supports multidimensional representations and partial reproducibility of a larger decoder subspace, with weak reproducibility of its specific block partition. This could reflect genuinely distributed computation, nonidentifiable decompositions, or insufficient learning; geometry alone does not distinguish them.

Next freeze a bounded group-ablation study on the mixed models: remove decoded group contributions while holding other gates/codes fixed, and measure task-margin changes and output KL relative to the same intact reconstructed model. Use scalar and shuffled-group controls, report removed energy/active dimensions, and avoid equating arbitrary high-energy damage with selectivity. Choose any role matching or candidate groups on training tasks only; test matching/roles on the existing source-separated evaluation pairs without further selection. Geometric versus functional role matching can distinguish unstable coordinates from stable functions. The tiny evaluation panel remains development evidence, not new-task discovery. Keep all finalweights frozen and original task exclusions untouched.

## Audit and reproduction

Independent Torch64 direct-factor SVD reproduces1024conditional covariance spectra without using the QR-covariance calculation. SVD-derived bases reproduce12overlap matrices; assignments/supportJaccards and allsummaryaggregates are checked. Eight explicit orthogonal coordinate changes preserve decoded contributions. Max relative spectrum error6.83e-15, rank error8.0e-15, overlap error4.45e-16 and rotation error6.33e-16. These are numerical/structural audits, not causal-role validation.

Run `analyze_block_structure.py` and `audit_block_structure.py`. Large frozen codes/dictionaries are archived as `emlp_m12_structure_inputs.pt` in primary artifacts. ResultJSON retains every group covariance, eigenvalue, occupancy and matching matrix.

[Protocol](validation/block_structure_protocol.json), [all groups and controls](validation/block_structure.json), [audit](validation/block_structure_audit.json), [M11 fidelity results](DIVERSE_CAUSAL_SUPERVISION.md).
