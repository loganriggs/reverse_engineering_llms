# Continuous versus hard blocks: joint fidelity comparison complete

M112 now includes the prespecified task intervention tests and amplitude controls. Continuous block shrinkage removes artificial selection jumps, but does not provide a uniformly more faithful computational representation. It improves some final-position response errors while worsening natural retention and often worsening changed-input response fidelity. Much of the final-position gain can be matched by shrinking the hard-gate intervention.

## What is being compared

Hard and continuous block models were adapted from identical historical parameters within seeds71/72, using the same132updates and natural-KL plus encoded causal-response supervision. The hard controls reproduce their historical checkpoint tensors bit-for-bit. This is matched adaptation from hard-trained starts, not a from-scratch architecture comparison. See CONTINUOUS_TRAINING_NATURAL_RESULTS.md for training and natural evaluation.

For each task pair, we encode both endpoint writes. We then add their encoded donor-minus-recipient difference to the native recipient MLP1 write, preserving the rest of the native background. The teacher is the corresponding native donor-minus-recipient intervention. This tests encoded changes separately from reconstructing the entire state.

Three conditions: hard encoded change; continuous encoded change; and hard change multiplied by ||continuous change||/||hard change||. The last uses candidate vectors only, with no native-teacher norm or output tuning. Its direction remains hard-gated but its total local norm matches the continuous candidate. Zero cases are explicit; no zero-hard/nonzero-continuous ineligibility occurred.

The input-position site patches only the token positions changed between prompts; the final-position site patches the last token. “Familiar-task holdout” below means the14source-group pairs, not the input-position intervention site.

## Equal-family full-output response fidelity

The error is squared final log-probability response error weighted by native output probabilities, divided by native teacher-response energy. Each family first gets a ratio, then families are averaged. Zero response scores1; lower is better. A value below1 means improvement over doing nothing, not full faithfulness.

| Cohort | Seed | Intervention site | Hard | Continuous | Norm-matched hard |
|---|---:|---|---:|---:|---:|
| Training diagnostic | 71 | Changed input | 0.647 | 0.790 | 0.805 |
| Training diagnostic | 71 | Final | 1.510 | 0.970 | 0.875 |
| Training diagnostic | 72 | Changed input | 0.827 | 0.894 | 0.907 |
| Training diagnostic | 72 | Final | 1.329 | 0.907 | 0.924 |
| Familiar-task holdout | 71 | Changed input | 1.005 | 0.928 | 0.989 |
| Familiar-task holdout | 71 | Final | 1.252 | 1.000 | 0.875 |
| Familiar-task holdout | 72 | Changed input | 0.957 | 0.939 | 0.968 |
| Familiar-task holdout | 72 | Final | 1.464 | 1.132 | 0.797 |
| Whole-task holdout | 71 | Changed input | 0.756 | 0.827 | 0.823 |
| Whole-task holdout | 71 | Final | 1.285 | 0.802 | 0.793 |
| Whole-task holdout | 72 | Changed input | 0.917 | 0.931 | 0.878 |
| Whole-task holdout | 72 | Final | 1.371 | 0.881 | 0.843 |
| Whole-task competent5 | 71 | Changed input | 0.770 | 0.811 | 0.821 |
| Whole-task competent5 | 71 | Final | 1.173 | 0.739 | 0.763 |
| Whole-task competent5 | 72 | Changed input | 0.861 | 0.908 | 0.875 |
| Whole-task competent5 | 72 | Final | 0.991 | 0.825 | 0.800 |

Across the three all-family cohorts and two seeds, continuous beats original hard at final positions in6/6comparisons. It beats norm-matched hard in only1/6of those equal-family comparisons. At changed input positions it beats original hard in2/6comparisons (both familiar-task holdouts). Thus removing the gate discontinuity does not consistently improve encoded response directions. The qualified-five subset is also mixed and retained separately.

## Aggregation changes the answer

Pooled ratios weight families in proportion to native teacher-response energy. They answer a different question from equal-family averages. For whole-task final-position responses:

| Seed | Hard pooled error | Continuous pooled error | Norm-matched hard pooled error |
|---|---:|---:|---:|
| 71 | 0.117 | 0.109 | 0.136 |
| 72 | 0.261 | 0.200 | 0.478 |

Here continuous beats both hard controls for both seeds. The norm-matched control is therefore not a universal explanation of every gain. Conversely, the very low pooled errors do not imply broad family-level fidelity: the corresponding equal-family errors are substantially higher. All family rows, pooled/equal-family response and answer-margin errors, local errors, and candidate/native energy ratios are retained in validation/continuous_task_comparison.json. No favorable aggregate was selected as the sole result.

## State reconstruction remains worse

Whole-task intact replacement KL rises from about0.132/0.131 for hard to0.227/0.227 for continuous (seeds71/72). Top1 agreement falls from75.0%/74.1% to66.1%/61.6%. These intact replacements reconstruct MLP1 writes at all positions, unlike the site-specific native-background difference tests.

The earlier natural evaluation similarly showed KL0.46–0.50 for continuous versus0.17–0.18 for hard, with worse KL, CE and local reconstruction on every evaluated document. We must not combine those state-fidelity losses and the selected causal-response gains into an unsupported overall “better representation” claim.

## Evidence and scope

The training diagnostic contains one fixed pair per11families; it is not the full training set. The familiar-task holdout has14pairs and the whole-task panel56pairs across7families. All pairs, both endpoints and eligible source/final sites were retained. The two lower-competence whole-task families (role binding and spatial inverse) are shown in the all-family analysis and excluded only in the predeclared competent-five view. All whole-task families remain excluded from featurizer fitting; these are reused development evaluations, not pristine confirmation or new-task discovery.

3022native forwards completed, with640exact baseline/teacher replays. All2868candidate response vectors and648intact reconstructed states were independently audited, including NumPy encoders/decoders and norm-matched deltas. Eight source-site endpoint cases were ineligible due to existing token-position alignment restrictions. No magnitude-control-specific ineligibility occurred.

The audit checked29,700metrics and648decodes. Independent decode error was below2.39e-7; norm-matched deltas matched exactly. Raw activation-energy sums required relative rather than absolute tolerance: an initial1.49e-8summation difference exceeded an inappropriate absolute1e-8check. The correction is recorded; local-energy checks use scaled1e-10, output metrics retain absolute1e-8. The analyzer also initially used presentation labels instead of stored region enums, producing an empty-slice failure before any output; that schema correction is recorded.

Independent SQLite aggregation verifies336summary quantities, with maximum scaled discrepancy3.53e-16. Identity rows are exactly zero. All experiments and audits are terminal. Full arrays/logits remain local, referenced with hashes by the receipts; all scripts, protocols, metrics and reports are in the published packet.

## Consequence for fundamental computational units

We have separated three requirements that can conflict: preventing artificial extractor boundaries, reconstructing native states, and preserving native causal responses. Continuity is a meaningful prior for the first, but is insufficient for the latter two. The current continuous map should not be promoted as a validated manifold or semantic unit extractor.

No result here establishes native neuron redundancy, cancellation-dominated MLPs, or a discrete native taxonomy. Those claims concern the implementation of the native model, while this comparison concerns how we describe its writes and changes. The artificial-gate finding remains valid even though this particular repair sacrifices fidelity.

This fixed adaptation/comparison is closed. Do not add optimization steps, tune gains, change the loss calibration, or choose a seed based on these results. The next useful theoretical question is what constraints allow sparse decoding to remain continuous without the amplitude loss imposed by this shrinkage rule. State the assumptions, test them in known-mechanism toys and existing encoders, and distinguish global code-space guarantees from behavior on the native data manifold before proposing another training sweep.
