# Bilinear-attention gauge canonicalization: completed small-model comparison

Follow-up: [metric × legibility-prior plan and known-rule two-head toy results](FACTORIAL_PLAN_AND_TOY_RESULTS.md). This follow-up separates uniform versus usage-weighted reduction from native-bond, WTG-bond, pair-response, and intervention objectives. It is a hand-built toy and does not establish interpretation of a trained model.

Flagship metric follow-up: [fixed uniform–corpus and intervention mixtures on the 546M head](MIXED_METRIC_RESULTS.md). A half uniform–corpus fit improves held-out sequence head-output fidelity on FineWeb, WikiText-103, and TinyStories at matched rank; adding a pair-level QK2 intervention objective does not improve sequence-level interventions. Sparse and human-legibility priors remain untested on this head.

Sequence-objective follow-up: [directly optimize the summed first-layer head response](SEQUENCE_OBJECTIVE_RESULTS.md). With disjoint FineWeb fit/selection/test documents and two separate test corpora, rank-64 sequence fits beat the initial WTG cut and a direct pair-refit control. A 25% sequence-level QK2-constant objective improves that intervention and a held-out QK2 key shift. Full-model behavior and component legibility remain open.

Full-model follow-up: [propagate the saved cuts through all 18 layers](FULL_MODEL_BEHAVIOR_RESULTS.md). On 18,432 next-token positions across disjoint FineWeb, WikiText-103, and TinyStories samples, the natural sequence fit has the lowest original-to-edited KL in both seeds of each corpus. A zero-head edit has roughly 47–54 times its KL. CE shifts are tiny and do not establish an accuracy gain; component legibility remains open.

Exact-gauge legibility-prior follow-up: [factor-weight versus activation sparsity on the fixed rank-64 cut](EXACT_GAUGE_SPARSITY_RESULTS.md). Orthogonal rotations preserve the entire attention function to numerical precision. Activation objectives lower effective active-component count on held-out corpora, but the gain concentrates on common tokens and independently fitted component bases align weakly. Factor-weight L1 drops sharply without comparable pair-response sparsity. These are sparsity proxies, not semantic components.

Label red team: [post-fit token-form labels and exemplars for the exact-gauge components](COMPONENT_LABEL_TRANSFER_RESULTS.md). At 262,144 pairs per corpus, activation-sparse gauges do not beat random or SVD gauges on coarse held-out label enrichment. The label taxonomy is dominated by broad word-form classes; the subsequent full-model ablation tests context localization separately.

Component-causality red team: [full-model deletion and matched partial deletion of exact-gauge components](COMPONENT_ABLATION_RESULTS.md). Top activation-sparse components have larger uncalibrated effects because they carry more head response. After calibrating all edits to the same head-output error on separate documents, their downstream KL is no more localized across contexts than random-basis components in this limited assay.

Trained follow-up: [two-head toy-language reduction and red team](TRAINED_TOY_LANGUAGE_RESULTS.md). The pair-function extraction is exact; whole-sequence and pair-frequency environments differ beyond sampling noise, while the task is too redundant for accuracy to distinguish the reduced heads.

Stronger trained follow-up: [necessary two-head dual-output language](DUAL_OUTPUT_LANGUAGE_RESULTS.md). Ten seeds establish a real two-head dependency; rank-one reduction usually preserves the task, while a difficult seed separates uniform, data-aware, task, and intervention criteria.

Joint function follow-up: [exact combined-table reduction, metric comparison, and sparse key dictionary](JOINT_FUNCTION_TOY_RESULTS.md). On ten trained dual-output toy checkpoints, a rank-two reduction of the summed token function preserves the rare withheld rule in every seed, including the difficult seed that fails a same-raw-factor-count headwise reduction. Uniform, training-frequency, and half-mixture metrics all preserve the task at rank two; none reliably rescues a rank-one bottleneck. A label-free sparse rotation of the rank-two function's key dictionary aligns with the two task bits and gives selective post-fit bit-flip effects. This is an interpretable **function-level** basis, not recovery of physical head computations.

Legibility-prior follow-up: [key-family and query-conditional group penalties](DUAL_GROUP_PRIOR_RESULTS.md). The supplied token-family partition and a broader intervention panel do not rescue the hard held-out rule at rank one; group concentration is not itself a mechanistic success criterion.

Identifiability control: [shared versus duplicated attention computation](SHARED_COMPUTATION_IDENTIFIABILITY.md). Two graphs have exactly the same external function, including input-edit responses, while one internal-node intervention distinguishes their reuse structure. Shortest-description structure is a prior, not proof of the implemented graph.

Full-width flagship follow-up: [matrix-free first-layer heads](FULL_WIDTH_FLAGSHIP_RESULTS.md). The width-128 physical bond is reduced without an eight-token domain quotient. Uniform and corpus measures, a branch-swap control, and a matched direct-fit comparison separate gauge truncation from the metric it is meant to optimize.

Context-transfer follow-up: [length, rare-token, cross-corpus, and sequence-output tests](FLAGSHIP_CONTEXT_TRANSFER_RESULTS.md). The original four-position metric overweights self-pairs; length-128 fitting and larger ranks improve long-context head fidelity but still do not establish downstream CE or semantic legibility.

This experiment tests Glen Evenbly's [weighted trace gauge (WTG) and full-environment truncation (FET)](https://arxiv.org/abs/1801.05390) on a repeated-token, bilinear-attention circuit. It distinguishes three claims:

1. A fixed attention-network representation has a reproducible internal gauge.
2. Its bond can be compressed more faithfully using the full loop environment than by a local SVD.
3. The resulting gauge identifies a canonical decomposition of the **function**.

The first two claims hold in this controlled setting; the third fails under an exact QK-branch swap.

## Object and metric

Four existing trained, one-layer checkpoints from `tensor_language/runs` have head widths 8, 16, and 32, vocabulary size 100, and causal length 8. Three have one head; the fourth has two width-8 heads, both evaluated separately while the other head remains fixed in the full model. The original first-layer head is represented exactly on tokens by

\[
T_{ijabo}=\mathbf 1_{j\leq i}
\left(\sum_r U_{iar}V_{jbr}\right)B_{ijab}C_{bo},
\]

where \(U,V\) are the first rotary QK branch, \(B\) is the second QK score including the head scale, and \(C\) is the OV map. Positions \(i,j\) remain explicit: the stored rotary tables are rounded separately, so collapsing to an exact relative-position table would be numerically false. The full-rank token-table forward matches the saved model at max absolute residual error below \(1.3\times10^{-14}\).

The compression target is the internal QK1 bond: insert a rank-\(r\) matrix \(X\) in place of its identity. This is an after-RoPE bond operation. The expected squared error averages over valid causal token pairs under a specified token distribution. At \(i=j\), only equal query and key tokens are allowed. The experiments use uniform tokens and a Zipf-1 sensitivity check. A separate set of 128 random sequences measures the resulting first-layer logits. No sequence samples are used to choose the decomposition.

Methods:

- **Local SVD** first truncates QK1 while cutting away QK2 and OV.
- **WTG** forms the exact double-layer bond environment, solves the dominant eigenoperator problems, fixes the weighted trace gauge, and retains its largest bond coefficients.
- **Full-environment ALS** alternates exact least-squares updates for a rank-\(r\) bond matrix using that environment. Its objective is the same best-scaled state-fidelity objective as FET, although this is an ALS implementation rather than a line-by-line port of Evenbly's update. Convergence is local; it is not a global-optimum certificate.
- **Collapsed functional SVD** truncates a different unfolding, \((i,a)\) versus \((j,b,o)\), of the entire token function. Invalid self-pairs are zero-completed. Its rank and parameter count are **not** matched to an internal-bond rank; it is a descriptive functional baseline.

## Truncation outcomes

Relative squared token-function error under uniform valid-pair weighting:

| Checkpoint / head | Bond rank | Local SVD | WTG truncation | Full-environment ALS | Collapsed functional SVD |
|---:|---:|---:|---:|---:|---:|
| Width 8 / sole | 2 | 0.002064 | 0.000844 | **0.000366** | 0.186 |
| Width 16 / sole | 2 | 0.011753 | 0.003687 | **0.000505** | 0.133 |
| Width 32 / sole | 4 | 0.000259 | 0.000169 | **0.000073** | 0.142 |
| Width 8 / head 0 of 2 | 1 | 0.032234 | 0.022367 | **0.018958** | 0.306 |
| Width 8 / head 1 of 2 | 1 | 0.001120 | 0.001075 | **0.000325** | 0.368 |

At these tight ranks, the loop-aware truncations improve the exact function metric, and the full-environment solution is best. The same ordering holds for relative logit MSE on the sampled sequences: width 16/rank 2 gives 0.01153, 0.00365, and 0.000526 respectively. At higher ranks the differences narrow; all ranks, sequence outcomes, alternate metrics, and objective traces are in the JSON files. The collapsed SVD's large error reflects a different global unfolding (especially position and causal-mask structure), not a fair resource-matched defeat. Its valid-domain weighted error need not decrease monotonically because its SVD optimizes an unweighted, zero-completed tensor.

The Zipf-1 width-16/rank-2 run gives 0.011738, 0.003839, and 0.000424 under its own metric; this is a sensitivity check with an artificial token frequency law, not a language-model corpus result. Cross-metric scores for every solution are also saved.

## Gauge and interpretation checks

- An invertible, non-orthogonal QK1 bond gauge changes the raw factors while preserving the exact token function to below \(3.2\times10^{-14}\). WTG-truncated token functions agree across that transformation to below \(10^{-11}\) in all recorded ranks. Normalized WTG coefficients agree to below \(1.4\times10^{-14}\), and the normalized boundary matrices are identity to below \(1.3\times10^{-12}\). The bridge control has cycle entropy about zero, and WTG and full-environment truncation coincide there.
- **Exact branch-swap counterexample.** Swapping the two QK branches leaves the entire head token function identical to within \(7.2\times10^{-15}\). Yet cycle entropy on the now-targeted bond changes from **0.659 to 0.286 bits** (width 8), **0.959 to 0.383 bits** (width 16), and **0.673 to 0.593 bits** (width 32). In the two-head checkpoint, head 0 changes **0.258 to 0.303** and head 1 **0.336 to 0.640** bits. This is a representation change, not an internal gauge transformation of one fixed bond. It proves that these bond entropies are not invariants of the attention function and cannot alone identify which multiplicative branch is the physical or preferred mechanism.
- **Causal stress.** Setting QK2 to a constant while keeping QK1 and OV fixed generally preserves the tight-rank ordering here. For width 16/rank 2, intervention error is 0.01341 (local), 0.00363 (WTG), 0.000628 (full environment). Thus the in-distribution win does transfer to this one branch intervention; it does not establish broad causal fidelity. Every intervention score is retained in the JSON.

The small checkpoints do not test a joint decomposition that redistributes computation between heads. The earlier flagship gauge work also found that simple private-bond rotations bought little compression, so the small-model bond-truncation gain must not be read as a full-model gain.

## Bounded 546M-checkpoint pilot

`flagship_pilot.py` uses only cached weights from the 18-layer, nine-head bilinear model, without loading its full module or using the GPU. It examines first-layer heads 0 and 4 on two independently chosen sets of eight token IDs and four positions. The model's QK RMS normalization and bf16-rounded rotary tables are included. The raw post-RoPE QK1 bond has width 128, but on each selected domain its exact score matrix has rank 32. An SVD refactors that *restricted function* into a 32-dimensional bond before WTG. It is a domain quotient, **not** a gauge transformation of the full 128-dimensional physical bond.

An independent sequence calculation of all nine heads, including OV mixing, agrees with the sum of the token tables to max absolute error \(5.7\times10^{-14}\). The restricted QK quotient and QK-branch swap also pass exact-function checks. The checkpoint and chosen token IDs are recorded in `flagship_results.json`.

Uniform valid-pair relative squared error on the selected domain:

| Token seed | Head | Bond rank | Local SVD | WTG truncation | Full-environment ALS |
|---:|---:|---:|---:|---:|---:|
| 17 | 0 | 4 | 0.06007 | 0.06812 | **0.01014** |
| 17 | 4 | 4 | 0.00384 | 0.00288 | **0.00136** |
| 23 | 0 | 4 | 0.03685 | 0.06323 | **0.00408** |
| 23 | 4 | 4 | 0.02206 | 0.00766 | **0.00319** |

Here full-environment optimization wins all four rank-4 cases, but WTG truncation is **worse than local SVD in both head-0 cases**. The WTG is still gauge-consistent; canonical coefficients alone need not give the best truncation when cycle entropy is nonzero. At rank 8, full-environment errors are 0.000130, 0.0000288, 0.0000235, and 0.0000461 in the same case order. The seed-23/head-0 rank-8 ALS ran its 600-step cap and was still improving slowly, so that number is an upper bound on the best achievable error, not a converged optimum.

The exact QK swap again preserves the selected-domain function while changing cycle entropy: seed 17/head 4 gives **0.612 versus 1.593 bits**. This extends the representation-dependence finding to the flagship weights, but only on the declared token subset. No conclusion follows about full-vocabulary compression, later layers, CE loss, or whether a recovered bond names the model's physical computation. Exact WTG here uses a dense \(32^2\times32^2\) transfer eigensolve; applying it directly to a full width-128 head would require a matrix-free implementation and a tractable full-domain environment.

## Reproduce

Use a Python environment with PyTorch, NumPy, SciPy, and einops. For example, the local environment used here is `/home/loganriggs/Coding/neuron_circuits/.venv/bin/python`.

```bash
OPENBLAS_NUM_THREADS=1 /home/loganriggs/Coding/neuron_circuits/.venv/bin/python run.py \
  --checkpoint /home/loganriggs/Coding/reverse_engineer/tensor_language/runs/L1_d16_h1/model.pt \
  --ranks 2 4 8 12 --output results_d16.json
```

The script asserts exact-forward matching, gauge invariance, WTG boundary conditions, agreement between literal and environment errors, and a bridge control. JSON contains all metrics and optimization traces; companion NPZ files contain the fitted bond matrices and WTG gauge matrices. The original model checkpoints are referenced by path and SHA-256 rather than copied into this repository.

The CPU flagship pilot runs with `OPENBLAS_NUM_THREADS=1` and `flagship_pilot.py`; its output is `flagship_results.json`.

## Decision

Evenbly's methods are useful for **bond-level compression within a chosen copy-tensor network**, especially when the bond is genuinely cyclic. They do not solve function-level canonicalization. The branch-swap counterexample is an exact, architecture-native reason to keep those objectives separate. The selected-domain flagship pilot supports testing a matrix-free environment on larger token coverage, but the next mechanistic experiment should compare candidate function-equivalent factorizations on held-out, branch-specific interventions; merely collecting more WTG spectra would not answer which computation the model uses.
