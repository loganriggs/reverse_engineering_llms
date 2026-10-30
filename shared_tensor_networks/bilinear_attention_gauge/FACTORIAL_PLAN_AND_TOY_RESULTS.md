# Canonicalization, reduction, and legibility: factorial program

The proposed experiment has two independent choices. **Measure:** what input
distribution defines a good functional approximation? **Legibility prior:**
among similarly faithful reduced circuits, what structure do we prefer? Keep
those axes separate. Weighted trace gauge (WTG) fixes a bond coordinate system
for one specified network and measure; it does not itself identify a semantic
mechanism or choose a unique function-level factorization.

## First completed test: known-rule two-head toy

`toy_factorial.py` makes a first-layer, two-position bilinear-attention pair
function with two heads, each containing a common and a rare query-key rule.
It retains rank one of the original rank-two QK1 bond per head. QK2 remains a
separate multiplicative path and each head has a two-dimensional OV output.
The toy uses the exact pair function: no sample data are needed to construct
its environment. The `usage` measure is an *invented* independent-token
frequency law, not a language corpus. Three seeds vary background weights.

The measure axis is uniform, usage-weighted, or a 50:50 mixture. After WTG
initialization, four further objectives were tried at a single coefficient
`lambda=0.02`: no prior, L1 on native bond entries, L1 on WTG-normalized bond
entries, L1 on the resulting pair-response norms, and fidelity after replacing
QK2 by a constant score. All fits use four starts and a CPU Adam optimizer.
Analytic gradients are checked against central differences; the exact gauge
check is asserted. The JSON files retain per-head and summed-head metrics.

Mean relative squared error over six head/seed cases; lower is better:

| Fit measure / objective | Common-rule error | Rare-rule error | QK2-intervention error on uniform pairs |
|---|---:|---:|---:|
| Uniform / fidelity | 0.967 | 0.000 | 0.094 |
| Uniform / canonical-bond L1 | 1.000 | 0.000 | 0.088 |
| Usage / fidelity | 0.024 | 0.145 | 0.848 |
| Usage / canonical-bond L1 | 0.028 | 0.162 | 0.831 |
| Hybrid / fidelity | 0.282 | 0.002 | 0.316 |
| Hybrid / canonical-bond L1 | 0.328 | 0.003 | 0.295 |

**Interpretation:** at rank one, the uniform measure preserves the large rare
rule and mostly sacrifices the common rule. The invented usage measure strongly
favors the common rule and impairs rare-rule and uniform intervention fidelity.
The hybrid sits between them. That is a deliberately capacity-limited example
of why the measure must be specified before calling an axis "important."
At this one weak prior strength, none of the legibility penalties gives a
compelling gain over fidelity alone. A lambda/error-budget sweep is necessary.

An exact non-orthogonal QK1 gauge changes the native entrywise L1 from 1.35
to 1.7985 while changing the pair function by at most `9e-16`. The pair
response L1 is unchanged. The WTG-relative bond L1 is unchanged to `2e-15`
under the same gauge in all three seeds. Thus native-weight sparsity is not a
function-level prior; WTG-relative or response-level sparsity is better posed.
This checks internal gauges, not QK branch swaps or cross-head redistribution.

The toy is hand-built, so rule labels are known for evaluation. It does **not**
show discovery on a trained model, task accuracy, held-out language
generalization, or a sparse explanation of the 546M model. Four-start
optimization is not a global certificate; some near-kink native- and
canonical-L1 runs have gradient norm above `1e-4`, so those comparisons remain
exploratory.

Reproduce with `OPENBLAS_NUM_THREADS=1` and the project Python environment:

```bash
cd /home/loganriggs/Coding/reverse_engineering_llms/shared_tensor_networks/bilinear_attention_gauge
for seed in 0 1 2; do
  OPENBLAS_NUM_THREADS=1 /home/loganriggs/Coding/toy_models_of_tensor_networks/tn_venv/bin/python \
    /home/loganriggs/Coding/reverse_engineering_llms/shared_tensor_networks/bilinear_attention_gauge/toy_factorial.py \
    --seed "$seed"
done
```

## Matched-error sweep

`toy_pareto.py` sweeps prior weights from 0 to 1. A candidate may spend at
most **0.02 additional relative function MSE** above the best rank-one fit for
that head and measure. Candidates with gradient norm `>=1e-4` are excluded
from selection. All three seeds were run separately and merged; the merged
`toy_pareto.json` retains every raw fit and selected candidate. This is an
*oracle diagnostic on a hand-built function*: lambda selection sees the same
toy function and intervention being reported. It is not held-out transfer.

Mean across six head/seed cases, choosing the candidate with smallest
uniform-pair QK2-intervention error within that error allowance:

| Fit measure | Fidelity only: intervention MSE | Canonical L1: intervention MSE | Intervention prior: intervention MSE |
|---|---:|---:|---:|
| Uniform | 0.094 | 0.088 | 0.088 |
| Usage | 0.848 | 0.781 | 0.768 |
| Hybrid | 0.316 | 0.248 | **0.160** |

On the hybrid measure, the intervention-prior selection spends the allowed
reconstruction error: mean training MSE rises from 0.105 to 0.120. Its lower
intervention error is therefore a meaningful Pareto trade, **not a free
interpretability gain**. The native-L1 candidate reaches 0.700 intervention
MSE under the usage measure, but that prior is gauge-dependent and is not a
stable description of the function.

The effective number of active pair responses (an inverse-concentration
index, not a count of semantic rules) changes only modestly. For example,
under hybrid fitting it is 4.57 for fidelity only, 4.43 for canonical L1,
4.29 for pair-response L1, and 4.17 for the intervention prior. Those
reductions also spend error and do not establish more recognizable concepts.
The sweep therefore sharpens the next test: use trained models and evaluate
rule recovery and interventions on *held-out* combinations and intervention
types, instead of trusting this in-sample sparsity proxy.

```bash
cd /home/loganriggs/Coding/reverse_engineering_llms/shared_tensor_networks/bilinear_attention_gauge
for seed in 0 1 2; do
  OPENBLAS_NUM_THREADS=1 /home/loganriggs/Coding/toy_models_of_tensor_networks/tn_venv/bin/python \
    /home/loganriggs/Coding/reverse_engineering_llms/shared_tensor_networks/bilinear_attention_gauge/toy_pareto.py \
    --seeds "$seed" --output "toy_pareto_seed${seed}.json"
done
OPENBLAS_NUM_THREADS=1 /home/loganriggs/Coding/toy_models_of_tensor_networks/tn_venv/bin/python \
  /home/loganriggs/Coding/reverse_engineering_llms/shared_tensor_networks/bilinear_attention_gauge/toy_pareto.py \
  --merge toy_pareto_seed0.json toy_pareto_seed1.json toy_pareto_seed2.json \
  --output toy_pareto.json
```

## Next comparisons and gates

1. **Pareto sweep, not one lambda.** For each measure and prior, match
   reconstruction error and active-component budget. Compare native-weight,
   WTG-weight, pair-response, and class-pair/group sparsity. A supervised
   class-rule penalty is a separate condition and must not count as unsupervised
   discovery. Check the same proposed component under QK branch swaps and
   jointly across heads.
2. **Trained toy language.** The first retrieval task was redundant; see
   [trained toy results](TRAINED_TOY_LANGUAGE_RESULTS.md). A harder
   [dual-output task](DUAL_OUTPUT_LANGUAGE_RESULTS.md) passes the
   two-head-ablation gate across ten seeds. Native, canonical, pair-response,
   and [family-group priors](DUAL_GROUP_PRIOR_RESULTS.md) have now been tested
   with held-out combinations and several interventions. None rescues the
   difficult rare rule at rank one. The next test needs explicit reusable
   components and a prediction of which component responds to which
   intervention. Keep generator labels out of unsupervised fitting and
   model selection.
   An [exact shared-versus-duplicated attention control](SHARED_COMPUTATION_IDENTIFIABILITY.md)
   now proves that external function data alone cannot always identify
   computation reuse. A noisy intervention assay distinguishes those graphs
   and transfers to a different patch strength and input law. The remaining
   challenge is to infer candidate components rather than supply their
   locations and graph families.
3. **Small real checkpoints.** The existing vocabulary-100 heads permit exact
   all-token environments. Use a real held-out text distribution for the
   data-aware measure, report pair-frequency coverage, and compare uniform,
   data-aware, and floored/hybrid measures on full pair and sequence outputs.
4. **546M first-layer heads.** The matrix-free full-width operator and
   corpus-free uniform Monte Carlo comparison are now complete for heads 0
   and 4; see [full-width results](FULL_WIDTH_FLAGSHIP_RESULTS.md). These
   results avoid both a dense `128^2 x 128^2` transfer matrix and a tiny
   token-domain quotient. A local-window FineWeb corpus metric and matched
   direct-fit control are also complete on head 0. The next tests are
   rare-token strata, unseen combinations, longer contexts, downstream CE,
   and interventions. Uniform Monte Carlo is not exact all-vocabulary
   enumeration; corpus-pair fidelity is not a semantic result.

Across all stages, report **function reconstruction, downstream task
performance, causal fidelity, rule recovery, and new-rule discovery** as
separate outcomes. A low reconstruction error is not evidence that an axis is
human-legible or that the physical model implements the inferred module.
