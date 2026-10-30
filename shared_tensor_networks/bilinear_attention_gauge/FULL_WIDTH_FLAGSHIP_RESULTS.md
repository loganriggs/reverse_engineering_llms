# Full-width, corpus-free first-layer head reduction

`matrix_free_environment.py` applies the double-layer bond environment without building its \(d^2\times d^2\) Gram or transfer matrix. It computes the two weighted-trace-gauge (WTG) boundary operators by power iteration, then truncates the physical QK1 bond. The bond remains width **128** throughout; unlike the earlier eight-token pilot, there is **no domain quotient**.

`flagship_matrix_free.py` draws token IDs uniformly from the entire **50,304-token vocabulary**, and causal position pairs uniformly among the ten pairs in a length-four sequence. Self-pairs use the same token on query and key. This is a **corpus-free Monte Carlo approximation of a declared uniform metric**, not an exact enumeration of the vocabulary function and not a learned data-frequency measure. The QK2 score squared and OV output norm squared weight each QK1 bond sample. The relative squared error is the weighted squared difference in that head's pair contribution after inserting a rank-limited QK1 bond, divided by the original contribution energy. It is not full-model CE or a sequence-level logit metric.

The factor extraction matches the earlier direct first-layer implementation on randomly selected token/position pairs to maximum absolute error below \(1.1\times10^{-18}\). On three small random environments, matrix-free Gram and transfer applications, WTG cuts, and gauge-transformed truncated scores match explicit dense references within numerical precision. The local baseline is the exact rank-r SVD of the **unweighted sampled QK1 score matrix**, computed through a width-128 QR core. A separate dense-score check matched its optimum to below \(7\times10^{-14}\). This baseline does not optimize the QK2/OV-weighted metric.

## Independent-sample results

Each line uses 16,384 fit pairs and **16,384 independent validation pairs**. Errors are relative squared pair-function errors; lower is better.

| Head | Seed | Rank | WTG fit | WTG validation | Local SVD fit | Local SVD validation |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 17 | 8 | .6282 | .6418 | .6641 | .6705 |
| 0 | 17 | 16 | .3580 | .3787 | .3830 | .3921 |
| 0 | 17 | 32 | .1093 | .1217 | .1187 | .1264 |
| 0 | 18 | 8 | .6369 | .6548 | .6574 | .6721 |
| 0 | 18 | 16 | .3548 | .3778 | .3781 | .3936 |
| 0 | 18 | 32 | .1110 | .1234 | .1200 | .1264 |
| 4 | 17 | 8 | .0257 | .0251 | .0254 | .0252 |
| 4 | 17 | 16 | .0201 | .0197 | .0198 | .0195 |
| 4 | 17 | 32 | .0134 | .0134 | .0132 | .0130 |
| 4 | 18 | 8 | .0260 | .0262 | .0260 | .0262 |
| 4 | 18 | 16 | .0205 | .0205 | .0201 | .0202 |
| 4 | 18 | 32 | .0138 | .0138 | .0136 | .0135 |

Head 0 gains modestly from the full environment; at rank 32, validation error is about .122–.123, compared with .126 for local SVD. Head 4 is far easier to compress under this *particular measure*, with rank-eight error about .025–.026. WTG does not improve over local SVD there. Different heads therefore warrant separate budgets; a common rank is not justified by this pilot.

The smaller 4,096-fit-pair head-0 seed-17 run gives WTG validation errors .6465/.3873/.1340 at ranks 8/16/32, compared with .6418/.3787/.1217 at 16,384 fit pairs. A 512-pair smoke test had a much larger fit/validation gap, so low-sample measurements would have overstated compression. Even 16,384 pairs cover a tiny fraction of the complete vocabulary-by-vocabulary domain and only four positions. The independent validation tests the *same uniform sampling law*, not rare-token, corpus, long-context, task, or causal generalization.

This establishes that a full physical-bond, weight-only **measure** can be approximated and checked without loading a language corpus or selecting eight special tokens. It does not establish human-legible directions, a canonical attention *function* across QK branch swaps, or mechanistic relevance. The next stage should compare (1) corpus-weighted versus uniform measures with matched independent evaluations, and (2) class/group or intervention-aware legibility objectives at matched pair-function error. The dual-output toy provides known component provenance and held-out combinations for that second stage. A component should earn a name only if its ablation and transfer predictions survive held-out input and internal interventions.

## Reproduce and artifacts

`full_width_results/` stores JSON metrics and NPZ bond matrices for both heads and seeds, plus a 4,096-pair sample-size check. The external checkpoint is referenced, not copied. On this host numerical jobs are pinned to CPU 2 to avoid intermittent process crashes on another core:

```bash
cd /home/loganriggs/Coding/reverse_engineering_llms/shared_tensor_networks/bilinear_attention_gauge
taskset -c 2 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/loganriggs/Coding/neuron_circuits/.venv/bin/python flagship_matrix_free.py \
  --head 0 --seed 17 --n-train 16384 --n-val 16384 \
  --output full_width_results/head0_seed17_n16384.json
```
