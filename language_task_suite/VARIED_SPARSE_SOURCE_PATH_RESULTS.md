# Task-output priors for sparse source-path selection

On the competent two-layer varied-language model, a **verb-output-aware** ranking of second-layer attention sources preserves the dense model's rule-edit effects better than rankings based on attention scores, residual reconstruction, or generic output sensitivity. This is a local causal-path result, not a sparse-weight student or a claim of a unique canonical circuit. It also does **not** improve fidelity to the exact language generator: the sparse model inherits the teacher's errors and usually adds some.

## Experiment

[`eval_varied_source_sparse_selection.py`](eval_varied_source_sparse_selection.py) exactly expands the final query's second attention block into per-head, per-source `QK1 × QK2 × V` contributions. It ranks those contributions separately for each original and edited prefix, retains the top K source positions **per head**, sums the retained vectors, then applies the native output projection, residual, final RMSNorm, and unembedding. Full unmasked source replay agrees with native logits to at most 7.7e-6 across all panels.

The two frozen teachers are the original [`surface_mixture_v2_varied_soft_models`](surface_mixture_v2_varied_soft_models/) checkpoint and its [`surface_mixture_v2_varied_functional_observed_p10`](surface_mixture_v2_varied_functional_observed_p10/) score-concentration counterpart. Each was evaluated on three independent 256-stream generator panels (seeds 20261120–22), totaling 11,430 valid edit pairs per model. The edited families include lexical verb, induction, previous place, cue, and nuisance edits. No fitting uses these panels. Seed 20261122 was added after the ranking choices were made, as a fresh replication; seeds 20–21 were development panels, so treat the results as exploratory rather than a locked benchmark.

The rankings are random position; absolute score product; source-vector norm before or after the head output projection; and projected norm on centered unembedding rows for all vocabulary, twelve object tokens, twelve random non-verb tokens, or the twelve verb tokens. The verb set is a known task-output prior. It uses current input activations and does not account for the final RMSNorm Jacobian, so it is neither data-free nor an optimized weight gauge. Object and random-vocabulary contrasts match the verb contrast's row count; the all-vocabulary control tests generic output sensitivity.

At K=4, the gate retains on average 21.9% of source positions across the variable-length edit prefixes. The table averages the four substantive verb-edit families, weighted by pair count across three panels. Attention MSE measures the block output before final normalization; model KL compares sparse to dense predictions; edit L1 measures the difference between sparse and dense **signed probability changes** under a valid edit. Smaller is better.

| Ranking, K=4 | Original attention MSE | Original model KL | Original edit L1 | Observed-pair model KL | Observed-pair edit L1 |
|---|---:|---:|---:|---:|---:|
| Random position | .866 | .7244 | .5344 | .7571 | .5459 |
| `|QK1×QK2|` | .0309 | .00673 | .0826 | .00771 | .0880 |
| Pre-output vector norm | .0250 | .00586 | .0776 | .00649 | .0819 |
| Post-output vector norm | **.0238** | .00647 | .0813 | .00722 | .0860 |
| All-vocabulary contrast | .0258 | .01011 | .1017 | .01134 | .1090 |
| Object-token contrast | .0255 | .00811 | .0910 | .00854 | .0937 |
| Random-vocabulary contrast | .0294 | .01042 | .1000 | .01178 | .1083 |
| Verb-token contrast | .0503 | **.00531** | **.0720** | **.00552** | **.0751** |

Thus the criterion giving the best vector reconstruction is different from the one best preserving the modeled task behavior. At K=8 (33.6% of positions), verb contrast still has the best model KL and edit L1: .00166/.0384 for the original teacher and .00172/.0395 for the observed-pair teacher. The original model's K=4 verb contrast has edit L1 .077 lexical, .065 induction, .068 previous-place, and .077 cue; the score-product ranking has .089/.079/.077/.086. The same ordering holds in the observed-pair model.

Source localization alone is insufficient. At K=1, verb contrast includes the edited position in at least one head for 94.5% of lexical edits, 76.7% of induction edits, and 97.5% of previous-place edits in the original model, yet its edit L1 remains large. At K=4 it almost always includes the edited position, while missing context paths and cancellations still change the effect. The selected position can change between original and edited prefixes because the gate is input-dependent.

**Oracle caveat.** Across substantive edits, the dense original model's mean effect L1 against the exact generator is .2444. K=4 verb contrast has .2603, versus .2541 for pre-output norm; the observed-pair model's dense value is .2285 and its K=4 verb contrast value is .2431. The task-aware selector best preserves the *teacher's* causal response, but it does not fix the teacher's mistakes. Reported oracle KL and per-family effects are in the raw JSON files, separate from model KL.

## Research implication and stopping point

This is evidence that task-output priors can improve **which source paths to retain** at fixed local source count. It does not show that the paths form human-readable independent units, that a sparse-parameter student can learn them, or that an extra attention layer helps. The local ranking sweep is complete. A next architecture experiment should turn such output-aware source/path targets into a trainable sparse or factored student, compare it with the flat and active-depth controls at equal **total nonzero parameter** budgets, and evaluate KL, exact-oracle rule edits, and actual layer participation on diverse tasks with an entire rule family withheld from causal-target fitting. Do not count merely nonzero added-layer weights as evidence of useful depth.

Reproduce with the local environment from the repository root:

```bash
for model in surface_mixture_v2_varied_soft_models surface_mixture_v2_varied_functional_observed_p10; do
  for seed in 20261120 20261121 20261122; do
    /home/loganriggs/Coding/neuron_circuits/.venv/bin/python \
      language_task_suite/eval_varied_source_sparse_selection.py \
      --model-dir language_task_suite/$model --nstreams 256 --length 72 \
      --seed $seed --keeps 1,2,4,8,16 --device cuda \
      --output language_task_suite/$model/source_sparse_selection_n256_seed$seed.json
  done
done
```
