# Correlated training law: function reduction versus readable gauge

We trained ten fresh, one-layer, two-head bilinear-attention retrieval models. Each sequence contains one A token, one B token, a filler, and a query that either keeps or swaps the two output roles. Training samples A and B equal with probability .9, draws the swap query with probability .2, then removes the withheld A0/B1/swap combination. Every individual token and each query/key table cell still appears elsewhere. There are 2,500 supervised training steps per seed. The exhaustive test has 96 token-order/bit/query cases. Both heads matter: removing either head gives at most .625 exhaustive joint accuracy in any seed.

The exact combined token-function extraction agrees with direct model logits to at most 2.5e-14. Thus all rank-two and gauge comparisons use the same observed model functions. The gauge fit has no task labels; known A/B variables and outcomes are used only for evaluation.

## Reduction chooses a different function

Four of the ten full models fit their training distribution but fail the withheld combination. A rank-two approximation of their complete 4×6 token-function table often removes the offending residual. The empirical reduction minimizes squared table error weighted by 100,000 **unlabeled** sequences from the correlated training law. The uniform reduction uses all table cells equally, with no sampled inputs.

| Function | Exhaustive joint accuracy | Withheld-combination accuracy | Exhaustive BCE | Uniform table MSE relative to full model |
|---|---:|---:|---:|---:|
| Original full models | .950 | .600 | .0734 | 0 |
| Uniform rank two | .988 | .900 | .0164 | .0166 |
| Training-frequency rank two | **1.000** | **1.000** | **.0088** | .0207 |
| Half uniform/frequency rank two | .988 | .900 | .0119 | .0178 |

Uniform rank two repairs three of four rare failures; frequency rank two repairs all four. Frequency weighting has **higher** uniform table error and mean exhaustive logit MSE (2.56 versus 2.06 for uniform), so its task gain is not better imitation of the original model. The four failing models had negative withheld margins before reduction; three become positive under uniform rank two and all four under frequency rank two. This is a concrete example of the user's proposed distinction between function faithfulness and downstream task loss. It is a toy result, not evidence that low-rank compression generally improves generalization. The withheld *combination* was never used in training or frequency fitting; its constituent query/key cells are represented in the complete weight-derived table.

## Gauge chooses coordinates within that reduced function

Each rank-two function is then held fixed while 4,096 orthogonal rotations are searched. The reported A/B alignment and intended-bit effect fraction are post-fit diagnostics. They do not identify physical heads. Whole-sequence response L1 is evaluated on the actual sampled training law; all-pair edit L1 averages the 15 possible key-token swaps without task labels.

| Reduction; gauge | A/B alignment, all ten | Effect fraction, all ten | A/B alignment, six competent full models | Effect fraction, six competent |
|---|---:|---:|---:|---:|
| Uniform; SVD | .803 | .702 | .824 | .723 |
| Uniform; all-pair edits | **.958** | **.984** | **.970** | **.991** |
| Uniform; training-sequence response | .910 | .899 | .888 | .844 |
| Frequency; SVD | .810 | .706 | .830 | .726 |
| Frequency; all-pair edits | **.962** | **.984** | **.974** | **.992** |
| Frequency; training-sequence response | .808 | .710 | .801 | .686 |
| Frequency; uniform-sequence response | .697 | .519 | .705 | .521 |

The frequency-reduced, training-sequence gauge improves its own mean response L1 from 24.07 for the all-pair-edit basis to 23.15, but its A/B alignment falls from .962 to .808 and intended-bit effect fraction from .984 to .710. The same conflict remains in the six full models that already solve the withheld combination. A known-role edit positive control and key/output/individual-component sparsity also favor separate A/B axes. All rotations preserve each selected reduced function and its predictions exactly.

The failure mode is coactivation: reducing the L1 norm of *summed sequence contributions* rewards sum/difference coordinates for frequently co-occurring A and B. The objective measures sparsity at a different level than an individual token response or finite token edit. A sparse sequence code can therefore be less selective under component deletion. All-pair edits include role-invalid substitutions, so this is a declared synthetic edit prior, not natural-language intervention discovery.

## Limits and next experiment

This task still has only two intended variables and a linear final readout, and the held-out set is small but exhaustive for this grammar. The function-level component deletions are not physical-head interventions. The next test needs a third useful variable, correlated frequent contexts, a rare independent rule, and held-out *edit families*, then an assessment of whether the selected gauge transfers to finite interventions on the actual trained model. On the 546M model, use many fresh contexts and distinguish path-output fidelity, task prediction, and selective full-model effects. Do not choose a gauge using the labels on which it is later judged.

Reproduce: `taskset -c 2-5 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/loganriggs/Coding/neuron_circuits/.venv/bin/python train_correlated_dual_output_language.py --seeds 10 --steps 2500`, then run `dual_joint_edit_gauge.py --family correlated` under the same environment. Per-seed checkpoints and records are in `correlated_dual_models/`.
