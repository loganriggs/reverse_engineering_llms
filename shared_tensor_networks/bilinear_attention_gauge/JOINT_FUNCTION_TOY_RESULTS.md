# A positive task-grounded canonicalization result on trained two-head language

The earlier [necessary two-head toy](DUAL_OUTPUT_LANGUAGE_RESULTS.md) has two independently needed output bits, A and B, and a rare query that swaps their output positions. Ten trained checkpoints solve all 96 exhaustive sentences, including the 12 rare-query A0/B1 sentences withheld during training. Bondwise rank-one QK1 reduction per head loses that held-out rule in the difficult seed 2. The later [real-model sparse-gauge red teams](EXACT_GAUGE_SPARSITY_RESULTS.md) showed that pair sparsity alone does not imply human legibility. This experiment changes **what is canonicalized**: the complete *effective token function summed across heads*, rather than either head's physical QK1 bond.

For each checkpoint, extract the exact role-valid pair table \(T_{qko}\) from the trained weights, with two query types, six key tokens, and two output logits. Summing the two head tables gives \(T\). Flatten query and output into a \(4\times6\) matrix \(M\). This extraction is **data-free after training**: it uses weights and all role-valid query–key pairs, not sampled sentences or target labels. The direct table forward matches the saved model on all 96 sentences to below `1e-10`. Every one of the 12 role-valid query–key pair cells occurs in at least one *nonheld-out* sentence; the withheld joint A0/B1 rare-query sequence is not supplied to the reduction.

## Which function is reduced matters

Use ordinary SVD to truncate \(M\), or apply rank-one SVD separately to each head's \(4\times6\) effective table. These have the **same 20 raw factor entries**: joint rank two uses \(4\times2+2\times6=20\); two headwise rank-one tables use \(2(4+6)=20\). After factor-gauge redundancy, their generic degrees of freedom are 16 and 18 respectively. They are different function classes, so equal factor count does not isolate every capacity effect. Neither is the earlier rank-one **QK1-bond** truncation, which keeps QK2 and OV factorization fixed.

| Effective-table representation | Raw factors | Perfect exhaustive / held-out seeds | Mean relative table MSE |
|---|---:|---:|---:|
| Combined rank one | 10 | 0/10 (one seed's holdout alone survives) | .437 |
| Two separate head tables, rank one each | 20 | 9/10 | .0305 |
| **Combined function, rank two** | **20** | **10/10** | **.00491** |
| Head-preserving joint 8×6 table, rank two | 28 | 10/10 | .00503 |

The combined rank-two table remains perfect on the withheld rare combination even in **seed 2**, where two separate rank-one head tables give only **87.5%** exhaustive accuracy and **0%** held-out rare accuracy. Across seeds, combined rank-two table error ranges from nearly zero to **.0351**, yet classification remains 100%. The head-preserving joint fit also succeeds, so the advantage is not dependent on literally deleting head identity; allowing a **shared key dictionary across heads** is enough at a larger representation budget. This is a concrete example of function-level reduction retaining a compositional task rule that a local headwise reduction can lose.

## Does a label-free sparsity prior find the two task variables?

Write the combined rank-two matrix as \(M=LD\), with \(D\in\mathbb R^{2\times6}\) a key-token dictionary. An orthogonal rotation \(R\) gives the *exactly same reduced function*: \(M=(LR^{\mathsf T})(RD)\). Search 4,096 rotations to minimize \(\|RD\|_1\). This key sparsity prior sees **no A/B labels or sentence targets**. A separate normalized factor-L1 prior over both \(L\) and \(D\) is also tested. Only after selection do we compare the learned dictionary rows with the true A-bit contrast \((-1,+1,0,0,0,0)\) and B-bit contrast \((0,0,-1,+1,0,0)\), allowing sign and permutation.

| Basis of the same rank-two function | Mean matched A/B cosine | Minimum across 10 seeds | Mean intended bit-flip energy fraction |
|---|---:|---:|---:|
| Plain SVD | .884 | .737 | .829 |
| 128 random rotations per seed, mean | .875 | — | .812 |
| **Key-dictionary L1 rotation** | **.975** | **.929** | **.997** |
| Normalized factor-L1 rotation | .975 | .931 | .998 |

The key-dictionary L1 falls from mean **17.20 to 14.78**. Its A/B alignment is close to a label-aware best-rotation ceiling of **.976**. Only about **4.8%** of sampled random rotations equal or exceed its alignment on average. The exact rank-two function and 100% task accuracy do not change under this rotation.

The final column is a **post-fit intervention evaluation**, not a fitting objective: remove one dictionary component from the reduced function, flip the A token, B token, or filler token, and measure which bit-flip response the removed component had carried. After post-hoc matching to A or B, the intended bit's squared effect is **99.7%** of the component's A+B+filler bit-flip effect under key L1, versus **82.9%** under plain SVD. The unaffected bit retains 100% classification accuracy after one sparse-component removal across the exhaustive sentences; the affected bit falls to about 50%. These interventions are on the **reduced explanatory function**, not on an identified physical node of the trained model. They show selective task semantics of this representation, not recovery of the model's actual computation graph.

This positive toy result explains why the choice of canonical object matters. Local WTG fixes a gauge for a chosen head/bond topology; the combined effective function is invariant to redistributing equal contributions across heads and can expose a smaller shared dictionary. Sparsity then chooses an interpretable orientation **within** that low-rank function space. It also explains why a prior that succeeds here need not work for the 546M head: the toy has only six role-valid key tokens and two known task variables, while the large model has many tasks, token types, positions, heads, and potentially no globally sparse token basis. On the toy, the A/B labels validate a component after fitting; on real language, semantic labels and causal tasks still need discovery and independent testing.

Reproduce with `dual_joint_function_canonical.py`. It reads the ten saved `dual_output_seed*.pt` checkpoints, runs exact all-pair SVDs and rotation sweeps on CPU, asserts direct-model agreement and pair-cell coverage, and writes `dual_joint_function_canonical.json` with all per-seed singular values, scores, dictionary rows, random controls, and bit-flip effects.
