# Exact weight-only first-layer token-pair function

The v8 bilinear transformer has a useful data-free canonical **functional
object** at layer 0. For a fixed query position `i`, key position `j≤i`,
head `h`, query token `a` and key token `b`, let `x_t` be the token embedding
after the block's RMSNorm. Apply the model's rotary transform at `i` to
each Q projection and at `j` to each K projection. Define

`M_h[a,b;i,j] = <Q1_h(x_a,i), K1_h(x_b,j)> / d_h`,
`N_h[a,b;i,j] = <Q2_h(x_a,i), K2_h(x_b,j)> / d_h`,
`S_h = M_h ⊙ N_h`, and
`P_h[b,:] = W_O,h V_h(x_b)`.

The head's contribution from key `j` to query `i` is exactly
`T_h[a,b,:] = S_h[a,b;i,j] P_h[b,:]`. Summing `T_h` over heads and causal
keys gives the entire first-layer attention output. The [implementation](eval_v8_first_layer_token_function.py)
constructs `M`, `N`, `S` and `P` for **all 368×368 token pairs** at a chosen
position pair, with no sampled language data. It therefore removes the
internal QK factor basis from the effective score. It does **not** identify
which original QK branch has a unique semantic role, and it does not
canonicalize later layers' continuous residual functions.

We independently checked the table by masking the actual model's V output
to a single key position and comparing the resulting head-summed attention
output against the table prediction on 16 randomly chosen token pairs.
Maximum absolute output error is 2.8e-9 and 1.9e-9 for teacher seed 1 at
`(i,j)=(31,7)` and `(63,31)`, and 5.7e-7 and 1.9e-6 for teacher seed 4.
All pass the predeclared 1e-5 check. A random invertible post-RoPE QK-bond
change leaves one QK score matrix invariant to <3e-16 in float64. This
gauge check is **for the fixed position pair**; an arbitrary such matrix
need not commute with RoPE across all positions.

For a weight-only compression baseline, we measured the singular spectrum
of each collapsed score matrix `S_h` and the query unfolding of the full
output tensor `T_h`. The latter has the same singular values as
`S_h diag(||P_h[b,:]||)`, since
`T_h T_hᵀ = S_h diag(||P_h[b,:]||²) S_hᵀ`. The number below is the smallest
rank retaining **99% squared Frobenius energy** at `(i,j)=(31,7)`:

| Teacher | Heads 0–3: score `S` ranks | Heads 0–3: full-output query ranks |
|---|---|---|
| Seed 1 balanced | 38, 64, 17, 55 | **11, 36, 5, 29** |
| Seed 4 balanced | 24, 28, 4, 26 | **6, 11, 3, 9** |

The OV value map makes the represented **output function** more
compressible across query-token directions than score-space SVD alone
suggests. This is a weight-only mathematical result, not a preservation
claim for final language-model behavior. A second position pair `(63,31)`
shows the same pattern but different ranks; for seed 1 they are score
17/74/24/40 and output 4/45/5/13, and for seed 4 score 19/29/4/33 and
output 4/11/2/10.

The tables are position-specific. Equal-offset pairs shifted by 32
positions differ in collapsed-score Frobenius norm by roughly
**0.16–0.24%** across these runs. The model caches rotary sine/cosine in
bfloat16, so exact translation invariance of the continuous rotary formula
does not hold numerically. This matters for any claim of a single
relative-position token table.

This establishes a clean starting point for the requested two-stage
interpretability experiment: first construct the exact functional tensor,
then optimize a reduction for a chosen purpose. Next compare weight-only
SVD truncation, broad-stream data-weighted truncation, and task/causal
weighted truncation at matched rank and storage. Evaluate held-out token
pairs, whole section–color combinations, next-token KL, exact-oracle
prediction, key/relevant-color causal fidelity and irrelevant-color
spillover. A low tensor approximation error alone cannot certify an
interpretable or behavior-preserving circuit.

The full tables and spectra are saved for
[seed 1 at `(31,7)`](results/v8_layer0_token_function_seed1_q31_k7.json),
[seed 1 at `(63,31)`](results/v8_layer0_token_function_seed1_q63_k31.json),
[seed 4 at `(31,7)`](results/v8_layer0_token_function_seed4_q31_k7.json),
and [seed 4 at `(63,31)`](results/v8_layer0_token_function_seed4_q63_k31.json).
Each JSON links to its `.pt` `M/N/S/P` arrays.
