# Reinserting reduced first-layer token functions into the v8 teachers

The [exact token-pair construction](V8_FIRST_LAYER_TOKEN_FUNCTION.md) and
[rank-matched reductions](V8_TOKEN_FUNCTION_REDUCTION_RESULTS.md) were local
first-layer results. This experiment replaces all layer-0 attention scores at
**one causal relative offset** with a token-pair score table, then lets the
remaining transformer run normally. It asks whether a local reduction
preserves the *full model's* next-token distribution and section-edit
responses. The [runner](eval_v8_first_layer_replacement.py) makes the patch
after the attention output projection; it subtracts the original contribution
at that offset and adds the contribution computed from the replacement score
and the unchanged OV value vector. All other offsets, layers, and weights are
untouched.

The exact table is built at absolute query/key positions (31,28), offset 3,
for each of the two independently trained, balanced v8 teachers. Using that
same table at every position of offset 3 introduces a small bfloat16 RoPE
translation discrepancy. Its full-model teacher KL is approximately 1e-7,
and its section-key effect error is below 8e-5 L1 on all panels, so it is a
practical identity control. Removing the entire offset is the causal-size
control. We also tried offset 24 in seed 1, but deleting that offset changed
the section-key effect by only .014 L1 on its panel, making it a weak test of
section behavior. Offset 3 deletes more of the model's overall computation.

The offset-3 factor fits use 4,096 training, 2,048 development, and 4,096
held-out v8 streams at fixed positions (31,28), with a 0.1-weight uniform
output prior for the data-aware method. The same 1,515 distinct token pairs
occur in training for either teacher; 981 of 4,096 test occurrences are pairs
unseen in training. At rank 8 the data-aware method lowers local sampled-pair
relative output L2 error versus output SVD in every head of both teachers.
For seed-1 head 1 it falls .129→.044; for seed-4 head 1, .075→.023. It
raises unseen-pair error in those heads (.205→.288 and .230→.266), so this
remains a distribution-relative tradeoff.

Two fresh full-model stream/causal panels were evaluated per teacher. Each
stream panel has 128 length-256 sequences. Each causal panel contains 50
exact-support section-key edits and smaller relevant/irrelevant-color edit
groups, drawn from 300 generated streams with a cap of 50 key edits. The
following numbers show full-model teacher KL over **all next-token sites**
and mean L1 error in the *section-key intervention effect* relative to the
unpatched teacher. A/B denote independent evaluation seeds 72001/73001 and
72002/73002. Lower is better.

| Teacher, method | All-site teacher KL A/B | Section-key effect L1 error A/B |
|---|---:|---:|
| Seed 1, zero offset | .01513 / .01350 | .04213 / .01892 |
| Seed 1, score SVD r8 | .000071 / .000079 | .00172 / .00139 |
| Seed 1, output SVD r8 | .000061 / .000070 | .00247 / .00132 |
| Seed 1, data + prior r8 | **.000009 / .000011** | .00224 / .00192 |
| Seed 1, output SVD r16 | .000008 / .000009 | **.00142 / .00078** |
| Seed 1, data + prior r16 | **.000004 / .000005** | .00176 / .00152 |
| Seed 4, zero offset | .04825 / .05218 | .02843 / .03159 |
| Seed 4, score SVD r8 | .000353 / .001121 | .00395 / .00446 |
| Seed 4, output SVD r8 | .000160 / .001053 | **.00218 / .00333** |
| Seed 4, data + prior r8 | **.000089 / .000865** | .00254 / .00374 |
| Seed 4, output SVD r16 | .000072 / .001119 | .00099 / .00117 |
| Seed 4, data + prior r16 | **.000049 / .001101** | **.00086 / .00106** |

The data-aware reduction has lower all-site teacher KL than score and output
SVD at both ranks, both teachers, and both panels. Its section-key effect
is **not consistently better**: output SVD or score SVD can win at rank 8,
and seed-1 output SVD wins at rank 16. These intervention differences are
small relative to the native key-edit effect (about 1.6–1.8 L1), but they
are larger than the exact-table replay floor. The choice of reduction metric
therefore matters even after reinsertion. Local sampled-pair error and
overall model KL do not suffice to select the most causally faithful
factorization for a given task. The zero-offset controls also show that
offset 3 matters to the model overall while its effect on section-key edits
is modest (.019–.042 L1); no single fixed offset has yet been shown to
carry the section-binding computation.

This is **not** whole-attention compression, attention-factor uniqueness, or
evidence that the factors are human-legible. It does not use whole-task
training holdouts: both teachers were trained on the section task, and the
factor training/dev/test split is by generated stream, not by entire rule or
section–color combination. Causal panels are held out from factor fitting
and use exact-support edits, but they test preservation of a known behavior,
not discovery of a new task.

The next attention experiment should identify the positions and paths that
actually mediate section-key binding: source header, relevant color, and
answer query can be separated by variable distances. Then compare
weight-only, broad-data, and task-causal reductions at matched budgets on
held-out section–color combinations. If no small fixed-offset family
mediates much of that behavior, stop the offset-specific reduction path and
move to path- or circuit-level interventions. A legibility claim additionally
requires sparse components with selective necessity, sufficiency, and low
spillover—not only low KL.

Artifacts: [seed-1 exact offset-3 table](results/v8_layer0_token_function_seed1_q31_k28.json),
[seed-4 table](results/v8_layer0_token_function_seed4_q31_k28.json),
[seed-1 reductions](results/v8_token_reductions_seed1_q31_k28.json),
[seed-4 reductions](results/v8_token_reductions_seed4_q31_k28.json),
[seed-1 panel A](results/v8_layer0_replacement_offset3_seed1_pilot.json),
[seed-1 panel B](results/v8_layer0_replacement_offset3_seed1_test2.json),
[seed-4 panel A](results/v8_layer0_replacement_offset3_seed4_pilot.json), and
[seed-4 panel B](results/v8_layer0_replacement_offset3_seed4_test2.json).
