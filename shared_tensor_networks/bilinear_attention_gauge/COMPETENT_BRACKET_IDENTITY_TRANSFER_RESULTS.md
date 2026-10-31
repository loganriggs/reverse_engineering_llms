# Frozen bracket axis transfers across six unseen token-ID variants

The [earlier unseen-token test](UNSEEN_TOKEN_EDIT_TRANSFER_RESULTS.md) found that a bracket axis fitted from the `Ġ(`/`Ġ[` substitution lost much of the rank-four head path on brace edits. The model also failed the brace-closing task, so that test could not distinguish a token-identity problem from a change in the bracket relation. We now test six **different token-ID pairs for the same parenthesis-versus-square contrast**, all absent from both original gauge-fitting panels. The rank-four layer-13/head-8 cut is weight-derived; its one-axis orientation is the previously frozen answer-free edit-PCA gauge. Neither is refitted on these new identities.

For each identity, two prefixes and three arithmetic expressions form a discovery competence screen. Different prefixes and new expressions form validation. We predeclared selection at at least five of six correctly ordered discovery prompt pairs. All six identities pass six of six, so no family is dropped. The validation prompts each differ by exactly one token ID. Competence means the native model prefers `)` after the parenthesis form and `]` after the square form. We also record the expected closer's rank against the full vocabulary, since a two-answer choice alone is weak evidence.

| Unseen opening-token pair | Discovery correct | Validation correct | Both expected closers top-1 | Rank-four versus native head path error | Frozen axis versus rank-four path error |
|---|---:|---:|---:|---:|---:|
| `(`/`[` | 6/6 | 6/6 | 5/6 | .010 | .022 |
| `:(`/`:[` | 6/6 | 3/3 | 3/3 | .020 | .060 |
| `((`/`[[` | 6/6 | 3/3 | 0/3 | .049 | .055 |
| `Ġ((`/`Ġ[[` | 6/6 | 3/3 | 0/3 | .023 | .022 |
| `=(`/`=[` | 6/6 | 3/3 | 2/3 | .040 | .079 |
| `Ġ"(`/`Ġ"[` | 6/6 | 3/3 | 1/3 | .027 | .059 |

Across the 21 held-out validation pairs, native forced-choice accuracy is **21/21**; both expected closers are top-1 in **11/21**, and every expected closer is in the top three. Mean absolute native head-path margin effects range from 2.61 to 7.38 across families. Pooling absolute path differences, rank four differs from the full native head by **2.29%** and the frozen axis differs from rank four by **4.39%**; all 21 axis paths have the rank-four path's sign. A bootstrap resampling the six token-identity families gives 95% intervals [1.46%,3.61%] and [2.79%,6.56%] respectively. The top-1 subset has essentially the same axis result (4.29%). These are positive transfer results for the *same symbol distinction* under new BPE encodings and prompt formats.

They narrow the earlier failure: changing token ID alone does not destroy the axis, whereas shifting to braces did on that incompetent task panel. This does **not** prove a universal bracket concept. The identity families were designed to preserve the parenthesis/square relation, some expected closers are merely top-three rather than top-one, and the validation count is modest. The rank-four path and one-axis path are sufficiency-style patches at a selected head, not evidence that this head is necessary or that the axis was recovered without seeing parenthesis/square edits at all. A stronger next test would train the answer-free gauge on several *competent* bracket relations and hold out an entire relation or rule, then evaluate output predictions and reverse patches on independent prompts.

Reproduce with `taskset -c 2-5 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /home/loganriggs/Coding/neuron_circuits/.venv/bin/python flagship_competent_bracket_identities.py`. All prompts, token IDs, model ranks, margins, and paths are in `full_width_results/competent_bracket_identities.json`.
