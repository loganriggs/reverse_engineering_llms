# A joint rank-four cut preserves bracket paths and improves natural KL

The [output-aware reduction](OUTPUT_AWARE_HEAD_REDUCTION_RESULTS.md)
improved natural-text KL but lost 66% of the native bracket-head edit
path. We tested whether a rank-four projector must make that tradeoff.
For the same final-position pre-output-projection head state, let `P=UUᵀ`
with four orthonormal columns. The new **answer-free** fitting objective is

`(1−λ) E_natural||W(I−P)z||² / E_natural||Wz||²`
`+ λ E_edit||W(I−P)d||² / E_edit||Wd||²`.

Here `W` is the frozen head output matrix, `z` a natural head state, and
`d` an edited-minus-base head-state response. The natural moment uses
the same 2,048 FineWeb training contexts as the output-aware cut. The
edit moment uses 80 original parenthesis/square edits for the bracket
head and 44 noun-number edits for the agreement head. Every agreement
edit sharing one of the six validation token-swap identities is excluded.
No answer tokens, task margins, or downstream gradients enter fitting.
The edit categories are deliberately supplied, so this is intervention
supervision, not blind discovery. We fitted the declared grid
`λ={0,.001,.01,.03,.1,.3,1}` from weight, output-aware, and edit-PCA
starts; the best training objective is retained at each point. All
candidates remain orthogonal rank-four cuts of the same 128-dimensional
head state.

Every cut is tested on the same 576 held-out natural windows per head
across FineWeb, WikiText-103, and TinyStories at lengths 32 and 64,
plus 21 competent unseen-token-ID bracket pairs or 20 competent
lexical/format-shift agreement pairs. The native head path is the
finite change in the task logit margin caused by transplanting the
complete edited head-state difference into the base run.

| Head | Cut | Equal-panel mean KL ↓ | Top-1 flips / 576 | Native head-path error ↓ | Competent pairs still correct |
|---|---|---:|---:|---:|---:|
| Bracket | Weight-only control | .000977 | 7 | 2.29% | 21/21 |
| | λ=0, natural only | **.000803** | **5** | 66.21% | 21/21 |
| | λ=.001 | .000804 | **5** | 64.70% | 21/21 |
| | λ=.01 | .000806 | 6 | 43.37% | 21/21 |
| | λ=.03 | .000809 | 6 | 9.18% | 21/21 |
| | **λ=.1** | **.000810** | **6** | **2.05%** | 21/21 |
| | λ=.3 | .000811 | 6 | 2.44% | 21/21 |
| | λ=1, edits only | .005392 | 15 | 3.21% | 21/21 |
| Agreement | Weight-only control | .001839 | **4** | **5.16%** | 20/20 |
| | λ=0, natural only | **.001673** | 8 | 8.93% | 20/20 |
| | λ=.001 | .001673 | 8 | 8.86% | 20/20 |
| | λ=.01 | .001679 | 8 | 8.41% | 20/20 |
| | λ=.03 | .001690 | 8 | 8.05% | 20/20 |
| | λ=.1 | .001707 | 8 | 9.40% | 20/20 |
| | λ=.3 | .001735 | 8 | 12.24% | 20/20 |
| | λ=1, edits only | .002375 | 11 | 25.44% | 20/20 |

The bracket λ=.1 cut has a **17% lower** equal-panel mean KL than the
weight-only cut while matching its finite head-path fidelity. A paired
10,000-sample context bootstrap gives λ=.1 minus weight ΔKL
**−0.000167**, 95% interval **[−0.000311, −0.000059]**. Its head-path
error is 2.05% versus 2.29% for weight; a bootstrap over the six
token-identity families gives an interval for that difference of
**[−1.16, +0.31] percentage points**. Thus the edit-path difference
from weight is unresolved; the important result is that the large path
loss at λ=0 has largely disappeared without giving up the natural KL
advantage on this panel.

After inspecting the grid, we froze λ=.1 and tested **24 entirely new
arithmetic expressions**, four per each of the same six held-out
opening-token identities. This is an independent *context* check, not
an independent identity check. All 24 pairs are native-competent and
remain correct under every tested cut. λ=.1 misses the native finite
head path by **2.10%**, versus **2.83%** for weight-only, **64.57%** for
λ=0, and **2.69%** for λ=.3. The six-family bootstrap interval for
λ=.1 minus weight path error is **[−1.75, +0.07] percentage points**;
their small difference remains uncertain, but the recovery relative
to λ=0 transfers to new expressions.

Agreement is a useful negative control. Edit-output proxy error falls
as λ rises, yet the finite downstream path does not approach the
weight-only cut's 5.16% error; edit-only fitting reaches 25.44%.
The agreement output-aware KL advantage over weight also remains
uncertain under the paired context bootstrap. Optimizing a head's
linear output response is not enough to guarantee its downstream
finite margin effect. A later reduction objective may need local or
finite downstream sensitivity, potentially with explicit task
supervision, for this head.

This experiment produces a **candidate head-specific Pareto
improvement** on the bracket panel, not a universal canonical form.
The six token identities are chosen examples of one bracket relation,
and the fresh confirmation reuses those identities. The corpus panels
sample windows and may share source documents across context lengths;
their bootstrap does not model that dependence. The λ=.1 setting was
selected after viewing the first bracket validation paths, so the
24-expression confirmation, not the original 21-pair score, is the
stronger evidence of transfer. Whether this cut supports a stable,
human-readable sparse or grouped gauge is the next stage of the
canonicalization program.

Reproduce fitting with `flagship_head_joint_moment_fit.py`, the complete
grid evaluation with `flagship_head_joint_moment_eval.py`, the fresh
context check with `flagship_joint_bracket_confirmation.py`, and paired
analysis with `analyze_head_joint_moment_eval.py`. Fitted moments,
bases, per-window KL, finite paths, and summaries are saved under
`full_width_results/head_joint_moment_*` and
`full_width_results/head_joint_bracket_confirmation.json`.
