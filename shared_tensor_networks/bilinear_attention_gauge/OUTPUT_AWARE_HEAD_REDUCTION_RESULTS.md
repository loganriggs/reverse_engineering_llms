# Output-aware data reduction improves KL but loses a competent edit path

The [larger KL panel](LATER_HEAD_RANK4_KL_SCALE_RESULTS.md) tested a
dataset-aware cut made by ordinary PCA of 256 natural head states. That
prior minimizes raw state reconstruction, not what the head's output
projection uses. We now test a more relevant unlabeled objective while
keeping the **same orthogonal rank-four intervention class**.

For a 128-dimensional pre-output-projection head state `z`, head output
matrix `W`, and orthonormal four-column basis `U`, minimize

`E[||W(z - UUᵀz)||²] / E[||Wz||²]`.

This objective uses 2,048 FineWeb training windows, half of length 32
and half length 64, with no task answers or edit labels. We optimize the
orthogonal projector with QR-constrained gradient steps from three
initial subspaces: the original weight cut, original corpus PCA, and a
new 2,048-context corpus PCA. All three starts converge to the same
reported training loss to the displayed precision. A separate 256-window
FineWeb slice is used only to assess reconstruction transfer. The two
older frozen bases and the new 2,048-context PCA are controls.

| Head | Cut | Validation output MSE fraction ↓ | Validation raw head MSE fraction ↓ |
|---|---|---:|---:|
| L13 H8 / bracket | Weight only | .729 | .911 |
| | Corpus PCA, 256 | .739 | .853 |
| | Corpus PCA, 2,048 | .728 | **.841** |
| | Output-aware, 2,048 | **.703** | .870 |
| L11 H3 / agreement | Weight only | .761 | .921 |
| | Corpus PCA, 256 | .823 | .899 |
| | Corpus PCA, 2,048 | .767 | **.867** |
| | Output-aware, 2,048 | **.735** | .880 |

The output-aware cut wins on its held-out *linear head-output* proxy
for both heads. We then run the complete model on the same 576 natural
windows per head used by the prior KL check: FineWeb, WikiText-103, and
TinyStories at lengths 32 and 64. We also retest the 21 competent
unseen-token-ID bracket edits and 20 competent lexical/format-shift
agreement edits. No basis is refitted on any evaluation item.

| Head | Cut | Equal-panel mean KL ↓ | Top-1 flips / 576 | Native head edit-path error ↓ | Edit pairs correct |
|---|---|---:|---:|---:|---:|
| Bracket | Weight only | .000977 | 7 | **2.29%** | 21/21 |
| | Corpus PCA, 256 | .002467 | 10 | 80.25% | 21/21 |
| | Corpus PCA, 2,048 | .001889 | 11 | 77.53% | 21/21 |
| | Output-aware, 2,048 | **.000803** | **5** | 66.21% | 21/21 |
| Agreement | Weight only | .001839 | **4** | **5.16%** | 20/20 |
| | Corpus PCA, 256 | .001763 | 8 | 72.31% | 18/20 |
| | Corpus PCA, 2,048 | .001934 | 8 | 24.23% | 20/20 |
| | Output-aware, 2,048 | **.001673** | 8 | 8.93% | 20/20 |

Each KL mean weights the six corpus × length cells equally. Relative
edit-path error is mean absolute error of the finite logit-margin effect
from transplanting the edited head state, divided by mean absolute native
head effect. These are different estimands. Paired bootstrap over the
natural windows gives output-aware minus weight equal-panel KL difference
**−0.000173**, 95% interval **[−0.000318, −0.000067]**, for bracket.
For agreement the difference is **−0.000164**, interval
**[−0.000459, +0.000100]**; its KL advantage is unresolved. The
bootstrap treats windows as independent and does not account for possible
source-document overlap between the two length panels.

This is a concrete conflict between fidelity targets. The more
carefully fitted data-aware projector improves ordinary next-token KL
for the bracket head yet removes two-thirds of its native bracket-edit
path. All 21 bracket answers remain correct, again showing that task
accuracy does not certify path preservation. More data alone improves
ordinary PCA somewhat but does not close the bracket path gap. For
agreement, the output-aware cut is closer to the path (8.9% error) but
still trails the weight cut (5.2%); its natural KL difference is uncertain.

The next reduction experiment should explicitly mix natural-output
and edit-response moments, and predeclare both held-out KL and finite
path gates. That would test whether one rank-four cut can approach a
Pareto improvement instead of silently switching which behavior it
faithfully represents. The natural-output proxy is linear in the head
state; it is not itself downstream KL or a canonical function. The
selected heads and edits also do not establish a whole-model
compression result or a necessary answer circuit.

Reproduce the unlabeled fits with `flagship_output_aware_head_fit.py`,
the full-model comparison with `flagship_output_aware_head_eval.py`, and
paired uncertainty with `analyze_output_aware_head_eval.py`. Their
bases, per-window KL, finite paths, and summaries are saved under
`full_width_results/output_aware_head_*`.
