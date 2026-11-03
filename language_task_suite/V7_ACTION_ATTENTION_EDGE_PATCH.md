# V7 action rule: one attention edge carries most of the tested edit response

The V7 depth-2 bilinear/RMSNorm model already passes the local conditional-action
behavior gate, while its cross-section lookup remains unlearned. Earlier
whole-kind local bond compression found that a rank-16 weight-only basis
retains more action response than an unlabeled other-kind PCA basis. This
experiment asks a more specific question: **does the learned action edit pass
through a particular bilinear attention score and value path?** It does not
try to infer a circuit for the unlearned cross-section task.

[`eval_v7_action_attention_edge_patch.py`](eval_v7_action_attention_edge_patch.py)
draws supported V14 narrative edits from fresh streams. Each edit changes
exactly one adjective in the current clause and has a different deterministic
action answer under the visible-prefix oracle. The original and edited
prefixes have the same length and differ at one source key position. For each
attention block and head, the script runs the edited model while replacing
one of these activations with its original-prefix value:

- The bilinear QK score at answer query → edited adjective key.
- The complete score row at the answer query, for comparison.
- That key's head-specific value vector before output projection.
- The score and value together.

Patching the immediately preceding key is a locality control. A same-input
source-activation replay is checked numerically against the unpatched model.
All patches are evaluated on the **full downstream model output probability**
of the edited target. The score is the mean decrease in edited-target
probability relative to the native edited run; it cannot be added across
patches because the model is nonlinear.

| 128 fresh supported edits | Seed 35051 | Seed 35052 |
| --- | ---: | ---: |
| Native edited-minus-original target probability | .931 | .902 |
| L2 attention head 3: original score on edited source edge | .688 | .699 |
| Same head: original complete query score row | .691 | .699 |
| Same head: original source value vector | .879 | .872 |
| Same head: original source score **and** value | .921 | .899 |
| Same head: original score on preceding-key control | ~0 | ~0 |
| Same head: original preceding-key value control | 0 | 0 |
| Largest other-head/source-value effect | .014 | .019 |

The changed adjective's **one QK edge** accounts for nearly all of this
head's score-row effect. Restoring both that edge and its value almost
eliminates the edited-target response on both panels. The value patch has a
positive response-removal effect in 126/128 and 128/128 cases; the combined
patch does in all 256. The first attention block and other heads have much
smaller *single-patch* effects on this measure.

This is strong local causal localization of an **eligible, easy action rule**
in one trained model. It is not a weight-only circuit extraction or proof of
unique computation: the patch uses activations from the original prefix,
downstream RMSNorm/MLP layers remain active, and multi-head redundancy or
interactions could be missed by single-patch effects. The two panels vary
data seed, not model seed. The V7 language distribution still fails lexical
and discourse realism screens, so this is a mechanism-method control rather
than the final realistic-DGP interpretability result.

Next method comparison: derive an explicit head-3 QK/OV weight-path candidate,
including its input-dependent layer-2 residual basis, and test whether a
weight-only or task-blind canonical/sparse reduction selects the same edge
at matched held-out token NLL and action-edit fidelity. Preserve whole-kind
holdout for fitting any data basis. Stop this single-rule detour if the
weight-only candidate does not generalize to a second trained model seed;
then redirect effort to a more variable sampled document process with
multiple behaviorally learned rules.

Reproduce the two panels from the saved V7 block-2 checkpoint:

```bash
python language_task_suite/eval_v7_action_attention_edge_patch.py \
  --checkpoint language_task_suite/results/document_mixture_v7_cross_depth2_4_2000_block2.pt \
  --seed 35051 --output language_task_suite/results/v7_action_edge_patch_seed35051.json
python language_task_suite/eval_v7_action_attention_edge_patch.py \
  --checkpoint language_task_suite/results/document_mixture_v7_cross_depth2_4_2000_block2.pt \
  --seed 35052 --output language_task_suite/results/v7_action_edge_patch_seed35052.json
```
