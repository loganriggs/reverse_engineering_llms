# Weight-only QK reduction of the localized V7 action head

The [activation patch](V7_ACTION_ATTENTION_EDGE_PATCH.md) localized a learned
conditional-action edit to the changed adjective's score/value edge in
layer-2 attention head 3. The present experiment asks whether **model weights
alone** prioritize QK components that retain that behavior, rather than
requiring action labels or even activation samples to choose them.

The head has two multiplicative QK branches. RoPE groups each branch's 16
coordinates into eight two-dimensional rotary pairs. The score is the
product of the two branch sums. Removing a pair means dropping its two
coordinates from both the query and key dot products for that branch;
all other heads, the OV path, MLPs, and residuals remain native. Keeping
all eight pairs in both branches replays the full model within `1e-5` on
held-out token NLL and action-edit response.

[`eval_v7_action_rotary_pair_reduction.py`](eval_v7_action_rotary_pair_reduction.py)
compares five rankings of pairs within each branch:

1. **Weight-only:** Frobenius norm of each pair's effective unrotated
   `Q_pair.T @ K_pair` matrix. No data or task labels enter selection.
2. **Task-blind data:** mean squared pair contribution on 2,400 sampled
   causal query-key positions in 300 V7 documents of non-narrative kinds.
   Narrative is a whole-kind holdout for this fit, though it was present
   during original language-model training.
3. **Supervised:** leave-one-pair-out loss of action-edit response on 32
   development edits, disjoint from evaluation edits.
4. Random pair ordering, and 5. lowest-weight-first ordering, as controls.

Two independent fit/test panels each use 200 fresh narrative documents and
128 supported action edits. The same trained model checkpoint is used in
both; this is **data-panel replication, not model-seed replication**.

| Pair budget per QK branch | Method | Token NLL, panels 1/2 | Action accuracy, panels 1/2 | Edit response, panels 1/2 |
| --- | --- | --- | --- | --- |
| 8/8 (native) | all | 1.494 / 1.488 | .907 / .952 | .917 / .926 |
| 2/8 | weight-only | 1.827 / 1.819 | .193 / .151 | .110 / .100 |
| 4/8 | weight-only | 1.536 / 1.529 | .795 / .759 | .737 / .766 |
| 4/8 | task-blind other kinds | 1.536 / 1.529 | .795 / .759 | .737 / .766 |
| 4/8 | supervised development | 1.533 / 1.527 | .696 / .651 | .743 / .727 |
| 4/8 | random control | 1.927 / 2.627 | .106 / .012 | .081 / .008 |
| 4/8 | low-weight control | 2.690 / 2.667 | 0 / 0 | .001 / .001 |
| 6/8 | weight-only | 1.497 / 1.492 | .863 / .837 | .840 / .873 |
| 6/8 | task-blind other kinds | 1.497 / 1.492 | .863 / .837 | .840 / .873 |
| 6/8 | supervised development | 1.498 / 1.506 | .870 / .873 | .871 / .901 |

The weight-only and task-blind data rankings produce **the same selected
pair sets at budgets four and six**, despite fitting on different document
kinds. Weight-only selection greatly beats the two controls at four pairs.
The supervised ranking improves edit response at six pairs but can worsen
general token loss; at four pairs it has lower action accuracy. Thus the
weight-only ranking is a credible coarse **importance ordering** for this
head, not proof that each selected pair is a human-legible feature.

This is a constrained physical-coordinate reduction. RoPE preserves a
meaningful pairing, but arbitrary equivalent parameter gauges and other
network topologies are not handled. The `Q_pair.T @ K_pair` norm is an
unrotated weight-only proxy, not a canonical decomposition of the full
position-dependent bilinear attention tensor. The eight-pair parameter
count is local to one head, not a measure of total circuit sparsity. At
two pairs, the rule mostly fails; at six it is useful but not fully faithful.
## Independent model seed

I trained a second depth-2, width-64 V7 model for the same 2,000 updates,
with seed 34802 rather than 34801; its training, calibration, and test
documents were redrawn. It learns the local action rule again (IID full-vocab
accuracy .938; fresh 128-edit response .848), but not the cross-section rule
(IID accuracy .206, composition accuracy .103). The dominant edited-token
path shifts from layer-2 head 3 to **layer-2 head 1**: restoring that head's
source score/value removes .802 of a .848 response, whereas the largest
other-head combined patch removes .062. This demonstrates a head permutation
across independently trained models, not one universal physical head label.

I reran the pair experiment on head 1 with two independent fit/test panels.
It uses the same non-narrative whole-kind fit and disjoint action-edit
development/test split.

| Pair budget | Method | Token NLL, panels 3/4 | Action accuracy, panels 3/4 | Edit response, panels 3/4 |
| --- | --- | --- | --- | --- |
| 8/8 (native) | all | 1.497 / 1.490 | .915 / .947 | .809 / .806 |
| 4/8 | weight-only | 1.574 / 1.567 | .691 / .755 | .456 / .482 |
| 4/8 | task-blind other kinds | 1.574 / 1.560 | .691 / .803 | .456 / .517 |
| 4/8 | supervised development | 1.579 / 1.572 | .818 / .840 | .596 / .591 |
| 4/8 | random control | 1.999 / 2.063 | .279 / .372 | .072 / .039 |
| 4/8 | low-weight control | 2.103 / 2.103 | .242 / .367 | .056 / .031 |
| 6/8 | weight-only | 1.504 / 1.496 | .867 / .931 | .727 / .740 |
| 6/8 | task-blind other kinds | 1.505 / 1.499 | .915 / .931 | .776 / .763 |
| 6/8 | supervised development | 1.509 / 1.503 | .915 / .936 | .767 / .760 |

The weight-only ordering **does transfer as a coarse importance signal**:
it beats both controls in the second model. Its four-pair action response is
only .456/.482 versus native .809/.806, and task-aware selection does better.
At six pairs, other-kind activation energy preserves more response on the
first panel and comparable response on the second. The strong claim that
weights alone identify a small, nearly complete action circuit does **not**
survive this model-seed check. This is a stopping point for the single-rule
reduction thread. The next main effort should return to a broader, more
realistic sampled document process with multiple learned rules, then compare
method transfer and causal specificity across those rules. The current V7
grammar remains far from natural language in lexical tail and discourse form.

Reproduce:

```bash
python language_task_suite/eval_v7_action_rotary_pair_reduction.py \
  --checkpoint language_task_suite/results/document_mixture_v7_cross_depth2_4_2000_block2.pt \
  --output language_task_suite/results/v7_action_rotary_pair_seed35151.json
python language_task_suite/eval_v7_action_rotary_pair_reduction.py \
  --checkpoint language_task_suite/results/document_mixture_v7_cross_depth2_4_2000_block2.pt \
  --fit-seed 35251 --test-seed 35252 \
  --output language_task_suite/results/v7_action_rotary_pair_seed35251.json
python language_task_suite/train_document_mixture_v7_cross.py \
  --depths 2 --steps 2000 --train-n 10000 --test-n 1000 \
  --vocab-calibration-n 10000 --seed 34802 --device cuda \
  --output language_task_suite/results/document_mixture_v7_cross_seed34802.json \
  --save-models
python language_task_suite/eval_v7_action_rotary_pair_reduction.py \
  --checkpoint language_task_suite/results/document_mixture_v7_cross_seed34802_block2.pt \
  --head 1 --fit-seed 35351 --test-seed 35352 \
  --output language_task_suite/results/v7_action_rotary_pair_model34802.json
```
