# Executable input featurizers: training readiness

M80 completed the training inventory and missing-input capture. No new featurizer has been fitted. The next stage is implementation and correctness preflight under a frozen protocol.

## Why this is a different experiment

Earlier scalar/block/shared featurizers encoded the native MLP write after it had already been computed. Their success could identify useful output directions without explaining how the MLP computes them. The historical training cache contains writes, final logits and causal targets, but not normalized MLP inputs. M80 adds those inputs for all140endpoints in the existing70training pairs, with exact replay of every historical write and final-logit vector. An independent NumPy bilinear calculation verifies every new input against its cached write. No evaluation input was captured for training.

## Coverage and limitations

There are11training families,14preserved source-group holdout pairs, and7whole-task excluded families. Current family names and exact token sequences are disjoint from the whole-task evaluation set. This does not establish absence of related semantics from pretraining or natural training text. Existing causal teachers cover138source,140final and140identity assignments; two source assignments are ineligible. This turn did not rerun causal teachers.

| Family | Training pairs | Both contrast correct | Both top-1 correct | Distinct prefixes through first changed token |
|---|---:|---:|---:|---:|
| agreement | 12 | 12 | 12 | 8 |
| antonym | 3 | 3 | 1 | 6 |
| capital_recall | 2 | 2 | 2 | 4 |
| country_language | 2 | 2 | 2 | 4 |
| entity_binding | 6 | 6 | 4 | 12 |
| number_successor | 4 | 4 | 4 | 8 |
| object_color | 4 | 3 | 2 | 6 |
| past_tense | 2 | 2 | 2 | 4 |
| plural_noun | 3 | 2 | 1 | 6 |
| pronoun_gender | 16 | 16 | 16 | 2 |
| sentiment | 16 | 16 | 7 | 32 |

Overall68/70pairs pass both-endpoint contrast and53/70both-endpoint top1. Contrast labels measure the specified answer alternatives, not general task mastery. Effective source-context diversity is much smaller than pair count for some families; pronoun gender has16pairs but only2prefixes through the source edit. Retain weak cases and report coverage rather than silently filtering or rescuing prompts.

## Frozen next comparison

Use128bilinear input coordinates, then dense, scalar, disjoint-block, or overlapping-group gates. All four kinds share443,776trainable parameters and initial arrays perseed. Dense uses128active coordinates; sparse versions use32or fewer, so equal parameter count is not equal active compute. Shared groups decode an atom once via union gating.

Within each kind, compare intact-KL supervision with intact KL plus allthreecausal corners: predicted changes on native backgrounds, native changes on predicted backgrounds, and predicted changes on predicted backgrounds. Center each response on its own background. This targets the compatibility failure exposed by M79. Natural retention, schedule, initial state and optimizer are matched within the objective comparison. Use two fixed seeds, with no best-seed selection.

The full-dictionary projection of native writes provides a privileged local-MSE reference for each decoder; it is neither an executable input predictor nor a behavioral upper bound. Known bilinear toys, independent losses, donor/recipient gradients, union counts, identity cases, holdout checks and actual forward replacement must pass before fitting. Detailed architecture, prospective warmup/schedules, calibration rules and evaluation are in validation/bilinear_input_featurizer_preflight_protocol.json. Numeric preflight tolerances and code receipts must be frozen before those checks execute.

This connects the conceptual hypothesis—context-dependent nonlinear features with reusable downstream effects—to an explicit training intervention. It does not assume that learned blocks will be semantic manifolds or literal native units. Those claims require subsequent transfer and causal-specificity tests.
