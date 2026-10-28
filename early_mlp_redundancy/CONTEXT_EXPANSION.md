# M17: expanded contexts retain all ten training families, unevenly

2026-10-28. A fixed context expansion and native competence screen are complete. No featurizer has been trained on it. The dataset supplies broader context coverage for the causal-featurizer loop while documenting where the native model itself fails.

## Fixed dataset and separation

Six wrappers applied to all80 familiar development pairs produce480 pairs/960 endpoints. Four templates are available for training; two entire templates are excluded from training:

| Template | Split | Prefix (followed by newline) |
|---|---|---|
| consider | train | `Consider the following text.` |
| read | train | `Read this passage.` |
| text | train | `Text:` |
| exercise | train | `For this exercise:` |
| passage | evaluation | `A passage begins below.` |
| sample | evaluation | `Sample text follows:` |

The previous66/14 source-token split is unchanged. Every changed source token is absent from every opposite-split prompt, including new prefixes. All960 tokenizations decompose exactly into prefix IDs and original task IDs. Templates contain generic framing, not new tasks, semantic categories, or unseen combinations. Two evaluation templates are held out from training, but native outcomes have now been inspected: this remains research development, not pristine confirmation. Four single-component families have context evaluation only and no lexical evaluation.

| Lexical split | Template split | Pairs | Pairs in native-passing cohorts |
|---|---|---:|---:|
| train | train | 264 | 246 |
| evaluation | train | 56 | 55 |
| train | evaluation | 132 | 129 |
| evaluation | evaluation | 28 | 28 |

Only the train/train quadrant is eligible for fitting. Evaluating the other three distinguishes lexical transfer, context transfer on training lexical items, and combined lexical/context transfer. Answer labels and the underlying model's prior vocabulary are not held out.

## Native competence and retained failures

Freeze gates before model outcomes: per template × lexical split × family, require at least75% of pairs to have both binary answer comparisons correct and at least50% of endpoints to have the native full-vocabulary top1 answer correct. Passing cohorts retain all their pairs, including individual errors. Training eligibility depends only on train/train outcomes. Evaluation failures cannot alter training choices. No prompt replacement or threshold tuning.

87 of96 cohorts pass. Train/train has33 of40 passing cohorts, preserving246 of264 context pairs. Allten families remain represented, but past tense has only one passing training template. These are the observed failures:

| Template | Lexical split | Family | Pairs | Both-endpoint binary rate | Full-vocabulary top1 rate |
|---|---|---|---:|---:|---:|
| consider | train | antonym | 3 | 0.333 | 0.333 |
| consider | train | past_tense | 2 | 1.000 | 0.250 |
| consider | train | plural_noun | 3 | 0.667 | 0.833 |
| read | train | past_tense | 2 | 1.000 | 0.250 |
| text | train | antonym | 3 | 0.667 | 0.500 |
| exercise | train | past_tense | 2 | 1.000 | 0.250 |
| exercise | train | plural_noun | 3 | 0.667 | 0.500 |
| exercise | evaluation | plural_noun | 1 | 1.000 | 0.000 |
| sample | train | antonym | 3 | 0.667 | 0.333 |

Training coverage after this gate:

| Family | Passing context pairs | Passing training templates |
|---|---:|---:|
| agreement | 48 | 4 |
| antonym | 6 | 2 |
| capital_recall | 8 | 4 |
| country_language | 8 | 4 |
| entity_binding | 24 | 4 |
| number_successor | 16 | 4 |
| past_tense | 2 | 1 |
| plural_noun | 6 | 2 |
| pronoun_gender | 64 | 4 |
| sentiment | 64 | 4 |

These counts are coverage, not independent sample counts: each source pair is repeated under several wrappers. In particular, larger pronoun/sentiment families should not silently dominate a family-balanced training objective. The native screen establishes usable targets, not improved featurizers or discovered computational units.

## Next training and causal-validation design

Before fitting, freeze a paired schedule from the four M11 mixed checkpoints (scalar/block, two seeds): original-context task exposure versus expanded-context task exposure, with the same source pair/family schedule and unchanged natural-text interleaving. Restrict expanded training choices to native-passing train/train cohorts, use family-balanced draws, and report the limited past-tense context coverage. Match optimizer updates and initial checkpoints; longer prompts mean tokens/FLOPs are not matched. No widths, gates or gain searches.

Native-output fidelity remains one outcome. The decisive additional check is intervention fidelity: construct the same fixed decoded perturbation from an initial frozen featurizer, apply it separately to the native MLP write and to each reconstructed write, and compare changes in answer margin and output distribution relative to their respective intact baselines. Freeze groups, examples and prefix/task sites before fitting. Using initial perturbations prevents a trained featurizer from appearing faithful simply by shrinking its own intervention vectors. Such fixed-direction transport is still weaker than identifying the final model's semantic units; report that limit.

Keep both regions from M16. Evaluate all three nontraining quadrants and natural-text forgetting, report native-failing cohorts separately, and preserve whole-task exclusions (`sequence_copy`, `spatial_inverse`, `weekday_successor`, `role_binding`, `tense`). The allowed lexical `past_tense` family is distinct from excluded `tense`. New-task discovery and semantic-manifold validation remain open. This is one fixed coverage comparison, followed by causal validation, not another search on old matching scores.

## Audit and reproduction

```bash
python prepare_context_expansion.py
python probe_context_expansion.py
python audit_context_expansion.py
```

The preparation script refuses protocol changes once frozen; the probe refuses to overwrite completed results. Before the protocol was frozen, preparation was corrected to use all changed token positions because multi-token pairs have a null single-token-pair field. No model outcome informed that correction or template choice.

Independent audit rebuilds changed source tokens from endpoint differences, verifies all960 tokenizer boundaries and cross-split exclusions, and replays all960 saved native logits with NumPy for margins and top1. All96 gates match; maximum margin discrepancy1.78e-15. Native screening required960 forwards. Large logit samples remain at `artifacts/bilin18_tensor_decomposition/emlp_m17_native_context_samples.pt`; small data/protocol/results/audit are bundled here.

[Protocol and expanded examples](validation/context_expansion_protocol.json), [all native outcomes and gates](validation/context_expansion_competence.json), [audit](validation/context_expansion_audit.json).
