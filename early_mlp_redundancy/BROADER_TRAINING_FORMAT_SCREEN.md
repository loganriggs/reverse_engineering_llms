# M55: one modest training-family addition from a fixed format screen

**Only object_color with two demonstrations passes the fixed competence gate.** It scores 75% both-endpoint contrast preference and 50% intended full-vocabulary top1 across four pairs. The other13 family/format cells fail. This is one potential supervised training family, not broad competence, a discovered task, or a verified computational unit.

## Prospective design and audit

The M54 design fixed seven previously nonpassing training families, two formats, four pairs per cell, and the original competence thresholds. The exact112 prompts/answers and token IDs were frozen before any model forward. Formats are explicit question/answer and two-demonstration completion. Exact demonstration source items are disjoint from query source items; arithmetic/Boolean primitives and some answer labels recur, so this is not lexical independence. All historical whole-family exclusions remain intact. No representation fitting or evaluation-set selection occurred.

Six of56 pairs have no valid disjoint single-token answer groups, all in animal_sound; they remain in the manifest with reasons. The100 valid endpoints were evaluated. Unequal token lengths are reported, not filtered by this competence gate. This differs from the older H1039 equal-length screen, which matters for interpreting the historical comparison.

Independent audit checks all112 prompts against the Hugging Face GPT2 tokenizer (producer used tiktoken), all100 final output vectors with NumPy, and all14 summary gates. Margins replay exactly. Fourteen direct Block.forward checks match the instrumented native model exactly. There were114 total forwards. No failed prompts were repaired or replaced after outcomes.

## Every family retained

Each cell reports valid pairs; both-endpoint preference / full-vocabulary intended-answer top1. Old cells count equal-length pairs, while new cells count all valid pairs. Query items are largely reused; Boolean syntax changes, and all prompt formats change, so this is a family-level comparison rather than an isolated format intervention on identical strings.

| Family | Historical H1039 | Explicit QA | Two demonstrations |
|---|---|---|---|
| comparative | 4; 0.25/0.00 | 4; 0.75/0.00 | 4; 0.00/0.00 |
| animal_sound | 1; 0.00/0.00 | 1; 1.00/0.00 | 1; 0.00/0.00 |
| object_color | 3; 0.67/0.67 | 4; 1.00/0.25 | 4; 0.75/0.50 |
| addition | 4; 0.00/0.00 | 4; 0.00/0.00 | 4; 0.00/0.00 |
| subtraction | 4; 0.00/0.00 | 4; 0.00/0.00 | 4; 0.00/0.00 |
| boolean_negation | 4; 0.25/0.25 | 4; 0.00/0.00 | 4; 0.00/0.50 |
| month_successor | 4; 0.00/0.00 | 4; 0.00/0.00 | 4; 0.25/0.25 |

Gate: at least4 valid distinct pairs, >=.75 both-endpoint preference, >=.50 intended-answer top1. Object-color completion is exactly at the accuracy thresholds. Only3/4 object-color pairs have equal token lengths; it would not pass the old aligned-pair minimum. Do not silently claim an aligned patching dataset.

The four correct top1 color outputs are white/black/green/red for snow/coal/grass/blood. The other four outputs begin with “a”, “a”, “the”, and “milk” instead of the intended color. Both correct and incorrect cases stay in any follow-up. Binary contrast preference can succeed while the model's most likely next token is not an answer; this is why the full-vocabulary gate is retained. Other families' failures may reflect formatting, conservative answer labels, tokenization, or competence limits, not absence of the underlying operation.

## Consequence for the featurizer loop

The screen supplies one human-authored candidate family outside the current ten-family panel. It does not yet provide verified causal supervision: a correct output need not depend on an identifiable early MLP operation. Before training, perform one bounded native causal availability check and evaluate the frozen representations against common targets from these same examples. Preserve existing whole-family transfer sets and prior-exposure labels. There is no reason here to tune further prompt formats or infer meaningful shared manifolds.

The concrete follow-up deliverable is a native direction/response bank for all eight color-completion endpoints plus a no-fit scalar/block/shared comparison. Its stopping criterion is completing the fixed bank and reporting its effect sizes and fidelity, regardless of success. This keeps the single-family excursion tied to the broader comparison. A training expansion, if justified afterward, must separately freeze sample/compute-matched controls; no fits start from this screen alone.
