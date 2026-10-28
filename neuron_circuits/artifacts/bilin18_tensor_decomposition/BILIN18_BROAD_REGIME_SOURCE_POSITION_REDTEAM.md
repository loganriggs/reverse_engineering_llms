# Edited-source equal-count subset red team

For each held-out pair, the finite effect of the actual edited source positions is compared with every equal-count subset of non-destination source positions. The output metric is fidelity to the complete frozen top-three program effect.

## Locked results

- **P0_edited_sources_are_exceptional_for_direct_read_programs: supported.** Observed: `{'families': ['agreement_attractor', 'ioi', 'capital_recall', 'negated_sentiment', 'repeated_sequence']}`
- **P1_direct_read_and_relay_regimes_separate: not supported.** Observed: `{'direct': ['agreement_attractor', 'ioi', 'capital_recall', 'negated_sentiment', 'repeated_sequence'], 'relay': []}`
- **P2_retrieval_payload_positions_are_specific: supported.** Observed: `{'observed': 1.0}`
- **P3_factual_recall_cue_is_specific: supported.** Observed: `{'observed': 1.0}`
- **P4_concrete_position_evidence_is_saved: supported.** Observed: `{}`

| Family | Site/program | Examples | Edited projection | Edited fidelity | Equal-count percentile |
|---|---|---:|---:|---:|---:|
| agreement_attractor | L11 H3,H2,H0 | 6 | 0.672 | 0.892 | 1.000 |
| ioi | L14 H4,H6,H7 | 6 | 0.941 | 0.997 | 1.000 |
| capital_recall | L13 H0,H6,H5 | 3 | 1.001 | 1.000 | 1.000 |
| sentiment_classification | L9 H7,H1,H3 | 3 | 0.109 | 0.207 | 0.778 |
| negated_sentiment | L5 H7,H8,H4 | 3 | 0.223 | 0.396 | 0.900 |
| repeated_sequence | L8 H3,H4,H6 | 4 | 0.994 | 1.000 | 1.000 |

## Concrete held-out examples

### agreement_attractor

- `The trees near the report` → `The book near the reports`; edited=[1]; effect=1.703/2.374; percentile=1.000; best subset=[1].
- `The cats near the tree` → `The book near the keys`; edited=[1]; effect=2.196/3.271; percentile=1.000; best subset=[1].
- `The report near the pictures` → `The reports near the book`; edited=[1]; effect=3.591/4.351; percentile=1.000; best subset=[1].

### ioi

- `John and Sarah went to the store. John gave the book to` → `John and Sarah went to the store. Sarah gave the book to`; edited=[8]; effect=2.776/2.963; percentile=1.000; best subset=[8].
- `John and Mary went to the store. John gave the book to` → `John and Mary went to the store. Mary gave the book to`; edited=[8]; effect=2.787/3.009; percentile=1.000; best subset=[8].
- `John and Alice went to the store. John gave the book to` → `John and Alice went to the store. Alice gave the book to`; edited=[8]; effect=2.536/2.691; percentile=1.000; best subset=[8].

### capital_recall

- `The capital of Russia is` → `The capital of Spain is`; edited=[3]; effect=12.753/12.737; percentile=1.000; best subset=[3].
- `The capital of France is` → `The capital of Germany is`; edited=[3]; effect=5.734/5.719; percentile=1.000; best subset=[3].
- `The capital of China is` → `The capital of Austria is`; edited=[3]; effect=9.287/9.279; percentile=1.000; best subset=[3].

### sentiment_classification

- `Review: The film was excellent. Sentiment:` → `Review: The film was awful. Sentiment:`; edited=[5]; effect=0.015/0.257; percentile=0.778; best subset=[8].
- `Review: The film was delightful. Sentiment:` → `Review: The film was boring. Sentiment:`; edited=[5]; effect=0.044/0.249; percentile=0.778; best subset=[8].
- `Review: The film was wonderful. Sentiment:` → `Review: The film was dreadful. Sentiment:`; edited=[5]; effect=0.039/0.356; percentile=0.778; best subset=[8].

### negated_sentiment

- `Review: The film was not wonderful. Sentiment:` → `Review: The film was not dreadful. Sentiment:`; edited=[6]; effect=0.080/0.358; percentile=0.900; best subset=[7].
- `Review: The film was not great. Sentiment:` → `Review: The film was not awful. Sentiment:`; edited=[6]; effect=0.096/0.437; percentile=0.900; best subset=[7].
- `Review: The film was not excellent. Sentiment:` → `Review: The film was not boring. Sentiment:`; edited=[6]; effect=0.228/0.574; percentile=0.900; best subset=[7].

### repeated_sequence

- `blue green; blue green; blue` → `blue yellow; blue yellow; blue`; edited=[1, 4]; effect=2.639/2.654; percentile=1.000; best subset=[1, 4].
- `orange purple; orange purple; orange` → `orange red; orange red; orange`; edited=[1, 4]; effect=2.632/2.653; percentile=1.000; best subset=[1, 4].
- `green yellow; green yellow; green` → `green black; green black; green`; edited=[1, 4]; effect=2.675/2.693; percentile=1.000; best subset=[1, 4].

## Guardrail

A high percentile shows that the edited source locations are unusually sufficient for this frozen program, not that their internal vectors have a unique semantic label. A low percentile can indicate a relayed representation rather than failure of the overall computation.
