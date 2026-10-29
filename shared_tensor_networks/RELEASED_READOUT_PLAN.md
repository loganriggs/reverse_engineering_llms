# Untying after shared training

Fixed before execution. Reuse the15completed high-budget d8/h4ten-head shared fits (five teachers, three initializations). Copy each final shared solution into ten independently parameterized heads, preserving its initial function exactly. Fit each head for at most200TRFevaluations on the identical512inputs and noisy source labels. No new labels or retry selection. Retain the original shared and random-independent results as fixed baselines.

Separately, start independent heads at the exact teacher for each of the five teachers (one run per teacher, not triplicated). This is a privileged local diagnostic. Total20model groups,200individual head fits. All independent models have740parameters; the original shared model has164. Continuation has additional compute and a trained initialization, so it is not a fair standalone optimizer ranking.

Transfer uses exactly ten normalized predicted source outputs for every method, with the same four held-out tasks and32/128noisy labels. Report source training/population error, population transfer floor and finite adaptation error. Aggregate all three released initialization outcomes within teacher. Verify initial-function equality and that accepted optimization does not increase empirical loss.

Question: does releasing sharing merely reveal poor optimization from random starts, or does it degrade generalization despite reducing empirical error? Mixed answers are permitted. Do not select successful heads, extend failed fits or tune transfer penalties after results. Close after200fits and audits.
