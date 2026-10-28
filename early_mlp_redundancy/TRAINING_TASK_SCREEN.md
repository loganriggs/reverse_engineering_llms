# EMLP-P7: the early response is useful for some training tasks, not a universal target

Training-only exploratory screen, 2026-10-28. Returns the bounded early-MLP study to the
competent-task → causal-target → featurizer loop. No featurizer was fitted.

## Data boundary and source diversity

Use the exact80 historical training pairs from the existing selector comparison. No evaluation
examples are exported or forwarded.76 pairs have a single changed token and equal lengths;
number-successor's four pairs change multiple tokens and receive competence checks only.
There are152 directional aligned examples but only106 distinct prefixes through the edited token.
All160 exported texts retokenize exactly to the saved GPT-2 token IDs.

| Family | Training pairs | Single-token pairs | Distinct pre-edit prefixes | Token pairs seen with multiple pre-edit prefixes |
|---|---:|---:|---:|---:|
| Agreement |12|12|1|0|
| Sentiment |16|16|4|4|
| Entity binding |12|12|12|4|
| Pronoun gender |16|16|1|0|
| Capital recall |4|4|1|0|
| Antonym |4|4|1|0|
| Plural noun |4|4|1|0|
| Past tense |4|4|1|0|
| Country language |4|4|1|0|
| Number successor |4|0|N/A|N/A|

Only sentiment and binding currently test the same token replacement under different source
prefixes. The other families add task/token diversity but mostly repeat or fix source context.
This distinction matters for evaluating a reusable context-conditioned computation.

## Native competence and intervention

Recompute native outputs on all160 endpoints. Use the inherited cohort gate: at least75% of pairs
have both forced-choice answers correct, and at least50% of endpoints have a correct full-vocabulary
top1 under the recorded answer groups. All ten training families pass; plural noun is exactly at
both thresholds. Sentiment's full-vocabulary rate is71.875%, binding75%, antonym75%, country-language
87.5%; the other families except plural noun are100%. Incorrect examples remain in the experiment.

For each of the76 aligned pairs, in both directions, install the donor MLP0 write at the edited
position and capture the induced MLP1 response there. Transplant only this MLP1 response into an
otherwise intact recipient. Since there is one changed token, earlier positions are identical;
later MLP0 writes cannot influence the response at this source position. Four same-site signed
permutations match its norm. Native persistent embeddings/value bus and the suffix remain live.

Measure final-output donor KL and change in the task's donor-favoring answer-group log odds.
The latter is divided by the natural donor-minus-recipient margin gap within family. These are
continuous recovery scores, not answer accuracy or an estimated fraction of model understanding.

## Results

Percent recovery: positive means moving toward the donor; negative means worsening the gap.
Random values are pooled over four controls per directional example.

| Family | Donor-KL recovery | Random KL recovery | Task-margin recovery | Random margin recovery |
|---|---:|---:|---:|---:|
| Agreement |17.12%|−.22%|8.82%|−.11%|
| Sentiment |19.51%|.68%|11.38%|.24%|
| Entity binding |.13%|2.33%|−.25%|1.10%|
| Pronoun gender |3.59%|.30%|1.27%|.18%|
| Capital recall |−.70%|.13%|−.39%|.12%|
| Antonym |−1.30%|.38%|−2.70%|−.19%|
| Plural noun |15.83%|1.87%|6.35%|.60%|
| Past tense |18.85%|1.34%|9.79%|.88%|
| Country language |.03%|.18%|.01%|.13%|
| Number successor |Not tested|Not tested|Not tested|Not tested|

The clearest descriptive signals are agreement, sentiment, plural noun and past tense. No
inferential usefulness threshold or confidence claim was preregistered for this screen. Choosing
these for the next diagnostic is outcome-informed exploration, not independent confirmation.

Binding is particularly informative: it has real pre-edit prefix diversity and native competence,
but this local early response does not recover its task signal. A generic “better featurizer” at
this one boundary is therefore not automatically a good target for every competent task. Other
positions, layers or persistent-state routes can matter. The current screen does not distinguish them.

## Decision and concrete next experiment

Do not train a universal featurizer against this target yet. The positive subset provides a
candidate causal-response target; negative families remain documented counterexamples. Preserve
whole-task exclusions (sequence copying, spatial inverse and the other evaluation families).

Next freeze two neutral preambles—`Consider the following text. ` and `Read this carefully: `—and
prepend each to every one of the36 training pairs in the four most responsive families. Recheck
native competence and single-token alignment on all72 new variants before using intended labels.
Keep all errors, report each family×preamble cohort, and stop after this fixed screen if the gate
fails; do not repair prompts until they pass. These are new prompt variants of known training tasks,
not new task discovery or untouched final validation. If competent, compare the causal response
with the original source-context target before deciding whether token-only or context-dependent
representations are needed. No held-out family needs to be opened for this step.

## Verification and reproducibility

1,072 native forwards:160 endpoint baselines plus152 source-response captures and760 interventions.
All80 training memberships,76 alignment selections and cohort rates independently checked. NumPy
recomputation on two saved examples agrees within8.89e-16 for KL and1.78e-15 for margins. Original
selection/eligibility metadata hashes remain unchanged. No audit forwards, fits, reserved task
forward passes or natural-document additions.

[Exported training panel and diversity audit](validation/training_source_context_audit.json)
· [Results](validation/training_task_responses.json) · [Audit](validation/training_task_responses_audit.json)
· [Runner](probe_training_task_responses.py) · [Metric audit](audit_training_task_responses.py)

The exported synthetic prompts, token IDs and answer groups let the runner operate without loading
old activation banks. The source exporter remains archival and needs the original workspace.
Sample logits are written to `/tmp/emlp_p7_logits.pt` and retained outside the compact packet.
Historical corpus/state tensors are not bundled. Results are training diagnostics, not held-out transfer.
