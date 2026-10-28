# EMLP-P8: source-response direction transfers better than its magnitude

Exploratory training-context screen, 2026-10-28. This tests whether the candidate MLP1 causal
response is tied to the original prompt prefix. It does not train a featurizer or establish
transfer to unseen lexical pairs or tasks.

## Fixed competence screen

Prepend exactly `Consider the following text. ` (A) and `Read this carefully: ` (B) to all
36 P7 training pairs in agreement, sentiment, plural noun and past tense. These families were
selected using P7 outcomes. All 72 new pairs remain single-token edits. Evaluate all 144 endpoints;
retain errors and use the inherited cohort gates (75% both-endpoint binary correctness, 50%
full-vocabulary top1). Do not repair prompts or relax gates after seeing results.

| Family | A binary / top1 | A passes | B binary / top1 | B passes |
|---|---:|---|---:|---|
| Agreement |100% / 100%|Yes|100% / 100%|Yes|
| Sentiment |93.75% / 21.875%|No|100% / 62.5%|Yes|
| Plural noun |75% / 75%|Yes|100% / 100%|Yes|
| Past tense |100% / 37.5%|No|100% / 25%|No|

Forced-choice discrimination can survive when unrestricted next-token behavior does not meet
our competence criterion. A failed cohort is not evidence that its feature disappeared.
Only the five passing cohorts enter the intervention experiment, including their wrong examples.

## Response-transfer experiment

For each of the 48 passing pairs, in both directions, generate the local MLP1 response caused
by replacing the edited position's MLP0 write with its donor value. Generate it separately in
the original prompt and the prefixed prompt. Apply each response at MLP1 in the otherwise intact
prefixed recipient. Also test the original response rescaled to the new response's norm and four
signed-permutation random controls matched to the new response norm.

All arms target the prefixed donor output distribution. Positive recovery reduces donor KL;
zero is the intact recipient. Neither donor recovery nor larger intervention effects imply a
more faithful natural computation. These are oracle response vectors, not learned predictors.

| Cohort | Mean cosine, old vs new | New-context response KL recovery | Old-context response | Old direction, new norm | Random |
|---|---:|---:|---:|---:|---:|
| Agreement A |.886|18.62%|46.36%|16.43%|.13%|
| Agreement B |.943|19.67%|34.23%|16.68%|.12%|
| Plural noun A |.868|12.95%|17.79%|13.15%|−2.49%|
| Plural noun B |.910|9.38%|7.18%|7.22%|−.95%|
| Sentiment B |.983|18.72%|15.46%|16.98%|.06%|

Task-margin recovery agrees in sign for these response arms, but measures a different quantity.
For agreement A/B the old response recovers 30.71%/19.52% of the task-margin gap; after norm
matching it recovers 8.48%/8.66%, versus 9.68%/10.32% for the new response. Full results are in
[the JSON](validation/preamble_transfer.json).

## Interpretation and limits

The same training token pair retains a broadly aligned, causally useful response across these
prefix changes. This weakens a strictly prompt-specific-vector explanation. A reusable direction
with context-dependent magnitude is a plausible candidate representation. The large raw old-vector
advantage for agreement is reduced by norm matching, so it must not be advertised as better
feature fidelity. Directional differences also remain; magnitude alone is not proven sufficient.

This is a narrow manipulation of two preambles, not general context invariance. Added tokens
change both content and absolute position; those effects are confounded here. Repeated lexical
pairs are not independent discoveries. No confidence intervals or population-level claim are
made. Failed competence cohorts stay reported. Whole-task evaluation exclusions remain untouched.

The next bounded discriminator is the missing factorial arm: new-context direction with the
old-context norm, on these same five cohorts. Together with the three existing arms, it separates
intervention magnitude from direction without a gain sweep or learned fit. This would complete
the causal target audit; it should then feed a lexical-held-out comparison of shared directions
versus context-dependent response predictors, rather than further preamble collection. Such
lexical validation would remain distinct from the original whole-task holdouts.

## Reproduction and verification

From the packet directory, using the same environment as the README:

```bash
python probe_preamble_competence.py --output validation/preamble_competence.json
python probe_preamble_transfer.py --output validation/preamble_transfer.json
python audit_preamble_screen.py
```

The scripts save diagnostic logits/vectors to `/tmp/emlp_p8_competence_logits.pt` and
`/tmp/emlp_p8_transfer_samples.pt`; the audit expects those outputs. They are not bundled.
144 competence forwards plus 1,056 intervention-study forwards. Independent NumPy checks cover
four competence outputs and seven intervention outputs (maximum metric error 5.33e-15), and all
96 old/new response-vector pairs. Membership, retokenization, gates, arm counts and pooled KL
arithmetic pass. This is a sampled numerical audit, not independent replay of every forward.
See [audit receipt](validation/preamble_audit.json).
