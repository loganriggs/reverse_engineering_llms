# Small language-model surface proposals fail the semantic gate

The sampled sports world has linked score, player, and quarter rules, but its
handwritten sentence bank repeats heavily. This pilot tests a different way
to expand the surface grammar: ask a locally cached small instruction model
for typed sentence forms, then reject outputs that omit or repeat slots,
add extra numbers/names, or contain obvious unsupported claims. It is a
**proposal-bank screen**, not a training corpus or accepted DGP variant.

[`propose_sports_surface_bank.py`](propose_sports_surface_bank.py) uses the
cached `Qwen/Qwen2.5-1.5B-Instruct` revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, temperature .9 and top-p
.95, six style prompts, and seed 37051. It samples 120 one-sentence proposals
for a result fact with four typed slots and 120 for a player-points fact with
two slots. No raw corpus prose enters the prompt or output file. A brief
comparison with the already-cached 0.5B *base* and 0.5B *instruct* models
showed frequent replacement of slots with invented teams and scores; the
1.5B model was used for the measured screen.

| Proposed fact | Attempts | Mechanical accepts | Distinct accepted strings | Dominant rejection |
| --- | ---: | ---: | ---: | --- |
| Winner, loser, two scores | 120 | 6 | 6 | 97 missing/repeated slots |
| Player and points | 120 | 41 | 29 | 41 missing/repeated slots; 31 lacked a supported scoring verb |

The mechanical gate is too weak even at this low yield. All six result
accepts use “triumphed over”; some add “convincing” or “decisive,” which
requires a margin threshold absent from the typed fact. One calls the score
a thing a team “clinched,” which is awkward. Only about two of the six are
plain, safe variants. Many accepted player forms omit the unit *points*,
say a player scored “for victory,” or invent being a star, dominant, or
known for scoring prowess. Several are awkward rather than natural sports
prose. These are visible in the [candidate bank](results/sports_lm_surface_bank_candidates_seed37051.json);
the word “accepted” there means **mechanically accepted**, not semantically
approved.

This attempt does not solve the original generator problem. Its low slot
adherence and residual hallucinations mean using the candidate bank for
training would corrupt known-rule ground truth. Enlarging the bank from
these proposals would not necessarily fix document-level discourse or
Zipfian form either. Stop this small-model template proposal path rather
than increasing its sample count. A next candidate must have a stronger
semantic contract: for example, real sentences tied to richer sampled world
facts with independent factual rejection, or a renderer whose grammar
derivations themselves carry the proposition dependencies. In either case,
audit complete-document and task-window form against held-out human text,
verify visible-prefix rules and supported edits, and train bilinear models
only after stating which rules are behaviorally eligible for weight analysis.

Reproduce the proposal screen:

```bash
python language_task_suite/propose_sports_surface_bank.py \
  --n-per-kind 120 \
  --output language_task_suite/results/sports_lm_surface_bank_candidates_seed37051.json
```
