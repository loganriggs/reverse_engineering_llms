# Longer live-ledger training exposes an answer-to-answer shortcut

The [initial live-ledger pilot](SPORTS_LIVE_LEDGER_PILOT_RESULT.md) showed that 1,000-step bilinear models got moderate heldout run accuracy but failed coherent +1-play edits. This follow-up trains two width-48, four-head, depth-4 bilinear transformers for 3,000 steps on the same 5,000 base RotoWire training samples, then evaluates disjoint heldout games. Both use full next-token cross-entropy and the same seed, batches per step, optimizer, and 400 untouched heldout documents. The control sees only base documents. The paired-edit arm expands each base document into its original text, a +1 home-score offset, and a coherent +1 scoring-play edit, making 15,000 training documents but taking the same 3,000 optimizer steps. The empirical article-form caveat still applies: this is a structured live feed, not a natural-prose realism pass.

| Arm | Untouched heldout `Latest` copy | Run first | Run second |
|---|---:|---:|---:|
| 3,000-step control | .970 | .505 | .930 |
| 3,000-step paired edits | .995 | .515 | .868 |

These are number-candidate accuracies on 400 heldout documents. The [control training record](results/sports_live_ledger_bare_3000_control.json) and [paired-edit record](results/sports_live_ledger_bare_3000_paired_edits.json) give full-token NLL and full-vocabulary site accuracy as well.

The [200-document causal panel](results/sports_live_ledger_3000_causal_n200.json) is more revealing:

| Arm | Copy responds to +1 offset | Affected run responds to +1 play, in sampler support | Unaffected run stays fixed, in support | Second run stays fixed if first printed answer is raised by 1 |
|---|---:|---:|---:|---:|
| Control | 185/200 (.925) | 178/357 (.499) | 222/357 (.622) | 39/200 (.195) |
| Paired edits | 197/200 (.985) | 212/357 (.594) | 237/357 (.664) | 45/200 (.225) |

The remaining 43/400 affected-run edits per arm cross the generator's run-selection boundary; the table reports only edits still satisfying the selection rule. Paired edits improve the affected-run response by about 9.5 percentage points, but neither arm passes a strong arithmetic gate. `Latest` copy is much more robust.

The last column is a separate, deliberately inconsistent-text diagnostic. The printed first run number is changed by +1, while all score lines and the second answer's mathematically correct target remain unchanged. Ordinary second-answer accuracy is 179/200 for the control and 172/200 for paired edits on this panel. After corrupting the first printed answer, second-answer accuracy collapses to 23/200 and 24/200. The models are using the earlier *answer token* heavily when predicting the later one. This explains why `.930` second-site heldout accuracy cannot be read as independent evidence that the model computes the second score difference. It does not establish the precise algorithm used after reading that token.

Decision: stop treating either run output as an identified subtraction circuit. The first run output is only about 51% accurate, and its coherent edit response is weaker than the second's. The second output is strongly teacher-forced by the first answer. A next generator version should randomize team/query order and avoid exposing the earlier correct answer before another arithmetic target; the next evaluation must report each target with and without earlier answer tokens. For weight-method comparison now, use the causally stronger `Latest` copy rule and other independently verified rules, while continuing the broader document-realism work. Keep reconstruction, heldout loss, causal fidelity, and discovery as distinct outcomes.
