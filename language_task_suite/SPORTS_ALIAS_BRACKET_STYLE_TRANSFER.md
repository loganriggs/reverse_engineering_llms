# Natural-report rule transfer across writing styles

Two depth-2 bilinear models trained on the RotoWire alias×bracket DGP passed
the bracket rule strongly on held-out RotoWire games. The question here is
whether that apparent circuit transfers to SBNation reports about held-out
games, using the *same* two controlled rule edits.

The [SBNation evaluator](eval_sports_alias_bracket_style_transfer.py) reads
the human report archive locally, uses the existing conservative
full-name→surname extractor and bracket-pair detector, samples two new
one-token surnames, and forms a 2×2 alias×bracket factorial. Of 1,292
unique held-out-partition SBNation games, 123 have both source opportunities
and **76** pass the shared 320-token BPE alignment rule. The [aggregate
evaluation](results/sports_alias_bracket_sbnation_transfer_seed98351_n76.json)
saves no report text. A one-reader inspection of eight additional eligible
source/target snippets found plausible name references and varied natural
parentheticals; that spot check is too small to certify all 76 cases.

| Evaluation panel | Seed 98201 alias paired exact / bracket paired exact | Seed 98202 alias / bracket |
|---|---:|---:|
| Held-out RotoWire, previous 100-game panel | .230 / .945 | .235 / .970 |
| Held-out SBNation, new 76-game panel | .079 / .079 | .079 / .250 |

The source change also shifts bracket position. The median bracket target
is GPT-2 token **18** in the previous RotoWire panel and **122** in the
SBNation panel, so the raw comparison does not isolate writing style.
The [position-control evaluator](eval_sports_alias_bracket_position_control.py)
draws all aligned held-out cases under the same selection code and balances
five 64-token position bins. RotoWire has 150 aligned cases, SBNation 76;
the common support yields **46 per source** (13/11/10/12/0 across bins).
The [aggregate result](results/sports_alias_bracket_position_control_seed98351.json)
also contains source/game hashes but no prose.

| Position-balanced panel, 46 per source | Seed 98201 bracket paired exact | Seed 98202 | Candidate-pair accuracy, seeds 98201 / 98202 |
|---|---:|---:|---:|
| RotoWire | .783 | .859 | .957 / .957 |
| SBNation | .109 | .261 | .587 / .609 |

The bracket edit still usually moves the logit preference in the right
direction on SBNation (.935 for both models on the matched panel), but the
full target probability response is smaller and the answer-site loss is
much larger. Correct edit direction alone therefore overstates rule
competence. Alias copying was only partial even on RotoWire and drops
further under this shift; it is not a clean cross-style circuit control.

The matched bins address one major confound but do **not** hold fixed exact
position, source-to-target span, surrounding vocabulary, report length,
parenthetical content, or game. This is a domain-transfer failure, not a
proof that a style variable alone caused it. The two reports also have no
full symbolic next-token oracle; their controlled answer sites are the
ground truth. Bracket closure is a positive control **within RotoWire's
distribution**, not yet a generally portable bracket algorithm.

For the larger research goal, this creates a useful whole-style holdout for
future featurizers and weight-only methods. A method should not be credited
with discovering a reusable bracket component merely because it explains
the RotoWire-trained model on RotoWire. First train a joint DGP with both
human styles or a semantically validated varied renderer, establish
behavioral transfer, then compare canonicalization/sparsity on held-out
style and supported edits at matched prediction fidelity. We should not
interpret a missing SBNation circuit in the present checkpoints: the
model did not learn that behavior reliably.
