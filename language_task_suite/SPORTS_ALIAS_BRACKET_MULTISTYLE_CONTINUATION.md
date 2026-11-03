# Mixed-human-style continuation: partial bracket transfer

The [cross-style evaluation](SPORTS_ALIAS_BRACKET_STYLE_TRANSFER.md) showed
that two RotoWire-trained depth-2 bilinear models perform poorly on the same
controlled bracket edit in SBNation prose. This experiment tests whether
exposure to the second human writing style helps, with an equal-update
RotoWire-only continuation as a control.

The [trainer](finetune_sports_alias_bracket_multistyle.py) starts from each
frozen 6,000-step RotoWire checkpoint and reinitializes AdamW. It makes
600 updates with batch size 8, learning rate 1e-4, and the original
alias/bracket site weights 500/100. The mixed arm draws four examples per
update from RotoWire and four from SBNation; the control draws eight from
RotoWire. Both use the same seed and the same cached train-game panels: 584
aligned unique RotoWire games and 249 aligned unique SBNation games, with
one sampled alias pair and bracket span per game. Repeated minibatches
reuse these examples. All evaluation games lie in the deterministic
held-out partition, disjoint from training across both sources.

The periodically inspected panel has 46 RotoWire and 46 SBNation reports,
balanced over coarse bracket-target position bins. It is the same panel for
all checkpoints. The records are [mixed seed 98401](results/sports_alias_bracket_multistyle_mixed_seed98401_600.json),
[RotoWire control 98401](results/sports_alias_bracket_multistyle_rwcontrol_seed98401_600.json),
[mixed 98402](results/sports_alias_bracket_multistyle_mixed_seed98402_600.json),
and [control 98402](results/sports_alias_bracket_multistyle_rwcontrol_seed98402_600.json).

| Starting model | Continuation | Matched RotoWire bracket paired exact | Matched SBNation bracket paired exact | Matched SBNation two-closer rank |
|---|---|---:|---:|---:|
| 98201 | none | .783 | .109 | .587 |
| 98201 | 600 RotoWire updates | .859 | .174 | .576 |
| 98201 | 600 mixed updates | .967 | .565 | .772 |
| 98202 | none | .859 | .261 | .609 |
| 98202 | 600 RotoWire updates | .859 | .261 | .663 |
| 98202 | 600 mixed updates | .957 | .663 | .848 |

The mixed models also improve SBNation whole-token NLL on this panel from
7.87→7.33 and 7.83→7.31 nats/token. RotoWire whole-token NLL **worsens**
from 4.58→4.94 and 4.55→4.80, despite improved bracket accuracy; task
competence and broad language fidelity should be tracked separately.
Alias paired exact on the SBNation matched panel rises only .087→.174 and
.054→.130. It remains a partial-learning control.

The [fresh evaluator](eval_sports_alias_bracket_multistyle_fresh.py) draws
from source valid/test splits, filters out every game key present in that
source's train archive, and keeps the held-out game partition. It yields 56
RotoWire and only **15 SBNation** aligned games. This panel is independent
by game from both training and periodically inspected panels. Its
[600-update result](results/sports_alias_bracket_multistyle_fresh_seed98451.json)
shows SBNation paired exact .133/.267 at baseline, .233/.233 after the
RotoWire-only controls, and .533/.533 after mixed continuation. On the same
fresh RotoWire panel, mixed continuation reaches 1.0/1.0. The tiny
SBNation sample supports the direction of the result but gives a wide
uncertainty range.

An **exploratory**, bounded extension to 1,200 mixed updates raises the
periodically inspected SBNation bracket score to .609/.707, and the
[fresh 15-game panel](results/sports_alias_bracket_multistyle_1200_fresh_seed98451.json)
to .567/.667. The 1,200-step fresh file includes the 600-step controls for
reference, **not** matched-update 1,200-step controls. The extension was
chosen after inspecting the 600-step result. It does not clear the
bounded-check behavior target of .8 paired exact in both seeds, so the
single-rule training sweep stops here.

This is evidence that additional style exposure improves the controlled
bracket behavior in these models. It is **not** evidence of an interpretable,
portable bracket circuit: the models still fail many SBNation cases, the
source reports differ in content and local syntax, alias coreference is
surface-heuristic, and natural prose has no full next-token oracle. A
weight-only canonicalization or sparse basis should be judged on a future
model that behaviorally passes both styles before claiming recovery of a
shared rule. The next main DGP work returns to a broader set of sampled,
semantically validated language tasks rather than further tuning this
one bracket case.
