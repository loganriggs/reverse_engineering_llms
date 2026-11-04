# Causal edit effects versus exact BPE prediction in reduced OV factors

The [larger joint panel](CORPUS_LIST_JOINT_FRESH_PANEL_RESULT.md) makes it
possible to separate three questions: whether a model reacts to an edit,
whether that reaction is specific to the edited rule, and whether the exact
next BPE token is top-ranked. The same 234 heldout FineWeb source documents
have four token-local arms: unshifted/shifted list numbers crossed with
round/square parenthetical closure. The two dense teachers here have received
explicit joint training on disjoint source documents. Rank-4 OV students
were fit from separate-rule training windows only.

For a bracket edit, the measured response is the change in the logit margin
`logit(round target) − logit(square target)` between round and square arms.
For a list edit, it is the change in `logit(shifted digit) − logit(original
digit)` between shifted and unshifted arms. The corresponding margin at the
*other* rule site measures spillover. A positive intended response is
necessary but does not imply exact next-token prediction. The
[causal evaluator](joint_list_bracket_causal.py) has a synthetic independent
two-rule fixture and stores per-method aggregate effects in the
[JSON result](results/corpus_list_bracket_joint_supervised_ov_rank4_causal.json).

| Teacher seed | OV method | Joint exact | Bracket edit response | Bracket edit spillover on list | List edit response | List edit spillover on bracket |
|---|---|---:|---:|---:|---:|---:|
| 71503 | Dense | .538 | 23.57 | .389 | 19.59 | .157 |
| 71503 | Weight SVD | .624 | 28.73 | .376 | 14.72 | .174 |
| 71503 | Data SVD | .389 | 22.63 | .469 | 14.96 | .144 |
| 71503 | Rule-site fit | .526 | 22.60 | .389 | 16.14 | .160 |
| 71504 | Dense | .534 | 26.84 | .475 | 16.27 | .123 |
| 71504 | Weight SVD | .436 | 29.54 | .424 | 8.50 | .128 |
| 71504 | Data SVD | .385 | 25.95 | .504 | 11.29 | .119 |
| 71504 | Rule-site fit | .551 | 26.45 | .553 | 12.85 | .130 |

The intended responses are positive on **all 468 paired edit edges per
rule** for every method and both seeds. Intended effects exceed mean absolute
cross-rule spillovers by roughly 48–104× in the reduced models. That is
evidence of selective *behavioral responses* to these controlled edits, but
the direction criterion is saturated and cannot rank the OV methods. The
weight-only SVD amplifies the bracket response beyond its teacher in both
seeds and loses more of the list response, illustrating why one large causal
effect is not an overall fidelity measure. The data and rule-site fits keep
bracket response magnitude closer to the teacher, yet do not consistently
win strict exact prediction. No method has been shown to recover a unique
or human-legible circuit.

An [error audit](audit_joint_bracket_token_errors.py) shows why bracket
two-way and causal direction can be high while exact-token prediction is
weaker. It classifies each top BPE token by whether it contains the correct
closing character, contains the wrong one, or contains no closing bracket.
This is a coarse character-rule diagnostic; a token containing the correct
character can still have the wrong adjacent punctuation.

| Dense model | Exact BPE token top-1 | Top token contains correct closing character | Top token has no closing bracket |
|---|---:|---:|---:|
| 71503 separate-rule | .510 | .750 | .245 |
| 71503 joint-trained | .661 | .923 | .077 |
| 71504 separate-rule | .418 | .637 | .363 |
| 71504 joint-trained | .684 | .892 | .108 |

Wrong top tokens frequently include alternatives such as `].`, `.)`, `,`,
or a lone bracket where the target BPE token also contains neighboring
punctuation. After joint training, the correct bracket *character* is often
present even when the exact BPE token loses. Both levels matter: the former
is closer to the planted closure rule, while exact-token accuracy measures
the actual language-model prediction. The
[aggregate token audit](results/corpus_list_joint_bracket_token_error_audit.json)
contains counts and token-level error categories, without source prose.

Next, the useful weight-method benchmark needs a non-saturated causal
outcome: compare edgewise response magnitudes, full-vocabulary exactness,
and background KL under matched retained connections. The current OV rank
test alone says little about sparsity or legibility. In parallel, expanding
the joint DGP beyond list numbers and brackets remains necessary for a
general method-selection result.
