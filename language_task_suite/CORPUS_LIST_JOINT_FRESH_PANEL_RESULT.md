# Fresh 234-document list × bracket panel

The earlier joint list×bracket test had only 20 FineWeb source documents.
This replication screens a disjoint 100,000-document range (offsets
235,000–334,999) and keeps all 234 eligible four-arm documents. Every selected
source has a local consecutive numbered list and an independently editable
parenthetical within the 256-token model context. The list shift and bracket
style edits are BPE-token-local. No candidate from this range was used to
train the dense models or fit the OV reductions.

The [builder/evaluator](eval_corpus_list_joint_fresh_panel.py) stores the
reconstructable 234-document tensor cache locally, but commits only aggregate
[scores and source hashes](results/corpus_list_joint_fresh_235000_335000.json).
Selected sources were exact-hash screened against the earlier list manifest
and SimHash-screened against one another. Cross-panel near duplicates beyond
exact matches were not screened, so document independence is approximate.

| Dense model | Joint exact, all four arms and both rules | Joint two-way | Bracket paired top-1 | List paired top-1 |
|---|---:|---:|---:|---:|
| Seed 71503 | .316 | .786 | .374 | .910 |
| Seed 71504 | .252 | .735 | .286 | .897 |

The large panel confirms that the list edit transfers much better than
full-vocabulary bracket closure. The joint exact criterion requires both
rule tokens to be top-1 for all four edits in a document. A two-way score
merely asks each correct token to beat its paired alternative. Thus the
models respond directionally to the edits but often prefer some unrelated
token at the bracket site. This is a capability limitation to gate any
claim about a complete two-rule circuit.

I also refit the fixed rank-4 OV methods from the preceding experiment,
unchanged, and evaluated them on this fresh panel. Fits still use separate
list-only and bracket-only training windows; this joint panel is test only.

| Model | OV method | Joint exact | Joint two-way | Bracket paired top-1 | List paired top-1 |
|---|---|---:|---:|---:|---:|
| 71503 | Dense | .316 | .786 | .374 | .910 |
| 71503 | Weight SVD | .389 | .769 | .472 | .900 |
| 71503 | Data SVD | .256 | .744 | .325 | .882 |
| 71503 | List-site fit | .278 | .786 | .346 | .902 |
| 71503 | Bracket-site fit | .325 | .765 | .393 | .887 |
| 71503 | Both-site fit | .329 | .786 | .389 | .904 |
| 71504 | Dense | .252 | .735 | .286 | .897 |
| 71504 | Weight SVD | .295 | .667 | .382 | .844 |
| 71504 | Data SVD | .162 | .714 | .226 | .857 |
| 71504 | List-site fit | .115 | .735 | .147 | .895 |
| 71504 | Bracket-site fit | .226 | .718 | .293 | .859 |
| 71504 | Both-site fit | .261 | .718 | .312 | .891 |

Weight SVD improves strict joint exact in both seeds, mainly by improving
bracket full-vocabulary top-1. This could be a useful denoising effect, but
it is not a canonicalization result or evidence of a human-legible circuit.
It also lowers two-way accuracy in both seeds. Rule-aware KL optimization
does not beat weight SVD on strict joint exact, despite the lower separate
rule-site teacher KL reported in
[the earlier fit](CORPUS_LIST_BRACKET_RULE_AWARE_OV_RESULT.md). Fidelity to
the dense teacher is a poor sole objective when the teacher itself often
misses the desired target. Descriptive differences here should not be
treated as statistical significance without paired per-document analysis.

## Explicit joint-supervision condition

To test whether exposure matters, the same builder found 216 joint sources
within the original training offset range 115,000–214,999. These are disjoint
from the 234 fresh test sources by offset and exact source hash. A fixed
1,000-step continuation of each separate-rule model mixed all four joint
arms with independent list and bracket windows, using full next-token loss
and marked-site weights. This is a **joint-supervised capability condition**,
not a test of zero-shot task composition. The [training script](train_corpus_list_joint_supervised.py)
and per-seed [71503](results/corpus_list_joint_supervised_seed71503.json) /
[71504](results/corpus_list_joint_supervised_seed71504.json) reports record
the fixed recipe and scores.

| Seed | Training condition | Fresh joint exact | Fresh joint two-way | Joint bracket top-1 | Joint list top-1 | Standalone bracket two-way | Standalone list exact |
|---|---|---:|---:|---:|---:|---:|---:|
| 71503 | Separate rules | .316 | .786 | .374 | .910 | .934 | .933 |
| 71503 | Joint continuation | .538 | .932 | .583 | .936 | .961 | .956 |
| 71504 | Separate rules | .252 | .735 | .286 | .897 | .880 | .867 |
| 71504 | Joint continuation | .534 | .876 | .575 | .927 | .919 | .878 |

Joint exposure improves unseen-source strict exact behavior in both seeds,
chiefly by improving bracket full-vocabulary prediction. Yet approximately
46% of fresh joint documents still fail the all-arms exact criterion.
The 216 training sources are a selected subset of natural pages and this
does not imply robust general-language composition.

Finally, I reran rank-4 OV reduction on the stronger teachers. The
[aggregate comparison](results/corpus_list_bracket_joint_supervised_ov_rank4.json)
still fits OV factors from separate-rule training windows only, then tests
the 234 joint documents. For seed 71503, weight SVD gives .624 joint exact,
data SVD .389, and balanced rule-site fitting .526, versus .538 dense. For
seed 71504, those are .436, .385, and .551, versus .534 dense. Thus the
preferred reduction changes by model seed. The rank-4 rule-aware fit improves
marked-site teacher KL over weight SVD, but that objective is not a stable
selector for higher strict joint accuracy. These reductions do not establish
canonical circuits or human legibility.

The next methods experiment should measure causal response *magnitude and
specificity* on targeted edits, matched with background fidelity, before
calling a reduced OV factor interpretable. The broader sampled-language DGP
still needs more independent rules and document forms; these two rules are
only a corpus-grounded pilot.
