# Sports-document V3: more true facts with a small grammar worsen repetition

The [V2 study](SPORTS_DOCUMENT_DGP_V2_DISCOURSE_AUDIT.md) found that moving
clauses improves variation near answer sites but does not make whole reports
natural. A screen of corpus sentences suggested that real basketball prose
usually combines point totals with other numerical statistics or context.
V3 tests the narrow hypothesis that **adding those statistics to the sampled
world and typed sentence forms** is enough to improve the form.

[`sports_document_dgp.py`](sports_document_dgp.py) now samples five rebound
and assist totals alongside the existing players' point totals, without
changing the original V1/V2 random stream when `stat_richness=0`. At
`stat_richness=1`, each player clause may mention points plus rebounds or
assists using one of four richer single-player or four richer paired-player
forms. All mentions are drawn from the sampled plan. The audit independently
parses each rendered player segment and checks its numeric facts, in addition
to the three original visible-prefix rule sites and selective edits. The
extra statistics are **contextual world facts**, not newly scored rule sites.

The controlled form test compares V2 and V3 on the **same 256 sampled worlds**
in each game partition, cropping each document to its first 55 words:

| Partition | V2 whole-document repeated trigrams | V3 whole-document repeated trigrams | V2 winner/margin/top windows | V3 winner/margin/top windows |
| --- | ---: | ---: | --- | --- |
| Training games | 69.9% | 76.8% | 64.7 / 63.4 / 62.7% | 70.8 / 71.3 / 72.2% |
| Held-out games | 70.8% | 78.1% | 64.0 / 64.5 / 64.3% | 71.6 / 71.8 / 73.0% |

Word types in those 256 first-55-word panels are 799→796 on training games
and 798→786 on held-out games. The richer sentences are longer, so the fixed
crop contains fewer distinct player names; their repeated stock phrases
also increase cross-document trigram overlap. The site windows worsen too.
The comparison is a negative result for this **particular small typed
grammar**, not evidence that additional world state is unhelpful in general.

The semantic audit sampled 100 plans in each of eight source-style ×
form-entropy × paired-clause conditions on both game partitions: **1,600
plans**, three visible-prefix answers and three selective edits per plan.
It also independently verified **19,252** emitted point/rebound/assist
numbers against the sampled state. This establishes the renderer mechanics
but cannot make its prose natural.

**Decision:** keep `stat_richness` as a negative control and do not use V3 as
the main weight-interpretability training distribution. The source screen
does not imply that more numerical fields alone solve the problem; natural
reports also choose what to mention, link events, compare players and teams,
and vary syntax. The next candidate should use a probabilistic mention and
discourse plan with a much larger validated realization bank, or test a
corpus-conditioned renderer with explicit factual rejection checks. Preserve
the game-group split and compare actual samples, answer-site windows, and
whole documents before a model sweep. As before, only behaviorally learned
rules are eligible for weight analysis.

The paired aggregate is
[`sports_document_dgp_v3_paired_form_n256.json`](results/sports_document_dgp_v3_paired_form_n256.json).
The train and held-out semantic audits are under [`results`](results/);
raw source reports remain local outside Git.
