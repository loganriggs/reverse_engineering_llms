# Sampled sports-document DGP V1: linked rules, incomplete language realism

The objective is a variable sampled language distribution with known rules for
testing whether weight-based interpretability methods recover computations a
model actually learned. Capability is a per-rule behavioral eligibility check.
This is a **candidate distribution**, not the accepted replacement for the
earlier synthetic language.

[`sports_document_dgp.py`](sports_document_dgp.py) samples two basketball
teams from a RotoWire or SBNation source game, a winner, two final scores,
and five players from the participating rosters. It samples synthetic player
point totals and realizes a guarded corpus-derived score sentence, variable
single- or paired-player clauses, and three recaps in randomized order. The
recap targets are the winner, score margin, and highest scorer *among the five
listed players*. Sampling controls include source-style mix, player-frequency
exponent, margin limit, sentence-form entropy, paired-clause rate, and
paragraph-break rate. The roster correction keeps names on participating
teams; point totals remain invented, and the text does not fully model an
actual game or natural reporting decisions.

The [`audit`](audit_sports_document_dgp.py) checks each answer from the
visible prefix, without reading the latent plan. For each plan it swaps team
roles, increases the winning score, and boosts a non-leading player's points.
Each edit must change only its intended answer. Across eight settings
(RotoWire/SBNation score style × low/high form entropy × separate/paired
player clauses), all **2,400 plans** passed their three original sites and
all **7,200 selective edits**. All six recap orders occur in each arm. This
validates the marked local rules and intervention mechanics; it is not a
full next-token distribution oracle.

On independent panels of 64 first-55-word documents:

| Surface | Word types | Cross-document repeated-trigram share |
| --- | ---: | ---: |
| Linked synthetic report control | 202 | 90.1% |
| Sports V1, varied RotoWire forms, paired player clauses | 481 | 52.2% |
| Sports V1, varied SBNation forms, paired player clauses | 462 | 56.4% |
| Natural RotoWire reports | 833 | 16.0% |
| Natural SBNation reports | 1,132 | 5.4% |

The most varied V1 arm in this panel is much less repetitive than the linked
report, but still far from natural sports writing. Its target-site windows
repeat trigrams across documents about 62% of the time, so the gap persists
where the rules are tested. The natural controls are genre-matched but not
content- or writing-context-matched. These statistics are descriptive form
checks, not proof of semantic or distributional equivalence.

Example from the sampled distribution:

> The host Phoenix Suns beat the visiting Oklahoma City Thunder, 125-123, on
> Tuesday. Archie Goodwin scored 23 points, while Kyle Singler added 17 points.
> Andre Roberson scored 24 points, while Brandon Knight added 16 points.
> Markieff Morris posted 30 points. In the final result, Phoenix Suns won.
> The highest scorer in this group was Markieff Morris. The two final totals
> differed by 2 points.

The example makes the remaining defect visible: the document still follows
result → player list → recap. The first sentence may come from a report
template, but most later prose comes from a small typed grammar. Source
template guards reject many unsupported claims, but are not a complete
semantic certificate. Thus V1 is an intermediate multi-rule training control,
not evidence that the language-realism objective has been met.

**Next experiment:** vary discourse acts and rule-bearing clauses across a
document, and compare natural-form statistics at matched target windows as
well as whole documents. Split source games and score templates by group
before fitting a model. Train 1–4-block bilinear models on multiple DGP
settings with ordinary next-token loss; gate winner, margin, and top-player
behavior separately using held-out predictions and the three edits. Only
then compare weight-only decompositions, sparse gauges, and full QK×OV paths
for rules that actually pass. Preserve cross-rule and whole-form holdouts.

Reproduce the aggregate audit with:

```bash
python language_task_suite/audit_sports_document_dgp.py \
  --plans 300 --form-n 64 \
  --output language_task_suite/results/sports_document_dgp_audit_n300.json
```

This requires the official RotoWire and SBNation archives at the local paths
declared in the generator. Raw archives stay outside Git. The aggregate JSON
and scripts are committed. A [1–4-block bilinear training scaffold](train_sports_document_pilot.py)
has passed a one-step depth-1 smoke test; no substantive model has yet been
trained on this V1 DGP. Its current random-world split and exposed test word
types are explicitly diagnostic and must be replaced for a strong transfer
claim.
