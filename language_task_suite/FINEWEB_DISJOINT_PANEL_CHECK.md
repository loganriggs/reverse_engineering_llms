# Disjoint-panel check of toy-language realism and dependency cues

This check continues the [toy-language proposal](https://www.lesswrong.com/posts/GncmyuKmp526RJpmj/ambitious-mech-interp-w-tensor-transformers-on-toy-languages): build a *known joint data-generating process* whose competing language rules can be queried causally, while keeping clear which aspects resemble natural language. The new 192 × 513-token local FineWeb panel (`fineweb_n192_skip7000.pt`) has zero identical token rows in common with the 480-window panel used for earlier language-suite development. SHA256 of its token array is `10d66676c804569eaa501d0c3c425f357d1d4305eb2581f1e9a5403504f054c0`. The separate `fineweb_n192_skip11000.pt` panel also has zero identical rows with either panel. It was previously used elsewhere in the tensor-language repository; we inspected only its hashes and broad surface counts here and reserve its semantic contents from generator tuning. These facts establish disjoint token windows, not independent source documents or a globally untouched sample.

At a matched **128-token window**, the v6 generator's original .30 quote/.30 source rates again overproduce obvious task markers. The audit replays every generated prefix and independently checks the exact source and narrative oracles. Each configuration below uses 192 generated streams and the same 192 FineWeb rows, seed 12001. Rates are fractions of windows with at least one cue; the different tokenizers limit direct comparison.

| Quote / source rate | Source-answer sites | Quote: toy / FineWeb | Question: toy / FineWeb | Colon: toy / FineWeb | Parenthesis: toy / FineWeb | Repeated word, first 40: toy / FineWeb |
|---|---:|---:|---:|---:|---:|---:|
| .30 / .30 | 108 | .870 / .375 | .380 / .167 | .620 / .328 | .000 / .365 | .280 / .164 |
| .15 / .25 | 26 | .672 / .375 | .203 / .167 | .505 / .328 | .000 / .365 | .275 / .164 |
| .08 / .15 | 3 | .406 / .375 | .182 / .167 | .469 / .328 | .000 / .365 | .277 / .164 |

Thus tuning the quote marginal alone can nearly match FineWeb but removes almost all source-answer supervision in short streams. The colon, parentheses, and lexical-repetition gaps remain. The .08/.15 configuration is **not** a calibrated natural-language model. These are small-window surface comparisons, not estimates of natural dependency prevalence. The machine-readable audits for [.30/.30](surface_mixture_v6_fineweb_skip7000_audit.json), [.15/.25](surface_mixture_v6_fineweb_skip7000_lower.json), and [.08/.15](surface_mixture_v6_fineweb_skip7000_calibrated.json), plus the [selection code](fineweb_dependency_panel_skip7000.py), make the check reproducible.

I also inspected a fresh cue-selected dependency panel from the 192-window set. The selection seed is 20261103, with at most five previously unused windows per cue; only row IDs and judgments are stored in [the JSON](fineweb_dependency_panel_skip7000.json), not source prose. “Yes” means a local relation is reasonably clear, not that it gives a deterministic next-token target.

| Proposed relation | Yes | No | Unclear | Implication |
|---|---:|---:|---:|---|
| Named-source attribution | 1 | 1 | 3 | Broad attribution regexes overstate clean source binding. |
| Correction of a prior fact | 0 | 4 | 1 | Generic correction words remain weak evidence. |
| Alias / name equivalence | 3 | 0 | 0 | Promising, but only three eligible windows after exclusions. |
| Local numeric update | 0 | 5 | 0 | Number words mostly mark descriptive statistics and dates. |
| Numbered-item scope | 5 | 0 | 0 | Clear repeated evidence for a *scope* relation, though not yet a predictive target. |

The earlier 480-window pilot found source attribution 4/5 and numbered scope 5/5. The new panel weakens the case for a **generic** source-attribution cue while corroborating scope. It does not invalidate v6's intentionally sharp source-binding task, and five cue-positive examples are much too few for prevalence claims. The source model's prior failure to use the query key is a separate competence failure, not a finding about FineWeb.

The next DGP revision should sample a latent hierarchy of numbered sections and their scoped facts, then render several surface forms and mix them into the same stream as the existing bigram/skip and narrative rules. A query should ask for a fact from one of two sections with different answers, and supported edits should change the section key, the relevant fact, or an irrelevant section independently. Preserve exact prefix distributions and record per-site oracle labels. To avoid the observed rarity trap, maintain a natural-rate generator for realism checks and use conditional training examples or importance weighting for rare but diagnostic sites; report both distributions separately. Freeze the new grammar and its evaluation before examining the reserved panel's semantic content. Then train a model, establish query-key competence, and only afterward compare weight-only versus data-aware canonicalization and sparse/causal bases on whole-rule and whole-surface holdouts.
