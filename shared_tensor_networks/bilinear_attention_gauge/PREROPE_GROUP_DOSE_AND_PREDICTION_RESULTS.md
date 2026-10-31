# Is the lexical response group special downstream?

The [conditional lexical result](PREROPE_CONDITIONAL_LEXICAL_RESULTS.md)
showed that an exact-gauge response group inside the fixed mixed rank-96
pre-RoPE QK1 cut responds strongly to the key tokens `" as"` and `" so"`
on new documents. This follow-up compares the identity-start group with
three groups recovered from independently randomized optimization starts and
three Haar-random rank-10 controls. It also tests 25%, 50%, and 100%
deletions and inspects the most affected predictions.

`flagship_prerope_group_dose_panel.py` uses **exactly the same 96 documents
and token spans** as the new-document conditional test (32 per corpus). The
script asserts document pairing. For identity-start/full deletion, all 96
per-document KL and pre-projection head-response-energy values agree
bit-for-bit with the prior artifact. Each candidate group removes its own
QK1 contribution only when the key ID is 355 or 523. These are paired
interventions on one fixed reduced circuit; no new gauge or reduction is fit
here.

Aggregate per-document KL at full deletion:

| Corpus | Identity group | Three independent-start groups | Three random rank-10 groups |
|---|---:|---:|---:|
| FineWeb | 3.68e-4 | 0.98–1.67e-4 | 1.08–2.82e-5 |
| WikiText-103 | 7.90e-5 | 1.43–3.33e-5 | 2.24–5.40e-6 |
| TinyStories | 6.23e-4 | 1.08–3.28e-4 | 1.65–6.76e-5 |

At the same deletion fraction, every independently found group has more
downstream KL than every random group in each corpus, even at 25% deletion.
The KL rises smoothly with dose, approximately quadratically at smaller
doses. This argues against an effect visible only after full erasure.

**The important qualification is response magnitude.** The discovered
groups remove far more head-response energy on these two keys than random
rank-10 groups. For example, on FineWeb at full deletion the identity group
removes 2,695 units of summed pre-projection head-response energy, versus
129–235 for the random groups. When a smaller discovered-group dose is
compared with a larger random-group dose at roughly similar energy, the
random group can have comparable or greater KL: on FineWeb identity at 25%
removes energy 168 and causes KL 1.71e-5, while one random group at 100%
removes energy 235 and causes KL 2.82e-5. On WikiText identity at 25%
removes energy 86 and causes KL 4.88e-6, while a random group at 100%
removes energy 70 and causes KL 5.40e-6. The KL/energy ratio is not uniformly
higher for the discovered groups. Therefore this test supports **selective
activation on the two keys**, not a uniquely sensitive downstream channel.
Group sizes also differ (six to ten axes), so equal deletion fraction alone
does not equalize perturbation size.

`flagship_prerope_edited_exemplars.py` inspects the three highest-KL
documents per corpus and the three most affected prediction positions in
each. These are descriptive examples selected for effect size, not an
independent validation set. High-effect contexts include `"as is"`,
`"as well as"`, `"as the away team"`, `"as pretty as"`, and `"so easy"`.
The effect often peaks immediately after the key. For a TinyStories
`"as fast as"` continuation, the model's probability of `" he"` changes
from 0.592 to 0.519 after deletion. Other contexts show smaller and varied
changes. There is no single demonstrated next-token rule or consistent
accuracy direction.

The current mechanistic hypothesis is now narrow: this subspace contributes
strongly to QK1 attention scores at certain `" as"`/`" so"` keys, across
several uses of those tokens. To distinguish a lexical lookup from a
reusable grammatical computation, the next experiment should remove one
occurrence at a time, stratify `" as"` by construction (comparison, role,
coordination, etc.), and test those strata on new documents. Those labels
must be used for validation or explicit supervised fitting, not described
as unsupervised discoveries.

The [construction-stratified matched-dose follow-up](PREROPE_AS_CONSTRUCTION_DOSE_RESULTS.md)
has now performed a controlled version on fresh curated prompts. First
versus second `" as"` effects survive matching post-`c_proj` residual
energy within identical sentences, but three Haar rank-ten groups show
the same pattern and bracket the discovered group's KL. This supports
context-dependent lexical activation, not a distinctive grammatical
causal channel; this single-lexeme branch is closed for now.
