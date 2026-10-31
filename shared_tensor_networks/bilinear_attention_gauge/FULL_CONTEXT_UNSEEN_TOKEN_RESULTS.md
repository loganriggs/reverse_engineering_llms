# The punctuation direction transfers to unseen gauge-fit key IDs, but pair-response remains stronger

The [same-position finite-axis test](FULL_CONTEXT_FINITE_AXIS_RESULTS.md)
found a punctuation-associated full-context QK1-key axis on four familiar
punctuation tokens. This red team changes the token IDs. It keeps the axes,
gauge, deletion strength, rank-256 reduction, and matched controls fixed.
Eight new punctuation tokens and eight space-prefixed-word controls are
selected **without model-effect scores**: two punctuation IDs in each of
four FineWeb-training frequency bands (20–59, 60–149, 150–399, 400–1199),
then the nearest-frequency unused word ID. All 16 IDs are absent from the
4,096 key positions used to fit the full-context gauge. They are also absent
from the older pair-response rotation's 2,048 sampled training token IDs
(two appear only in that rotation's validation sample). The underlying
rank-256 subspace itself was fitted on a broader FineWeb pair sample, so
this is an **axis-fitting holdout**, not a claim that no prior stage ever
saw those token IDs. Seven pairs match
their training counts exactly; the remaining pair has counts 1019 and 1017.
Some punctuation tokens are composite marks such as `.'` and ` ?`; this is
a token-form test, not a claim about abstract punctuation independent of
tokenization.

For 24 new, disjoint 64-token documents per corpus (FineWeb test,
WikiText-103, TinyStories), substitute the two members of a pair at the
**same key position**, three documents per pair per corpus. Delete one
selected QK1-key coordinate in the rank-256 explanation and run all 18
layers. As before, the score is the fraction of pooled squared
*final-position, full-vocabulary logit change* arising on punctuation
variants. The paired next-token log-probability effect is a separate,
more task-specific readout.

| Coordinate | FineWeb | WikiText-103 | TinyStories |
|---|---:|---:|---:|
| Full-context punctuation axis 7 | .980 | .957 | .974 |
| Full-context punctuation axis 2 | .822 | .898 | .953 |
| Same-gauge space-word axis 4 | .256 | .083 | .060 |
| Training-energy-matched random axis | .221 | .226 | .198 |
| Earlier pair-response punctuation axis 16, partial deletion | **.986** | **.993** | **.998** |

Full-context axis 7 exceeds `0.8` punctuation fraction on **all eight**
token pairs when pooled across the three corpora; axis 2 also exceeds `0.8`
on all eight. One rare pair has low absolute response energy, so its
fraction alone is weak evidence, but the other seven pairs are not all
driven by one familiar period token. Axis 7 minus the same-gauge control
has paired document-bootstrap 95% intervals `[.396,.954]`, `[.738,.939]`,
and `[.795,.941]` across corpora. Versus the random control the intervals
are `[.585,.910]`, `[.533,.897]`, and `[.575,.896]`.

The next-token target is somewhat more variable: axis 7's
punctuation-effect fractions are `.940/.993/.974`, while axis 2's are
`.813/.776/.987`. These are promising finite effects on these substituted
inputs, not a discovered language task. The older pair-response axis 16
remains at least as selective in full-vocabulary logits. Its partial
deletion strength (`0.167`) was fixed by matching predicted *training*
deletion-effect energy to full-context axis 7. Axis 7 minus the pair-response
control has full-logit fraction differences `−.006/−.037/−.024`; the
document-bootstrap intervals are `[−.048,.013]`, `[−.102,−.005]`, and
`[−.053,−.008]`. This does not support prioritizing full-context gradient
sparsity over the simpler pair-response prior for this label. Matching the
training effect does not exactly match test-domain first-layer dose.

The reduction itself passes a separate fidelity gate on these **same edited
inputs**. Mean native-to-rank-256 last-position KL is
`0.000206/0.000171/0.000113`; the pooled relative MSE of the
punctuation-minus-word *input-edit response* is
`0.00119/0.00165/0.00040` across FineWeb/WikiText/TinyStories. Thus the
reduced model reproduces the native model's tested token-edit response
closely enough that these component effects are not merely an arbitrary
off-function surrogate. An identity first-layer projection changes logits
by less than `1.9e-5`, gauge rotation by less than `4.0e-5`, and a
zero-strength deletion matches its own baseline exactly.

This evidence is stronger than concentration or pair-label enrichment:
it is a finite, multi-corpus, frequency-matched, held-out-key-ID prediction.
It remains a **function-level decomposition of the chosen reduced model**.
The native model has no unique corresponding coordinate to delete, and the
token substitutions also change other first-layer branches and values.
The output effect is predominantly a broad logit-vector response rather
than a named task success/failure. A next mechanistic step would identify
specific output-token consequences, test more natural minimal-pair edits,
and trace which earlier-position signals later layers use. Another
full-context sparsity sweep is not justified by the current comparison.

Reproduce with `flagship_full_context_same_position_edit.py --n-docs 24
--seed 1331 --token-panel unseen`, followed by
`augment_full_context_pair_control.py --source
all_head_full_context_same_position_n24_s1331_l64_f100_unseen.json`,
`analyze_full_context_finite_axis.py` on the augmented panel, and
`flagship_unseen_token_reduction_fidelity.py`. The exact token texts,
frequencies, per-document results, bootstrap summaries, and fidelity
records are in the corresponding `full_width_results/` JSON artifacts.
