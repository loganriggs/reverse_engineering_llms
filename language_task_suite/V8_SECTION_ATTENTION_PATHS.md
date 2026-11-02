# Which attention paths carry v8 section lookup?

The [first-layer reinsertion test](V8_FIRST_LAYER_REINSERTION_RESULTS.md)
preserved full-model predictions at a fixed relative offset but did not
identify the section-binding mechanism. Here we directly remove
**semantically specified query/key attention edges** in each of the three
bilinear-attention layers of two independently trained, balanced v8 teachers.
This tests causal necessity of path families and provides concrete targets
for later sparse/canonical reductions.

The [runner](eval_v8_section_attention_paths.py) parses exact-support section
questions to locate the queried section ID, each section header ID, each
fact color, and the final answer position. For a selected edge `(query,key)`
it computes that layer's native bilinear QK product times its V vector,
projects through `W_O`, and subtracts only that contribution via a forward
hook. Other edges, layers, and model weights remain unchanged. An empty-edge
hook replays the baseline exactly. In an independent layer-0 masked-V check,
the selected-edge subtraction matched removing the key's value vector at
the chosen query to maximum absolute error `4.77e-7`.

The cleanest selectors remove **all** section header IDs or **all** fact
colors at a specified query position; their edge positions stay fixed under
a supported query-key edit. Selecting only the *matching* header/color uses
the generator's ground-truth section label, so those targeted results in the
JSON are exploratory and must not be treated as an unsupervised circuit
discovery. The table below uses only the fixed-selector families. Two fresh
panels per teacher each contain 50 supported key edits. Panel A has 15
relevant-color and 10 irrelevant-color edits; panel B has 27 and 18. Values
are mean L1 difference between the native section-key edit effect and the
effect after ablation, over the model's full answer distribution. Larger
means that edge family matters more for preserving the trained computation.

| Edge family removed | Seed 1 A/B | Seed 4 A/B |
|---|---:|---:|
| Layer 0: query key → all section headers | .314 / .489 | .169 / .336 |
| Layer 0: answer → all fact colors | .002 / .007 | **.564 / .814** |
| Layer 1: answer → all fact colors | **1.067 / 1.274** | .003 / .003 |
| Layer 1: answer → query key | .004 / .004 | .395 / .802 |
| Layer 2: answer → query key | .772 / 1.001 | .523 / .291 |

The same qualitative split appears on both panels. In **seed 1**, the
answer directly reads fact colors mainly in layer 1; in **seed 4**, mainly
in layer 0. Both models use layer-0 attention from the query-key token to
section headers and later answer-to-query-key attention. Seed 4 additionally
uses answer-to-query-key attention in layer 1. These are *path-necessity*
observations, not yet a complete multi-layer circuit: intervening on a path
can change intermediate states and downstream computations, and the edge
families may interact non-additively.

The changes are usually damaging relative to the exact generator, which
supports interpreting them as useful paths rather than arbitrary teacher
differences. Baseline key-edit effect-to-oracle L1 is .434/.492 for seed 1
and .540/.552 for seed 4. Removing the dominant answer-to-colors route
raises that error to 1.404/1.668 for seed 1 and .997/1.263 for seed 4.
Removing layer-2 answer-to-key raises it to 1.087/1.377 and .877/.774.
Layer-0 query-key-to-all-headers raises it to .603/.875 and .658/.822.
Removing seed-4's layer-0 answer-to-all-colors also reduces its erroneous
response to irrelevant-color edits (oracle-effect L1 .296→.244 on A and
.513→.152 on B) while damaging relevant-color use; the learned path is
useful but not perfectly selective.

This helps explain why a rank-reduced **single fixed offset** was an
incomplete test. The source headers and fact colors are at variable
distances from the query and answer, while some useful attention runs into
the query-key token rather than directly into the answer. It also shows
that two teachers with similar section-level competence can allocate the
computation differently across layers. A canonicalization or sparse
student that preserves only one seed's dominant layer will not necessarily
transfer as an explanation to the other seed.

The next experiment should test **sufficiency and interactions**: retain
only these candidate routes in one layer at a time or patch their
contributions between matched supported examples, then measure section
prediction, teacher fidelity, exact-oracle fidelity, and spillover onto
other toy-language rules. Use whole section–color combination holdouts for
the featurizer-training stage. For interpretation, compare sparse factors
or shared-dictionary groups that specifically preserve these path effects
against weight SVD and broad-data reduction at matched total budgets.
Do not infer a unique algorithm from the current necessity tests.

Records: [seed 1 A](results/v8_section_attention_paths_seed1_panel74001_full.json),
[seed 1 B](results/v8_section_attention_paths_seed1_panel74002_full.json),
[seed 4 A](results/v8_section_attention_paths_seed4_panel74001_full.json), and
[seed 4 B](results/v8_section_attention_paths_seed4_panel74002_full.json).
