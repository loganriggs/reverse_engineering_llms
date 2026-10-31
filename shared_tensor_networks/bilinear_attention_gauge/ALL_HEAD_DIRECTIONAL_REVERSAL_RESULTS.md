# Pair-level labels do not survive a full-context directional test

The [earlier full-model test](ALL_HEAD_COUNTERFACTUAL_CAUSALITY_RESULTS.md)
found a small punctuation-key edit effect for response-varimax axis 16 in a
rank-256 shared-key explanation of the first attention layer. Its pair-level
label was `space_word -> punctuation`: a space-prefixed word query and a
punctuation key. Axis 165 in the *same exact gauge* had the opposite label,
`punctuation -> space_word`. This follow-up asks whether those labels predict
the direction of a **new key-edit response** in the full 18-layer model.

## Design

The two axes and their validation pair-energy-matched deletion doses were
fixed before this test. We used 48 new 128-token windows (16 each from
FineWeb, WikiText-103, and TinyStories), excluding documents from previous
full-model and causal panels. Every window began with an earlier natural
space-word key and a later natural space-word query. The four conditions
crossed query form (natural word or period/colon) with key form (natural word
or comma/semicolon). The edit under a fixed query is the response to changing
just the chosen key from word to punctuation. Two random axes are retained as
reference controls.

The selected-pair calculation isolates only that one query/key interaction,
including both QK branches, RoPE, OV, and the sum over nine heads. The
full-model calculation includes all context keys and all downstream layers.
For each query form, its component-effect fraction is the pooled squared
change in the key-edit **logit response** after partial deletion of the axis,
divided by the pooled squared baseline logit key-edit response. This is a
finite counterfactual in a chosen reduced explanation, not a physical-node
ablation or mediation fraction.

## Result

At the isolated **punctuation-key cell**, axis 16 contributes mean squared
pair-output energy 216.4 versus 54.7 for axis 165. At the isolated
**punctuation-query/word-key cell**, axis 165 contributes 123.1 versus 11.3
for axis 16. Thus the two axes do distinguish the two labelled cells in the
selected-pair table. However, the *key-edit differences* of those pair
contributions are not directionally selective: for axis 16 they have energies
225.8 under a word query and 242.3 under a punctuation query; for axis 165,
206.8 and 194.2. A high value in one cell does not imply a selective edit
effect, because the word-key baseline also contributes.

The full-model key-edit response fractions are:

| Axis | Word query | Punctuation query |
|---|---:|---:|
| Response-varimax 16 (`word -> punctuation`) | 0.000489 | **0.018067** |
| Response-varimax 165 (`punctuation -> word`) | 0.000562 | 0.001165 |
| Random 94 | 0.000288 | 0.001162 |
| Random 106 | 0.000242 | 0.000814 |

The predeclared double-dissociation contrast was `(axis16 - axis165 under
word query) - (axis16 - axis165 under punctuation query)`. Its value is
**-0.01697**, with a corpus-stratified, paired-window bootstrap 95% interval
**[-0.03106, -0.00782]** (10,000 draws). The proposed ordering would have
made this contrast positive. All three corpora show the larger axis-16
fraction under punctuation queries: FineWeb 0.00794 versus 0.000390 under
word queries; WikiText 0.00300 versus 0.000138; TinyStories 0.03909 versus
0.000725. The largest punctuation-query window contributes 33.8% of that
axis's total component-effect energy, but removing its five largest windows
still leaves a punctuation-query fraction 0.00762 versus word-query 0.000211
after each condition is trimmed separately. The effect is heterogeneous;
the pooled fraction should not be read as a typical-window fraction.

The rank-256 baseline closely tracks the native 546M model on this panel:
mean native-to-reduced query KL is 0.00019–0.00071 across the four conditions,
relative squared error of the native key-edit logit response is 0.000358
under word queries and 0.000860 under punctuation queries,
and relative squared error of the query/key interaction logit response is
0.000618. The identity-projection implementation check has maximum absolute
logit discrepancy `1.96e-5`. This supports evaluating the reduced explanation
on these inputs; it does not validate its component labels.

## Interpretation and next test

The new test **rejects the proposed full-context directional interpretation**
of axis 16. It does not negate the earlier, narrower finding that deleting
axis 16 alters the punctuation-key edit response more than a word-key control
under natural word queries. A component can be associated with a particular
pair table cell and still have a different edit-response profile after the
rest of the sequence and model are included. The selected-pair key-edit
contrast already fails to cleanly separate these axes, so downstream effects
are not the only possible explanation.

The next legibility-prior comparison should optimize a **context-level
response signature** or a named task's counterfactual effect, with complete
held-out edit families and energy-matched within-gauge controls. Pair
response sparsity and coarse token-form labels remain useful diagnostics, but
should not choose axes for a causal ontology by themselves. Keep the reduction
metric (uniform, natural, mixed, intervention-preserving) separate from this
legibility objective; the [research matrix](CANONICALIZATION_RESEARCH_MATRIX.md)
lays out the full comparison.

Reproduce with `flagship_shared_key_directional_reversal.py`,
`flagship_shared_key_directional_pair_probe.py`, and
`flagship_shared_key_directional_summary.py`. Full records and the summary are
in `full_width_results/all_head_shared_key_directional_reversal_n8_s111-117.json`,
`all_head_shared_key_directional_pair_probe_n48.json`, and
`all_head_shared_key_directional_summary_n48.json`.
