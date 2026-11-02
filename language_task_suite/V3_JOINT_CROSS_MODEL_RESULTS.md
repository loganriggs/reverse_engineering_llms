# The joint-language bracket computation moves across blocks

The [first joint-model mode study](V3_JOINT_MODE_RESULTS.md) found
task-selective first-layer score-table axes for verbs, objects, and outer
bracket closing. This report asks whether that three-way first-layer split
recurs in an independently trained model. It does **not**, despite similar
task behavior. Finite block interventions locate a major difference in
which trained stage the outer-bracket behavior depends on.

## Capability gate

The second model uses the same three-block width-64 four-head bilinear
architecture, eight-object training lexicon, exact-oracle auxiliary losses,
batch size 32, length 128, and 3,000 training steps, but a different model
and stream seed. Its [held-out evaluation](surface_mixture_v3_nested_varlen_depth3_seed2/heldout_edits_seed20261254.json)
uses 256 fresh streams. Verb/object KL is **0.095/0.110**, inner/outer
closer KL **0.0004/0.0001**. On 111 inner and 243 outer relevant opener
edits, effect-L1 is 0.0014/0.0007 and edited answer accuracy is 100%.
Thus both seeds implement a competent bracket task. Object induction remains
weak in the second seed as well: its model effect TV is 0.122 against the
oracle 0.250, with effect-L1 0.399 on 613 edits. This limits conclusions
about that particular object mechanism.

The same rank-8 shared token-score basis remains a useful first-layer
functional reduction. On [independent fit, prediction, and edit panels](surface_mixture_v3_nested_varlen_depth3_seed2/shared_token_basis_seed20261260.json),
task-blind joint-stream weighting gives whole-model teacher KL **0.00110**
and outer-opener effect-L1 **0.0006**, versus **0.00496/0.0038** for uniform
vocabulary weighting and **0.02918/0.0129** for marginals fitted on
no-bracket streams. The full weight-derived token table replays the native
model with teacher KL below 1e-6. These numbers say the data-aware metric
compresses the function well; they do not locate its bracket circuit.

## The first-layer axis split fails to recur

The [second-seed mode scan](surface_mixture_v3_nested_varlen_depth3_seed2/basis_mode_ablations_seed20261250.json)
uses 2,048 new unlabeled fit streams, a new 64-stream development panel,
and a separate 128-stream held-out prediction and grammar-valid edit panel.
The selected object mode, head 3 key 0, raises object KL by **0.510** on
holdout and has smaller verb KL damage **0.070**. But the development mode
with maximum outer-opener edit damage is head 1 query 0, which raises
outer-edit error only **0.031** on holdout and simultaneously raises verb
KL by **1.192**. The verb and outer selections coincide. There is no
comparably clean first-layer outer-bracket axis in this scan, even though
the rank-8 baseline preserves outer edits (effect-L1 0.0006).

This is a failed replication of the **layer-local axis split**, not a
failure of bracket behavior or proof that no bracket feature exists.
Head and axis labels can move under ordinary training symmetries; the
causal participation tests below show a deeper difference than relabeling.

## Where does outer closing depend on the network?

In the [exact first-layer token-table intervention](joint_depth3_layer0_participation_seed20261280.json),
the native and full-table models agree to mean KL below 3e-7. Zeroing all
first-layer heads while retaining the two downstream trained blocks gives:

| Model seed | Native outer-edit L1 | First-layer-zero outer-edit L1 | First-layer-zero outer KL |
|---|---:|---:|---:|
| First seed | ~0.0004 | **1.230** | **2.522** |
| Second seed | ~0.0004 | **0.036** | **0.014** |

The [block-output ablations](joint_depth3_block_participation_seed20261290.json)
use the same 128-stream and 121-outer-edit panels for both models. Each
intervention zeros one trained attention block's output and leaves the
other blocks active:

| Removed attention output | First seed outer-edit L1 | Second seed outer-edit L1 |
|---|---:|---:|
| None | ~0.0004 | ~0.0004 |
| Block 0 | **1.418** | 0.040 |
| Block 1 | 0.001 | **1.922** |
| Block 2 | **1.974** | 0.023 |

Outer-close prediction KL follows the same pattern: block-0/1/2 ablation
gives 2.841/0.0002/1.077 in the first model and 0.011/3.895/0.005 in
the second. Inner closing has a different pattern: all three block
ablations raise its KL substantially in both models. These interventions
establish **different causal dependence across depth for the same outer
bracket task**. They do not by themselves identify a minimal algorithm:
small single-block effects can reflect redundancy, and blocking a residual
path can alter downstream inputs in ways that obscure a clean stage label.
As an implementation cross-check, zeroing the complete first-layer score
table and hooking block 0's attention output to zero produce exactly equal
logits (maximum absolute difference 0) on eight length-64 streams for each
model. This ties the token-table and block-intervention results together.

The practical correction is to compare decompositions at a common
*behavioral or causal-path level*, not assume that the same task will occupy
the same first-layer head and eigenvector across seeds. A stable
canonicalization target may need to treat equivalent computations across
different layer allocations as related, while still tracking which paths
the particular trained model actually uses. Next test a path or featurizer
method on both seeds at matched prediction and edit fidelity. Require
separate reports for bracket, verb, object, nuisance edits, and genuinely
unseen task-family transfer. The synthetic streams remain a poor match to
FineWeb form; this result concerns controlled computation, not natural
language realism.
