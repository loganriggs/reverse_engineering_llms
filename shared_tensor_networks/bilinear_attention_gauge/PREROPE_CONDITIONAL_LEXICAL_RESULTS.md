# Conditional causal test of a pre-RoPE response group

The [exact-gauge group analysis](PREROPE_GAUGE_AND_GROUP_RESULTS.md) found a
reproducible response subspace inside the mixed rank-96 pre-RoPE cut of the
546M model's layer-0 head-0 QK1 bond. Its pair energy was enriched on a
post-hoc set of connective-like keys, but frequency-matched labels were
uneven. This follow-up asks whether those keys are also where removal of
the group changes the **full model's next-token distribution**.

`flagship_prerope_conditional_group_ablation.py` keeps the selected rank-96
cut fixed and removes the exact-gauge group's QK1 contribution only when a
key position has a specified token ID. The edited attention runs through all
18 layers. The reference is the unablated rank-96 model, not the native
model. The zero-mask edit reproduces its logits exactly; an all-key mask
matches an equivalent full-bond deletion to at most 3.6e-5 maximum logit
difference. This validates the conditional edit against the existing bond
operation. The edit is a counterfactual intervention on the reduced circuit,
not a claim that the model stores the group as an explicit module.

## Discovery panel and narrowing the label

The first full-model panel used 16 windows per corpus: eight documents under
each of two seeds on held-out FineWeb, WikiText-103, and TinyStories. It
compared the 55 tokenizer IDs whose decoded text strips to one of ten
connectives against 20 token-form/frequency-matched ID sets, and separately
against 20 masks that select the **same number of nearby key positions in
each document**. All comparisons use the same input windows. The target
exceeded every control in aggregate KL and removed pre-projection head
response energy on every corpus. The observed effect remained larger after
dividing aggregate KL by aggregate removed-head energy, although that ratio
is not a calibrated causal sensitivity measure.

That set-level result was misleadingly broad. A separate edit for each
target token ID showed that on FineWeb, the token `" as"` accounts for about
73% and `" so"` for about 21% of the set's aggregate per-document KL;
on WikiText, `" as"` accounts for about 82%; on TinyStories, `" so"`
accounts for about 89%. The effects of separate edits need not sum exactly
to a combined edit because output KL is nonlinear. This is evidence for a
**narrow lexical trigger**, not a recovered generic connective or syntax
computation. The original broad label and its control results remain in the
JSON for audit.

## New-document holdout for `" as"` and `" so"`

After observing that concentration, the two token IDs (355 and 523) were
fixed and tested on entirely new document IDs. There are 32 windows per
corpus, from seeds 31 and 37; no document ID overlaps the earlier 16 per
corpus. Controls are ten two-token sets, each sampled from the ten nearest
training-frequency neighbors of the target ID in the same coarse token-form
category, plus ten within-document nearby-position masks with exactly the
same target-key count. The ten frequency draws contain nine distinct sets.

| Corpus | Target keys in 32 windows | Target mean KL per position | Largest matched-token control mean KL | Largest nearby-position control mean KL |
|---|---:|---:|---:|---:|
| FineWeb | 19 | 1.15e-5 | 1.68e-7 | 1.99e-7 |
| WikiText-103 | 19 | 2.47e-6 | 3.03e-8 | 1.38e-7 |
| TinyStories | 24 | 1.95e-5 | 4.44e-8 | 1.20e-7 |

The target's **sum** of per-document KL, used to rank against control sums,
is 3.68e-4, 7.90e-5, and 6.23e-4 in table order. It exceeds every control
sum in each corpus. Its removed head response energy also exceeds every
control, so a large part of the downstream effect is simply that this group
responds much more strongly to these keys. The aggregate KL/removed-energy
ratio also exceeds all nearby-position controls, and all frequency controls
on FineWeb and TinyStories, but the ratio is affected by nonlinear model
responses and key context. The final per-token split is recorded separately
from this combined-token holdout.

An independently recorded per-token edit on these same new documents
assigns 97.8% of the combined KL to `" as"` on FineWeb and 100% on
WikiText; `" so"` does not occur in the selected WikiText windows. On
TinyStories, `" as"` and `" so"` contribute about 62% and 38% of the
combined effect. The near-additivity here is an observation about these
samples, not a general property of KL under interacting edits.

The effect remains highly concentrated: the three highest-effect windows
contribute 86.7%, 89.3%, and 73.3% of target aggregate KL on FineWeb,
WikiText, and TinyStories. Only 16, 11, and 15 of the 32 windows contain a
target key. These observations limit any claim about broad generality.

## Interpretation and next experiment

The strongest supported statement is that a response group selected without
semantic labels from a fixed reduced attention bond has a large, replicated
causal response when `" as"` or `" so"` occupies the key position. The
group itself was obtained from an exact internal gauge; the *subspace* was
more stable across starts than its individual axes. The lexical hypothesis
was selected after inspecting a prior holdout, so the new-document test is
the relevant replication.

This still does not tell us whether the model uses the group to represent a
linguistic relation, a token-specific lookup, or a particular next-token
prediction pattern. A decisive next test should examine edited predictions
and context classes for the high-effect windows, compare independent-start
group projectors and random rank-matched groups under the same token mask,
and test smaller intervention strengths. If a coherent behavior survives
these checks, the corresponding group signature can become a causal target
for the broader featurizer program.
