# Few-shot transfer to unseen raw rules and compositions

The [internal-head experiment](UNLABELED_HEAD_RULE_RECOVERY_RESULTS.md)
fitted four-dimensional dictionaries to trained QK1×QK2×V responses
without task labels or output readout weights. Here we test whether their
individual atoms help on **new tasks**. The original models were trained on
four *mixed-feature* count-sign outputs. They were never trained to output
the four raw-bit majority signs or any XOR of two raw-bit majorities.

For each of ten trained models and each weight-only or empirical-usage
rank-four cut, we generated 20 independent episodes of **64 labeled
sequences per new task**. A one-atom probe chooses its atom and sign from
those labels; a two-atom probe chooses one of six atom pairs and the sign
of their aggregate-activation product. No probe threshold is tuned. The
held-out test has 20,000 fresh seven-key sequences per model, shared
across episodes but disjoint from their labeled examples. We score four
raw-bit majority tasks and all six XOR-of-majorities compositions.

| Basis of the same internal-head function | One-atom raw-majority accuracy | One-pair XOR accuracy |
|---|---:|---:|
| Weight-only SVD | 74.61% | 65.11% |
| Weight-only random rotation, 32 per cut | 79.92% | 68.60% |
| Weight-only edit L1, all token pairs | **99.69%** | **99.37%** |
| Weight-only edit L1, direct D flips excluded | 93.56% | 87.56% |
| Empirical-usage SVD | 74.65% | 65.07% |
| Empirical-usage random rotation, 32 per cut | 79.10% | 67.93% |
| Empirical-usage edit L1, all token pairs | **99.68%** | **99.37%** |
| Empirical-usage edit L1, direct D flips excluded | 96.09% | 92.36% |

Under the all-pair edit prior, the weight-only one-atom accuracy ranges
from 99.0% to 100% over model seeds, and one-pair XOR accuracy from 97.9%
to 100%. The empirical results have similarly high ranges. The direct-D
edit holdout is less reliable, particularly for new tasks involving D:
one-atom D-majority accuracy falls from 100% to 87.7% for the weight-only
fit and 92.7% for the usage fit. This matches the weaker D component
recovery found in the preceding intervention audit.

The four raw-bit directions have **99.995%** mean coverage by the
weight-only four-dimensional subspace and **99.997%** by the usage-weighted
subspace. All gauges of a given cut span exactly that same subspace, so a
flexible full-space probe would have the same information. The result is
specifically that the edit-sparse basis exposes useful primitives and
their pairwise compositions to **small sparse probes**. That is stronger
than reconstruction or coarse semantic alignment, but it does not mean
the original model computes the new XOR tasks, nor that the featurizer
invented the tasks: the evaluator specified them and supplied 64 labels
per task to choose the atom/pair and polarity.

The next test should train a model on a more diverse set of compositions,
withhold an entire composition family and its direct edit family from
featurizer fitting and model selection, then compare a learned
shared-dictionary/group-gate architecture with exact edit-sparse rotations
at matched function fidelity. It should report full-space task capacity
separately from one- or two-component accessibility, and explicitly test
whether recovered groups transfer under a changed token distribution.

Reproduce with `dense_weak_new_task_transfer.py` after the internal-head
factor script. The saved `dense_weak_new_task_transfer.json` contains
per-seed, per-episode, per-task scores and the coverage audit.
