# Overlapping groups transfer, but do not beat physical heads at matched error

The [query-conditioned attention benchmark](CONTEXTUAL_COMPOSITION_CANONICALIZATION_RESULTS.md)
has five models that solve four primitive, six XOR, and six AND tasks. This
follow-up fits featurizers on frozen head outputs from **primitive and AND
queries only**. All six XOR queries are absent from featurizer training,
group fitting, and model selection. The native models had been trained on
XOR; this is a whole-family featurizer holdout, not zero-shot model
generalization.

## What was learned

First, a 12-atom flat sparse autoencoder learns an encoder and decoder
from allowed-query activations. It keeps the four largest absolute atom
coordinates per example. We trained both 12- and 22-atom flat controls;
the latter has more parameters but uses the same four-coordinate budget.

We tried a differentiable shared-dictionary model with eight
sample-dependent group gates, binary overlapping atom membership, two
active groups, and a top-four atom cap. It was initialized from the flat
dictionary and used straight-through gradients for both gates and
memberships. Sparse initial memberships led to much worse held-out
reconstruction: 39–44% mean XOR head relative MSE at the two tested
membership costs, versus about 11% for the flat autoencoder. Those are
**failed fits**, not evidence that overlapping groups are inherently bad.
Initializing all memberships active instead makes the gates vacuous. We
therefore did not use the gradient-trained result to compare semantic or
causal quality.

The second, more reliable fit freezes the 12-atom dictionary and learns
eight binary groups by clustering unlabeled top-four activation supports
from new allowed-query examples. A group contains four, six, or eight
atoms. At inference, one or two groups are chosen to minimize head
reconstruction error, followed by the top four coordinates in their
**union**. If an atom belongs to both groups, it is decoded once. This
exhaustive small-set inference uses the current head activation but no
task labels or XOR examples. A direct unit check verifies that overlapping
membership cannot double-count a shared atom.
This gives the group model a stronger and more expensive per-example
selector than the physical-head top-magnitude baseline.

For six atoms per group, the fitted graph has 48 of 96 possible
group-to-atom links. All five fits share eight atoms across multiple
groups; three fits produce seven distinct memberships rather than eight.
This is genuine overlap, with some redundant learned groups.

## Held-out XOR and intervention results

The table averages five model seeds on fresh uniform sequences. Head
error is relative squared reconstruction error. Edit-path error compares
the **finite change in model logit** from toggling one bit of one key
token, before and after the featurizer. The edited token and query are
held fixed across methods. This edit family was not used to fit groups.

| Featurizer | Active atoms | XOR accuracy | XOR head error | XOR logit error | XOR edit-path error |
|---|---:|---:|---:|---:|---:|
| Physical heads, top 4 | 4 | 97.41% | 0.050 | 0.118 | 0.208 |
| Physical heads, top 3 | 3 | 93.15% | 0.108 | 0.244 | 0.409 |
| Flat dictionary, top 4 | 4 | 92.46% | 0.108 | 0.272 | 0.459 |
| Overlap, size-6 groups, one active | 4 | 92.53% | 0.109 | 0.271 | 0.453 |
| Overlap, size-6 groups, two active | 4 | 93.01% | 0.105 | 0.257 | 0.436 |

At a similar reconstruction error near 0.106–0.109, two overlapping
groups modestly improve on the *flat* dictionary's XOR prediction and
edit path, but do not improve on the physical-head baseline. Physical
top-three uses fewer active coordinates and has a lower edit-path error.
Under shifted token frequencies, XOR accuracy is 91.8% for two groups,
91.4% for physical top-three, and 90.9% for the flat dictionary. These
small differences are descriptive across five seeds, not a demonstrated
general advantage.

Group size matters. With four members per group, one active group drops
to 80.7% XOR accuracy and 22.7% head error; two groups recover 92.8%
and 10.6%. Size-eight groups produce mostly duplicate memberships
(four to six distinct groups among eight) and bring no material gain.
This is a capacity/coverage tradeoff rather than evidence for a natural
hierarchy or computational DAG.

The important distinction is between *representational capacity* and
*recoverable computation*. The overlapping-group model really does share
atoms and choose groups per sample, but its groups were fitted to
activation-support similarity. Their ability to reconstruct a held-out
query family does not establish that a shared node implements a reusable
algorithm. A stronger prior would need to improve held-out intervention
fidelity over the physical-head and flat controls at matched error, or
recover independently verified operations. This specific clustering
prior has not cleared that bar. Stop adding group sizes or membership
penalties to this toy without a new causal criterion.

Reproduce the flat and differentiable attempts with
`contextual_composition_shared_groups.py`, then the support-cluster
comparison with `contextual_composition_group_supports.py`. The JSON files
with matching names contain all five seeds, learned memberships, uniform
and shifted evaluations, and per-family finite edit-path measurements.
