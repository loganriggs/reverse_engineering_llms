# Does a sparse all-head gauge yield readable token behavior?

The [exact-gauge comparison](ALL_HEAD_GAUGE_PRIOR_RESULTS.md) found that
direct token-response L1 makes contributions more concentrated without
changing the rank-256 nine-head attention function. Concentration alone is
not legibility. `flagship_shared_key_label_transfer.py` now tests a limited
post-fit question: do coarse query/key token-form labels proposed on one
FineWeb document split transfer to new documents and corpora?

## Design

The underlying function-aware subspace and seven of its exact orthogonal
gauges are fixed before this test. For each gauge, choose the 20 components
with the most nine-head pair-output energy on FineWeb documents 15,000–17,500.
For each chosen component, propose the query-token-form → key-token-form
category with the largest enrichment relative to the original pair-output
energy, among categories carrying at least 2% of that original energy.
Categories distinguish space, digits, punctuation, space-prefixed words,
capitalized words, word fragments, and mixed strings. No category enters
the gauge fit. Evaluate the proposed category unchanged on FineWeb documents
17,500–20,000, WikiText-103, and TinyStories.

The first 4,096-pair run had response-energy effective sample sizes of only
86–123, so its rankings were unreliable. We repeated the test with 16,384
pairs per law under both raw energy and a prespecified robustness condition:
clip each law's *original* pair-response energy at its 99th percentile,
applying the same per-pair multiplier to every component. The clipped
effective sizes increased to 593–739. A 32,768-pair clipped replication
reached 1,228–1,391. Clipping is a sensitivity analysis, not part of the
trained model or the gauge fit.

## Held-out enrichment

Mean log2 enrichment of the 20 selected component labels on the 32,768-pair
clipped panel (higher is better for this **coarse label proxy**):

| Gauge | FineWeb test | WikiText-103 | TinyStories |
|---|---:|---:|---:|
| Supplied function-aware basis | 0.676 | 0.630 | 0.617 |
| Random rotation | 0.497 | 0.562 | 0.537 |
| Weight varimax | 0.851 | 0.999 | 0.624 |
| **Response varimax** | **1.463** | **1.176** | **1.165** |
| Direct response L1, native start | 0.852 | 0.966 | 0.785 |
| Direct response L1, second native start | 0.833 | 0.916 | 0.760 |
| Direct response L1, random start | 0.794 | 0.669 | 0.608 |

All three external corpora favor response varimax on this panel, although
its component responses are *less* concentrated than the supplied basis
under the effective-coordinate measure. Direct L1 improves concentration
the most, yet is not the strongest label-transfer condition. This separates
two notions that are easy to conflate: **few active components per example**
and **one component associated with a recognizable token form**.

The 16,384-pair raw and clipped panels both show a mixed ordering: response
varimax and weight varimax often beat the supplied basis, while the direct-L1
advantage varies by corpus. The 32,768-pair clipped panel strengthens the
response-varimax result, but the category taxonomy is deliberately coarse.
Most proposed labels concern punctuation and space-prefixed words. A positive
log2 enrichment, even relative to a random rotation, does not identify a
semantic rule or a physically localized head circuit. Component sets and
chosen labels vary with gauge; the table compares each condition's own
post-fit proposals, not a matched set of semantic mechanisms. No full-model
component deletion or prediction of a named task was tested in this assay.
A subsequent [full-model input-edit test](ALL_HEAD_COUNTERFACTUAL_CAUSALITY_RESULTS.md)
finds a narrow punctuation-sensitive response for one response-varimax
component at matched edit energy. It remains an intervention on the reduced
explanation, not a uniquely identified physical node.

## Decision

The fixed-function gauge experiment has reached its useful stopping point
as a sparsity-proxy study. The next test needs **named behaviors and selective
causal effects**: derive candidate tasks from component exemplars without
using those labels to fit the gauge, then evaluate new contexts, matched
component deletions, random subspaces, and head-local controls. The existing
[two-bit](JOINT_FUNCTION_TOY_RESULTS.md) and
[three-bit](THREE_COUNT_CANONICAL_RESULTS.md) trained toys provide positive
known-rule controls for this pipeline; they do not validate the real-model
components automatically.

Artifacts: `full_width_results/all_head_shared_key_label_transfer_n4096.json`,
`all_head_shared_key_label_transfer_n16384.json`,
`all_head_shared_key_label_transfer_n16384_clip99.json`, and
`all_head_shared_key_label_transfer_n32768_clip99.json`.
