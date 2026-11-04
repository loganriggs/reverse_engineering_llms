# Corpus-grounded nonce-copy pilot

## Question and scope

Can a small bilinear language model learn a genuinely document-local literal-copy rule in varied human prose, and will it transfer to new source documents and unused code combinations? This pilot is a behavioral prerequisite for later weight-based circuit analysis, not yet a canonicalization or sparsity result.

The generator selects FineWeb documents containing a repeated, three-letter uppercase organization-style acronym. It consistently substitutes a sampled, opaque three-letter fictional alias throughout each selected document. Two versions of the same source differ only in that alias, so the later occurrence has a paired causal target. The opaque codes each tokenize as three BPE pieces; the model must predict all three in order. The natural surrounding text is retained as language-model training context, but only the marked copy sites have an exact rule-based answer. Source facts are not asserted to remain historically true after alias substitution.

This task is deliberately narrow. The entity-source whitelist and strict exclusions remove obvious unit/currency/economic acronyms that would make the substitution semantically misleading. The whitelist is a surface prior, not an entity recognizer. Two independent 1,000-document screens found 12 and 11 eligible/aligned documents. We reviewed examples during construction; the selected population is only about 1% of FineWeb, so this is not a full natural-language DGP.

## Data and checks

The source split uses nonoverlapping FineWeb stream offsets: validation 95,000–104,999, training 115,000–214,999, and test 335,000–344,999. The base cache selected 177, 1,846, and 166 source documents respectively. Selected documents were screened for exact hash matches and near duplicates at SimHash distance ≤3 across splits. Training codes come from a 256-code pool; the test's second panel uses a disjoint 256-code pool on the *same held-out source documents*. This isolates code-combination transfer from document transfer. The code-level holdout does not hold out individual letters.

The validation fixture checks paired token alignment, prefix-only causality at the target, three-piece tokenization, and exact reversibility. The evaluation teacher-forces alternative candidate sequences after the same visible prefix and also measures unrestricted greedy generation. Full-sequence preference between the two sampled aliases is an easier diagnostic than exact generation; it is not evidence that the model can emit an arbitrary code.

The local `.pt` caches reconstruct FineWeb text and remain outside git. Aggregate manifests contain counts and source hashes, but no source prose.

## Results before the eight-draw comparison

| Model | Training | Test seen-code paired preference | Test seen-code exact greedy, both worlds | Test unseen-code paired preference | Test unseen-code exact greedy, both worlds |
|---|---|---:|---:|---:|---:|
| Depth 2, width 64 | nonce/list/bracket mixture, 3,000 steps | 3.0% | 0% | 3.0% | 0% |
| Depth 2, width 64 | nonce-focused, 6,000 steps | 85.5% | 0.6% | 68.1% | 0% |
| Depth 4, width 128, RMSNorm | nonce-focused from scratch, 6,000 steps | 98.8% | 24.7% | 93.4% | 6.0% |

The larger model's training-first-256 exact greedy score was 91.8%, compared with 30.5% on validation and 24.7% on test familiar codes. That gap, and the 6.0% score on unused code combinations, disqualify a claim of robust open-vocabulary copying. Its held-out seen-code first/second/third piece top-1 accuracies were 50.6%, 83.7%, and 80.1%. The first piece is the main bottleneck. The depth-2 focused model also forgot list and bracket behavior inherited from its initial checkpoint, so the focused experiment is diagnostic rather than a competent three-rule model.

The controlled comparison reuses exactly the same selected source documents, validation set, test set, and unused-code panel, but draws eight alias pairs per training source. It has 14,768 training pairs rather than 1,846. The build resumed successfully from an atomic partial cache after an illegal-instruction process crash. Tensor equality confirms the held-out panels match the base cache exactly. Both conditions use the same depth-4 width-128 RMSNorm architecture and 6,000-step budget; each seed is compared with itself.

| Seed | Training aliases/source | Validation exact greedy, both worlds | Test seen-code exact greedy, both worlds | Test unseen-code exact greedy, both worlds | Test seen-code sequence NLL | Test unseen-code sequence NLL |
|---|---:|---:|---:|---:|---:|---:|
| 72104 | 1 | 30.5% | 24.7% | 6.0% | 4.68 | 7.63 |
| 72104 | 8 | 50.8% | 39.2% | 15.7% | 3.11 | 5.30 |
| 72105 | 1 | 22.6% | 25.3% | 4.8% | 4.60 | 8.45 |
| 72105 | 8 | 46.9% | 45.2% | 18.1% | 3.02 | 5.22 |

Eight-draw training improves exact generation in both seeds, but still leaves most unused-code combinations incorrectly generated. Seen-code paired *two-choice* sequence preference was 98.8%/99.4% in the two augmented seeds, well above exact generation; evaluating only that easier measure would overstate competence. The first-256 training score is not directly comparable between caches because the augmented cache's first 256 pairs come from fewer distinct source documents. The extra pairs change sampling frequency over alias realizations and reduce repeat exposure to any particular pair; they do not add independent source documents. Two seeds support a directional effect, not a precise population estimate.

If exact held-out generation remains weak, stop this narrow copy-task detour and broaden the multirule DGP rather than interpreting a nonexistent robust circuit. If it improves, repeat across seeds and put copying back into a capable joint model, then apply weight-only canonicalization and sparse decomposition under behavioral and causal fidelity checks. Track behavioral competence, reconstruction, task prediction, causal fidelity, and new-rule discovery separately.
