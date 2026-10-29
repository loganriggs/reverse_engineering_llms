# Standard-data long-training control

Depth3 stack/tree restored the audited8010-update checkpoints, including Adam moments, and continued the identical training recipe to70260updates. Both final train/validation metrics and saved logits passed independent NumPy replay. First30epoch curves match the archived originals exactly; origin checkpoint hashes are recorded. No best-checkpoint selection.

| Model | Standard8010 accuracy | Standard70260 accuracy / CE | Extra70260 accuracy / CE | Standard final train accuracy |
|---|---:|---:|---:|---:|
| stack | 84.30% | 83.22% / 0.9667 | 87.34% / 0.4442 | 96.32% |
| tree | 83.34% | 82.88% / 1.6498 | 89.46% / 0.4063 | 99.26% |

Longer standard-data training fits the training set increasingly well but does not close the extra-data validation gap. Final validation CE worsens substantially, especially for the tree. Thus the extra-data gain cannot be explained solely by giving this fixed optimizer recipe more updates on the standard pool. This does not separate distinct-example count from the extra split distribution or establish the best possible regularized standard-data performance.

Standard long sees17961575examples versus17981640for extra; optimizer updates are exactly70260 in both. The last standard epoch is partial (263complete epochs+39batches). Architectures differ in parameter count, and the comparison still has only one initialization and one development validation split.

Runtime failure retained: initial long-tree launch on core1 died with SIGSEGV before writing a status file. Same source retried on core0 completed and passed audit. Root cause remains unknown; this is not evidence about model optimization. See results/svhn_long_standard/runtime_failure.json.

Next: five fresh initialization seeds and a wider shared stack with approximately matched tree parameter count, using the prespecified protocol in SVHN_EXPERIMENT_PLAN.md. Keep optimizer-family experiments separate from claims about extra data.
