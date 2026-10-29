# Fashion-MNIST occurrence-tree pilot

All 19 configurations completed. This is a one-seed, five-epoch development pilot with 2,048 training examples and 5,000 validation examples from the official training split. Official test untouched. Fully connected bilinear stack versus fully untied occurrence tree, equal-width and approximately parameter-matched controls. RMSNorm is applied at each bilinear node, without learned gains. No augmentation.

Independent NumPy replay verified all 19 final validation logits, CE and accuracy; maximum relative logit error 1.946e-5. Float64 correctness checks verify independent occurrence parameters, matched initial functions, input gradients and summed occurrence parameter gradients, including normalization.

| Model | Parameters | Validation accuracy |
|---|---:|---:|
| d2_norm0_stack | 52544 | 76.38% |
| d2_norm0_tree | 102720 | 76.56% |
| d2_norm0_tree_budget | 54060 | 74.74% |
| d2_norm1_stack | 52544 | 80.92% |
| d2_norm1_tree | 102720 | 80.38% |
| d2_norm1_tree_budget | 54060 | 77.42% |
| d3_norm0_stack | 54592 | 63.86% |
| d3_norm0_tree | 207168 | 59.50% |
| d3_norm0_tree_budget | 57024 | 56.38% |
| d3_norm1_stack | 54592 | 78.22% |
| d3_norm1_tree | 207168 | 80.38% |
| d3_norm1_tree_budget | 57024 | 71.82% |
| d4_norm0_stack | 56640 | 17.44% |
| d4_norm0_tree | 416064 | 10.20% |
| d4_norm0_tree_budget | 50440 | 20.44% |
| d4_norm1_stack | 56640 | 76.92% |
| d4_norm1_tree | 416064 | 78.20% |
| d4_norm1_tree_budget | 50440 | 45.14% |
| d0_norm0_mlp | 55050 | 77.94% |

The normalized depth-4 stack/tree reach 76.92%/78.20%; the narrow budget tree reaches 45.14%. The equal-width tree uses 416,064 parameters versus 56,640. The two-hidden-layer ReLU baseline reaches 77.94% with 55,050 parameters. This suggests a width bottleneck in the budget tree under this recipe, not a proven intrinsic tree limitation. Five epochs do not establish convergence. RMSNorm improves all paired pilot settings; broader normalization/optimization claims require replication.

## Interruptions and evidence

Producer originally exited139 (segmentation fault) after14fits. First resume completed3more then exited132 (illegal instruction in a native operation reached through Adam). Second resume completed the remaining2. Causes remain undiagnosed. Completed fits were preserved; incomplete fits restarted from their fixed seed. Original and resume sources, interrupted result records and crash trace retained. All final outputs independently replay successfully, but this does not establish hardware/runtime reliability or audit every training step.

## Structural facts worth separating from learning outcomes

For these bias-free bilinear stacks and their untied trees, f(-x)=f(x), including RMSNorm: the first quadratic nodes are even, and subsequent compositions preserve evenness. Thus neither family can distinguish opposite standardized inputs; parameter untying cannot repair this restriction. In a balanced paired dataset with opposite inputs assigned different labels, accuracy is at most one half.

Without RMSNorm, logits are homogeneous of degree p=2^L: f(cx)=c^p f(x). Positive global input scaling leaves argmax predictions invariant, but changes confidence and cross-entropy. Large logit magnitudes and unchanged predicted classes can coexist. This does not imply unchanged training dynamics.

For one normalized node with pre-normalization z and positive input rescaling c, the output becomes z/sqrt(mean(z^2)+epsilon/c^4). At epsilon=0 this removes positive radial scale exactly (away from zero); finite epsilon breaks exact scale invariance. Subsequent normalized layers remain even but the model is no longer polynomial. Sixteen independently perturbed architecture/depth/normalization cases verify evenness, unnormalized homogeneity and the single-node normalization identity in float64. These are elementary architecture restrictions, not claims of novel theory.

## Next experiment, already launched

User requested substantially more data and adequate training time. run_fashion_full.py uses all55,000 available training examples, the same5,000 validation examples,30epochs, RMSNorm depths2–4 with three architecture controls and one ReLU baseline. Per-epoch model/optimizer checkpoints support crash recovery. It records full train/validation curves;30epochs is a first budget, not a convergence assertion. Still one seed and development validation only. Next inspect plateau/overfitting, then replicate seeds and sample-size curves. SVHN with its additional training split is the next larger-data task; no SVHN training has run yet.
