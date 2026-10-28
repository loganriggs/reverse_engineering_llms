## H1050 — Exact token-content program rescues factual transfer; sparse residual contribution is small

Previous goal turn was progress: H1049 established a large seen/excluded-country gap despite strong native competence. This turn exported and validated the tokenwise content computation, then supplied it equally to all frozen conditional residual action models. No fitting or head-localization search.

Program: raw1152dimensional pretrained token embedding → native RMS normalization x0 → firstblock residual/embedding scalar mixture → second RMS normalization → full firstblock1152x1152value projection → nine cached value heads. It computes x0 and cached values without attention/QK or a model prefix. This extends H1023's head0lexical program to the full cached vector; it is an extraction of existing model computation, not a newly learned rule.

Numerical validation: exact match (max error0, relative L2error0) against native pre-attention11 cached values and embedding inputs on the factual panel and192existing natural training documents,3072token occurrences/1553unique token IDs. Content replacement without a residual edit reproduces each factual native logit vector exactly. Natural-text checks establish numerical agreement of this tokenwise program, not natural behavioral generalization.

Cost is explicit:1327106float32weight scalars,5308424tensor bytes (~5.31MB), plus the external pretrained embedding table57950208scalars (~231.8MBfloat32). One1152x1152matrix multiply (~1.327million MACs) per token plus normalizations. Output retains2304exact content values per token (x0+cache). Conditional candidates additionally use at most16residual coordinates per token, a frozen PCA encoder, and a147456parameter residual action decoder. Native recipient residual and downstream suffix remain. This is not a48total-coordinate replacement or a compressed whole model. Unused other-port decoder weights from the original checkpoints are not required for conditional inference.

Intervention: add only the sparse predicted residual change to the native recipient residual; generate donor x0/cache directly from donor embeddings with the exported program. Every scalar/block/shared method gets identical content access. Controls include no residual change and dense rank128residual projection. All frozen models/geometry remain unchanged. Comparison with H1049compressing all streams is therefore a different representation contract with an explicit exact-content allowance.

Excluded-country paired relative KL:

| Conditional residual edit | Capital | Language |
|---|---:|---:|
| Zero residual change |0.02243|0.01460|
| Trained scalar16 |0.01798|0.01169|
| Trained block max16 |0.01949|0.01315|
| Trained shared max16 |0.01938|0.01346|
| Dense residual128 |0.01537|0.01121|

For reference, H1049trained scalar with all streams compressed was0.7994capital/0.7778language on the identical examples. Conditional scalar donor-answer top1is87.5%capital/100%language; zero-residual control already achieves87.5%for both. Thus most rescue comes from preserving token content, not a successful sparse contextual explanation. Sparse residual edits add modest benefit. Seen-country conditional scalar errors are similarly small0.01746capital/0.00716language. The large lexical gap largely disappears under the explicit content allowance.

Interpretation: losing token-derived cached information was the dominant bottleneck for these factual transfer failures. A reusable fixed tokenwise map preserves arbitrary country content much better than the narrow fitted activation subspace. This supports separating token identity/content from contextual computation when designing conditional featurizers. It does not establish a general-purpose content/context decomposition for residual-dependent tasks, prove a semantic manifold, or show that these action dictionaries explain the model. The pretrained embedding table and value map are substantial retained machinery, and arbitrary unseen-token behavior still depends on that machinery.

Verification:192native natural-prefix runs,288suffix evaluations, frozen source hashes unchanged, max16active residual coordinates. Exported code/weights have recorded hashes and explicit manifest costs. Zero fitting rows. Independently recomputed KL on32saved logit examples, max1.45e-14. Original factual native competence and all cohort outcomes retained. No new natural documents or fresh-task claims.

Next bounded generality check: apply the same exact content program and zero-residual/sparse/dense residual controls to the other eight whole-family crossfit folds, without retraining. This prevents treating a cache-dominated factual success as a general representation success. Compare initial/final residual decoders and the zero-residual baseline, with the same conditional information allowance for every architecture. Then use the residual-dependent failures to specify a conditional residual-training objective; do not expand the factual task mechanism into another head-level detour. The broader goal remains incomplete.
