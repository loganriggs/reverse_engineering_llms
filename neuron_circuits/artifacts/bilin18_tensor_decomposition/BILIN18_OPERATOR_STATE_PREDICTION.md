## H832 — Training-only state compression predicts some local computation, but remains inaccurate

Move from repeated explanatory interventions to prediction on unused recipient documents. H831 established downstream consequences of crossed state-dependent responses but showed that local geometry alone is an incomplete guide to final fidelity.

### Prediction benchmark

Capture raw pre-MLP activations at zero-indexed blocks1 and2 for the original2048 BSF training documents (16 positions each:32,768 states/site). Select512 documents from the H824 panel after excluding EVERY H825 native endpoint document and H827 recipient document. Use one hash-selected position per validation document, without output-based selection. These documents were captured unedited in H824 but had no previous causal response evaluation. They are new to this prediction benchmark, not new to the entire research corpus or model pretraining. The original reserved test remains unused.

Fit only from training states. Evaluate eight fixed orthonormal Gaussian raw perturbation directions per site, chosen by seed832+layer independently of response targets. The target is the exact local RMS-plus-bilinear MLP derivative at the held-out state. No response labels or task labels enter the fit. Evaluate all512 documents and eight probes, including weak responses.

Compare training global mean raw state, token mean raw state, global/token mean normalized shape with recipient radius, and global PCA shape ranks4/16/64 with radius. For shape models, radius r=sqrt(mean(b^2)+eps) is ONE explicitly supplied state scalar. Reconstructed shapes are renormalized, converted back to raw states using that radius, and the entire RMS/bilinear derivative is recomputed from the reconstructed state. The true baseline's normalization derivative is NOT supplied. PCA reads the original normalized activation only to encode its retained coordinates. All methods retain the full fixed MLP weights; this is state compression, not compression of all model parameters or proof of runtime savings.

### Held-out results

Relative RMSE is sqrt(sum prediction squared error / sum true response squared norm). A zero-response predictor has relative RMSE1. Errors are pooled over documents and probes; mean cosine is secondary.

| State representation | Block1 relative RMSE | Block2 relative RMSE |
|---|---:|---:|
| Global raw mean |.974|1.011|
| Token raw mean |.855|.932|
| Global mean shape + radius |.970|.991|
| Token mean shape + radius |.839|.896|
| PCA4 shape + radius |.924|.957|
| PCA16 shape + radius |.857|.894|
| PCA64 shape + radius |.729|.745|

Token-conditioned state is informative. Token shape+radius is better than PCA16 at block1 and approximately tied at block2. PCA64 improves on both but retains large response errors: about73–75% relative RMSE. Its mean response cosines are .727/.688, versus .606/.493 for token shape+radius. This is useful predictive structure, not an adequate reconstruction of the local operator or semantic explanation.

Document bootstrap2048 resamples, preserving all eight probes per document, gives PCA64-minus-token-shape RMSE intervals[-.125,-.093] at block1 and[-.162,-.138] at block2. PCA16-minus-token-shape intervals are[.0027,.0345] and[-.0137,.0103]. These are descriptive intervals conditional on this fixed probe set and training fit; they do not account for multiple method comparisons, other probe seeds or task generalization.

### Resource and coverage distinctions

Each PCA model carries r encoded coordinates plus one radius scalar, a1152-by-r dictionary and a global shape mean. Token models use an external token identity lookup and9,016 stored1152-dimensional training token means; they are not parameter-matched to PCA. Token-plus-radius additionally observes the radius scalar. Raw fixed baselines receive neither radius nor contextual coordinates. Do not equate token lookup storage with one learned continuous coordinate or claim an architectural winner from these unmatched capacities.

109/512 validation tokens are unseen among sampled training positions and use the predefined global-mean fallback. They remain in all metrics. Novelty here concerns sampled training token identities, not the language model's vocabulary or pretraining. Training and validation states are drawn from the same existing natural-text source pool, with exact document separation; broader distribution shift remains untested.

No BSF or shared-group model has yet been trained at these exact raw pre-MLP sites. Existing early BSFs were fit to different activation sites and cannot be inserted as a valid matched comparator without adaptation. This benchmark establishes the fixed/token/PCA baselines for that work; it does not complete the requested architecture comparison.

### Verification

2560 three-block prefix forwards (2048 training +512 validation), zero full-model forwards. Source/model/checkpoint hashes tracked and unchanged; selected documents exclude prior causal endpoints; no reserved-test evaluations or new corpus documents, natural-document pool26,112. PCA basis orthogonality error1.55e-15, probe orthogonality error<=6.66e-16, all metrics finite. Reconstructing a state from its FULL normalized shape and recorded radius recovers it within1.31e-16 relative error, checking that radius bookkeeping itself does not introduce the approximation. Analytic derivative formula is the H830-verified RMS-plus-bilinear expression. Fits, captured states, per-document/probe errors and validation selections are saved.

### Next informative comparison

The next model should separate lexical baseline from contextual residual instead of forcing a small global PCA basis to represent both. Fit token-conditioned residual PCA on training data, with explicit fallback for unseen tokens, and compare reconstruction-trained versus response-weighted state bases at matched ranks. Treat the present512 documents as development validation after these results; preserve remaining unused documents for a frozen confirmation. Then train BSF/shared-group representations at the exact same state site and compare at declared storage/active-coordinate budgets. Carry a frozen subset through whole-model KL after local prediction checks. The purpose is to discover which compact state variables predict computation and transfer across contexts, not to relabel low reconstruction error as a meaningful manifold.
