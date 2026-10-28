## H834 — Frozen lexical/context routing transfers to new documents and probe directions

Confirm the H833 development finding without refitting. Before capture/evaluation, lock all H832/H833 token means, global/residual/weighted bases, ranks16/64, the presence-based routing rule and new probe seeds. Use ALL315 remaining H824 documents after excluding H825 native endpoints, H827 recipients and H832's512 development documents. One hash-selected position per document; eight new orthonormal probes per site (seed834+layer), independent of fitting and development probes. The original reserved test remains untouched.

The routing rule uses token-conditioned contextual residuals for tokens present among training sampled positions and global PCA of the same rank otherwise. No count threshold or outcome-selected branch. All models retain the same explicitly supplied radius scalar and recompute the complete local RMS/bilinear derivative from their reconstructed state.

### Frozen confirmation results

Relative response RMSE, pooled over the315 documents and eight probes; zero response would score1:

| Site | Rank | Global PCA | Token residual PCA | Weighted token residual | Routed residual | Routed weighted |
|---|---:|---:|---:|---:|---:|---:|
| block1 |16|.866|.732|.727|.720|.716|
| block1 |64|.740|.623|.620|.605|.603|
| block2 |16|.902|.809|.798|.799|.792|
| block2 |64|.759|.688|.683|.670|.667|

Lexical/context decomposition and the fixed fallback improve prediction on new documents/probes. At rank64, routed weighted minus global PCA error is -.137 at block1, descriptive document-bootstrap95% interval[-.154,-.119], and -.092 at block2[-.104,-.080]. Response weighting adds a smaller gain over routed residual PCA: -.00205[-.00384,-.00036] and -.00323[-.00520,-.00125]. These4096-resample document intervals preserve probe dependence within each document, but are conditional on the fixed probe set and are not corrected for all method comparisons.

The error remains large (.603/.667), so this is improved approximation of local computation, not recovery of a complete operator or manifold. No full-model prediction distribution is evaluated here.

### Transfer pattern repeats

There are257 seen-token and58 unseen-token documents. Rank64 results:

| Site | Token stratum | Global PCA | Residual PCA | Weighted residual |
|---|---|---:|---:|---:|
| block1 | seen |.745|.570|.567|
| block1 | unseen |.722|.791|.784|
| block2 | seen |.764|.652|.648|
| block2 | unseen |.740|.821|.813|

The development finding replicates: token-conditioned bases are substantially better on seen tokens but worse on unseen tokens. Routed models equal the corresponding conditioned model exactly on seen tokens and global PCA exactly on unseen tokens, by construction. The independent evidence is that this frozen rule improves the newly evaluated aggregate and that the prerequisite seen/unseen ordering repeats; equality on routed branches itself is not a discovery.

“Seen” refers to sampled featurizer training positions, not the model's vocabulary or pretraining. These confirmation documents came from the already captured H824 pool and existing natural corpus. They are unused relative to the prior causal/prediction evaluations, not globally untouched research data. After this result all remaining H824 documents have been used in this sequence; do not silently reuse them as fresh confirmation for future architecture choices.

### Resource and interpretation limits

Routed methods retain both a global basis and a conditioned basis, plus9016 token means and the token lookup. Weighted encoders/decoders are separate matrices. Their storage exceeds global PCA and ordinary shared-basis PCA; equal ranks mean equal active contextual-coordinate counts, not matched total parameters or runtime. All methods still use the full MLP weights to map reconstructed state to derivative. The validated claim is improved state-dependent response approximation under these access assumptions, not model compression or architecture superiority at equal resources.

Random probes test local operator behavior, not meaningful tasks or known circuit directions. The state-conditioned prediction framework is useful because it exposes lexical transfer limits, but it has not yielded unsupervised task labels or semantic computational units. Existing response-specific shared groups and native early BSFs remain distinct representations requiring exact-site comparison.

### Verification and next step

315 three-block prefix forwards, zero whole-model/test forwards. Frozen source/model/fit hashes unchanged; excluded documents disjoint; all metrics finite. Probe orthogonality error1.55e-15. Routed unseen reconstructed states equal global states exactly. No new corpus documents; natural-document pool26,112. Saved captures, selected documents, probes and per-document errors support independent audit.

Now compare BSF and shared-dictionary featurizers trained on this EXACT pre-MLP state site and training split, giving competitors the same lexical baseline/radius access. Report storage and active dimensions separately and retain fixed/PCA controls. Use development documents for checkpoint selection, not this confirmation panel as a fresh test. Before making a downstream-faithfulness claim, freeze a small representative evaluation subset and test local-response approximations through full-model KL, including zero-response references and both random and natural directions. New architecture confirmation will require additional document-disjoint occurrences from the existing corpus pool. The goal remains to identify reusable computational structure, not merely improve this local error metric.
