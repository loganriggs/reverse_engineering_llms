## H797 — Native L2H5 routing is primarily value-repetition sensitive, with smaller key modulation

Previous turn was progress: H796 validated frozen L2H5 causal interaction-removal fidelity across vocabulary and observed more-negative routing for matching identities under partial source-output swaps. H797 removes the hybrid-state concern for the routing observation by evaluating fully native prompts.

Cross same/different keys with same/different value words in both established formats. For each cell, query either key after the two source clauses. Source positions and total token length must match across all eight cells of an item/format; no outcomes enter this alignment rule. Capture frozen L2H5's first-value→second-value signed coefficient, its two Q/K factors and normalized-residual cosine similarity. Repeated-key/conflicting-value cells are deliberately not assigned a unique correct answer. Query choice is later than the measured edge and supplies a causal invariance control.

### Routing result

| Format | Keys | Values differ: mean edge coefficient | Values match: mean edge coefficient | Value-matching change |
|---|---|---:|---:|---:|
| Vertical bars | Different | -.0294 | -.4462 | -.4168 |
| Vertical bars | Same | -.0316 | -.4672 | -.4356 |
| Colon/newline | Different | -.0336 | -.4936 | -.4600 |
| Colon/newline | Same | -.0358 | -.5068 | -.4710 |

The large negative value-matching contrast is present even when keys differ. Descriptive item-bootstrap intervals are [-.4484,-.3862] and[-.4962,-.4217] for different keys in the two formats. The primary repetition-sensitive routing effect therefore exists on native inputs and is not solely an artifact of the earlier MLP-output substitutions.

Key identity still modulates the response. With different values, matching keys changes the coefficient by only about-.0022 in each format, with intervals spanning zero. With matching values, the key-matching changes are-.0210 and-.0132. The key×value interactions are-.0188 and-.0110, with descriptive intervals below zero. Thus “key-independent” would be too strong; value repetition dominates the tested routing contrast, with smaller context modulation.

The coefficient is a signed weight, not a probability, and negative routing does not directly establish inhibition of an output word. Q/K factor signs remain consistent with earlier observations: matching values raises the first factor and makes the second more negative.

### Competence limits

With distinct keys and distinct values, candidate accuracy is only65.6% in vertical bars and55.0% in colon/newline; full-vocabulary intended top1 is53.1%/50.0%. These cells fail the established75% contrast-accuracy standard, so this panel does not establish a successful binding algorithm or identify why binding succeeds.

Matching-value cells have100% candidate accuracy regardless of whether keys match; intended full-vocabulary top1 is96.9–100%. Same-key/different-value cells have conflicting assignments and no accuracy claim. The routing factorial remains measurable in all cells, but behavioral interpretation must respect these differences.

### Identity versus general similarity remains unresolved

Normalized-input cosine increases when values match: for example, with different keys it rises from .597 to .801 in vertical bars and .698 to .836 in colon/newline. A descriptive leave-item-out prediction analysis gives routing-coefficient RMS errors:

| Predictor | RMS error |
|---|---:|
| Format-specific intercept | .2267 |
| Linear input cosine plus format | .1111 |
| Quadratic input cosine plus format | .0885 |
| Key/value identity factorial plus format | .0532 |

These post hoc small regressions diagnose predictive structure, not competing causal explanations. Identity and cosine covary; words overlap across held-out item indices, so this is not lexical generalization. The identity model's lower error does not prove that the head represents a discrete equality feature, and a quadratic scalar cosine is not an exhaustive model of learned geometry. No manifold or learned compact operator is validated here.

### Verification and exclusions

208 native model forwards. All16 vertical-bar item factorials and10 colon/newline factorials survive alignment. Six colon/newline item factorials (48 planned contexts) are excluded by tokenizer-length/position differences. Source-role assignment and single-token answers are verified. Changing the future query produces exactly zero change in both source-edge Q/K factors, their product and input cosine. All coefficients finite; model/manifest/runner hashes unchanged. Original reserved test untouched; natural-document pool remains26,112.

Routing summaries use one query copy per cell, avoiding treating the identical future-query routing observations as independent data. Accuracy uses both queries where unambiguous. Bootstrap intervals cluster by item within each format, are descriptive and not multiplicity-adjusted; formats share items.

### Research decision

The strongest current description is a native value-repetition/similarity-sensitive negative read from the first source to the second, with modest key-context modulation. We have a verified algebraic edge and a transferable counterfactual causal effect, but its role in native prediction remains untested.

Next scale or remove this exact edge in fully native prompts, including the competent matching-value cells, and measure intended-token log-probability, full-vocabulary KL and answer contrasts separately. Retain ambiguous and low-competence conditions only for appropriately limited geometric/response comparisons. This will test whether the negative read changes native prediction in the way a repetition-discounting hypothesis suggests, rather than assigning function from its sign. Any subsequent data-based canonical form or shared-dictionary featurizer should preserve that measured native response as well as the earlier finite source-intervention effects.
