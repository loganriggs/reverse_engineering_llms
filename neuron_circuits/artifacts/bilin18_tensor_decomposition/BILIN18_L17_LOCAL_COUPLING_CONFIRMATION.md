

## H404 — Fresh local-coupling confirmation with separate-source normalization

Confirm the frozen H403-selected pairs(0,6),(5,6) on512 new complete documents and six new geometry-preserving rotation draws. Independent rehashing excludes all6,848 earlier completed-panel documents; there are now7,360 unique complete documents across listed panels. Same corpus/model, not training holdout or cross-domain confirmation.

Let K=||mixed Taylor response||², Jh=||first-order response h||², Jk analogously, and S=||intact final raw state||². Primary comparison retains H403 exactly: ratio of pooled K/S to pooled(Jh+Jk)/S, compared actual versus rotations. Before fresh outcomes, add T=S*K/(Jh*Jk), averaged sitewise, then compare actual versus rotated means. T corresponds to normalizing each input direction by its own first-order output norm and expressing the resulting curvature in intact-state units. This normalization changes input magnitudes; it is a distinct coordinate-dependent diagnostic, not another input-Gram-matched intervention. There is no clipping or exclusion of small derivatives.

Frozen gates: {'geometry': True, 'finite_difference': True, 'primary_twofold': True, 'response_normalized_twofold': True}. Both primary and additional diagnostic require the two simultaneous upper actual/rotated ratio bounds below.5. Four thousand paired document/rotation-draw bootstrap resamples give separate joint2 bands per metric, not a joint8 band across all metrics. Finite native double mixed-derivative audits use step.001 at four fixed sites per pair/condition. Geometry is checked after float32 casting. No native output KL or finite-amplitude behavior is claimed by this derivative experiment.

|Pair|Absolute curvature ratio|First-order ratio|Primary coupling ratio (joint95%)|Separate-source T ratio (joint95%)|
|---|---:|---:|---:|---:|
|[0, 6]|1.40490|10.23046|0.13732 [0.12208,0.15447]|0.06191 [0.05984,0.06406]|
|[5, 6]|1.22775|10.87734|0.11287 [0.10034,0.12697]|0.05341 [0.05162,0.05526]|

Audits: {'max_relative_native_double_mixed_FD_error': 2.355733746625732e-05, 'geometry': [{'seed': 20270804, 'norm_scaled_gram_max_error': 3.74287186359969e-08, 'norm_scaled_residual_dot_max_error': 2.5260855157165207e-08}, {'seed': 20270805, 'norm_scaled_gram_max_error': 3.5384525808776545e-08, 'norm_scaled_residual_dot_max_error': 2.5463033039966236e-08}, {'seed': 20270806, 'norm_scaled_gram_max_error': 3.8003603643594794e-08, 'norm_scaled_residual_dot_max_error': 2.5042989383742398e-08}, {'seed': 20270807, 'norm_scaled_gram_max_error': 2.9770472867042634e-08, 'norm_scaled_residual_dot_max_error': 2.2733287690453108e-08}, {'seed': 20270808, 'norm_scaled_gram_max_error': 3.26984526295119e-08, 'norm_scaled_residual_dot_max_error': 2.2766768916222284e-08}, {'seed': 20270809, 'norm_scaled_gram_max_error': 3.711194014804722e-08, 'norm_scaled_residual_dot_max_error': 2.3972730570188426e-08}], 'minimum_individual_first_order_energy': 2.7935828581584192}. Independent saved-array/document checks: {'means_max_absolute_error': 7.105427357601002e-15, 'document_audit': {'excluded_complete_document_hashes_independently_rehashed': 6848, 'fresh_unique_complete_documents': 512, 'fresh_unique_windows': 512, 'overlap': 0, 'completed_unique_total': 7360}}. Site metrics, each individual derivative energy, cancellation terms and bootstrap contrasts are retained. The separate-source normalization is computed per site before averaging, unlike the primary ratio of pooled energies.

Outcome: all four frozen gates pass. Primary normalized-coupling ratios confirm at0.1373,[0.1221,0.1545] for heads0/6 and0.1129,[0.1003,0.1270] for5/6. Separately normalizing each source gives T ratios0.06191,[0.05984,0.06406] and0.05341,[0.05162,0.05526]. The effect therefore survives removal of the summed-denominator dominance explanation under this explicitly defined additional metric. It is not explained solely by head6 contributing most of a summed first-order energy.

Actual absolute mixed-curvature energy is nevertheless larger than rotations: ratios1.4049,[1.2619,1.5641] and1.2277,[1.1028,1.3668]. First-order energy ratios are10.2305,[9.7678,10.7150] and10.8773,[10.3855,11.3925]. Natural orientations are highly responsive in the local linear map, with lower mixed curvature relative to that response—not absolutely noninteracting directions. Head6 still supplies98.766% and99.631% of the summed actual first-order energies; the separate-source diagnostic specifically addresses reliance on that sum.

Arithmetic checks: maximum finite-difference discrepancy2.36e-5; maximum normalized geometry discrepancy3.80e-8; minimum individual first-order squared norm2.794, with no denominator clipping. No document contributes more than0.853% of total T in any of the four pair/family summaries. These diagnostics make a few near-zero derivatives or a single dominant document an unlikely explanation for the pooled confirmation, without proving uniform behavior at every site.

This is a reproducible local property of two learned source orientations in this model/layer. It does not establish globally independent features, a compact feature dictionary, semantic specialization or cross-context portability. The response-normalized diagnostic is coordinate-dependent and does not preserve source input norms after rescaling. Direct finite-amplitude tests already showed curvature truncation can fail at large edits. These limits remain part of the result.

### Updated hypothesis tournament: twenty explanations and discriminating tests

Priorities are qualitative, not numerical posterior probabilities. Several explanations can coexist.

|#|Explanation|Current evidence|Discriminating test|
|---|---|---|---|
|1|Learned source orientations align with high-sensitivity directions of the local MLP.|Leading; broad discovery and fresh selected-pair confirmation.|Predict advantage from Jacobian geometry on unseen contexts.|
|2|The low coupling ratio is only head6 dominating a sum.|Weakened by fresh separate-source normalization.|Retain per-source normalization across more pairs.|
|3|A few native bilinear factors carry the sensitivity advantage.|Unresolved, mechanistically actionable.|Freeze factor ranking on old documents and test retained/removed sets on held-out documents.|
|4|The advantage is distributed across many factors.|Strong alternative to3.|Measure held-out reconstruction and ablation curves against matched random sets.|
|5|Large factor contributions cancel to make mixed curvature small.|Plausible, not universal in energy decomposition.|Track signed factor contributions and finite removal effects, preserving cross terms.|
|6|Input RMS correction alone creates the pattern.|Insufficient as a universal account; remains a component.|Separate fixed-scale bilinear and RMS derivative terms under factor perturbations.|
|7|Only input norms and mutual angles matter.|Contradicted for absolute sensitivity/error by full-Gram rotations.|Extend geometry-preserving controls to other layers.|
|8|High-dimensional generic geometry explains ordinary additivity.|Supported for actual and random orientations.|Test matched norms across widths or models.|
|9|The effect is caused only by a few small derivative denominators.|Weakened by fresh minimum-energy and document-influence audits.|Examine fixed predeclared derivative strata without deleting tails.|
|10|The effect is a document-panel artifact.|Weakened for the two frozen pairs by fresh confirmation.|Repeat on different text domains or corpora.|
|11|Head6 is special; the pattern is not pair-generic.|Leading scope alternative; all-pair universal gate failed.|Analyze held-out head6 versus non-head6 pair families.|
|12|The property is caused by local output-direction alignment rather than factor sparsity.|Unresolved.|Compare factor support and output-vector overlap at matched derivative energy.|
|13|Low-dimensional context regimes select different response operators.|Unresolved.|Cluster only discovery contexts and predict operators on held-out contexts.|
|14|A shared operator transfers across documents.|Not established by context-specific derivatives.|Predict derivative vectors from frozen cross-context operators.|
|15|Single-source laws are compact because of architecture, not features.|Supported by exact rational derivation.|Count reused parameters and test transfer instead of counting local coefficients.|
|16|Output-distribution geometry determines which errors matter behaviorally.|Supported by earlier stage reversals.|Score selected factor interventions through native KL on fresh data.|
|17|Local relative decoupling survives large finite interventions.|Not generally supported; mixed predictor fails4× gates.|Freeze amplitude ladders and retain failures.|
|18|Pairwise weak coupling extends to many simultaneous sources.|Untested.|Compare complete multi-source quadratic law and summed source laws.|
|19|The locally responsive directions correspond to semantic features.|Untested; no semantic labels follow from these metrics.|Controlled linguistic contrasts after a reusable causal mechanism is identified.|
|20|The useful unit is a context-conditioned derivative/curvature operator.|Supported as a local predictive description, not as a recovered reusable unit.|Test compact parameterization, context transfer and causal factor support.|

Narrowing: prioritize3/4/5/12—whether the confirmed sensitivity/curvature relation has sparse native-factor support, distributed support, or cancellation/alignment structure. Before creating another derivative-metric variant, inspect the existing native-factor work to avoid duplicating earlier compression failures. Then freeze a factor-selection rule on an older panel, with signed contribution accounting, and test retained-factor reconstruction plus factor-removal interventions on held-out documents. A successful attribution ranking alone is insufficient; intervention and held-out prediction must agree. The broad goal remains open. Future fresh panels must exclude7,360 complete-document hashes from completed panels.
