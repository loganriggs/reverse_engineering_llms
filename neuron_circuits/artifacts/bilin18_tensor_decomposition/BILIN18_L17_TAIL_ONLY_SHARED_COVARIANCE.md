

## H388 — Tail-only shared covariance from disjoint discovery documents

Construct K_tail as the equal-site mean conditional unembedding covariance after removing the top32 tokens and renormalizing the remaining probabilities at each of512 sites from128 older H365 discovery documents. This averages within-site conditional covariance, not covariance after pooling different means. Freeze before evaluating responses. Discovery/evaluation complete-document hash sets are disjoint. The current evaluation panel is the reused H384512-document panel, not fresh confirmation.

Replace each recipient tail covariance by trace(C_tail)/trace(K_tail) × K_tail. Retain exact top32 covariance and the actual head/tail mean outer product. Compare exact, isotropic, full-distribution shared K, and tail-only shared K_tail. Derive both Gram and RHS consistently from each metric; same native full-vocabulary KL, eight-step safeguard, both bases and ranks16/64, all39 responses. The actual recipient tail trace and mean and the full target response are still required. No response-based tuning or template selection enters construction.

|Basis|Head|Family|Exact64|Isotropic64|Full shared64|Tail shared64|
|---|---|---|---:|---:|---:|---:|
|probability|0|actual|0.8599|0.8157|0.8256|0.8302|
|probability|0|moment|0.7478|0.6766|0.6900|0.6989|
|probability|0|donor|0.7479|0.6773|0.6903|0.6993|
|probability|1|actual|0.8012|0.7426|0.7487|0.7562|
|probability|1|moment|0.7901|0.7294|0.7344|0.7430|
|probability|1|donor|0.7919|0.7312|0.7363|0.7448|
|probability|6|actual|0.8158|0.7565|0.7621|0.7679|
|probability|6|moment|0.8352|0.7666|0.7753|0.7806|
|probability|6|donor|0.8396|0.7710|0.7796|0.7846|
|random|0|actual|0.8132|0.7687|0.7767|0.7857|
|random|0|moment|0.7287|0.6726|0.6759|0.6882|
|random|0|donor|0.7272|0.6703|0.6734|0.6861|
|random|1|actual|0.7737|0.7131|0.7278|0.7375|
|random|1|moment|0.7610|0.6996|0.7116|0.7229|
|random|1|donor|0.7638|0.7032|0.7155|0.7264|
|random|6|actual|0.7738|0.6621|0.7235|0.7336|
|random|6|moment|0.8056|0.7462|0.7631|0.7718|
|random|6|donor|0.8128|0.7541|0.7712|0.7796|

Frozen gates: {'tail_shared_within5': False, 'tail_beats_full_shared': False, 'tail_beats_isotropic': False}. Approximation requires all36 exact-minus-tail-shared joint upper bounds <5pp. The two superiority criteria require all36 joint lower gains over full-shared or isotropic >0.

|Contrast|Point range (all36)|Global range of simultaneous bounds|
|---|---:|---:|
|exact_minus_tail_shared|0.89 to 5.50 pp|-0.17 to 6.56 pp|
|tail_minus_full_shared|0.46 to 2.30 pp|-0.60 to 3.36 pp|
|tail_minus_isotropic|0.40 to 7.14 pp|-0.66 to 8.20 pp|

Discovery audit: {'discovery_documents': 128, 'evaluation_documents': 512, 'disjoint_document_hashes': True, 'full_covariance_replay_relative': 5.290993439845721e-16, 'tail_covariance_min_eigenvalue': 9.299306243704338e-05, 'tail_covariance_trace': 14.66154647515964, 'mean_discovery_tail_mass': 0.19576699808067488}. Positive-definite tail template and its trace independently verified. Input hashes checked. Exact and isotropic prior-run replays pass; an independent saved-array check verifies all three H387 grids within 0 absolute KL. All16/64 Gram directions retained; maximum normal-equation residual 4.4e-12. Baseline agreement, finite scores, and exact saved-grid minima pass. Norms and step choices remain saved. Independent audit fields are added by this reporting script.

Uncertainty: 4,000 paired recipient-document/control-draw resamples; joint108 bands across three contrasts ×36 settings. Discovery template, bases and donor assignments are fixed: these intervals do not resample template-learning uncertainty. Figure contains rank64 point estimates; JSON includes both ranks and confidence bands. A failed population mean template does not rule out conditional templates or every shared approximation.

All three frozen gates FAIL. The tail-only template improves on the full-distribution template in all36 point estimates, by0.46–2.30 percentage points, but only13/36 simultaneous lower gains exceed zero. Gains over isotropic are0.40–7.14pp, with30/36 positive simultaneous lower bounds. Thus matching the template to the tail helps descriptively, but neither universal superiority criterion passes.

Relative to exact recipient covariance, tail-only shared fitting loses0.89–5.50pp; the largest simultaneous upper loss is6.56pp, above the five-point margin. Twenty-five of36 upper bounds satisfy the margin, and33/36 lower loss bounds exceed zero. Rank64 recovery is68.61–83.02% with the tail-only template versus72.72–85.99% exact,66.21–81.57% isotropic, and67.34–82.56% full-distribution shared. The full-versus-tail mismatch therefore explains some improvement but does not make this unconditional average template universally sufficient at the declared accuracy.

These results do not prove that context-specific semantic features are necessary. A mean template can miss conditional structure; discovery-template sampling uncertainty is not included in the evaluation bootstrap; and all methods still use the target response. H384's fresh top512 confirmation remains stronger evidence for an approximate context-selected constraint description than any tested unconditional tail template. H385–388 distinguish consistency, scalar stabilization, shared anisotropy, and matched tail averaging, but none supplies an executable response generator.

Step back: further small covariance refinements now risk explaining the oracle rather than the model. The highest-priority unresolved hypothesis is that local, input-derived response structure supplies the coefficients, with output geometry chiefly determining which errors matter. The main alternatives are: first-order contextual response is sufficient; finite bilinear terms are essential; normalization curvature is essential; or target-aware step selection is doing substantial work. These can be separated more directly than by another template variant.

Next experiment: predict the MLP-only response using the analytic first-order derivative at the intact residual for each actual/control source payload, including input RMS normalization, bilinear MLP derivative, and final RMS derivative. Keep the direct residual source subtraction excluded, matching the existing MLP-only target. Fit low-rank coefficients to this predicted response, not to the exact finite response. Compare fixed unit step and a step selected solely against the predicted full response with the existing exact-response oracle; use the true response only for evaluation and explicitly labelled oracle controls. Include the unrestricted linear-response prediction to separate response-model error from rank-restriction error. This removes access to the exact finite target from the practical prediction path, although it still uses native weights and does not constitute parameter compression. Freeze the contrasts and tolerances before outcomes; retain failed or unstable sites. Future fresh panels exclude5,824 complete-document hashes. Broad goal remains active.
