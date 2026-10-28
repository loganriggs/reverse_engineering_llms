## H495 — Source covariance transfer of the structured program

Freeze source4/cross64 from the natural-source structured model. On the same512 H491 receiving contexts, compare the64 original centered natural sources with64 newly drawn covariance-matched Gaussian sources and64 Gaussian isotropic sources in the same head6 output span. Each source is assigned to eight receivers, both signed unit interventions. No representation refit or per-sample normalization. The source pool is generated and saved before intervention outcomes. This is exploratory on reused receivers and single random source pools, not fresh confirmation or an equivalence test.

Covariance Gaussian is G C/sqrt(n_cal−1), where C is the old centered calibration-write matrix and G has independent standard normal entries. Its covariance equals the empirical calibration covariance in expectation. Isotropic sources use an orthonormal basis of the128-dimensional head-output span, multiplied by sqrt(trace(calibration covariance)/128). Thus the two synthetic families have the same expected squared norm, not identical realized norms or distributions. Natural held-out sources are unchanged. No synthetic vector is rescaled to match an individual natural donor.

| Source family | Realized mean energy / calibration expected energy | Source4 retained energy |
|---|---:|---:|
|natural|1.111|97.939%|
|covariance_gaussian|1.168|98.431%|
|head_isotropic|0.996|3.207%|

| Source family | Compressed full-source KL recovery −/+ | Exact source4-projection KL recovery −/+ | Compression recovery against projected target −/+ |
|---|---|---|---|
|natural|96.037% / 96.121%|98.896% / 98.730%|96.883% / 96.900%|
|covariance_gaussian|95.608% / 95.383%|98.712% / 98.702%|96.855% / 96.683%|
|head_isotropic|16.930% / 15.595%|17.795% / 17.566%|89.471% / 89.488%|

Natural sources reach96.04/96.12% compressed full-source recovery; covariance Gaussian95.61/95.38%. Their source4 exact-projection recovery is~98.7–98.9%, and compression fidelity conditional on the projected target is~96.7–96.9%. These point estimates are close, but no covariance/natural equivalence interval was specified or established. Realized Gaussian energy is about5.2% higher than natural-pool energy, and its norm distribution differs. Do not treat the controls as individually norm matched or use the comparison to isolate covariance from every higher-order distributional difference.

Head-isotropic recovery is16.93/15.59%; exact projection already gives only17.79/17.57%. Its retained source energy is3.21%, versus97.94% natural and98.43% covariance Gaussian. Thus the main isotropic failure occurs at source projection, before cross-map compression. Even conditional on projected isotropic sources, compressed recovery falls to89.47/89.49%, indicating some additional transfer limitation in the cross-map approximation. The frozen natural-source program is not universally adequate throughout the head output span.

Effect magnitudes also differ. Mean native full KL is about.00290/.00280 natural, .00225/.00228 covariance Gaussian and.00150/.00147 isotropic. Equal expected raw source norm does not imply equal behavioral effect. All reported recoveries normalize by each family's own native effect. No independent semantic conclusions can be read from these magnitude differences.

Interpretation: substantial conditional fidelity survives removal of actual donor-specific text content when calibration covariance is preserved, whereas spreading energy isotropically in the same head span breaks the compact source representation. This supports the importance of distributional geometry for the data-based canonical form. It does not imply absence of semantic information in real sources, sufficiency of covariance for all model behavior, or universal transfer across heads/models. This revisits earlier covariance controls under the newer structured program, without their retained-output correction, but still with an exact native intact baseline write.

Audits: head output span rank128 and orthonormality checked; natural sources lie in that span within tolerance; expected energy scales derived from fixed calibration only; all source vectors/norm summaries saved; natural full-target errors replay H491 exactly; finite arrays. Both synthetic source pools and per-case full/projected errors archived. Source calibration/receiver documents unchanged; total18880 tracked documents. One random pool per family limits source-sampling inference. No confirmation intervals, semantic labels or CE claims introduced.

Next broaden the structural test across heads. Apply the same prescribed source4/cross64 construction separately to all nine layer17 heads, using the same calibration/donor IDs, training1024 and evaluation receivers. Recompute each head's source axes from its calibration writes; retain exact normalization and quadratic core. Report calibration source energy, exact source4-projection fidelity and compressed fidelity separately, with head6 as a numerical reference. This tests whether the method or rank budget transfers, not whether one frozen head6 basis is shared across heads. Keep the initial all-head comparison exploratory on reused receivers and do not select only favorable heads; reserve fresh confirmation for specific follow-up claims. It can reveal whether head6's near-rank1 source geometry is exceptional, a central limitation on claims about fundamental computational units. Observed-token CE, semantic interpretation and native-baseline removal remain open.
