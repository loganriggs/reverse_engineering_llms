# Shared context-role compression of full MLP response

No per-state SVD or output-axis restriction. Context-only rules need not integrate to a state function.

| Context code | Context-only response | Symmetric response | Both-role response | Cross response | Symmetric natural R2 | Cross natural R2 | Context-only cycle RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| pca0 | 0.005 | 0.005 | 0.005 | 0.005 | -0.049 | -0.049 | 0.000 |
| pca16 | 0.120 | 0.102 | 0.011 | 0.140 | 0.166 | 0.251 | 0.006 |
| pca64 | 0.291 | 0.269 | 0.049 | 0.358 | 0.399 | 0.539 | 0.012 |
| pca128 | 0.414 | 0.408 | 0.112 | 0.520 | 0.552 | 0.697 | 0.018 |
| pca256 | 0.573 | 0.584 | 0.236 | 0.709 | 0.715 | 0.841 | 0.023 |
| pca512 | 0.755 | 0.783 | 0.472 | 0.888 | 0.867 | 0.948 | 0.025 |
| pca1152 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| operator16 | 0.150 | 0.136 | 0.020 | 0.186 | 0.197 | 0.298 | 0.008 |
| operator64 | 0.336 | 0.326 | 0.082 | 0.422 | 0.447 | 0.593 | 0.015 |
| operator128 | 0.467 | 0.469 | 0.160 | 0.583 | 0.605 | 0.746 | 0.020 |
| operator256 | 0.622 | 0.638 | 0.303 | 0.757 | 0.760 | 0.873 | 0.023 |
| operator512 | 0.792 | 0.817 | 0.540 | 0.912 | 0.895 | 0.962 | 0.024 |
| random16 | 0.023 | 0.028 | 0.006 | 0.038 | -0.024 | -0.008 | 0.006 |
| random64 | 0.068 | 0.091 | 0.009 | 0.125 | 0.045 | 0.099 | 0.011 |
| random128 | 0.128 | 0.168 | 0.019 | 0.229 | 0.135 | 0.231 | 0.015 |
| random256 | 0.246 | 0.321 | 0.062 | 0.421 | 0.307 | 0.457 | 0.020 |
| random512 | 0.486 | 0.589 | 0.226 | 0.717 | 0.595 | 0.758 | 0.024 |

Locked gates: {'compact_response': False, 'compact_integrable': False, 'operator_advantage': False, 'consistency': True}. Full-rank replay relative RMS 2.05e-05; state-difference identity 5.46e-06; operator metric identity 1.99e-16.

Rank is only the encoded midpoint/context role. Full change vectors, exact normalization and mean-linear term remain; all native Left/Right/Down weights are retained. No per-state SVD. Encoder/decoder store2*1152*r values (PCA/random may share transpose). This does not establish total r-read or parameter compression.


## H236 — Shared context codes remove per-state SVD, but small codes fail full-response coverage

**Frozen comparison.** Using the exact quadratic identity
F(x')−F(x)=2B((x+x')/2,x'−x), fit one shared context encoder on the old2048 normalized
natural inputs. Compare PCA, a weight/operator-aware encoder, and nested random
codes at ranks0/16/64/128/256/512/full. The operator metric G satisfies
||J_raw(v)||_F²=vᵀGv; its analytic identity is independently verified. The encoder
minimizes discovery expected isotropic-Jacobian reconstruction error, not held-out
response error. It is generally oblique. All encoders froze before test19752–19815
(256 sites, all nine heads,5% and100% removals).

Four rules retain the exact mean-linear term. With centered midpoint m, change d,
and context map P: context-only uses2B(Pm,d); symmetric usesB(Pm,d)+B(m,Pd);
both uses2B(Pm,Pd); cross uses2B(Pm,d)+2B(m,Pd)−2B(Pm,Pd). The last three are
integrable quadratic state functions. Their constants are calibrated using old
discovery writes only. The context-only rule has no claimed natural-state decoder.

| Code | Context-only response | Symmetric response | Both response | Cross response | Cross natural R2 |
|---|---:|---:|---:|---:|---:|
| PCA64 | .291 | .269 | .049 | .358 | .539 |
| Operator64 | .336 | .326 | .082 | .422 | .593 |
| Random64 | .068 | .091 | .009 | .125 | .099 |
| PCA128 | .414 | .408 | .112 | .520 | .697 |
| Operator128 | .467 | .469 | .160 | .583 | .746 |
| PCA256 | .573 | .584 | .236 | .709 | .841 |
| Operator256 | .622 | .638 | .303 | .757 | .873 |
| PCA512 | .755 | .783 | .472 | .888 | .948 |
| Operator512 | .792 | .817 | .540 | .912 | .962 |
| Full rank | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

All scores are full1152-dimensional outputs. Both <=64 compactness gates fail.
Operator64 improves context-only fidelity over PCA64 by .045 (conditional paired
row-bootstrap95% [.039,.051]), below the locked .10 threshold. Cross-rule gain is
.064 [.057,.070]. Operator512 cross fidelity is .912 [.904,.920]. Small-edit
results are similar, so the low-rank deficit is not simply large-edit curvature.
PCA64 preserves38.8% of discovery context variance and36.8% of operator-metric
variance; operator64 trades some context variance (35.8%) for39.5% operator-metric
variance. The metric works as intended but does not reveal a tiny shared code.
The rank0 mean-state linear term predicts only.005 of response energy; this is
not an optimized affine predictor and does not rule out a better affine fit.

**Consistency is substantive.** Context-only triangle cycles reach .012/.015
relative RMS at PCA/operator64 and .025/.024 at512, normalized by the head1
response. Nonzero cycles mean these are not globally consistent state functions,
even if useful as response estimators. The symmetric/both/cross rules agree with
independent differences of their state functions and close cycles to numerical
precision. Full-rank replay relative RMS2.05e-5; state-difference identity5.46e-6;
operator metric identity1.99e-16. Eighteen tests pass, including the metric and
oblique-projection integrability formulas. No per-state SVD is used.

**Cost and interpretation.** Rank counts only the encoded midpoint/context role.
Full change vectors and exact endpoint normalization remain, the mean-linear term
is retained, and all native Left/Right/Down weights remain. State decoders also
retain full-input factors in the cross terms. This is not total-r-read or parameter
compression. The midpoint uses both known normalized endpoints but never their
changed MLP output. Broad context dependence persists in the tested linear codes;
hundreds of context reads, not16–64, are required for high full-response fidelity.
This does not prove an impossibility for nonlinear codes or another factorization.

**Step back: learned mechanism or generic bilinear geometry?** H235's rotating
response spaces and H236's broad context requirement could occur in many bilinear
maps. Before promoting either to a learned computational ontology, test local
channel coordination. Keep the real upstream states/head writes fixed and compare
the native L8 MLP with Right-row permutations and Down-column permutations, using
three fixed random permutations per family. These preserve each matrix's spectrum
and marginal weights while breaking distinct learned bindings. Jointly permuting
Left/Right rows and matching Down columns is a function-invariant control.
Refit each control's own global response basis on discovery; compare matched and
wrong-context source-local rank16/64 charts, natural-payload versus isotropic
spectral alignment, full-tangent fidelity, and shared PCA-context-code curves.
If shuffled controls reproduce the pattern, the current geometry is insufficient
evidence for learned units. If Right pairing matters but Down binding does not,
input conjunction structure is a sharper target. If both matter, quantify their
separate contributions. This tests local coordination conditional on a trained
upstream network, not an entirely untrained model. Freeze permutations/thresholds
before new test19816 onward. Do not add semantic names based only on these norms.

**Records.** BILIN18_L8_CONTEXT_ROLE_COMPRESSION.md, PNG/SVG, config, frozen context
encoders/operator metric, full-output responses, natural scores, per-head results,
cycles, paired row errors/intervals and test normalized inputs. Corpus bounds were
verified:20000 rows ×513 token columns. All runs completed; goal remains open.
