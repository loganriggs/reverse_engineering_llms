## H428 — Direct validation of signed polynomial predictions

The previous goal turn made progress: H427 separated numerator and denominator changes and produced signed-amplitude predictions. H428 tests their numerical implementation against direct double-precision MLP product evaluations at amplitudes -4, -1, +1 and +2, none used to recover the polynomial coefficients. It uses the first64 H425 recipient documents (256 token sites), actual head6 signals, the first raw-donor draw, and the first Gaussian draw. This is a deliberately scoped implementation audit, not a fresh statistical confirmation.

At each changed input x+a*u, the runner directly calculates ((x+a*u)L)*((x+a*u)R)-(xL)*(xR), rather than evaluating the saved polynomial expansion. Full output energy and core/added/union prediction errors are measured using the same fixed baseline Fisher metric. Saved H426 coefficients independently predict each of those quantities at each new amplitude. All source files and frozen factor groups are hash checked.

| Numerical check | Maximum discrepancy | Frozen tolerance | Outcome |
|---|---:|---:|---|
| Per-site energy/error relative difference | 4.65e-12 | 1e-7 | Pass |
| Pooled energy/error relative difference | 6.26e-14 | 1e-8 | Pass |
| Pooled combination-score absolute difference | 1e-14 | 1e-8 | Pass |

Independent saved-array inspection confirms the expected shape, finite values and maximum relative discrepancy. The config retains inherited H426 metadata; only the three explicitly frozen H428 tolerances and the result's three numerical gates apply here. No H426 statistical gate is replaced or changed.

Interpretation: the polynomial is a reliable implementation of this controlled response on the audited subset. This is stronger than fitting its construction endpoints, but it is still an architectural/numerical result. It does not confirm H427's full-panel sign-asymmetry estimate, establish a natural-versus-Gaussian population difference, test exact input RMS or exact KL, or identify semantic computational units. No new documents; cumulative distinct documents remain10944.

Next scientific direction: distinguish source skewness from even-order structure and ratio normalization. Pair natural source coefficients h with 2*mu-h, and pair Gaussian coefficients mu+z with mu-z, without norm/angle matching. Pairing cancels odd central moments while retaining each pair's centered second and fourth products. It preserves empirical second moments about frozen mu, but can change the empirical mean and covariance about that empirical mean; report those explicitly. In a quadratic response expanded around mu, pair differences in energy depend on centered linear and cubic terms, whereas pair averages retain constant, quadratic and quartic terms. A nonzero source mean means this is not the same operation as simply reversing the whole intervention. Analyze unnormalized energy and factor-cross numerators as well as normalized scores. Use independently selected recipients/draws for any chosen scientific confirmation, and keep the fixed donor pool limitation explicit.

The broad feature/computation goal remains open. This result removes an implementation concern and makes the next source-moment control interpretable; it does not complete the ontology.
