# Empirical activation metrics for hidden extraction

300fixed extractions from50stored models, each with population projection and32/128label adaptation:900fits independently audited. No source retraining or new target labels. Same fixed rank4quadratic→10products and rank10quartic-union decompositions as the previous panel, now using activation column norms and SVDs estimated on512source inputs,512fresh unlabeled inputs, or4096fresh unlabeled inputs. Fresh pools are nested and fixed per teacher.

Analytic Gaussian moments do not choose the empirical features; they are used only to evaluate final population error. Source512adds neither labels nor input observations. Fresh pools add unlabeled observations. Ranks remain a known architecture budget rather than an automatically discovered manifold dimension. All original failed fits are retained.

## Quadratic consensus: median teacher transfer MSE

| Model origin | Analytic metric | Source512 | Fresh512 | Fresh4096 |
|---|---:|---:|---:|---:|
| shared | 0.000851505 | 0.000851505 | 0.000851505 | 0.000851505 |
| separate_width | 0.147431 | 0.15238 | 0.144275 | 0.150379 |
| released | 0.00121496 | 0.00121342 | 0.00122379 | 0.00123005 |
| teacher | 0.00121496 | 0.00121342 | 0.00122378 | 0.00123005 |

Transfer averages four whole held-out tasks at128noisy adaptation labels/head, then all original initialization outcomes within each teacher. Five teacher aggregates form the median; these are reused models, not fresh-teacher confirmation.

## Ranges matter

| Model origin | Extraction | Metric pool | Median transfer | Full teacher range |
|---|---|---|---:|---:|
| released | quadratic_consensus | source512 | 0.00121342 | 0.000988386–0.0366749 |
| released | quadratic_consensus | fresh512 | 0.00122379 | 0.000987888–0.0543583 |
| released | quadratic_consensus | fresh4096 | 0.00123005 | 0.000989198–0.0355607 |
| released | quartic_pca | source512 | 0.00134304 | 0.000980742–0.0769758 |
| released | quartic_pca | fresh512 | 0.00148045 | 0.0010041–0.06251 |
| released | quartic_pca | fresh4096 | 0.00131655 | 0.00100034–0.0592116 |
| teacher | quadratic_consensus | source512 | 0.00121342 | 0.000988386–0.00290537 |
| teacher | quadratic_consensus | fresh512 | 0.00122378 | 0.000987888–0.00235647 |
| teacher | quadratic_consensus | fresh4096 | 0.00123005 | 0.000989198–0.00254292 |
| teacher | quartic_pca | source512 | 0.00134304 | 0.000980742–0.0159991 |
| teacher | quartic_pca | fresh512 | 0.00148045 | 0.0010041–0.00905289 |
| teacher | quartic_pca | fresh4096 | 0.00131655 | 0.00100034–0.00429705 |

The main compression result survives without an analytic metric: released-model quadratic consensus is about.00121using existing source inputs, versus.03436for their original source outputs. It does not repair all failed shared starts. Random-start independent models remain poor under every metric pool. Increasing unlabeled sample count does not monotonically improve transfer for these particular draws. All empirical and analytic methods agree for shared models because they preserve the same full feature span.

## Why higher-degree metric estimates can be noisy

For a Gaussian monomial z=product_i X_i^a_i with independent standard-normal coordinates, an iid empirical second moment has relative variance

Var(mean(z²))/E[z²]² = (product_i E[X_i^(4a_i)] / product_i E[X_i^(2a_i)]² − 1)/M.

This follows directly from variance of an iid sample mean and factorization of Gaussian moments; E[X^(2k)]=(2k−1)!!. It concerns second-moment estimates, not the standard deviation of estimatedRMSnorms or a full PCA-error theorem.

| Feature | Relative variance × M | Relative SD at512inputs | Relative SD at4096inputs |
|---|---:|---:|---:|
| bilinear_independent | 8 | 0.125 | 0.044 |
| quartic_distinct | 80 | 0.395 | 0.140 |
| squared_bilinear | 1216/9 | 0.514 | 0.182 |
| quadratic_repeated | 32/3 | 0.144 | 0.051 |
| quartic_repeated | 1280/7 | 0.598 | 0.211 |

For example, a product of four independent Gaussian coordinates has ten times the relative second-moment variance of a product of two. Squared bilinear features can be noisier still. Exact rational moment calculations agree with9-point Gaussian quadrature. These examples motivate caution with quartic empirical metrics; they do not prove that degree alone determines stability for the correlated learned features or that quadratic extraction universally wins.

## Verification and decision

Independent audit recomputes empirical norms from direct hidden activations, verifies leading right-singular eigenspaces and coefficient reconstruction, checks empirical product normalization and forward values, then replays every adaptation normal equation and exact Gaussian error. All300extractions/900evaluations pass. No teacher feature directions or held-out labels enter rank/subspace selection.

This removes the analytic-metric assumption for the observed good-basin compression result. It does not remove the fixed-rank prior, Gaussian task setting, small network size or random-start failures. Further work should target those limitations or synthesize the theory/evidence, rather than simply repeating more empirical pool sizes.
