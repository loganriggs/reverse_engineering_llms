## H908 — A shared context-dependent map transfers better than a small lexical subspace

The previous turn made progress: H907 found strong language MLP effects at L13/14 but poor recovery from four-dimensional discovery subspaces. H908 keeps the full held-out country-specific input change and approximates its processing context instead. Use discovery means from the first four pairs, separately by task/copy and template, plus pooled task and all-family means. Last four pairs are evaluated with live downstream computation at L12/13/14. Layer choices and evaluation contexts are reused from screening; this is not fresh confirmation.

### Exact algebra clarifies the controls

For normalized recipient u and source change d, a bilinear MLP obeys F(u+d)−F(u)=J(u)d+Q(d)=J(u+d/2)d, where J is its input Jacobian and Q is the quadratic change. Bias cancels. A fixed discovery mean m therefore defines two different approximations: J(m)d approximates the midpoint partner, while J(m)d+Q(d) approximates the recipient partner. Adding Q is not automatically more faithful when m is closer to the midpoint than to u.

The native recipient MLP output remains the offset in every condition; only its finite change is approximated. No input projection discards the country-specific change. All mean fits are frozen before evaluation. Family/template labels choose the diagnostic mean, so this is not yet a task-blind operator-selection mechanism.

### Language task effects at L13/14

Error ratio is effect RMSE versus the exact full donor substitution divided by the no-change RMSE. KL compares exact-edited and approximate-edited output distributions. The opposite-family mean uses copying discovery examples in the same template; pooled task means average both task templates.

|Layer|Partner approximation|Effect-error ratio|Exact-to-approximate KL|
|---|---|---:|---:|
|13|native_linear|0.434|0.04227|
|13|own_linear|0.334|0.00558|
|13|own_quad|0.290|0.04644|
|13|task_pooled_linear|0.331|0.00607|
|13|task_pooled_quad|0.304|0.04763|
|13|opposite_linear|0.790|0.03614|
|14|native_linear|0.595|0.11354|
|14|own_linear|0.389|0.00554|
|14|own_quad|0.292|0.10462|
|14|task_pooled_linear|0.392|0.00634|
|14|task_pooled_quad|0.300|0.10535|
|14|opposite_linear|0.698|0.01956|

Own-context midpoint approximations reach effect-error ratios0.334/0.389, compared with roughly0.80–0.86 for the previous rank4 input-subspace fits. Pooled task means perform similarly (0.331/0.392), suggesting that a context-dependent linear map can process unseen country changes without requiring those changes to share the small discovery span. This is a deliberate change of hypothesis and information budget: the mean-partner method retains the full source change and original weights, so it is not a matched-capacity compression victory.

Copy-context means are worse on language (ratios0.790/0.698, KL0.03614/0.01956), supporting context specificity. On copying, own-context midpoint means yield KL0.00398/0.00324, whereas pooled task means give0.02889/0.02931. Thus a single task mean does not reproduce all contexts, and the controls support reusable context-dependent processing rather than one universal map.

### Margin fidelity can reward distributionally wrong approximations

Adding the quadratic term to the own mean improves language effect-error ratios to0.290/0.292, yet worsens KL from0.00558→0.04644 at L13 and0.00554→0.10462 at L14. At L14, even the opposite-family-plus-quadratic condition has ratio0.274 while KL is0.09598. A target margin alone can favor a substantially less faithful intervention distribution.

Likewise, native-linear-only effects are not a reliable substitute for the full change: native J(u)d omits Q, whereas the midpoint form accounts for the exact quadratic effect through its evaluation point. These controls identify a finite-intervention geometry issue rather than proving that quadratic computation is unimportant.

### What this establishes and what remains

The data are consistent with a shared context-dependent transformation acting on many lexical directions. They do not establish a low-dimensional manifold, a compact learned dictionary, or automatic task labels. J(m) retains the original bilinear weights and may be high rank. Current means are diagnostic context summaries; downstream attention and other sites still participate in the task.

Next freeze these means and test additional country/language pairs and prompt variants with competence and tokenization checks specified before looking at intervention outcomes. Keep full-distribution fidelity alongside target margins, and retain failed competence cells. Independently compare means fitted from disjoint discovery country sets before treating the operator as canonical. This is more informative than further shrinking the lexical span on the same small panel. Binding still needs its separate attention/value-bus investigation; broader shared-dictionary/manifold objectives remain open.

### Verification

1,248 full-model forwards. Identity patches give zero logit difference; cached activation replay error5.11e−7. Exact full effects reproduce H906 at all three layers. Independent direct evaluations verify the midpoint identity on32 cached language endpoint pairs per layer, maximum relative error4.17e−15. Source/model/frozen-mean hashes unchanged and all KL finite. No natural or reserved-test evaluation; document pool remains26,112. Goal active.
