## H939 — Intervention coverage recovers known computation; effect loss is not extra information

Directly test the user's proposal to train a representation for causal prediction. A known teacher embeds8 signal and8 nuisance coordinates through a random orthogonal16D basis. Its4 outputs are disjoint products of signal-coordinate pairs. In native training states, nuisance coordinates nearly duplicate signal coordinates; in evaluation they become independent. This creates a controlled observational shortcut without changing the true computation.

Train12 same-capacity students:4 learned bilinear products with normalized left/right factors, output mixing and bias; four objectives, three matched initializations and minibatch sequences,1500 Adam steps each. The true product count4 is supplied as an architectural prior, but no true axes, task labels or factor assignments enter optimization.512 training states and512 ambient random unit-norm interventions are fixed. Evaluate2048 new states per familiar/decorrelated panel plus paired true-component edits, whose axes are used only for evaluation. Paired edits have norm sqrt(2), not the unit training norm.

Objectives: native outputs only; native plus intervened endpoints; native plus finite-effect error; native plus shuffled finite-effect targets. Endpoint and effect methods see the same teacher endpoint information. Native/effect losses use the same output-energy normalization. The effect objective weights correlated endpoint errors differently; it is not a new source of causal information. No activation-reconstruction objective, convex constraint, or LLM training is involved.

| Objective | Familiar native-output error | Decorrelated effect error | Matched true-pair cosine |
|---|---:|---:|---:|
| Native only |3.88–4.28%|90.3–99.0%|.616–.715|
| Intervened endpoints |.25–.67%|5.39–12.74%|.9961–.9995|
| Explicit effects |.01–.28%|.19–7.39%|.9993–1.0000|
| Shuffled effects |7.53–7.75%|79.9–80.3%|.774–.779|

Errors are norm ratios to true outputs/effects; ranges are three seeds, not uncertainty intervals. Pair cosine uses optimal one-to-one matching and allows swapping each product's two factors and sign ambiguity. This scores recovery of known computation rather than merely cross-seed agreement.

On paired-component edits, endpoint error4.23–9.51%, effect loss.15–5.67%, native-only76.5–83.0%. Thus intervention-trained factors generalize to an edit family not supplied during training. The explicit-effect loss beats endpoints in2/3 seeds and loses slightly in the third at this fixed budget. Both recover nearly exact true axes, so claiming unique identifiability from the effect loss would be unwarranted. Finite training, objective weights, seed variability and known-rank architecture remain confounds. Native-only failure may partly reflect poor conditioning and limited optimization; this does not prove asymptotic non-identifiability, since nuisance correlation is not exact.

Verification: teacher finite-change product expansion agrees to2.33e-15; independent embedding of true teacher into Student agrees to4.44e-15; input directional-gradient finite difference error1.58e-11; saved model effect metrics replay to1.11e-16. All12 fits, full curves and datasets saved; source hash unchanged; allfinite. Original model forwards0 and reserved-test evaluations0; natural document pool unchanged.

Decision: retain intervention coverage as essential in the proposed prior comparison. A causal effect loss is a promising training emphasis, but not automatically a better prior than learning on the same intervened endpoints. Next test convex-hull versus unrestricted and contrast-anchored factors under matched causal data, with anchor and initialization randomness separated. Include a teacher whose components are never individually observed, so a state-prototype prior can fail for a principled reason; compare effective directions after offsets/centering and account for scaling. Then transfer the validated objective to an established model site with independent interventions, known mechanisms used explicitly as training or evaluation anchors, and fresh diverse tasks. This synthetic result alone does not validate an LLM manifold or a unique computational basis.
