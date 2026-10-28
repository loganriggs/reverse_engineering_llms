

## H385 — Tail token identities and fitting geometry versus target signal

Reuse H384’s 512 documents, 2,048 sites and 39 native response conditions. Keep the 32 largest recipient token probabilities at their original identities. Sort the remaining tokens by recipient probability and reassign their probability values using one fixed seeded permutation of tail ranks. This preserves every probability value, the entire spectrum, and the top32 probabilities exactly; only assignments among the remaining vocabulary change. Both frozen probability/random bases and ranks16/64 remain fixed.

For each metric form the centered logit Gram matrix G and target cross-moment b, without the cap derivative. Cross original/shuffled G with original/shuffled b and solve c=b pinv(G). Evaluate all four with full native capped-logit KL and the same eight-step target-dependent safeguard. Mixed G/b combinations are diagnostic quadratic fits, not necessarily any natural probability metric or realizable context. They still use the full target response. The shuffle changes both within-tail and top-versus-tail mean contributions through centering.

|Basis|Head|Family|Own G/b|Own G, shuffled b|Shuffled G, own b|Shuffled G/b|
|---|---|---|---:|---:|---:|---:|
|probability|0|actual|0.8599|0.3761|0.3462|0.7610|
|probability|0|moment|0.7478|0.3792|0.2994|0.5976|
|probability|0|donor|0.7479|0.3744|0.2966|0.5996|
|probability|1|actual|0.8012|0.3302|0.3364|0.6441|
|probability|1|moment|0.7901|0.3502|0.3330|0.6289|
|probability|1|donor|0.7919|0.3459|0.3317|0.6287|
|probability|6|actual|0.8158|0.3814|0.3614|0.6588|
|probability|6|moment|0.8352|0.3531|0.3494|0.6530|
|probability|6|donor|0.8396|0.3477|0.3640|0.6552|
|random|0|actual|0.8132|0.3810|0.5952|0.7341|
|random|0|moment|0.7287|0.3957|0.5824|0.6282|
|random|0|donor|0.7272|0.3907|0.5766|0.6266|
|random|1|actual|0.7737|0.3735|0.5874|0.6795|
|random|1|moment|0.7610|0.3919|0.5919|0.6590|
|random|1|donor|0.7638|0.3877|0.5901|0.6629|
|random|6|actual|0.7738|0.3816|0.5534|0.6642|
|random|6|moment|0.8056|0.3685|0.5713|0.7070|
|random|6|donor|0.8128|0.3678|0.5784|0.7153|

Frozen gates: {'tail_alignment5': False, 'geometry_dominates': False, 'rhs_dominates': False}. Tail alignment requires every original-minus-fully-shuffled joint lower bound >5 percentage points. Geometry dominance requires every geometry-minus-RHS joint lower bound >0; RHS dominance requires every corresponding upper bound <0. The latter are competing universal hypotheses; they cannot both pass.

Factorial contributions average each factor’s effect over the other factor: geometry = [(Roo−Rso)+(Ros−Rss)]/2; RHS = [(Roo−Ros)+(Rso−Rss)]/2. They sum to Roo−Rss. Interaction = Roo−Ros−Rso+Rss. Indices list G first, b second. These describe this fitting-and-safeguarding procedure, not a decomposition of native neural computation.

|Contrast|Point range, all36 settings|Global range of simultaneous bounds|
|---|---:|---:|
|own_minus_shuffled|4.60 to 18.44 pp|-1.09 to 24.13 pp|
|geometry_contribution|-6.75 to 11.50 pp|-12.44 to 17.19 pp|
|rhs_contribution|2.16 to 15.40 pp|-3.53 to 21.09 pp|
|geometry_minus_rhs|-21.43 to 7.98 pp|-27.11 to 13.67 pp|
|interaction|8.57 to 89.86 pp|2.88 to 95.54 pp|

Audits: source hashes verified; protected probabilities exactly unchanged; entropy invariant within 1.8e-15 nats. Independent audit verifies permutation bijectivity and saved-grid minima. Original G/b full grids replay H384 within 2.2e-12 absolute KL. All16/64 Gram directions retained. Maximum normal-equation relative residual 2.3e-12. Target baseline agrees across methods and bases. Saved arrays include coefficient squared-norm ratios and step choices, so extreme mixed solutions and safeguard suppression remain observable. Nonworsening versus baseline is guaranteed by step zero.

Uncertainty: 4,000 paired recipient-document/control-draw bootstrap resamples; one simultaneous max-absolute-deviation band over 180 contrasts (five effects ×36 settings). Fixed orientation, empirical donor assignments and one tail-rank permutation. This is a reused panel and does not estimate variability across permutations. Plot shows rank64 point estimates; JSON includes both ranks and all bands. Independent audit fields are added by work/audit_h385.py after the runner.

Results and interpretation: all three frozen gates FAIL. Tail shuffling has positive point loss in all 36 settings (4.60–18.44 percentage points), but only 30/36 simultaneous lower bounds exceed zero and 11/36 exceed five points. Therefore neither a universal positive loss nor the stronger universal five-point loss is established at the declared multiplicity. This failure is retained. The protected top32 tokens carry 81.47% of recipient probability mass on average.

The clearest signal is interaction, one of the five prespecified contrasts included in the joint band. Interaction is positive in all 36 simultaneous intervals: points 8.57–89.86 percentage points, smallest lower bound 2.88 points. At rank64 the interaction spans 37.89–89.86 points. Original G/b recovers 72.72–85.99%, fully tail-shuffled G/b recovers 59.76–76.10%, original G with shuffled b recovers 33.02–39.57%, and shuffled G with original b recovers 29.66–59.52%. Coherent use of a single metric is much better than either mixed system for these rank64 groups. This is not evidence that either component alone is the native computational unit.

Geometry-minus-RHS contributions vary by basis: −2.61 to +7.98 points for probability axes, versus −21.43 to −8.55 points for the random basis. Thus no universal dominance holds across bases. A factorial average can hide large changes of sign depending on which other component is used. These are properties of the diagnostic fitting procedure, not a causal allocation of model computation between geometry and a signal generator.

Mixed-system coefficient norms reveal severe extrapolation despite accurate linear solves. Maximum squared coefficient/target-displacement norm ratio is 41,111 in the probability basis and 69,881 in the random basis, both in original-G/shuffled-b rank64 fits. The corresponding coherent original/fully-shuffled rank64 maxima are only 13.04/20.72 and 17.87/18.34. Original-G/shuffled-b rank64 selects the zero step on 5.09%/3.42% of sites (probability/random). These are retained failures, not exclusions. Very large norms and the finite safeguard mediate some of the mismatch penalty; they do not prove semantic content in individual tail directions.

Hypothesis update: (1) keeping only top32 identities while randomizing the rest is not point-equivalent to the native fit, though the universal five-point criterion fails; (2) covariance and target cross-moment consistency is strongly supported as important for oracle recovery; (3) a unique dominant factor is not supported across both bases; (4) generic tail stabilization remains plausible because this experiment changed both statistical structure and consistency in its mixed arms; (5) semantic tail features remain unproven. H384's twenty-hypothesis table remains current with these refinements.

Next discriminator: decompose the original metric exactly into top32 conditional covariance, tail conditional covariance, and the between-group mean term. Replace tail covariance with a trace-matched isotropic surrogate while deriving BOTH Gram and RHS from the same surrogate metric; retain the original top32 contribution and between-group mean term. Compare that internally consistent replacement with the exact metric, and separately remove the between-group term. This tests whether detailed within-tail directions are necessary beyond generic stabilization without recreating the extreme mismatched normal equations. Any promising approximation still needs fresh-document confirmation and eventually a target-free coefficient generator. Future fresh panels exclude 5,824 document hashes. Broad goal remains active.
