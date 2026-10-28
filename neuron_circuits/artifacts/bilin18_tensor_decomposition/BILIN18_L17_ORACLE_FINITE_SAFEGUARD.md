

## H376 — Finite step-size safeguard restores aggregate local-oracle recovery

On the entire H372512-document panel, reuse H374’s rank16/64 oracle coefficients and frozen H373 probability-defined subspaces. Keep the native target fixed at full amplitude. Scale only the patch by the frozen grid {0,1/16,1/8,1/4,1/2,3/4,1,1.25} and choose the lowest target-to-patch finite KL at each site. This differs from H375’s amplitude trace, where both target and patch were scaled. All tail cases remain in the primary evaluation.

Selection uses the full native target, and coefficients already use the native response: this is a doubly target-dependent oracle diagnostic, not an executable compressed model. Including0 and1 makes non-worsening versus baseline and unguarded patch algebraic properties of the candidate set, not empirical discoveries. The question is how much useful recovery remains after enforcing that safeguard.

|Head|Family|Guarded16 recovery|Guarded64 recovery|Non-unit step fraction16|Non-unit step fraction64|
|---|---|---:|---:|---:|---:|
|0|actual|0.741|0.861|0.000%|0.000%|
|0|moment|0.564|0.747|0.000%|0.000%|
|0|donor|0.573|0.753|0.000%|0.000%|
|1|actual|0.653|0.799|0.195%|0.049%|
|1|moment|0.618|0.784|0.187%|0.057%|
|1|donor|0.621|0.786|0.203%|0.033%|
|6|actual|0.687|0.824|1.123%|0.537%|
|6|moment|0.655|0.846|6.160%|1.994%|
|6|donor|0.660|0.848|6.755%|2.254%|

Frozen gates: {'guard16_sufficient': False, 'guard64_sufficient': False, 'local_finite_agreement': True}. Universal80% lower-bound recovery still fails at both ranks. The local/finite agreement criterion passes: maximum difference from H374’s unguarded local recovery is1.56 percentage points across18 groups. The finite joint18 band has half-width0.0367; this was specified for the guarded estimand before outcomes and does not replace H374’s failed unguarded band. Rank16 point recovery spans56.4–74.1%, and rank64 spans74.7–86.1%.

Head6 moment/donor rank16 recovery improves from−83.2%/−116.4% to65.5%/66.0%, although only6.16%/6.75% of sites select a non-unit step. Unguarded patches were worse than baseline at1.73%/1.84% of those sites. Rank64 control recovery rises modestly to84.6%/84.8%, with non-unit steps at1.99%/2.25%. Baseline fallback is selected at less than0.1% of sites in each group. Step1.25 is allowed, so a non-unit choice need not mean damping.

Interpretation: a small set of finite failures can dominate mean KL while local approximations remain useful elsewhere. Step-size choice alone nearly recovers the aggregate performance suggested by the local metric. This does not establish optimal finite coefficients, universal sufficiency, or that the directions correspond to semantic features. The remaining rank16/64 errors cannot be attributed solely to the identified tail catastrophe.

Audits: {'h374_endpoint_replay_relative': 0.0, 'max_best_minus_baseline': 0.0, 'max_best_minus_original': 0.0}. Grid endpoints exactly replay H374 baseline and unguarded scores; selected KL is no worse than either at every evaluated site. All KL values finite/nonnegative within tolerance. Save complete per-site grid errors and choices, summaries and4,000 paired document/control bootstrap replicates. Frozen discovery geometry and empirical donor assignments remain conditional; reused panel, no new-document confirmation of this safeguard. Native weights unchanged.

### Updated hypothesis priorities

|#|Hypothesis|Evidence or next discriminator|
|---|---|---|
|1|Euclidean coefficient mismatch limits recovery|Supported by H374; oracle fitting improves same spans.|
|2|A few finite failures dominate average error|Supported by H374–376; rare non-unit corrections repair aggregate loss.|
|3|Initially unlikely tokens cause local extrapolation failures|Directly observed in H375 selected tails; population prevalence still needs testing.|
|4|Logit-cap curvature is necessary for the catastrophe|Disfavored in selected tails; linearized-cap path also fails.|
|5|Large activation norm alone explains tail failure|Disfavored by selected tail norms near the target norm.|
|6|Rank16 contains enough information for universal80% recovery|Not established; safeguarded recovery56–74%.|
|7|Rank64 is universally80% sufficient|Fails frozen criterion; some groups remain below80% even in point estimates.|
|8|Probability-defined axes are special behavioral features|Unresolved; compare fixed random spans with identical oracle fitting.|
|9|Any moderate-dimensional span can fit concentrated distributions|Leading alternative for random-span control.|
|10|Shared probability geometry captures transferable structure|Supported by H372, limited to tested heads/layer/metric.|
|11|Recipient-specific probability geometry adds head-dependent effects|Supported by H372 head6 attenuation.|
|12|A global geometry describes every head equally|Disfavored by heterogeneous transfer and recovery patterns.|
|13|Probability weighting is the main interference source|Supported by H368 factorial accounting.|
|14|Between-context covariance alone creates interference|Disfavored by H370–372 within-context cancellation.|
|15|Finite fitting can repair errors without changing the span|Established for selected H375 cases; not globally optimized.|
|16|A target-free safeguard can approximate the oracle grid|Open; predict dangerous token growth without evaluating native target KL.|
|17|Oracle coefficients can be predicted from compact input features|Open; requires independent train/test coefficient prediction.|
|18|Alternative ranks recover behavior without semantic structure|Open; compare random orientations and sample-independent controls.|
|19|The main limitation is input-side computation rather than output span|Still plausible; all current oracles require full native response.|
|20|Results generalize across layers and model instances|Untested in this sequence; preserve scope.|

Next highest-information control: fixed random rank16/64 spans, with the same per-site Fisher fit and finite safeguard. If random spans perform similarly, oracle success may largely reflect low effective dimensionality of concentrated token distributions rather than meaningful probability eigenvectors. If probability axes outperform, quantify that advantage before learning coefficients. Freeze random seeds and comparison gates before outcomes. Broader goal remains active.
