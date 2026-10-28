## H613 — Influential contextual interaction is distributed across heads

Decompose Attention5's contextual-value four-condition body-write interaction into its nine projected head contributions. Their sum reproduces the full contextual interaction. Test each head alone, full-minus-one-head, per-position norm-matched random controls, and small fixed head sets. The reference is removal of the full contextual interaction, rather than whichever intervention produces the smallest target norm.

Choose head sets of size1/2/3/4 by exhaustive minimization of mean relative squared write error on old-wording items0–3 only (eight endpoints). Freeze selections before downstream evaluation. Evaluate those discovery cases, eight held-out old-wording endpoints, and all16 new-wording endpoints. The panel was previously explored; this is adaptive item/wording transfer, not an independent benchmark or fresh vocabulary. The new-wording set includes both discovery lexical items and other items.

Locked selections (zero-based heads) are H7; H5+H7; H5+H7+H8; and H0+H5+H7+H8. These minimize the specified internal write objective on discovery; no claim is made that they optimize output KL, CE or target fidelity. No evaluation outcome is used to reselect them. Each head contribution still requires its observed four-condition routing/value data.

**No single head reproduces the full intervention well.** Four heads improve approximation but leave substantial error on held-out examples and wording:

| Split | Heads retained | Write relative error | MLP11 effect relative error | Relative output KL | Relative margin error | Relative intended-token CE error |
|---|---|---:|---:|---:|---:|---:|
|discovery|7|0.7591|0.8380|0.6182|0.5955|0.5181|
|discovery|5,7|0.6652|0.7531|0.5597|0.5824|0.4230|
|discovery|5,7,8|0.5891|0.7061|0.5677|0.5808|0.6770|
|discovery|0,5,7,8|0.5145|0.5248|0.2036|0.3769|0.2794|
|same_wording_holdout|7|0.7599|0.8339|0.5911|0.7651|0.6317|
|same_wording_holdout|5,7|0.6665|0.7509|0.4680|0.5182|0.4417|
|same_wording_holdout|5,7,8|0.5904|0.7084|0.5558|0.6080|0.7367|
|same_wording_holdout|0,5,7,8|0.5165|0.5239|0.1771|0.3661|0.3082|
|new_wording|7|0.8693|0.9009|0.9380|0.9186|1.0859|
|new_wording|5,7|0.7465|0.7379|0.5689|0.5807|0.8963|
|new_wording|5,7,8|0.6916|0.7125|0.4874|0.5819|0.8012|
|new_wording|0,5,7,8|0.6076|0.6132|0.3642|0.3530|0.7096|

MLP11 effect error is the distance between the approximate-removal and full-removal target vectors, divided by the full-removal shift from native. It is not the remaining native factorial-interaction norm. Relative output KL similarly compares the approximation with the full-removal distribution, normalized by full-removal versus unchanged. These are fidelity errors, not task error rates or explained shares. All interventions retain the correct copied name as full-vocabulary top1.

The four-head set has MLP11 effect error 0.524 on held-out old wording and 0.613 on new wording. Its relative output KL is 0.177 and 0.364, while intended-token CE fidelity is less uniform (0.308 and 0.710). Thus it is a useful partial approximation, not a faithful compact replacement under all objectives. Individual-head results likewise fail to identify one generally dominant carrier.

Leave-one-head-out results expose objective and context dependence. Omitting H4 is comparatively mild: old-wording holdout write error 0.106, target-effect error 0.108 and relative KL 0.007; new wording gives 0.105, 0.100 and 0.016. This retains eight of nine heads, so it is not substantial compression. Omitting H5 has relative KL 0.055 on old-wording holdout but 0.401 on new wording, with new-wording intended-token CE error 1.539. H7 was the best discovery single-head write approximation, but its new-wording single-head relative KL is 0.938. Head importance depends on context and the fidelity criterion.

The output interactions are exactly additive over heads, but their downstream effects are not additive causal shares. Norm-matched random head controls use one seeded direction per head/body position; they are directional checks. Selection rankings are not significance estimates across independent datasets.

Interpretation: this particular contextual interaction is spread across multiple heads rather than localized to one head. That does not imply high intrinsic computational dimension or rule out shared features across heads. Several heads may write into a common low-dimensional output subspace, and discrete head pruning is only one representation choice. Nor does preserving copying top1 establish that the approximated internal interaction is behaviorally necessary.

Next compare head pruning with a low-rank output-space representation of the combined contextual interaction. Fit directions only on discovery writes, freeze ranks and basis before held-out evaluation, and keep internal-write, MLP11-effect, output-KL and intended-token-CE fidelity separate. Such a result would test data-based compactness across heads rather than equating architectural head count with number of computational units. It would remain a conditional representation unless coefficients can later be predicted without observing all four context responses.

Audits: projected head sum error <=2.57e-8 relative; source/runner/manifest/selection unchanged; discovery restricted to eight old-wording endpoints; native logit replay <=2.48e-5; 1,056 interventions plus32 baseline forwards; all metrics finite and copying top1 correct. FineWeb remains 26,112 documents; broad goal active and incomplete.
