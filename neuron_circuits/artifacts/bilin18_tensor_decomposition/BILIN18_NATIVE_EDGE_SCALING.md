## H798 — Native edge function depends on context; negative routing does not imply answer suppression

Previous turn was progress: H797 established native repetition-sensitive negative routing but left the edge's behavioral function unresolved. H798 scales the exact frozen L2H5 first-value→second-value edge in native prompts while separating likelihood, accuracy and distributional fidelity.

Use scales0,.5,1,1.5 of the actual signed edge contribution at the second source, before c_proj. All other edges and heads remain intact. Scale0 removes the edge; scale1.5 strengthens its original signed contribution. Record full-vocabulary KL(native||scaled), intended-token log-probability, intended probability and accuracy. Conflicting-key/value cells have no assigned intended answer. Scale0 is independently implemented after output projection to audit equivalence.

### Repetition cells do not support a simple inhibition label

| Format | Key relation, same value | Intended log-probability change on edge removal | Mean intended probability, native → removed | Mean KL |
|---|---|---:|---:|---:|
| Vertical bars | Same key | -.3773 | .4685 → .3268 | .1017 |
| Colon/newline | Same key | -.1157 | .4764 → .4324 | .0190 |
| Vertical bars | Different keys | -.0339 | .4993 → .4694 | .0197 |
| Colon/newline | Different keys | +.1071 | .4314 → .4689 | .0165 |

For same-key/same-value prompts, removal lowers intended log-probability in both formats: descriptive item-bootstrap intervals are [-.5359,-.2200] and[-.2123,-.0425]. The negative edge thus supports native prediction confidence in these cells, despite earlier counterfactual interaction removal increasing donor promotion. Those experiments remove different quantities in different computational backgrounds and are not contradictory.

With different keys and the same value, removal improves intended log-probability in colon/newline prompts (interval [.0446,.1726]); the vertical-bar mean has interval [-.1258,.0555] and is uncertain. Key context can therefore change the behavioral consequence even though H797's routing coefficient was predominantly value-repetition sensitive.

No cell changes aggregate full-vocabulary or binary accuracy in this experiment. In competent matching-value cells, native top1 remains96.9–100%. The causal effect is on confidence and distribution, not established necessity for the discrete answer. Log-probability changes are in natural-log units; mean probability change and mean log-probability change need not have the same sign because they average different nonlinear quantities.

### Dose response is nonlinear

For same-key/same-value prompts, halving the edge gives intended log-probability changes -.0856 in vertical bars and+.0023 in colon/newline (the latter uncertain). Strengthening it to1.5 gives -.0307 and-.0903. For different-key/same-value colon/newline prompts, half strength gives+.0855 and1.5strength gives-.1268. Thus neither monotonic inhibition nor a universal benefit from increasing edge strength is supported. A small grid of scales cannot establish an optimal native strength.

For distinct keys and values, model competence remains weak (native candidate accuracy65.6%/55%). Removal KL is much smaller on average (.00162/.00066), but these cells do not validate a successful binding mechanism. Same-key/conflicting-value cases have mean removal KL .00153/.00062 and no intended-answer accuracy claim.

### Verification

1,248 model forwards:208 native references,832 scale runs,208 independent post-projection removal checks. Scale1 reproduces native logits exactly; native margins replay H797 exactly. Pre-projection edge removal and post-projection subtraction agree within1.65e-5 maximum full-vocabulary logit error. KL is computed in float64; minimum recorded KL is0 and all metrics finite. Source/model/input/reference/manifest hashes unchanged. Original reserved test untouched; natural-document pool remains26,112.

The inherited panel has16 aligned items in vertical bars and10 in colon/newline. Bootstrap averages future-query contexts within item, including duplicate prompts when keys match; those duplicates are not independent observations. Intervals are descriptive without multiplicity adjustment. This is an intervention on one native edge, not a trained replacement model or proof of a semantic manifold.

### Research decision

The verified computation is a repetition-sensitive signed read whose downstream meaning is context-dependent. Calling it a “repetition suppressor” from its negative coefficient would be wrong: removing it reduces confidence in the repeated answer for same-key prompts. Native likelihood, KL, top1 and counterfactual donor specificity must remain separate fidelity targets.

Next hold the injected value vector exactly fixed across same-key/different-key contexts and vary only the downstream context. The first source precedes the changed second key, so its value vector should be identical by causality; audit that fact. Standardize coefficient changes as well, since the native edge coefficients differ modestly across contexts. This will test whether context-dependent downstream readout accounts for the sign difference without changing the content direction. Such a verified response family would be a better target for a shared-dictionary or data-based canonical representation than a global semantic label on the head.
