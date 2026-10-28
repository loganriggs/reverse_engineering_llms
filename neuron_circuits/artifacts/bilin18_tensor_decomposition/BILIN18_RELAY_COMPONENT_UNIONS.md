## H862 — Nested component unions: distributed representation versus dictionary fragmentation

The previous goal turn made progress by testing individual components causally in H861. H862 now tests whether unions improve transfer, while separating decoder geometry, native encoding, and edit magnitude. All five architectures and both seeds remain included.

### Protocol

Rank groups by overlap with each H848 discovery-supervised task axis. Freeze nested prefixes at nominal8/32/64 atom slots. Signed scalar dictionaries select8/32/64 atoms; block/shared dictionaries select1/4/8 groups of8. Use one random nested group ordering per fit/task as a control. Evaluate every candidate on both agreement and sentiment, all78 H849 pairs and both recipient directions. Selection does not use downstream outcomes, but it is task supervised and the panel is reused research data.

Projection transfers the native donor–recipient difference projected onto the combined decoder span. Encoded transfer uses the selected active coordinates decoded once per atom, scaled by each input's native raw RMS. For overlapping groups, an atom is included if at least one selected group is active; multiplicity never amplifies it. The rest of the recipient state stays native. A third control rescales the random projection to the ranked projection's edit norm for the same pair. This norm control is a pairwise oracle, not an implementable single-input featurizer. The original discovery rank1 task axis is a supervised reference.

### Results at nominal64 atom slots

Effect slope is sum(component_effect × full_relay_effect) / sum(full_relay_effect²). A slope of1 matches the reference scale but does not guarantee per-example fidelity. Relative effect RMSE and outputKL are retained separately in the analysis.

|Architecture|Task|Seed1 projection|Seed1 encoded|Seed2 projection|Seed2 encoded|
|---|---|---:|---:|---:|---:|
|scalar48|agreement|0.732|0.426|0.694|0.554|
|scalar48|sentiment|0.679|0.592|0.521|0.397|
|scalar64|agreement|0.654|0.425|0.666|0.521|
|scalar64|sentiment|0.544|0.322|0.599|0.421|
|block8|agreement|0.668|0.407|0.704|0.590|
|block8|sentiment|0.717|0.555|0.614|0.539|
|shared_fixed|agreement|0.561|0.052|0.714|0.173|
|shared_fixed|sentiment|0.392|0.273|0.533|0.130|
|shared_learned|agreement|0.189|0.029|0.223|0.024|
|shared_learned|sentiment|0.228|0.058|0.141|0.000|

![Nested union curves](/home/loganriggs/Documents/Codex/2026-10-21/he/outputs/relay_component_unions.png)

Solid/dashed lines denote seeds8351/8352. Blue is ranked projection, orange is native encoded union, gray is norm-matched random projection, and purple is the supervised rank1 reference. Curves are descriptive; only two seeds and small numbers of lexical-pair clusters are available.

At64 slots, scalar/block projection slopes range0.521–0.732, while encoded slopes range0.322–0.592. The single discovery-supervised axis reaches0.872 for agreement and0.890 for sentiment, with relative effectRMSE0.211/0.178. Thus larger learned unions recover substantially more signal than individual components, but do not yet match an aligned task direction. This supports dictionary fragmentation and encoder loss as important explanations; it does not establish intrinsically high-dimensional computation.

Ranked projections have larger own-task slopes than norm-matched random projections in59/60 fit/task/budget comparisons. Random control slopes range-.0036 to.272. This weakens a pure edit-magnitude explanation but is not60 independent replications: fits, budgets and tasks share examples, and only one random ordering was used per fit/task. Direct pairwise edit-norm checks agree within6.37e-12 raw units. Shared-group encoded agreement remains especially weak:0.052/0.173 for fixed groups and0.029/0.024 for learned groups at64 slots.

### Geometry and interpretation limits

At nominal64 slots, scalar and block spans have rank64. Fixed-shared spans have ranks57–59; learned-shared spans42–47. Random unions are matched in group count, not necessarily actual rank for overlapping architectures. These are not strict equal-capacity architecture comparisons.

Rank64 scalar/block unions contain about56.6–68.7% of the discovery task-axis energy. Fixed-shared unions contain51.7–58.6%; learned-shared unions32.2–40.3%. Cross-seed union overlap remains limited: about0.22–0.32 for scalar/block,0.30–0.36 fixed-shared, and0.46–0.47 learned-shared, measured by twice projector intersection trace divided by total rank. Higher learned-shared overlap does not imply higher causal fidelity: its dictionaries cover fewer directions and its gates remain problematic.

A crucial distinction: needing many learned atoms does not prove the model's computation is intrinsically high-dimensional. A single task direction can be fragmented across a dictionary whose axes are poorly aligned with it. The rank1 supervised reference makes that alternative explicit. Likewise, successful projection with weak encoded transfer points to inference/routing losses, not absence of representable signal. Even the chosen top-overlap unions are a heuristic, not an optimal subset search.

### Hypothesis refresh: twenty live explanations

|#|Hypothesis|Current status / discriminating evidence|
|---:|---|---|
|1|Each concept is one stable learned group|Weakened by H860–861 instability and partial effects.|
|2|Stable partial scalar contributors exist|Supported narrowly by the matched sentiment atom.|
|3|Computation occupies intrinsically multidimensional manifolds|Unresolved; dictionary fragmentation can mimic it.|
|4|A low-dimensional task signal is spread across dictionary atoms|Strong candidate; compare within-union rank1 compression next.|
|5|Encoders underuse available decoder geometry|Supported by projection versus encoded gaps.|
|6|Group gates suppress usable directions|Supported for some shared groups with zero encoded effects.|
|7|Learned membership collapses coverage|Observed in earlier fits and reduced union ranks.|
|8|Finite training causes instability rather than architecture limits|Unresolved; needs converged/multiseed fits.|
|9|Task signal amplitude is attenuated by reconstruction training|Supported by earlier norm-restoration effects, not universally.|
|10|Greater edit magnitude alone explains targeted effects|Weakened: ranked projection wins59/60 own-task comparisons against equal-norm random edits.|
|11|Whole-model fidelity differs from activation fidelity|Supported by H857–859.|
|12|Local derivative metrics fully predict downstream fidelity|Contradicted by H857–859 ranking reversal.|
|13|Token means explain apparent contextual feature stability|Insufficient for scalar/block H860 correlations; still a confound to monitor.|
|14|Unseen-token fallback causes a different calibration regime|Suggested by H859; coverage is not randomized.|
|15|Off-axis coordinates modulate the same task coordinate's effects|Supported by earlier writer interventions; test union complements.|
|16|Task-selective directions are sufficient but not necessary|Still open: transfer alone does not establish necessity.|
|17|Effects generalize across lexical/format/distance changes|Partial evidence on reused H849, with gap sensitivity.|
|18|The same component supports multiple computations|Not established by two-task specificity; requires broader active-site tests.|
|19|Persistent value-bus computations require a different featurization site|Prior binding/copy/capital evidence supports site-specific testing.|
|20|Behaviorally equivalent representations admit a useful data-based quotient|Plausible research direction; no uniqueness/minimality result yet.|

### Next information-gaining experiment

Within each frozen ranked union, project the discovery task axis into the union and normalize it. Compare the resulting single direction, its orthogonal union complement, and the full union under causal transfer. Keep both seeds, both tasks, and norm controls. If one within-union direction matches the full union effect, the apparent need for many atoms reflects dictionary alignment more than a high-dimensional computational manifold. If the complement is required, test interactions and necessity. This diagnostic does not become unsupervised just because the dictionary was learned without task labels.

After resolving that distinction, extend manifold checks to binding/copy/capital at their verified active sites. The broader diversity, held-out composition, and necessity requirements remain unfinished. Do not retrain or label groups semantically solely from these reused task results.

### Verification

47,424 whole-model forwards; identity logit error0, full-donor replay error1.43e-05, orthogonality error4e-15. Explicit atom-union reference agrees exactly; nested ranked spans never lose task-axis energy beyond tolerance. Source hashes unchanged, all metrics finite. No reserved-test evaluations;26,112-document pool unchanged. Paired bootstrap uses4096 lexical-pair cluster resamples, only8 agreement and5 sentiment clusters; pointwise intervals uncorrected. Broad goal remains active.
