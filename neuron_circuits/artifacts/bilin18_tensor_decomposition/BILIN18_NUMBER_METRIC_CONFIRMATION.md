## H599 — Frozen response metric transfers to new wording and eightfold larger perturbations

Freeze the H598 coefficient map K, its 64-dimensional context basis, and the H584 number direction before inference. Test 384 prompts with the same H597 lexical inventory but rotated subject/attractor pairings and new wording: near/could observe and in front of/will examine. Cross regular/irregular nouns, subject/attractor number, and unfronted/fronted/relative-clause frames. Exact prompt overlap with the checked prior manifests is zero. This is combination, wording and dose transfer, not an independent vocabulary holdout or diverse-task confirmation.

All 24 competence cells pass; both-endpoint candidate correctness is 1.000 throughout, and full-vocabulary intended-group top1 ranges 0.5625–1.000. Retain all examples. Capture 1,920 native/signed-dose forwards, then execute 5,120 interventions at doses 0.0625 and 0.5. Compare the exact context switch, unchanged context, ordinary projection, fixed response metric, and per-case response oracle. No refitting occurs.

The fixed response metric improves relative margin error and relative reference KL over ordinary projection in all 16 type/frame/dose cells. The larger dose is eight times the smaller one. Coefficients use K dc and do not consult each case's target response or incoming perturbation; the observed paired context difference dc and incoming perturbation for executing the bilinear map are still required.

| Cell | Dose | Projection margin error | Fixed-metric margin error | Projection relative KL | Fixed-metric relative KL |
|---|---:|---:|---:|---:|---:|
|regular/fronted_near|0.0625|0.6719|0.3073|0.5338|0.4580|
|regular/fronted_near|0.5|0.6719|0.3163|0.5138|0.4158|
|regular/relative_near|0.0625|0.5310|0.1131|0.2994|0.2289|
|regular/relative_near|0.5|0.5103|0.1215|0.2676|0.1808|
|regular/fronted_ahead|0.0625|0.6546|0.1372|0.4294|0.1870|
|regular/fronted_ahead|0.5|0.6451|0.1447|0.4091|0.1457|
|regular/relative_ahead|0.0625|0.3986|0.1890|0.2695|0.2029|
|regular/relative_ahead|0.5|0.4067|0.2229|0.2579|0.1800|
|irregular/fronted_near|0.0625|0.4794|0.1197|0.1461|0.0645|
|irregular/fronted_near|0.5|0.4875|0.1254|0.1770|0.0612|
|irregular/relative_near|0.0625|0.4173|0.1206|0.2474|0.1037|
|irregular/relative_near|0.5|0.4091|0.1253|0.2342|0.0914|
|irregular/fronted_ahead|0.0625|0.7823|0.2146|0.4790|0.1042|
|irregular/fronted_ahead|0.5|0.7486|0.2358|0.5013|0.1174|
|irregular/relative_ahead|0.0625|0.4697|0.3540|0.2668|0.1884|
|irregular/relative_ahead|0.5|0.4594|0.3604|0.2418|0.1648|

These errors are relative to the discrepancy between the full context switch and leaving context unchanged. They are not task error rates or fractions of total model computation. Absolute reference KL is small: the full-versus-unchanged mean ranges roughly 6e-7–1.2e-5 at small dose and 4e-5–1e-3 at larger dose. The experiment tests fidelity to a particular local intervention.

The fixed metric's margin error remains about 11–36%, and relative KL about 6–46%. No cell meets the joint 10% margin/KL heuristic. Internal response errors remain about 67–76%, despite better output fidelity. This supports a response-informed coordinate rule while leaving considerable local information unrepresented. The target-informed oracle minimizes internal response norm error, so it need not win every behavioral metric.

Most likely interpretation: the useful notion of context similarity depends on the operation and incoming signal distribution. Ordinary Euclidean context similarity discards distinctions that matter downstream; a fixed response-informed rule transfers within this family. Alternatives still open include a broadly reusable response metric versus a grammar/number-specific metric, and downstream insensitivity hiding substantial internal differences. Larger-dose success weakens a strictly infinitesimal-only account but does not resolve those alternatives.

Next prioritize cross-signal and cross-task transfer over further wording variants. Freeze K and compare its behavior on copying, factual recall, entity binding and sentiment as well as agreement, with competence gates and retained failures. Compare fixed metric versus ordinary projection and per-case oracle for each operation; report internal error, output KL and target-token CE separately. A task-specific benefit would support conditional computational units; broad transfer would support more reusable units. Do not infer diverse-task success from this agreement-only panel.

Audits: source, manifest, metric/basis and number direction unchanged; zero checked prompt overlap; native replay maximum margin error 1.63e-5; context-switch identity relative error 2.14e-6; oracle normal-equation error 1.12e-14; finite metrics. The inherited old-span-containment field is a placeholder, not a new containment audit. No new natural-text documents; FineWeb remains 26,112. Broad goal remains active and incomplete.
