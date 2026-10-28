## H598 — A fixed response-weighted projection improves behavior without per-case target responses

Freeze the expanded 64-dimensional context basis A. Using only 64 old regular discovery input responses e, compute G=mean(J_e^T J_e), where J_e dc=T(dc,e). Construct the fixed coefficient map K=(A G A^T)^-1 A G. At evaluation use coefficients K dc, instead of ordinary A dc. The new case's incoming perturbation e and full target response are not used to choose those coefficients; e is still used to execute the same bilinear response map after context projection. The actual paired context difference dc is required, so this is not a predictor of context from raw text.

G is computed through the original MLP weights and discovery responses; no new-case outcome fitting occurs. This is a response-weighted fixed linear projection, not a new semantic basis or a claim of computational speedup. The H597 outcomes motivated this follow-up, so it is adaptive evaluation on that panel rather than a second independent confirmation.

Test full, unchanged, ordinary projection, fixed metric and per-case oracle on all H597 recipient cases: 2,560 interventions. Float64 KL/group-NLL are measured against the live full context switch. Relative errors use the full-versus-unchanged discrepancy, not total model performance.

| Fresh cell | Projection margin error | Fixed-metric margin error | Oracle margin error | Projection relative KL | Fixed-metric relative KL |
|---|---:|---:|---:|---:|---:|
|regular/fronted_old|0.8291|0.1991|0.1861|0.4911|0.2912|
|regular/relative_old|1.0156|0.2893|0.2346|0.3799|0.1389|
|regular/fronted_new|0.8062|0.3289|0.3086|0.5760|0.4438|
|regular/relative_new|1.0095|0.1626|0.0961|0.6689|0.2039|
|irregular/fronted_old|0.4546|0.1027|0.0840|0.1855|0.0510|
|irregular/relative_old|0.6859|0.2594|0.1621|0.3000|0.1028|
|irregular/fronted_new|0.6853|0.1514|0.1225|0.5212|0.1311|
|irregular/relative_new|1.3273|0.1472|0.1002|1.2264|0.0917|

The fixed metric improves both margin error and reference KL over ordinary projection in all eight cells. On new-wording irregular relative clauses, margin error drops 1.327 to 0.147 and relative KL 1.226 to 0.092, without using that case's full response to compute coefficients. The oracle remains better (0.100 margin, 0.077 relative KL), but much of its advantage is achievable by a fixed rule derived solely from old discovery data.

Internal error changes less: in that cell it falls 0.774 to 0.679, versus oracle 0.651. Thus the useful coordinate choice is informed by how contexts affect computation, not just how much context-vector variance they reconstruct. This is concrete support for response-weighted data-dependent equivalence, with a defined operation, perturbation distribution and fidelity objective.

Important limits remain. Errors are appreciable, especially regular fronted/new-wording cases. No cell meets a joint 10% relative-margin/relative-KL heuristic under the fixed metric; this is improved approximation, not full faithfulness. It is one MLP context-switch response within an agreement experiment, not a global canonical form. The map requires observed paired context differences, and its training metric reflects one number-associated perturbation distribution. No claim is made about unrelated tasks or signals.

Next freeze K as well as the basis and number direction before a new confirmation panel or dose/signal regime. Recheck competence and keep failures. Compare fixed metric, ordinary projection and oracle without further tuning. Testing other incoming-signal distributions will distinguish a generally useful response metric from one narrowly adapted to this number perturbation.

Audits: 64 old regular training responses, zero fresh training cases; K A^T identity error <=1.20e-7; metric system condition 31.55; solve residual <=2.00e-8; native-response replay <=1.24e-5. An independent forward-map audit on four seeded test vectors over all 64 training responses verifies the adjoint formula within 1.59e-7 and normal equations within 3.30e-7, and reduces training response squared error versus Euclidean projection (ratio 0.937). Source/bases/metric unchanged; finite float64 metrics. No new natural-text documents; FineWeb remains 26,112. Broad goal active and incomplete.
