## H919 — Small discovery-name value spans fail causal transfer

H918 established a causal carrier: the persistent values read by L14H4 strongly affect binding. H919 asks whether this carrier is a small reusable subspace. It is a linear baseline for the manifold/featurizer question, not a new BSF fit.

### Design

Fit uncentered SVD to four persistent-value differences between discovery name pairs. One copy/template-A record per pair supplies the fit; no transfer names contribute. Evaluate ranks 1, 2 and 4 at each of the frozen L8H3, L13H0 and L14H4 readers. Three seeded random subspaces at each rank are controls. Use the same head-specific basis across binding/copying and both templates. Project the donor-minus-recipient bus difference, retain native routing, and run the downstream model live.

Evaluation uses the four other name pairs, previously used in earlier experiments, not a pristine test set. The reference is H918's full bus swap. Compare answer-effect RMSE and KL(full edited distribution || projected edited distribution) against doing nothing. Ratios near one mean little improvement over no intervention; zero means perfect reproduction. Captured value energy is measured separately.

### Rank-4 transfer results

|Family|Reader|Value energy captured|Effect error / doing nothing|KL / doing nothing|Random rank-4 effect-error range|
|---|---|---:|---:|---:|---:|
|copy|L8H3|0.040|0.961|0.968|0.976–0.980|
|copy|L13H0|0.016|0.981|0.977|0.961–0.973|
|copy|L14H4|0.026|0.990|1.009|0.967–0.976|
|task|L8H3|0.040|0.947|0.899|0.946–0.985|
|task|L13H0|0.016|0.982|0.962|0.962–0.969|
|task|L14H4|0.026|0.986|0.976|0.961–0.979|

Ranks 1 and 2 also fail to recover substantial effects. The discovery rank-4 span reconstructs its four training differences to numerical precision, but captures only 1.6–4.0% of transfer differences. At L14H4, binding effect error remains 98.6% of doing nothing and KL remains 97.6%. No persuasive advantage over random controls appears. This is not a formal significance test; there are only four transfer pairs, and endpoints/templates are correlated.

The result rejects this small paired-difference span as a useful transferable representation. It does not establish that no nonlinear manifold exists, or that larger or differently sampled dictionaries cannot work. Four arbitrary lexical contrasts are weak coverage of name identity. Persistent values may preserve many distinct identities while sharing the rule that routes or reads them. Hence shared computation need not imply low-dimensional activation differences.

### Hypothesis refresh: 20 alternatives

1. Names have high-dimensional token-specific value codes; the shared computation is routing.
2. A shared downstream readout transforms many lexical codes without compressing them to a small span.
3. The relevant subspace is larger than four dimensions and needs broader discovery coverage.
4. The arbitrary discovery pair differences miss semantic organization present in individual name vectors.
5. Nonlinear or union-of-subspace geometry exists despite failure of a global linear span.
6. Routing encodes sender versus recipient role independently of name identity.
7. Routing primarily exploits repeated-token identity rather than an abstract role.
8. Position or recency heuristics explain much of this narrow binding panel.
9. Name-specific query/key interactions modulate routing, preventing simple coefficient transfer.
10. L14H4 carries an exclusion signal that suppresses the sender rather than directly representing the recipient.
11. The head directly promotes the alternative name through context-dependent downstream processing.
12. Signed attention coefficients reverse a common lexical write between copying and binding.
13. Later MLPs transform the same value direction differently across contexts.
14. Final normalization amplifies or attenuates apparently task-specific effects.
15. Several readers redundantly carry the same lexical information.
16. Reader interactions are nonlinear, so individual-head effects understate joint dependence.
17. The persistent stream supplies identity while current-layer values supply small but necessary context corrections.
18. Other source positions, rather than the changed sender position alone, supply the decisive recipient information.
19. The present templates reward shallow completion that will fail altered syntax or entity order.
20. Full-vocabulary effects arise partly from broad name-likelihood shifts rather than selective binding computation.

These are candidate explanations, not equally supported claims. H918 most directly supports persistent lexical information (1), and H919 makes insufficient coverage/high dimension (3–4) plausible. A shared routing/readout rule (2,6,7,12) is the highest-information next target because it can explain reuse without requiring activation compression. Position/template shortcuts (8,19) and suppression versus promotion (10,11,20) are essential competing controls.

### Next experiment

Use a fixed counterbalanced binding panel that independently changes which name is listed first and which name is the sender, across both templates. Retain all eligible cases and check full-vocabulary competence before interpreting intervention results. Measure signed routing to each name occurrence and separate correct-name promotion from sender suppression. Then transfer routing coefficients between name identities while retaining the recipient's own values and positions matched by role, comparing same-role and wrong-role donors. Include native/self controls and copying controls. This tests a reusable operation directly; it avoids treating successful compression as a prerequisite for shared computation.

### Verification and execution accounting

The retained run has 1,376 forwards. A first complete run failed only an overly strict static-token cache tolerance: 1.1444e-5 versus 1e-5. The panel's earlier 1e-4 tolerance was restored, then all 1,376 forwards were rerun; total compute is 2,752 forwards. This numerical tolerance revision is disclosed rather than hidden.

Identity edits exactly reproduce native logits, and full bus edits exactly replay H918. Native-cache maximum logit discrepancy is 1.24e-5. Basis orthogonality error is 1.22e-15; discovery rank-4 reconstruction relative error is 7.26e-16. Source hashes unchanged and results finite. Fits, singular values, endpoint metrics and random-control outcomes saved. No reserved-test evaluation or added natural documents; pool remains 26,112. Broad goal active.
