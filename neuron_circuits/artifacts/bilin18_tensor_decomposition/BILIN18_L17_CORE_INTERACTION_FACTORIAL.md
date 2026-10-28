

## H270 — Interaction structure reverses the basis ranking with exact coefficients

H269 suggested either cross-core interactions or coefficient fitting caused the basis-ranking reversal. H270 holds exact coefficients, mean, states, source interventions and basis fixed while crossing natural/Jacobian basis with within-core-only/all-core-partner quadratic terms. Repeat using full or624-restricted mean-linear response. Primary width32; widths16/64 descriptive. Both scales0.1/1 and actual/same-head matched-angle payloads retained. Full signed quadratic Gram matrices saved. This eliminates fitting as a necessary cause of the reversal in these exact truncations.

Frozen natural/Jacobian bases come from H269 discovery only. The reused128-document panel is followed by a frozen128-document replication excluding complete-document hashes from all five recent panels (1,472 unique documents). Both panels have four positions per document and nine source heads. Replication config, identical gates and panel construction were written before reading primary outcomes; no bases, widths or thresholds changed. Fresh text may have appeared in older studies or model training; this is not a global or training holdout. Same random control seed pairs designs within a panel, not individual documents across panels.

| Panel, actual scale | Natural within32 | Jacobian within32 | Natural all partners | Jacobian all partners |
|---|---:|---:|---:|---:|
|Reused, 0.1|16.80%|23.24%|57.20%|51.26%|
|Reused, 1.0|17.19%|23.25%|56.90%|51.17%|
|Fresh, 0.1|16.68%|23.93%|56.64%|52.04%|
|Fresh, 1.0|16.82%|23.86%|56.18%|51.80%|

Primary contrasts at full actual edit (percentage points, paired document-bootstrap95%):

- Reused, within: -6.06 [-8.03, -4.17].
- Reused, all: 5.74 [3.53, 8.14].
- Reused, interaction: 11.79 [9.90, 13.75].
- Fresh, within: -7.04 [-8.90, -5.27].
- Fresh, all: 4.39 [2.85, 6.14].
- Fresh, interaction: 11.43 [10.15, 12.73].

The interaction contrast is (natural−Jacobian advantage with all partners) minus (natural−Jacobian advantage within core). These are differences in response-error fidelity, including correlations/cancellation, not independent additive energy shares.

**Reused locked gates:** exact_reversal_full: PASS, interaction_gain_full: PASS, exact_reversal_624: PASS, interaction_gain_624: PASS.

**Fresh locked gates:** exact_reversal_full: PASS, interaction_gain_full: PASS, exact_reversal_624: PASS, interaction_gain_624: PASS.


Exact reversal requires the within-only natural advantage CI entirely below zero and all-partner advantage CI entirely above zero at BOTH actual scales. Interaction-gain gate requires>=5 points with positive paired lower bound, both scales. Both criteria are evaluated separately with full and624 linear terms. Width16/64 and angle controls are descriptive, not used to redefine passing thresholds.

**Mechanistic implication.** In this residual polynomial, natural-state subspaces become more useful when allowed to interact with their complement; residual-Jacobian subspaces perform better when restricted to interactions internal to the core. A ranking of candidate feature directions therefore depends on the allowed interaction structure. Cross-core couplings are sufficient to produce the observed ranking reversal without learned-coefficient error. This does not show that fitting is irrelevant to the earlier learned models or that natural PCs are universally preferable. The tested decomposition is invariant under orthogonal rotations within each fixed core, so the evidence concerns interacting subspaces rather than individual named axes.

**Scope.** All-partner models access the full input and have more interactions than within-core models. This is an explanatory factorial, not equal-parameter/compute performance. Exact coefficients are not optimal fitted-readout bounds. No semantic units or downstream repair are established. Same corpus/model replication reduces document-selection concerns but leaves external validity open.

**Next discriminator.** Determine whether useful cross-core couplings admit a compact partner representation. Freeze natural32 and Jacobian32 cores; choose partner spans on discovery data only using (a) residual-tensor contraction energy and (b) exercised cross-term response geometry, with matched random controls and partner widths. Evaluate exact retained cross terms before fitting coefficients, then separately test a learned predictor at matched storage. Reserve new documents for confirmation after selection; the H270 fresh panel is now used. This distinguishes compact conditional computations from diffuse couplings, and weight concentration from actual execution coverage. Semantic labeling comes after causal transfer and downstream relevance, not from naming high-energy axes.

**Audits.** Three relevant tests pass, including partition replay, full-core endpoint and invariance to within-core rotations. Maximum relative full-model replay: reused 9.06e-12, fresh 8.4e-12. H269 all-partner fidelities reproduce to maximum absolute difference 1.91e-14. Fresh panel hashes and absence of overlap with all1472 prior documents verified. Full project suite not rerun. Figures inspected. All runs complete; native weights and original compiler unchanged. Broad goal remains active.
