## H609 — Boundary matching removes most earliest interaction, but bilinear mixing creates a remaining interaction

Add one leading space to every unprefixed H607 prompt. This makes the aligned body token IDs identical across prefix presence, for each capitalization choice. The prefixed prompts are unchanged. Audit all 64 body comparisons: each prefixed token sequence after its four-token prefix exactly equals its corresponding unprefixed sequence. The first-word alternatives are now ĠHe/Ġhe in both prefix conditions and ĠA/Ġa in both. The final token remains unchanged.

Repeat the H608 native trace and all 23 predeclared final-position write removals with norm-matched controls: 128 native prompts and 1,504 interventions. All eight competence cells remain perfect. This is an input control, not a new fit. It changes the boundary conditions of the comparison; differences from H608 are not additive causal shares of the original interaction.

**Attention0's interaction shrinks by more than 99.6%, strongly implicating the token-boundary asymmetry in the earliest signal.** It does not become exactly zero. Later interaction remains substantial, and the normalized MLP11 interaction norm is about 57–60% of its former value. The boundary confound therefore matters greatly but does not account for all context dependence.

| Wording | Measurement | Original boundary norm | Matched boundary norm |
|---|---|---:|---:|
|0|Attention0 write|8.525559|0.023916|
|0|MLP0 write|428.251539|9.719131|
|0|MLP0 normalized input|0.509222|0.002150|
|0|MLP11 normalized input|2.727935|1.646987|
|1|Attention0 write|8.553003|0.032541|
|1|MLP0 write|479.901087|19.690745|
|1|MLP0 normalized input|0.501035|0.003602|
|1|MLP11 normalized input|2.408702|1.368058|

The small Attention0 residual is reported rather than interpreted as a new semantic interaction or assumed to be exactly roundoff. Rotary/cache precision and other implementation details would require additional controls to explain that residual. Large raw write norms at different layers are not directly comparable measures of causal importance because residual scales and skip coefficients differ.

We can explain the local MLP0 interaction algebraically. Let B be its ungated bilinear map, z=z00 its native normalized input, P=z10-z00, C=z01-z00, and I=z11-z10-z01+z00. The four-condition MLP interaction is exactly:

B(z11)-B(z10)-B(z01)+B(z00)
= T(P,C) + [B(z+P+C+I)-B(z+P+C)].

The first term is the direct bilinear cross response between the prefix-associated and capitalization-associated input changes. The second term propagates input non-additivity, which can include RMS normalization and inherited attention interaction; it must not be called a pure normalization effect.

| Wording | Full MLP0 interaction norm | Direct bilinear cross norm | Input-nonadditivity remainder norm | Cross/remainder cosine |
|---|---:|---:|---:|---:|
|0|9.7191|9.9034|2.7481|-0.2052|
|1|19.6906|19.8060|5.8146|-0.1666|

The direct cross term is comparable to the full local interaction; the smaller remainder partly cancels it. This is concrete evidence for local multiplicative formation after the boundary asymmetry is removed. It is a decomposition of this operation's output, not a claim that the same term explains a corresponding percentage of the later MLP11 interaction or task behavior.

As before, removing an early interaction only at the final position has little effect on MLP11. The strongest tested removal is Attention9 for old wording (remaining target norm 0.806), and Attention5 for new wording (0.839; Attention9 leaves 0.850). Rankings are exploratory on the same panel. These effects are conditional interventions, not additive attributions; earlier body positions remain untouched and can carry or regenerate interaction.

Updated interpretation: the original effect contained a large boundary-token component, but interaction also arises through the model's bilinear computation under aligned token IDs. Thus it is too strong to dismiss the entire result as tokenization, and too strong to interpret it as an abstract prefix feature. We now have an explicit local interaction term whose causal downstream role can be tested.

Next capture aligned body positions and subtract the MLP0 bilinear cross term or full interaction across those positions, with first-token/final-token/all-body contrasts and norm-matched controls. Leave prefix-only positions explicit rather than inventing unmatched counterparts. This tests whether early multiplicative formation is carried from earlier body positions to the later attention writers, addressing the main limitation of final-position-only removals.

Audits: all 64 aligned bodies exactly match IDs; all eight competence cells perfect; old/new manifests and runner unchanged; exact identity replay; final embedding interaction zero; 1,504 interventions finite. The native float32 residual accounting has maximum relative discrepancy 2.55e-4, larger than H608 because the controlled interaction is smaller. A float64 local MLP decomposition verifies the identity within 1.56e-12. Its replay of the native float32 four-way MLP0 difference differs by at most 0.349% (old wording) / 0.172% (new wording), consistent with cancellation sensitivity; local norms agree closely. An initial manifest-audit variable collision was corrected and the full run repeated before recording this result. No new natural-text documents; FineWeb remains 26,112. Broad goal active and incomplete.
