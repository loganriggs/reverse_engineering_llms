## H590 — Active suppression and differential amplification both contribute to the frame gap

The previous goal turn made progress by separating propagated response from terminal sensitivity. Capture all downstream attention/MLP final-position writes after the fixed Attention5 perturbation, including MLP5. At +/-0.0625 and +/-0.5 discovery amplitudes, reconstruct the final central residual response exactly as the scaled direct injected path plus 25 scaled write responses. Residual gains are products of subsequent block skip coefficients.

For a common scalar comparison, map each term through the matched unfronted prompt's native terminal-readout gradient. This is a local descriptive chart, not a causal percentage. The exact vector reconstruction and chart's finite-difference error are audited separately. Rank components by the most negative fronted-minus-unfronted mean chart contribution at small dose on regular nouns only. This selects MLP11, MLP17 and MLP9. Lock that selection before causal tests.

Then overwrite each selected operation's final-position output with its same-prompt unperturbed value while preserving the original Attention5 perturbation. Test each alone, jointly, every downstream write clamped, and an unchanged control on all three frames, both noun types and both doses. There are 4,608 intervention forwards, plus 960 capture forwards. Irregular nouns validate the selection on an already-explored panel; no untouched population claim.

| Type / clamp, small dose | Unfronted response | Relative-clause response | Fronted response | Fronted minus unfronted |
|---|---:|---:|---:|---:|
|regular/unchanged|1.2763|1.2251|0.6522|-0.6241|
|regular/mlp_11|1.3427|1.3010|0.8421|-0.5006|
|regular/mlp_17|1.0546|1.1492|0.5058|-0.5488|
|regular/mlp_9|0.9130|0.8878|0.4934|-0.4196|
|regular/joint|0.7946|0.9155|0.5095|-0.2851|
|regular/all|0.1524|0.1930|0.1391|-0.0132|
|irregular/unchanged|1.2786|1.0620|0.6711|-0.6075|
|irregular/mlp_11|1.3531|1.2364|0.8901|-0.4630|
|irregular/mlp_17|1.0748|0.9459|0.5384|-0.5364|
|irregular/mlp_9|0.9524|0.7891|0.5530|-0.3994|
|irregular/joint|0.8590|0.8590|0.6092|-0.2498|
|irregular/all|0.1650|0.1959|0.1598|-0.0051|

MLP11 behaves as an active suppressor under this intervention: clamping it increases the response, especially in fronted prompts. In irregular nouns, fronted response rises 0.671 to 0.890, while unfronted rises 1.279 to 1.353. MLP9 differs: clamping it reduces both responses, particularly unfronted (1.279 to 0.952 versus 0.671 to 0.553). MLP17 also supplies more amplification in unfronted contexts. Thus the frame gap combines active suppression and differential amplification; calling every negative frame contrast a suppressor would be incorrect.

The selected joint clamp reduces the small-dose gap from -0.624 to -0.285 on regular nouns and -0.607 to -0.250 on irregular nouns. The effect persists at half dose. A substantial gap remains, and the changes are not additive independent shares. Intervening on an earlier write changes later responses.

Clamping all downstream responses leaves only a small direct-path response (~0.14–0.20) and nearly removes the fronted gap. This is a strong algebra/control condition, not a natural computation or a claim that these operations can safely be deleted. Relative to the direct path, the network's net response is amplifying in all frames. “Fronted attenuation” describes a contrast with the unfronted frame, not overall attenuation below the direct path.

Next H591 examines MLP11 specifically using its exact bilinear response. Cross the recipient versus matched unfronted native MLP input with the recipient versus unfronted incoming perturbation. Replace only the resulting response around the recipient native output. This distinguishes context-conditioned local transformation from the form of the incoming perturbation. All other downstream computation remains live.

Audits: final response reconstruction max relative error 3.63e-5 (max coordinate rounding error 0.126); all-clamp direct-path response max relative error 4.91e-4; unchanged replay <=1.44e-5 margin. Mean absolute terminal-chart linearization error is 0.000218 at small dose and 0.01196 at half dose; exact residual accounting is separate from this approximation. Source/basis/selection unchanged; finite metrics. No new natural-text documents; FineWeb remains 26,112. Broad goal active and incomplete.
