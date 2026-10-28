## H744 — Frozen operation edits transfer to new names, less completely to new wording

A four-dimensional subspace in each of the second query and key vectors was frozen from H742. It approximates the edit between binding (identify the other participant) and copying (repeat the mentioned identity) at one source-to-destination edge of L13H0. These are supervised paired-edit subspaces, not unsupervised features or established manifolds. The full query/key-factor exchange is the reference intervention; all other edges and the first attention factor stay native.

The new panel crosses old/new names with old/new wording, in narrative and list formats. No basis or rank was refitted. Of 128 prompt pairs, eight involving Harold/Doris failed the token-validity or endpoint-length rule, leaving 120 pairs and 240 contexts. All 120 binding/copy endpoint matches had identical source/final tokens and positions. Every valid context was retained; comparisons require both task cells to pass the preregistered native-competence gate.

| Names / wording | Format | Downstream edit error | Random rank-4 error | Eligible sign transfers |
|---|---|---:|---:|---:|
| Old / old (replay, includes fit names) | Narrative | 5.9% | 99.4% | 32/32 |
| Old / old (replay, includes fit names) | List | 7.4% | 96.4% | 30/30 |
| New / old | Narrative | 12.4% | 99.4% | 24/24 |
| New / old | List | 8.7% | 96.1% | 26/26 |
| Old / new | Narrative | 11.4% | 99.5% | 18/22 |
| Old / new | List | 30.1% | 95.7% | 31/32 |
| New / new — diagnostic | Narrative | 16.1% | 99.2% | 18/24 |
| New / new — diagnostic | List | 33.8% | 95.8% | 22/26 |

Error is RMS difference from the full-factor intervention, normalized by its RMS change from native, for the Kevin-versus-Brian absent-value contrast. Random results pool three fixed seeds. Sign-transfer denominators include only cases where full-factor exchange reverses the native contrast; error includes all contexts. Observations share templates/names and are not independent statistical replications. Original-value intervention errors follow the same pattern: new names 12.3%/8.8%, new wording 10.4%/33.6%, narrative/list respectively.

Both new-name/new-wording binding cells fail native competence: narrative full-vocabulary top-1 42.9% <50%; list both-endpoint choice 71.4% <75%. They remain diagnostic. All other cells pass, although old-name/new-wording narrative binding is exactly at the 50% top-1 threshold. Native coefficient sign is no longer a universal operation label: some new-name binding and new-wording copying cases violate the original sign tendency despite aggregate task competence.

Interpretation: compact paired edits reuse substantial causal structure across names and some wording changes, substantially outperforming equal-rank random controls. Wording-dependent residuals and exceptions weaken the claim of a universal operation manifold. This result concerns a single surgically edited attention edge, not whole-model task switching or overall accuracy.

Verification: 5,280 model forwards; all frozen inputs/source hashes unchanged; native head factorization relative error 5.83e-7; old-panel coefficient replay error 5.84e-7; native-value identity error 4.77e-6; zero contrast exactly zero; finite outputs. No original reserved-test evaluations. Natural document pool unchanged at 26,112.

Next: decompose fresh-panel coefficient error into omitted query, omitted key, their bilinear interaction, and norm-restoration effects without fitting to these evaluation cases.
