## H818 — Frozen fresh lexical panel is ready for copying; IOI falls below the item gate

Before native evaluation, freeze the H815 dictionary file hash, memberships, geometry gate, layer/head, response arms, amplitudes, and eligibility rules in configs/bilin18_fresh_representation_locked.json. Plan 32 new copying values and 32 new IOI name pairs. Use the primary bars copying and market/parcel IOI surfaces without template selection. No dictionary/gate training, intervention-based exclusion, or correctness-based item removal occurs.

### Readiness results

| Task | Planned items | Aligned complete items | Candidate accuracy | Full-vocabulary accuracy | Eligible |
|---|---:|---:|---:|---:|---|
| Repeated-value copying | 32 | 32 | 100% | 99.22% | Yes |
| Indirect-object completion | 32 pairs | 7 pairs | 100% | 100% | No: fewer than eight items |

Copying uses book, pen, key, door, wall, roof, floor, window, garden, field, road, bridge, river, mountain, ocean, lake, stone, sand, grass, tree, leaf, flower, cloud, rain, snow, wind, fire, water, earth, sun, moon, star. All four same/different-key and query cells align per item. Candidate correctness uses the fixed train foil and is a weak check; full-vocabulary accuracy is therefore reported. This task still tests repeated-content completion rather than discriminating two different stored values.

IOI retains planned pair indices 0,2,3,7,10,12,15. Complete four-cell alignment includes both initial name orders and giver choices. Other pairs fail source/answer single-token validity or paired-query length/position alignment; the JSON records both context-level and item-level reasons, which are not additive exclusion counts. All retained pairs correctly discriminate both giver choices, but this does not override the frozen minimum-eight-item rule. No pairs were added after inspecting model outcomes.

None of the new value words or names overlaps the prior response panels used for representation fitting/evaluation. This is relative lexical novelty, not a claim that the words have never appeared anywhere in the research project or model training. Query keys, syntactic forms, selected layers, and intervention design are reused.

### Hypothesis refresh before larger causal validation

The recent evidence favors useful low-dimensional response structure but does not yet establish stable semantic groups. The following alternatives guide information-gaining tests rather than a single favored narrative:

1. Shared directions carry a broadly reused response component; frozen shared ablation should harm both groups on fresh items.
2. The shared directions mainly capture one source perturbation family, so reuse may disappear for a different head or source location.
3. Private blocks contain additional causal content only in some contexts; deletion effects should vary by group.
4. The geometry gate is locally suboptimal for KL; alternative-private advantages should replicate within particular fresh groups.
5. Apparent gate failures are single-token metric artifacts; full-KL ordering can disagree, as already seen in IOI.
6. One small held-out lexical subset drove the copying-group1 anomaly; fresh aggregate effects may remove it.
7. Groups track lexical identity rather than computation; assignments may follow new word categories without preserving causal behavior.
8. Groups track perturbation amplitude or local-state transplantation; gates may change within the same lexical item.
9. Multiple nearby fitting minima describe equivalent response spans with unstable labels, making labels less identifiable than projections.
10. Shared/private subspace rotations obscure unique atom identities even when decoding and causal replay are stable.
11. The fixed shared PCA block captures high variance but misses lower-variance causally important directions.
12. Jointly learned shared atoms could improve over the staged PCA-plus-private fit; the present result does not test that architecture fully.
13. Additional storage explains improvements over active-dimension-matched flat bases, rather than grouping itself.
14. Causal error depends on the local baseline and RMS geometry, so response-only gates omit useful conditioning variables.
15. Small total IOI output error reflects bypass paths rather than a faithful L6 representation; zero-module normalization remains essential.
16. Positive-dose response compression may fail for negative directions; symmetry is untested.
17. The finite response family is locally compressible without forming a globally meaningful activation manifold.
18. More than one simultaneously active group may be needed; current causal evaluation uses one, despite correct union algebra.
19. Task-dependent readers transform shared content differently; weak cross-task span transfer need not imply no reusable content.
20. The two-task benchmark is too narrow to support a computational ontology; new competent operations may require different units.

Priorities are the fresh replication in 1/3/4/6, while retaining the metric distinction in 5 and the bypass control in 15. Do not add hierarchy or a DAG solely to interpret unstable geometric labels. Larger fresh causal validation should precede further architecture selection on the old three-item holdout.

### Verification and next step

156 native model forwards: 128 copying and 28 IOI. All capture-run hashes unchanged, all outputs finite, and first-source value/routing invariance to the later query is exact. The dictionary still matches its hash locked before these evaluations. Lexical-overlap audits are empty for both tasks. Reserved test untouched; natural-document pool 26,112.

No fresh causal representation result is claimed yet. Next capture the eligible 32-item copying response family at coefficients 0/.05/.10, then replay the frozen shared/private, shared-only, private-only, zero, other-private, and full controls. Evaluate full-output KL and fixed-token responses by group without refitting or selecting a gate from fresh outcomes. IOI can be reported descriptively, but it cannot count as a successful frozen validation with seven items.
