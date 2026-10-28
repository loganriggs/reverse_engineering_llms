## H850 — The compact relay coordinate is assembled by multiple writers and is not a complete causal state

**Main result.** Discovery-selected writers account for much of the frozen relay coordinate on the new lexical/distance panel, and intervening on them recovers much of the relay's answer effect. No single writer supplies the whole coordinate. The result supports a distributed construction of a compact downstream direction, rather than identifying one neuron/head as the entire feature.

**Exact native accounting.** At the relay token, decompose raw preMLP4 into attention headsL0–4 and MLP writesL0–3, each scaled by subsequent residual-mixing coefficients. This yields49 writer terms. The original embedding contribution cancels in each donor-recipient difference because the relay token is identical. Head contributions use actual native head outputs and output-projection slices. This is an algebraic decomposition of native differences, not an intervention-based attribution of computation.

For each task, rank writers by RMS contribution along the frozen H848 direction on only the8 discovery pairs. Freeze the top3 before examining H849 writer interventions:

- Agreement: L4H5, L3MLP, L2MLP.
- Sentiment: L4H7, L3MLP, L4H6.

On the H849 panel, projection slopes against the total coordinate difference are respectively0.406/0.240/0.097 for agreement and0.326/0.252/0.192 for sentiment. These slopes quantify native accounting, not independent causal percentages. A shared whole MLP module does not establish that both tasks reuse the same internal computation or feature.

**Writer interventions.** At each selected writer's original output site, add the native donor-minus-recipient vector at the relay position. In a separate axial arm, add only its projection onto the task's frozen direction. Run each writer separately and all3 jointly. Downstream modules recompute naturally; multi-writer edits add native deltas to the already perturbed execution, so they are not equivalent to replacing every intermediate output with its donor value. All78 H849 pairs, both directions, retained. The panel is fresh relative to basis fitting, but already used for direction validation; writer results are exploratory follow-up.

|Task|Gap variant|Top3 joint edit|Native answer gap transferred|Relative to direct rank1 relay effect|Coordinate relative RMSE|
|---|---:|---|---:|---:|---:|
|agreement|0|full|25.5%|82.5%|0.084|
|agreement|0|axial|22.6%|73.2%|0.165|
|agreement|1|full|9.8%|95.2%|0.194|
|agreement|1|axial|6.6%|63.9%|0.237|
|agreement|2|full|1.7%|73.4%|0.277|
|agreement|2|axial|1.3%|59.4%|0.319|
|sentiment|0|full|30.7%|87.3%|0.190|
|sentiment|0|axial|24.9%|71.0%|0.227|
|sentiment|1|full|21.0%|82.1%|0.212|
|sentiment|1|axial|17.3%|67.4%|0.238|
|sentiment|2|full|24.0%|80.5%|0.218|
|sentiment|2|axial|21.1%|70.9%|0.240|

For sentiment, the top3 full-vector edits reproduce about80–87% of the direct rank1 relay effect across gap variants; axial-only edits reproduce67–71%. The strongest single head,L4H7, produces only about28–32% of that direct answer effect. Agreement's top3 full edits reproduce73–95%, with stronger gap dependence.

**A limit on the scalar explanation.** ForL4H7 alone, full-vector and axial edits produce the same preMLP4 coordinate change up to numerical effects, since they differ only in the orthogonal component immediately before that site. Yet their downstream answer effects differ: at gap0, full-vector head edit transfers11.3% of the native answer gap while axial transfers7.3%. AgreementL4H5 shows a similar contrast (11.4% versus3.6%). Therefore the single coordinate is not sufficient to predict every intervention's output. It summarizes most of the tested *full donor-state contrast*, but off-axis state and downstream interactions can matter for partial-writer interventions. This prevents promoting rank1 compression into an unconditional one-dimensional causal model.

**Numerical audit and repair.** The initial absolute1e-4 decomposition threshold failed: raw-state differences have norms roughly3,511–10,184, and headwise float32 summation differs by up to4.88e-4. The weighted reconstructed difference has max absolute error7.70e-4, but max relative vector error7.48e-7 and error along the studied coordinate9.68e-8. We preserved the failed absolute-audit result and reran with explicit scale-aware1e-5 relative reconstruction/coordinate thresholds, while retaining an absolute1e-4 identity-logit threshold. Head-sum relative error2.44e-7; zero-edit logit error1.91e-5; fresh native logits matchH849 within3.44e-5. This was a tolerance/scale issue, not evidence that exact model algebra failed.

The validated run used1,441 full forwards. Three completed preliminary audit runs also used1,441 each (one intended diagnostic edit did not execute because the shell lacked a `python` alias); total work for this experiment was5,764 forwards. Logs and failed diagnostic JSON retained. Sources/checkpoints unchanged, all metrics finite, original reserved test untouched, natural-document count26,112 unchanged.

**Next work tied to featurizer validation.** We now have a concrete benchmark at an exact site: two frozen causal directions and known partial writers, with an explicit counterexample to scalar sufficiency. Capture natural-data preMLP4 states using the existing training/development partitions, fit task-blind scalar/block/shared models at that site, and test whether they recover these directions and preserve the frozen causal contrasts. Keep task directions out of fitting and use fixed-subspace baselines. Distinguish mere decoder-span alignment, actual encoder activation, donor-difference fidelity and causal-effect fidelity. Complementary mechanistic work should splitL4H7's projected write into source positions, signed routing and local/persistent values; do not infer that native projection slopes alone identify an abstract operation.

Previous turn was progress: independent lexical/gap validation produced frozen directions. This turn identifies their contributing writers and a substantive boundary on the one-dimensional interpretation. The broad feature/computation goal remains active.
