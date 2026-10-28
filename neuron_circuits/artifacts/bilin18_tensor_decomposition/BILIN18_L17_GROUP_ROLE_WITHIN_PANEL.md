

## H416 — Within-panel raw and matched source derangements

Use H413512 documents as both recipients and donor pool, with fixed H410 factor groups. Six frozen document derangements preserve token position and exclude every self-donor. Raw source vectors are copied without scaling or projection; reordering them back replays the observed vector marginal bitwise. Matched controls use the same maps, then enforce the recipient actual-source norm and residual dot product inside the same head output subspace. The original recipient state remains fixed. No factor selection, fresh panel or natural full-forward rerun is involved.

One joint9 max-absolute95% band from4000 paired recipient/mapping bootstrap draws covers raw3 roles, matched3 roles, raw−actual head6, matched−actual head6, and matched−raw head6. Row/mapping resampling describes uncertainty conditional on this observed panel and mapping family; it is not a new independent population confirmation. Frozen gates require both head6 loss contrasts lower>.1; matched−raw head6 equivalence wholly within±.05; and head8/head4 role thresholds>.05/>.1 within each family. Geometry applies to the matched family; raw donors deliberately do not preserve each recipient's norm/angle.

|Role|Actual|Raw shuffle|Raw joint9 interval|Matched shuffle|Matched joint9 interval|
|---|---:|---:|---|---:|---|
|head6_core_add_cross_over_full|-0.167567|-0.232593|[-0.3239987679414689, -0.14118676173190575]|0.061980|[-0.029425617055067972, 0.1533863891544952]|
|head8_core_add_cross_over_full|0.115869|0.136296|[0.044889624399058775, 0.22770163060862195]|0.154780|[0.06337369079355745, 0.24618569700312062]|
|head4_additions_minus_core_finite_recovery|0.433501|-0.075127|[-0.1665330405117817, 0.016278965697781478]|0.372340|[0.28093368637909216, 0.46374569258865533]|

|Head6 contrast|Estimate|Joint9 interval|
|---|---:|---|
|raw−actual|-0.065026|[-0.15643201982490815, 0.026379986384655024]|
|matched−actual|0.229547|[0.1381411310614928, 0.320953137271056]|
|matched−raw|0.294573|[0.20316714778161937, 0.38597915399118254]|

Gates: {'derangements': True, 'geometry': True, 'actual_replay': True, 'raw_head6_loss10': False, 'matched_head6_loss10': True, 'head6_matching_equivalence05': False, 'raw_other_roles': False, 'matched_other_roles': True}.

|Family|Head6 Fisher energy/actual|Head8 Fisher energy/actual|Head4 finite effect/actual|
|---|---:|---:|---:|
|raw|1.077000|1.284215|0.828735|
|matched|2.531746|1.015445|0.599918|

Absolute effect ratios are descriptive. Audits: {'geometry': {'relative_norm': 4.3129710985239885e-07, 'normalized_residual_dot': 2.98190248736858e-07, 'relative_span_residual': 5.438937772413769e-07, 'native_relative_span_residual': 5.757932513006381e-07}, 'actual_arrays_replay_max_absolute': 4.85722573273506e-16, 'raw_vector_marginal_bitwise_preserved': True}. Independent array checks: {'primary_stats_replay': True, 'bijective_derangements': True, 'finite_arrays': True}. All mappings, source hashes and per-site metrics are saved.

Outcome: the raw-shuffle result reverses the previous leading interpretation. Raw within-panel natural donors retain appreciable head6 cancellation: cross/full−0.232593 with joint95% interval[−0.323999,−0.141187]. Norm/angle-matched donors give+0.061980,[−0.029426,0.153386]. The matched-minus-raw contrast is+0.294573,[0.203167,0.385979]. Thus matching materially changes the role; its±.05 equivalence gate fails. The matched-minus-actual contrast+0.229547,[0.138141,0.320953] passes the>.1 loss gate, but raw-minus-actual−0.065026,[−0.156432,0.026380] does not. Raw is not established equivalent to actual; it does retain negative alignment despite changed pairing.

This rules out breaking the particular source-recipient pairing as a sufficient explanation of cancellation loss in H414/H415. Here raw donors are drawn from exactly the recipient panel, without transformation, and the vector marginal at each token position is bitwise preserved. Cancellation persists in all six raw draws (head6 pooled cross/full ranges−0.26390 to−0.21605). No other corpus, Gaussian sampler, or natural-direction transformation is required for that persistence. H415's suggestion that pairing was the leading explanation is therefore weakened; norm/angle matching is now the leading mechanism to dissect. No prior result is deleted or relabeled.

The head4 control also changes strongly with matching. Raw added-minus-core finite recovery is−0.075127,[−0.166533,0.016279], whereas matched is+0.372340,[0.280934,0.463746], versus actual+0.433501. The raw other-role gate fails; the matched other-role gate passes. Head8 cross/full is+0.136296,[0.044890,0.227702] raw and+0.154780,[0.063374,0.246186] matched. Raw head8 remains positive, but its lower bound misses the predeclared+.05 threshold; matched clears it. These values show why a control must be described by its construction rather than simply called natural or random.

Head6 full Fisher energy is1.077× actual under raw shuffling and2.532× under matching; head8 is1.284×/1.015×, and head4 finite effect is0.829×/0.600×. All are descriptive. Among raw head6 draws averaged over mappings, the largest recipient document supplies2.619% of total Fisher energy and2.386% of summed absolute cross contributions; no observations were excluded. Numerical checks pass: bitwise raw marginal preservation, all six bijective maps without self-donors, matched geometric discrepancies below5.76e-7, and exact-source array replay within floating-point tolerance.

Hypothesis update: #13/#20 (particular source-context pairing is necessary for this role) is weakened by the untransformed marginal-preserving control. Two more precise alternatives now dominate: (a) replacing donor amplitude breaks a norm/direction correlation and changes how contexts are weighted in the pooled statistic; (b) replacing the donor angle directly changes the response geometry. They may coexist. The confirmed original source-dependent group relationships remain valid, but the mechanism behind control differences was not isolated until raw donors were included. No semantic coordination conclusion follows.

Next experiment: factorially separate amplitude and angle matching using the SAME six derangements and frozen groups. Four conditions are raw (donor norm/donor angle), norm-only (recipient norm/donor angle), angle-only (donor norm/recipient angle), and both (recipient norm/recipient angle). Score the same head6/head8 tangent metrics and head4 finite recovery difference. Raw and both endpoints already exist in H416. A particularly sharp audit is available: the local derivative is linear in source h, so norm-only scaling by a must multiply every per-site Fisher energy/cross term by a² and cannot change the per-site cross/full ratio. If its pooled ratio changes sign, that is reweighting across sites, not a per-site alignment reversal. Check this identity rather than treating every aggregate change as a new computational role. Angle-only matching can change per-site direction and therefore alignment. Freeze the contrasts and report both effects without assigning all change to one prematurely.

No fresh documents were consumed. Fresh-panel exclusion remains8896. The broad research goal remains open, and no experiment processes remain running.
