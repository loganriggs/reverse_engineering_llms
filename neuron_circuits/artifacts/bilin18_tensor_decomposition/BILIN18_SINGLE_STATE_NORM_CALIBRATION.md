## H857 — A consistent norm-aware encoder improves most relay effects but worsens natural local fidelity

**Main result.** The pairwise-oracle norm benefit partly survives in a genuine single-input encoding function. With no task directions or gain fitting, residual-norm retention improves answer-effect error in57/60 aggregate model/task/gap cells. But full-output task KL improves in only46/60 for learned coefficients (43/60 for LS), and natural-text reconstruction/local-MLP fidelity worsens for every model. This is a tradeoff, not a universally improved featurizer.

**Single-state rule.** Let C be token-centered normalized input, beta the decoder bias, and C_hat its learned or fixed-support least-squares reconstruction. Define:

`C_hat_norm = beta + (C_hat-beta) * ||C-beta|| / ||C_hat-beta||`.

Add the training-token mean, normalize the fullshape and restore that input's own rawRMS to define one raw reconstruction F(input). Form interventions as native recipient + F(donor)-F(recipient). This preserves recipient reconstruction error and is cycle-consistent by construction. It requires one additional residual-norm scalar beyond token identity and rawRMS already accessed. Gates, dictionaries and supports remain frozen. Bias is retained; zero reconstruction-norm cases would fail explicitly rather than receive an arbitrary direction (none occurred).

This differs fromH853/H854, which normalized after adding the decoded donor difference to the native recipient shape. Compare calibrated and uncalibrated variants withinH857; improvements relative to older numerical tables cannot all be attributed to the norm rule.

**Adjacent-prompt learned-coefficient examples.** Values describe preservation of the full raw relay effect, not native task accuracy. Full tables retain both seeds, all architectures/gaps, and independent LS variants.

|Task|Model|Mean full relay effect preserved, before→after|Effect relative RMSE, before→after|
|---|---|---:|---:|
|agreement|scalar48_8351|72.3%→85.4%|0.327→0.215|
|agreement|scalar48_8352|86.7%→95.5%|0.187→0.125|
|agreement|block8_8351|74.0%→88.9%|0.347→0.216|
|agreement|shared_fixed_8351|30.9%→50.3%|0.716→0.560|
|agreement|shared_learned_8352|31.4%→49.3%|0.732→0.592|
|sentiment|scalar48_8351|86.9%→98.4%|0.191→0.127|
|sentiment|scalar48_8352|63.4%→72.1%|0.457→0.422|
|sentiment|block8_8351|78.9%→89.6%|0.278→0.215|
|sentiment|shared_fixed_8351|62.4%→76.0%|0.453→0.426|
|sentiment|shared_learned_8352|8.7%→18.7%|0.912→0.825|

The60 aggregate cells share the same small lexical panel and are not60 independent replications. Some cells worsen; ratios above100% can represent overshoot. OutputKL must be checked separately from the task answer contrast. Model/seed variation remains material, and learnedshared still performs poorly despite scaling.

**Natural-data tradeoff.** On the fixed512 H851 development states, every calibrated variant worsens Euclidean residual reconstruction and four-probe localMLP derivative prediction relative to its corresponding uncalibrated variant. Examples:

- Scalar48 seed8351: residual relativeMSE .468→.543, localresponse relativeRMSE .654→.683.
- Block8 seed8351: .529→.632 and .704→.742.
- Fixedshared seed8351: .706→.927 and .809→.878.
- Learnedshared seed8351: .741→.988 and .835→.924.

Independent LS improves natural reconstruction/localresponse before calibration, while giving mixed task effects. Norm retention may amplify useful task signal and also omitted/noisy natural components. Natural local-response error is not a substitute for actual whole-output behavior; H858 directly measures natural nexttoken KL before concluding whether global behavioral fidelity worsens.

**Audits.**6,552 whole-model forwards plus local analytic-response evaluations on512 states/four frozen probes. Residual norm retained within2.14e-14; no per-state LS error increase. A raw reconstruction triangle closes within1.14e-13, consistent with oneF(input) rather than pairwise correction. Identity logits exact; full reference margins replay within1.44e-5. Maximum gain3.93 on natural states and2.58 on task states; gains not tuned/clipped using task outcomes. All hashes unchanged, metrics finite, original reserved test untouched, natural-document count26,112 unchanged. Broad goal remains active.

Next: assess full nexttoken distributions on a prospectively fixed subset of natural development occurrences with complete reconstruction replacement. Distinguish task-specific abstraction, local operator fidelity and whole-output KL/CE before choosing an objective or endorsing the calibrated encoding.
