## H858 — Worse activation/local-operator fidelity can coexist with better whole-output KL

**Main result.** On a prospectively fixed64-example natural-text development subset, residual-norm calibration lowers mean nexttoken KL for every one of the10 models, with both learned and LS coefficients. This reverses the inference one might have made from H857's uniformly worse reconstruction/local-response metrics. The evidence favors evaluating behavioral fidelity directly, while remaining preliminary because this is a small reused development panel and many paired intervals include zero.

**Protocol.** First64 H851 development occurrences, fixed before full-output outcomes. Truncate each original token window after its selected position; causal attention makes this the same prefix context. Replace the complete rawpreMLP4 state at that position with F(native input) for everyH857 variant. Compare the full nexttoken distribution with the unmodified model using KL(native||replacement) and total variation. This is full reconstruction replacement, not a donor-difference task intervention. All64 retained forKL; nexttokenCE is available only where the original128-token window includes the following token.

|Model|Learned meanKL|Learned+norm meanKL|LS meanKL|LS+norm meanKL|
|---|---:|---:|---:|---:|
|block8_8351|0.0862|0.0788|0.0772|0.0715|
|block8_8352|0.0845|0.0797|0.0814|0.0767|
|scalar48_8351|0.0534|0.0460|0.0529|0.0463|
|scalar48_8352|0.0474|0.0430|0.0474|0.0419|
|scalar64_8351|0.0511|0.0457|0.0502|0.0452|
|scalar64_8352|0.0646|0.0592|0.0626|0.0595|
|shared_fixed_8351|0.1637|0.1316|0.1307|0.1087|
|shared_fixed_8352|0.1887|0.1730|0.1571|0.1488|
|shared_learned_8351|0.1931|0.1645|0.1819|0.1593|
|shared_learned_8352|0.1959|0.1562|0.1892|0.1546|

Norm retention improves all20 aggregate mean comparisons, but not every document. Depending on model/variant, about45–72% of documents improve. Paired64-document bootstrap intervals are saved and uncorrected; many include zero. For example learned scalar48 seed8351 meanKL change is-.00741 with interval[-.01299,-.00255], while block8 seed8352 is-.00483 with interval[-.01969,.01177]. Mean improvements alone do not establish robust universal superiority.

NexttokenCE changes are available on56 examples per variant. They can have either sign, including occasional improvement relative to the original model. Such negative CE changes on this tiny subset are not evidence of a better language model, and CE is not a substitute for faithfulness to the original output distribution. Both metrics remain distinct in the records.

**Research implication.** We now have an implementable single-state modification with promising task and natural whole-output behavior, even though it increases activation error and localMLP response error. Thus neither reconstructionMSE nor the chosen local derivative metric is sufficient to rank representations for end-to-end fidelity. It does not follow that larger feature norms are always useful, that the learned groups are meaningful, or that all dimensions discarded by the representation are irrelevant. The fixed/learnedshared models remain substantially less faithful than scalars under these fits, even after calibration.

The relevant notion of a good representation must specify the purpose: activation reconstruction, local operator approximation, original-output KL, and ground-truthCE are different objectives. This supports the user's data-based/function-based canonical-form direction, but we have not established uniqueness, a minimal canonical form, a semantic manifold, or general reusable computation units.

**Next confirmation.** Freeze the per-state rule and all40 variant checkpoints/prediction procedures. Use a document-disjoint natural panel not used in H851/H852 fitting or selection, preferably the already trackedH837 pool with a clear scoped-freshness statement. Evaluate direct whole-outputKL/TV and paired uncertainty without tuning gains or selecting a seed. Retain uncalibrated and LS controls, token-coverage strata, and individual harms. Only then consider retraining with a different objective or using the calibrated featurizer as the default for manifold interpretation.

**Audits.**2,688 full forwards. Prefix-truncated native preMLP capture differs by at most.00391 in raw units; identity replacement changes logits by at most1.77e-5, validating numerical equivalence at the behavioral level. Source/model hashes unchanged, metrics finite; no reserved-test use; natural-document pool26,112 unchanged. This turn added6,552 task forwards/local diagnostics inH857 plus2,688 natural-output forwards here, and demonstrated why local proxy metrics cannot settle the research direction. Broad goal remains active.
