

## H307 — Gaussian sampling null partly reproduces gaps, not capture levels

For each18 group use H304 training uncentered covariance eigenvalues as a plug-in population spectrum, normalized trace1. Generate independent zero-mean Gaussian fitting/evaluation samples in that eigenbasis. Compare N1024 (four positions×256 documents treated independent) and N256 (dependence sensitivity, not an estimated effective sample size), three replicates, fixed seed20270131. Fit output PCA at ranks32/128, measure fitting capture, independent-test capture, and exact population capture from the known diagonal covariance. Observed reference is H306 equal-size within/cross-block specialist means. No new native-model execution.

| Rank | Condition | Within-sample capture | Transfer capture | Gap |
|---|---|---:|---:|---:|
|32|Observed|34.74%|7.87%|26.87pp|
|32|Gaussian N1024|27.41%|20.26%|7.15pp|
|32|Gaussian N256|37.63%|16.53%|21.10pp|
|128|Observed|68.62%|23.94%|44.68pp|
|128|Gaussian N1024|62.51%|46.59%|15.92pp|
|128|Gaussian N256|82.34%|38.43%|43.91pp|

Locked excess-gap gate FAILS: the observed gap does not exceed the largest replicate mean null gap by5pp at both ranks and both sample counts. Rank32 observed26.87pp exceeds the N256 maximum21.22pp by5.65pp, but rank128 observed44.68pp exceeds its N256 maximum43.99pp by only.69pp. Three replicate ranges are simulation diagnostics, not calibrated confidence intervals or p-values.

**Interpretation.** Under N1024, Gaussian sampling optimism is far smaller than observed. Under N256, the rank128 gap is almost reproduced, so a large gap alone cannot identify context-specific structure. Yet N256 predicts rank128 transfer38.43% versus observed23.94%, and within-sample82.34% versus68.62%. It also predicts rank32 transfer16.53% versus7.87%. Matching one difference does not match the joint evidence. Neither null simultaneously reproduces the observed capture levels at both ranks.

The Gaussian population covariance is itself estimated from data, contains sampling noise, and is an uncentered moment treated as a zero-mean covariance. It omits actual response means, non-Gaussian tails, document/site correlations and heterogeneous scales. N256 does not estimate those correlations. Thus the comparison diagnoses sensitivity to sampling assumptions; it neither rejects all sampling explanations nor proves semantic/context-driven variation.

**Checks.** Source covariance hash verified. Mean simulated sample trace agrees with normalized population trace1 within2%; reported gap identities and capture ranges verified. Independent-test captures agree closely with exact population captures averaged over the same learned bases. No selection of effective sample size to declare success; both locked cases are reported. All jobs terminal.

**Next discriminator.** Preserve empirical response vectors and four-position document clusters, then randomly reassign whole documents to equal-size blocks. Compare the original block statistics with these empirical repartitions. This tests whether block membership contributes beyond observed non-Gaussian response variation and within-document dependence. It does not identify semantic contexts by itself; residual approximations already used original discovery data, and permutation exchangeability must remain a stated assumption. No additional optimizer sweep is justified yet. Broad research goal active.
