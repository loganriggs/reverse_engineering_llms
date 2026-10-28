

## H305 — Head-specific output spaces remain broad and transfer poorly

Reuse H304 saved uncentered moments of missed cross responses after H300 refined64. Fit a separate output eigenbasis to each of18 training groups (768 documents), evaluate on the corresponding256-document validation group, and compare with the shared equal-group basis. Also diagonalize each validation moment solely to compute the exact in-sample optimal rank-k projection capture. This last quantity is an optimistic empirical upper bound for ANY rank-k orthogonal output projection on that group, not a trained predictor or transferable estimate. Source moment-file hash locked and checked. No new model execution or fresh documents.

| Rank | Shared validation mean | Specialist validation mean | Specialist training mean | In-sample validation optimum mean |
|---|---:|---:|---:|---:|
|8|2.64%|3.32%|8.86%|15.16%|
|16|4.87%|5.83%|14.50%|23.64%|
|32|8.51%|10.10%|23.28%|35.56%|
|64|14.92%|17.11%|36.19%|51.03%|
|128|25.93%|28.42%|53.36%|68.95%|
|256|43.89%|45.65%|72.97%|86.30%|

All four locked gates FAIL: specialists at ranks32/128 do not capture80% in every validation group; specialization improves rank32 mean by only1.59pp, below10pp; even the in-sample validation rank128 optimum does not reach80% in every group.

At rank32, specialist validation capture ranges5.84–15.06%; its in-sample validation optimum ranges21.35–60.30%. No group reaches80% even with directions fitted directly to validation. The empirical optimal ranks required for80% are81–269 across groups. At rank128 the in-sample optimum mean is68.95%, while training-fitted specialists capture28.42%. At rank256 these are86.30% and45.65%. The shared basis remains competitive and has a higher worst-group capture than specialists at the reported32/128/256 ranks.

**Inference.** The broad shared output spectrum is not explained merely by pooling18 different compact rank32 subspaces. Broadness remains within individual head/payload populations, at least for fixed linear output subspaces on these samples. Head identity alone therefore does not select an adequate compact output chart. This says nothing decisive about nonlinear, token/context-conditioned, or compositional units. It concerns the residual after approximations have already removed other structure.

There is also a large estimation/context gap: training-selected directions capture much less validation energy than directions fitted on validation itself. That gap is compatible with high-dimensional estimation noise, context-dependent directions, unequal document populations, and finite-sample optimism. Training uses three times as many documents as validation, so their in-sample spectra are not directly comparable evidence of population shift. The validation eigenspectrum is explicitly not held-out performance. No irreducible population rank or semantic ontology follows from this comparison.

**Checks.** Eigenvalue sums reproduce validation covariance traces to relative<1e-10; every training-selected validation capture lies below its corresponding empirical optimal bound. These checks supplement H304 moment/projection audits. Sources and eigenbasis arrays are saved. All jobs terminal; native model/compiler/base unchanged.

**Next discriminator.** Use equal-sized independent document blocks to compare in-sample output spectra and cross-block transfer, then a nested training-size curve for output PCA. This separates sample-size optimism from consistently weak transfer. Evaluate shared and per-head directions with the same sample counts. If transfer rises strongly with data, estimation is implicated; if in-sample compactness remains poor, a fixed small linear output space remains inadequate; if compact directions transfer only within identifiable contexts, test context-conditioned charts. Avoid another optimizer sweep until this distinction is clearer. Broad research goal remains active.
