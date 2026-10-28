## H439 — Exact compilation succeeds; weight-energy compression loses coverage

H438 transferred frozen shared source directions but still constructed a dense context-specific omitted operator. H439 compiles each of the16 fixed directions H into a fixed weight map T_H = D_omit[diag(L_omit H)R_omit + diag(R_omit H)L_omit]. For normalized baseline x=p/s, the omitted removal-direction write equals -(T_H x - 2 M_omit(x)*mean(x*H))/s. The second term is the RMS radial correction. The identity retains native float32 RMS epsilon in double arithmetic.

Exact compiled predicted responses replay H438 shared16 within4.3e-15 relative error (frozen tolerance1e-9); scored errors replay within2.39e-17 absolute difference. Thus new context-dependent writes can be computed using fixed maps rather than reconstructing factor Jacobians. This identity alone does not reduce storage or original model inference cost.

Weight-only compression chooses a shared output basis from the eigensystem of sum_H T_H*T_H^T, without activation or finite-target fitting. Test output ranks16,32,64,128,256 on reused H438 data. Project either both the linear-map and radial terms, or just the linear maps while keeping the radial correction exact. No fresh confirmation or new scientific significance gates; no new documents, total12480.

| Output rank | Weight-map energy retained | Both terms projected: low / median KL recovery | Linear maps projected, radial exact: low / median KL recovery | Map+basis scalar count |
|---|---:|---|---|---:|
|16|27.92%|83.26% / 79.36%|90.20% / 93.59%|313,344|
|32|34.60%|84.51% / 81.18%|91.62% / 94.81%|626,688|
|64|44.24%|87.66% / 83.34%|93.46% / 95.89%|1,253,376|
|128|58.00%|90.26% / 85.94%|94.84% / 96.76%|2,506,752|
|256|75.32%|93.88% / 88.30%|96.74% / 97.83%|5,013,504|

Exact compiled shared16 finite-KL recovery is98.21% low/99.41% median. Dense16 maps store21,233,664 scalars, exceeding the15,482,880 scalars in the three original omitted-factor matrices. Rank64 compressed maps plus output basis use1,253,376 scalars, but their best tested radial-exact recovery is93.46%/95.89%. Rank256 uses5,013,504 scalars and reaches96.74%/97.83%. These counts exclude baseline model, retained-factor weights, shared source transforms and temporary workspace. The maps represent additional intervention prediction conditional on a baseline, not replacement of the original model forward pass.

| Post-primary ablation | Low-RMS finite-KL recovery | Median-RMS finite-KL recovery |
|---|---:|---:|
|radial_only|83.722%|83.875%|
|linear_maps_only|90.184%|76.522%|
|full_compiled_cached_radial|98.208%|99.411%|

Neither radial correction nor fixed linear maps alone reproduce the full correction. In median contexts, the linear-map term alone barely improves union128, while adding the radial term raises recovery to99.41%. Therefore the computational account must retain their combination; separate terms are not independent semantic features.

A separate cached-baseline audit obtains the omitted baseline write from the already-computed native full MLP write minus retained baseline factors. It changes predicted corrections by at most2.84e-06 relatively and preserves the aggregate finite-KL scores to roughly1e-8. This supports baseline-write reuse at the tested precision. The primary exact identity used the double omitted baseline directly; the cache result is explicitly a subsequent audit, not an undeclared change to that numerical gate.

Construction and runtime accounting: fixed maps can be precomputed once. Compressed linear-map evaluation costs about16*output_rank*1152 multiply-accumulates to construct all16 writes at a context, plus output reconstruction and radial terms. A single source can first combine its16 map coefficients, changing the amortization tradeoff but not eliminating map storage. The experiment has not benchmarked end-to-end speed. The original baseline forward pass remains necessary, and a compressed map can trade accuracy for fewer additional operations without explaining ordinary inference by itself.

Interpretation: fixed source directions support an exact reusable computational description, but a naive weight-energy output basis does not retain the original behavioral fidelity at small ranks. That is informative negative evidence against equating weight compression with behavioral compression. It does not establish that no small compiled representation exists.

Next compare a baseline-activity-derived output basis with the weight-only basis at identical ranks and storage. Construct the activity basis from the older H434 calibration states and fixed T_H maps, without finite target errors, then evaluate on H438 transfer states. Preserve the exact radial correction. This tests whether the missing efficiency comes from optimizing over all possible input directions rather than those visited by the model. Because H438 outcomes are already inspected, this comparison is a mechanism diagnostic; any selected design needs another independent confirmation.
