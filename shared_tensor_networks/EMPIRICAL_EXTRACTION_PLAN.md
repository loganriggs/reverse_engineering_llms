# Empirical activation metric for hidden extraction

Fixed before evaluation. Reuse all50models and the same four held-out heads/noisy adaptation labels. No source retraining, no new target labels.

For each model compare the existing quadratic-consensus rank4→10products and quartic-union rank10extractions using three empirical metrics: the512source-training inputs,512fresh unlabeled Gaussian inputs, and4096fresh unlabeled inputs (nested prefix). Fresh pools use teacher_seed+7000and are shared across model methods/initializations. All column norms and leading singular subspaces come from evaluated activations on these pools; no analytic Gaussian Gram matrix or teacher directions select features. Exact Gaussian moments are used only for final evaluation.

Ranks remain fixed at4/10from the shared-architecture budget. Use uncentered SVD and the existing minimum-norm adaptation solver with32/128labels/head. Compare all outcomes to the previous population-metric extraction and original source-output baseline. Keep every failed original model. Do not tune rank, sample size or regularization using held-out performance.

Total300extracted representations (50models×3pools×2methods), each with population projection and two finite adaptation evaluations. The4096fresh pool adds unlabeled observations, not supervision; the source512condition adds neither. This tests removal of the analytic-metric assumption, not distribution-free or native-LLM feature discovery.
