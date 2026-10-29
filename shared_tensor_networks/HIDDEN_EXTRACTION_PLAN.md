# Hidden-feature extraction after independent fitting

No new source training. Reuse15shared,15random-independenth4,15released, and5teacher-start independent models. Preserve source and whole-head adaptation data.

Compare four fixed transfer representations:
1. Ten normalized predicted source outputs (previous endpoint).
2. All within-head hidden products:10shared or100independent features. No cross-head products.
3. Normalize every hidden quadratic by its GaussianL2norm, take the leading four left singular vectors in whitened coefficient space, then form their ten unordered products. Four is the predeclared shared-architecture width, not selected using teacher truth or held-out labels.
4. Leading ten left singular vectors of the normalized within-head quartic-product union, as a same-dimensional compression control.

All SVDs are uncentered function-space decompositions. Use the known Gaussian input distribution, a privileged synthetic convention; no held-out task labels choose extraction. Fit32/128noisy adaptation labels with the existing minimum-norm rule; also report exact population projection floors. Compare all50models, averaging initializations within teacher. The100feature union has a different adaptation dimension and should not be presented as budget matched.

Evaluate hidden-subspace alignment only afterward using teacher information. A four-dimensional quadratic subspace is not four individually identified bilinear components; SVD basis functions need not be rank-two quadratics. Retain this distinction and all failed original fits. Stop after extraction/audit, without tuning ranks or regularization based on transfer.
