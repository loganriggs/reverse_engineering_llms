

## H344 — Allocation within locally sensitive input directions

Reuse H339 head6 edits, H341 intact-state output Fisher metric, and the fixed compiler. Compute its coefficient Jacobian J at intact normalized input x directly, without a discovery-mean split. The normalized-input sensitivity is K=JᵀMJ. Factor via W=M^(1/2)J and diagonalize WWᵀ, then recover right singular directions in the full1152-dimensional normalized-input space. These are local directions that vary by recipient, not shared feature identities. They are not restricted to the raw head source span.

For each normalized secant edit d, measure ||d||², projection energy onto the active sensitivity span, and dᵀKd. Aggregate across sites and documents before forming ratios. Sensitivity per unit edit equals active projection fraction times gain within that span. Rank uses eigenvalues >1e-10 times the largest at each site.

|Condition|Input energy|Active projection fraction|Tangent response energy|Within-active gain|Top4 sensitivity directions: response fraction|
|---|---:|---:|---:|---:|---:|
|actual|2571.3917|21.31%|1.9496|0.003557|88.46%|
|moment|2571.3916|16.05%|8.5341|0.020675|98.44%|
|donor|2571.3916|17.41%|9.0710|0.020267|98.41%|

|Control/actual|Input energy|Active projection fraction|Within-active gain|Sensitivity per unit edit|
|---|---:|---:|---:|---:|
|moment|0.99999996|0.753|5.812|4.377|
|donor|0.99999997|0.817|5.697|4.653|

Natural edits have more total projection into the sensitive span, but markedly lower gain there. This rejects the simple descriptive explanation that controls amplify responses because they enter the active span more. Allocation among directions and across contexts matters. The pooled within-active gain can change through both within-site directional alignment and weighting of sites with different spectra; this analysis does not separate them. Top4 refers to each site’s four largest sensitivity eigenvalues, not the four compiler output modes selected in H341.

Locked descriptive gates: rayleigh2: PASS, projection2: FAIL, within_active2: PASS. No fresh documents or new uncertainty intervals.

Audit: spectral response energy versus exact H342 tangent terms relative error 1.89e-14; active rank range [16, 16]; minimum smallest/largest eigenvalue ratio 3.36e-06. Input energies match to <4e-8 relative in aggregate. Full eigenvalues and document-level five-component accounting saved; source hashes verified. Native weights unchanged.

Next discriminator: separate within-recipient directional sensitivity from cross-recipient weighting, using paired per-site ratios and common site weights. Freeze a basis-independent statistic for fresh-document replication before calling this a stable organization principle. Broad research goal remains active.
