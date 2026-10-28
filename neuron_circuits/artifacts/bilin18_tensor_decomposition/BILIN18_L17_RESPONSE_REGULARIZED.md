

## H285 — Response supervision partly resolves the state/response tradeoff, with limited head transfer

Freeze the H28064 read coordinates, normalization and H279 base; restart its KL adapter. Eight fits cross response strengths0/.1/1/10 and Adam rates.1/.3,400 updates, checkpoints every100. Loss is intact KL +.1 times correction energy/base state-error energy + strength times base training KL times response-error/base response-error energy. Response error uses exact quadratic sufficient statistics of feature differences and native residual-response targets, so every response update sees the full training response distribution. This is MLP-output MSE regularization, not conditional-KL regularization. The statistic loss and gradient match explicit samples to1e-12 in a focused test, including split accumulation.

Response supervision uses full actual/angle removals on heads0–5, first768 documents. Heads6–8 are excluded from this additional loss and from model selection, but the existing base was previously trained using them: they are not globally unseen heads. The next256 validation documents select the lowest intact-KL checkpoint among those with <=2% worsening vs H280 and response error<=base separately for actual and angle fit heads. Seven checkpoints qualify; selected lambda10/rate.1/step300. All40 checkpoint records and weights retained, including repeated initialization checkpoints. No global optimum claim.

Fresh128 documents exclude3008 prior unique complete-document hashes. Independent full document/window/file hashes and exact window-content audits pass. All metrics finite. Conditional response evaluation uses the native intact-output anchor; intact KL is measured separately without that anchor. Bootstrap clusters all sites and included heads by document. Original thresholds remain fixed; intervals condition on frozen fitted candidates.

| Model | Fresh intact KL |
|---|---:|
|base|0.058878|
|h280|0.053935|
|optimization_control|0.052759|
|selected|0.052792|

Selected versus H280: 2.12% intact KL reduction, paired95% [0.04,4.21]%.

| Payload/head group | Response MSE reduction vs base | Response MSE reduction vs H280 | Conditional KL reduction vs base |
|---|---:|---:|---:|
|actual_fit|4.01%|9.79%|13.32%|
|actual_heldout|-6.70%|2.36%|-20.52%|
|angle_fit|0.56%|2.91%|2.81%|
|angle_heldout|-2.57%|0.69%|-3.86%|

Locked gates: state_retention: PASS, response_gain: FAIL, base_response_retention: PASS, heldout_transfer: FAIL.

State retention passes. Base response retention passes on supervised heads. The joint response-gain gate fails because angle improvement vs H280 is2.91%, below5%, despite actual improvement9.79%. Held-out transfer fails: actual/angle MSE improvements vs H280 are2.36%/.69%, below5%; both remain worse than base by6.70%/2.57%. Thus the tested state/response tradeoff is not unavoidable on supervised heads, but broad transfer has not been established. All-head averages mask this difference. No independent semantic unit is identified.

**Secondary optimization control.** After primary outcomes, an audit recognized that additional optimization could explain intact gains. The zero-response-strength runs were already in the locked sweep, but their fresh comparison was added after primary evaluation and is labeled exploratory. Select their checkpoint by validation intact KL only (lambda0/rate.1/step300), then re-evaluate on the SAME fresh panel; no new holdout claim. Primary outputs are preserved with _primary suffix. No primary gates changed. This compares equal-rate/equal-step restarts differing in response-loss strength; checkpoint choices themselves remain validation-selected.

Selected regularized versus zero-strength control: -0.06% intact KL reduction, paired95% [-1.44,1.22]%.
- actual_fit: regularized response-MSE reduction vs extra-fit control 10.52% [9.73,11.36]%.
- actual_heldout: regularized response-MSE reduction vs extra-fit control 2.97% [2.45,3.46]%.
- angle_fit: regularized response-MSE reduction vs extra-fit control 3.10% [2.83,3.37]%.
- angle_heldout: regularized response-MSE reduction vs extra-fit control 0.70% [0.55,0.87]%.

The zero-strength control is necessary when attributing the state gain; comparison to H280 alone conflates optimization time and response loss. Primary conclusions are retention plus uneven response improvement, not that response supervision itself improves intact KL. All raw values remain available.

**Next discriminator.** Reverse the response-supervised head split at fixed feature map and optimization budget, with a zero-strength restart control declared in advance. Assess whether the response gains follow the supervised heads. If they do, limited coverage of the response distribution is a better explanation than intrinsically difficult held-out heads. If heads6–8 stay worse even when supervised, examine their geometry and available read directions. Afterwards compare all-head supervision against the split fits before increasing feature capacity. Keep state and response metrics separate and use fresh documents after selection. No semantic interpretation should be assigned solely from successful regression.

All jobs terminal. Original model/compiler/base unchanged. Research goal remains active.
