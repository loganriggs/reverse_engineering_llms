## H864 — Task information at the relay is causally relevant but not required for most correct answers

H863 showed that a single supervised direction approximates most of each learned union's intervention effect. H864 tests the more demanding question of local necessity. Removing that direction's native pairwise contrast reduces answer-margin distinctions, but generally leaves the selected answer intact. Even erasing the full raw-state contrast at this one position does not remove most native behavioral discrimination.

### Protocol

Retain all20 nominal64-slot unions, five architectures and both seeds/discovery axes. For recipient state b0 and donor b1, add one-half of the projected difference. Apply the reciprocal edit to b1. Both contexts then have the same coordinate in the targeted span, while preserving their native complementary coordinates and all other model state. Test union, within-union rank1, and orthogonal complement. Include original supervised task axes, two fixed Gaussian rank1 controls, and the full raw-state midpoint. All156 recipient contexts, both agreement and sentiment, remain included.

The paired behavioral distinction is margin(context0) minus margin(context1), with the same answer groups. Gap fraction removed is the reduction in the sum of these paired distinctions divided by the native sum. This is distinct from the earlier donor-transfer slope and cannot be equated with a mediation fraction. Midpoints may be outside the native data manifold; local erasure is not global model ablation.

### Main results

Values separated by slashes are seeds8351/8352. Native binary answer-group accuracy is100% for both tasks. Full-vocabulary correctness is a different metric: native sentiment top-group accuracy is76.7%, agreement100%.

|Representation erased|Task|Native pair-gap removed|Binary accuracy after erasure|
|---|---|---:|---:|
|scalar48|agreement|6.1%/5.6%|100.0%/100.0%|
|scalar48|sentiment|15.7%/12.7%|98.3%/98.3%|
|scalar64|agreement|5.7%/5.6%|100.0%/100.0%|
|scalar64|sentiment|12.8%/14.2%|98.3%/98.3%|
|block8|agreement|5.7%/5.7%|100.0%/100.0%|
|block8|sentiment|17.1%/14.5%|98.3%/98.3%|
|shared_fixed|agreement|4.6%/6.0%|100.0%/100.0%|
|shared_fixed|sentiment|9.6%/12.9%|98.3%/98.3%|
|shared_learned|agreement|2.2%/2.3%|100.0%/100.0%|
|shared_learned|sentiment|5.8%/3.5%|100.0%/100.0%|
|supervised_axis|agreement|7.9%|100.0%|
|supervised_axis|sentiment|22.4%|98.3%|
|full_midpoint|agreement|9.4%|100.0%|
|full_midpoint|sentiment|25.0%|96.7%|

The scalar/block rank1 directions remove about5.6–6.1% of the native agreement distinction and12.7–17.1% of sentiment distinction. The complete raw relay midpoint removes9.4% and25.0%, respectively. Agreement remains100% correct even after full relay erasure; sentiment drops to96.7% binary accuracy, compared with98.3% for most scalar/block rank1 erasures. Thus these are partial task-carrying coordinates, not established necessary concept units.

For scalar/block unions, rank1 erasure closely approximates full-union erasure. Complement-only erasures have negligible net task-gap effects. Gaussian direction controls also have nearzero effects. Full-vocabulary sentiment top-group accuracy falls from76.7% to61.7% under full relay midpoint, and to63.3% under the supervised sentiment axis midpoint. Answer-group decisions therefore conceal some distributional changes; raw logits, margins, top-group decisions and native-to-erasedKL are all retained.

### Interpretation and next action

The useful signal is low-dimensional relative to the selected union, but the model's task performance remains supported by other state. This may reflect redundant routes, information at other positions, or causal contribution that is too small to flip robust margins; this experiment does not distinguish those explanations. Earlier donor swaps demonstrated transferability, not necessity, and H864 makes that distinction explicit.

To avoid extending conclusions from two tasks at one site, next apply controlled route-contrast erasure at the already verified early boundary across all five competent task families. Separate residual sequence, persistent value bus, and embedding anchor, including combined erasures. Binding, copying and capital retrieval have strong prior value-bus transfer evidence; the new question is whether erasing their bus contrast diminishes native discrimination, and whether other routes compensate.

### Verification and scope

10,452 full-model forwards, exact identity logit replay, full-donor replay error1.43e-05; erased-coordinate relative residual1.14e-15. Basis orthogonality passed, all inputs unchanged and outputs finite. No reserved-test use;26,112-document pool unchanged. All5/8 lexical pairs per sentiment/agreement task retained with templates/gaps/orientations.4096 lexical-cluster bootstrap intervals are saved, uncorrected and exploratory on reused research data. Broad goal remains active.
