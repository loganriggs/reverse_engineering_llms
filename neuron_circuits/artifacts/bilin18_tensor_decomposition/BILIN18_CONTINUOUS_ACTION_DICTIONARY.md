## H1047 — Continuous action dictionaries fit causal targets and transfer within tasks

Previous goal turn was progress: H1046 checked surrogate derivatives and established hard-selection descent mismatch. This turn held the untrained direct-change support rule fixed and trained continuous decoder directions instead. No gate surrogate was used.

Each method receives both endpoint states, projects their difference into the same frozen expanded-training PCA128perstream, and selects coordinates with the original untrained scalar/block/shared rule. Decoder columns are now learnable and unit-normalized. Three1152x128decoders give442368parameters for every method. Shared atoms are decoded once under the group union. Encoder directions and group membership stay frozen; this is a learned action dictionary, not a full learned BSF, nonlinear manifold, or learned semantic taxonomy.

Training:80pairs/10families,160fixed steps with16visits/family, Adam0.001from the PCA decoder, bidirectional finite paired-edit KL plus1e-4mean squared column-direction displacement. No full-state reconstruction objective. All pairs covered; final checkpoint only. Same schedule, parameter count, and input access across architectures. Max48active coordinates; shared unions sometimes smaller. No evaluation-dependent stopping or learning-rate search.

Continuous derivative checks on one training pair per architecture agree with central finite differences to within0.044%relative error across the tested steps/directions. This is a local implementation check, not proof of all gradients. Unlike the earlier importance-only fits, the actual full80pair equal-family training objective decreases strongly:

| Method | Initial training KL | Final training KL | Reduction |
|---|---:|---:|---:|
| Scalar |0.3281|0.1513|53.9%|
| Block |0.8283|0.1123|86.4%|
| Shared |1.0946|0.0851|92.2%|

Training losses exclude the tiny displacement penalty. Shared has the best final training fit, despite its worse initialization; this is not evidence of superior generalization.

Paired relative KL, initial→trained:

| Panel | Scalar | Block | Shared |
|---|---:|---:|---:|
| Same-family lexical |0.4475→0.3263|0.6322→0.3936|0.7161→0.4345|
| Historical copying/spatial |0.8448→0.8435|0.9166→0.9232|0.9220→0.9250|
| Weekday development |0.7498→0.7630|0.8416→0.8593|0.8974→0.8982|

All three improve on familiar-family lexical examples, while excluded-task transfer remains weak. Same-family donor-correct top1after edits rises to40%scalar,40%block,32.5%shared. Copying/spatial remains0%, weekday12.5%. Native recipient context is preserved, so these are edit operators, not autonomous model replacements.

The result separates previously conflated claims: actual finite causal supervision can successfully fit continuous action geometry and help within-family generalization; the failed STE importance recipe was not a sufficient test of that idea. However, training-fit superiority of shared groups does not translate to the best lexical development fidelity, and no whole-task generalization advantage has appeared. Fixed random groups and frozen encoder directions remain strong limitations. With442kparameters and only80pairs, memorization or narrow-family specialization is plausible. The current experiment cannot assign a unique cause to the remaining transfer gap.

Verification:2578suffix calls including960student training passes, full initial/final training audits, derivative probes, and fixed evaluation. Initial encoders reproduce H1044within3.49e-6KL after unit-column normalization; the small numerical difference is explicit. Decoder unit-norm error1.20e-7. Zero-change and reverse-sign behavior exact on checked examples. Frozen encoder/source hashes unchanged, no training/evaluation prompt overlap. Independently recomputed KL from18saved evaluation examples, max4.31e-14. No fresh task discovery or validated manifold claim.

Next decisive test: leave each of the10competent training families out in turn, refitting encoder PCA, normalization, and decoder using only the other nine families. Compare all three methods and their untrained controls under the same fixed per-family exposure, without selecting methods or thresholds from held-out outcomes. This creates a broader within-bank whole-task transfer test and prevents the encoder from seeing the held-out family's states. The bank was already competence-screened, so it is not pristine external validation; report that limitation. Fix the procedure before running, retain native errors, and measure actual training fit for each fold. Do not infer reusable shared computation solely from the strong in-sample shared loss.
