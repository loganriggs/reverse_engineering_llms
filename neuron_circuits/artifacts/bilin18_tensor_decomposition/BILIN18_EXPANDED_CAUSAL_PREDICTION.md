## H1012 — More lexical discovery improves new-noun prediction but does not identify computational units

Previous turn was progress: H1011 separated actual upstream-state prediction from native-corner graph replay and exposed poor lexical coverage. This turn keeps the six model families and ridge0.1fixed while expanding discovery. The dataset was written and hashed before model evaluation:24new noun pairs provide192discovery examples, added to the prior32; eight other noun pairs provide32new lexical and32new joint-context examples. All forms are single GPT2tokens and disjoint from inspected prior subject-number authority vocabulary. Reserved BSFtest data untouched. Novelty is relative to those inspected authorities, not every historical research prompt. Candidate nouns follow a fixed list filtered only on tokenization and vocabulary overlap; most are occupational/person nouns, limiting diversity.

New lexical examples use near/behind, joint examples under/above; both reuse the same8test noun pairs and must not be treated as independent replications. Test nouns and attractors are disjoint from new training nouns. Native capability is100%on all256new examples; nine-edge native-corner graph error5.30%training,6.05%new lexical,5.87%new joint. The graph therefore remains a viable reference on this new panel.

| Student | New lexical: old32 / expanded224 training | New joint: old32 / expanded224 training |
|---|---:|---:|
| Constant mean |76.9% /36.0%|76.8% /34.2%|
| Norms/cosines15 |35.4% /27.9%|36.1% /21.1%|
| PCA4 linear |72.4% /34.7%|72.7% /32.8%|
| PCA4 quadratic |69.0% /31.6%|68.9% /27.2%|
| PCA8 linear |72.6% /36.3%|72.9% /35.7%|
| PCA8 quadratic |71.6% /34.4%|71.9% /33.9%|

Scores are relative L2error predicting joint damage, summing nine separately predicted edges. Old H1011models are replayed unchanged on exactly the same new states. Students receive five clean source states, five source changes and subject-position residual, never suffix outcomes. Source magnitudes/alignments remain competitive with generic low-dimensional projections. Data size and lexical diversity increase together; this experiment does not isolate their effects.

Tradeoffs: on historical familiar-noun/context-OOD examples, norms15worsens21.1%to40.2%, quadraticPCA8worsens15.2%to42.1%. Historical lexical scores improve but historical-fresh norms15worsens33.3%to36.9%. Training expansion is not uniform progress. Constant-mean improvement is large, demonstrating that altered target scale/distribution explains part of the apparent gain. The analysis cannot equate better aggregate scores with discovered reusable units.

Per-component caution: expanded norms15errors range17.5–60.8%on new lexical and17.1–51.8%on new joint. No candidate establishes accurate recovery of every main effect and interaction. Expanded PCA8captures38.2%new lexical/33.5%new joint centered input energy, still modest.

Post-result paired bootstrap resamples8noun pairs (all their templates and numbers together),2000draws,seed1012. Old-minus-expanded norms15joint-error improvement95%interval is[-6.0,+18.1]percentage points on new lexical and[+6.2,+22.4]on new joint. Expanded mean-minus-norm improvement intervals include zero on both panels ([-7.9,+21.8]and[-0.9,+24.9]). These small conditional intervals do not justify a robust feature advantage over mean. They condition on the chosen eight noun pairs, training fit and task family; no cross-task evidence.

Audits:416unique token sequences across old/new captures;224training rows;zero train/eval token-hash overlap;new panel hash unchanged; independent feature construction within8.9e-16; independent PyTorch fitting within3.52e-10; frozen H1011predictor replay exact on original160states. New computation512partial sequences,2816suffix sequences,256native sequences, no backwards. Captures, predictions, coefficients, identities and teacher effects saved. No running process from this turn; natural-document pool26112unchanged. Broad goal active, H961pending.

Updated hypotheses: (1) magnitude and effective intervention strength explain much apparent transfer; (2) discovery lexical coverage controls generic subspace usefulness; (3) source-position features omit downstream context controlling sensitivity; (4) pooled output objectives sacrifice low-amplitude domains and small edges; (5) fixed residual bases are poorer units than local weight-defined multiplication with context-dependent transport; (6) native corner sparsity need not imply sparse predictable coefficients. Current evidence supports1/2as contributors but cannot distinguish3–6.

Next avoid architecture shopping on this now-open test. Use discovery-only factorial/transport diagnostics to distinguish source magnitude from downstream sensitivity. In particular compare equal-magnitude source edits and context swaps while measuring native main/pair responses, then construct a context-dependent predictor before freezing another genuinely fresh evaluation. Keep source-specific computational features and multi-task causal anchors as the research direction; this single-task pilot is not a substitute for the BSF/manifold/bootstrap goal. Increasing data and changing target means alone is not completion.
