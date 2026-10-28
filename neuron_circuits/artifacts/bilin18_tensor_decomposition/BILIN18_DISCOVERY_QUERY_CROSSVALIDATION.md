## H935 — Discovery-only model selection weakens the case for necessary discrete groups

Two lexical folds split the existing96 discovery examples into48/48, with both templates, task/copy families, and all three domains in each fold. Refit groups inside each fold. Compare linear ridge, eight kmeans groups with shared continuous residual correction, and eight random-hyperplane groups with the same correction, retaining three seeds per grouped method. Select alpha separately per head/model/seed from1e-5 through1 by summed validation persistent-write squared error, then refit all96 discovery examples. Twenty-one frozen models saved before any research-transfer scoring.

| Head | Linear CV error | Learned groups range | Random groups range |
|---|---:|---:|---:|
| L8H3 | .144 | .159–.164 | .164–.200 |
| L13H0 | .145 | .140–.159 | .152–.169 |
| L14H4 | .372 | .363–.387 | .372–.416 |

Errors are square-root summed write-error energy divided by summed native-write energy, pooled across both lexical folds. They are local native-state predictions, not causal-effect errors or whole-model fidelity. All selected penalties are1e-4 or1e-3, below H934's smallest1e-2. None is at the expanded grid boundary. Thus earlier continuous-model weakness partly reflected regularization. Learned partitions no longer have a consistent advantage over a flat continuous predictor; random geometry is often close after correction. This does not prove that discrete context is absent.

Random groups use three seeded normalized hyperplanes and training-median thresholds; they can encode meaningful context by chance. Empty cells fall back to global query centers and position means. Occupancy and partition complexity are not matched exactly. Selection pools raw write energy, so it need not optimize each task equally or predict downstream KL. Grouped and linear parameter counts differ. Keep all models and seeds for live causal evaluation without another selection step.

Verification: lexical halves disjoint; all96 scored indices discovery-only; normal-equation max residual3.78e-15 and independent primal/dual weight difference1.21e-13. All finite, input hashes unchanged. Zero model forwards, research-transfer evaluations, or reserved-test evaluations. Full source cache contains other rows but none was indexed for fitting/scoring. Frozen model hash c4fe3d46e8f3aa4e6982d327b73bfacb7b2916b7b9b81d04e96a555622ce29c9.

Next: evaluate these frozen fits with live queries and native/donor persistent values in all three heads, measuring both output KL and causal-effect error on the existing research-transfer panel. This is a representation-family discriminator, not yet manifold verification. Then return to the broader feature/manifold evidence with explicit pass/fail criteria rather than letting a long mechanism-tracing branch replace that deliverable.
