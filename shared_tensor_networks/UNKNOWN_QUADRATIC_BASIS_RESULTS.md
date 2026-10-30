# Unknown basis recovery, and why two outputs do not validate reuse

The predeclared240-case CPU experiment completed without failed positive-definite reference estimates. Eighty noiseless cases recovered planted rows up to sign/permutation (minimum cosine at least1-3e-16). With measurement noise SD.001, worst matched-axis cosine across all cases was.999909; with SD.01 it was.990075. At the latter noise level, median clean prediction MSE ranged.000162–.000249 across the four dimension/conditioning groups. These are small-noise controls in dimensions4/8 and condition numbers1/3, not broad robustness guarantees.

Each case uses d(d+1)/2 unit-norm probes and two scalar outputs per probe:20 scalar measurements for d4,72 for d8. Held-out predictions use512 fresh dense unit-sphere inputs. The fixed reference output is positive; the second output has deliberately separated component weights. There is no adaptive acquisition, learned feature encoder, training optimizer or native model in this experiment.

## Identification within the stated architecture

For h=B x and square-gate outputs, Q0=B^T B and Q1=B^T Lambda B. Polarization of scalar observations recovers each Q. Let R=Q0^(-1/2), and U=B R. Then U^T U=I, so

    R Q1 R = U^T Lambda U.

If all diagonal entries of Lambda are distinct, its orthonormal eigenvectors identify the rows of U up to signs/permutation, hence B=U Q0^(1/2). The positive reference readout fixes the row scaling convention; allowing arbitrary positive reference weights would leave the usual scaling freedom. This is an elementary simultaneous-diagonalization argument, not a novelty claim.

One output leaves B and O B indistinguishable for every orthogonal O. If Lambda has repeated entries, orthogonal mixing inside the repeated block also preserves Q1. The script explicitly rotates two axes by0.6 radians and verifies both controls in80 cases, retaining different axes despite identical relevant output functions. These ambiguities persist even with all possible input-interface probes. They do not apply to unrestricted access to hidden gate weights or interventions directly on those gates.

## A crucial falsifiability limitation

For ANY symmetric Q1 and positive-definite Q0, the whitening construction produces an orthogonal diagonalization and therefore a shared-square representation. Thus two such outputs never reject this architecture. Their apparent common basis is a valid algebraic representation, not independent evidence that the physical model uses those reusable intermediates. In a model planted from this restricted family, distinct generalized eigenvalues do identify its normalized basis; that conditional fact must not be promoted into unconditional circuit discovery.

The noiseless positive control and nonidentifiability controls therefore answer different questions: whether recovery works under supplied assumptions, and whether observations force those assumptions. They also clarify why low output error is insufficient as a mechanistic score.

## Next discriminating test

Freeze the basis inferred from two outputs. Obtain a third independently generated output and fit only its diagonal readout. Under the shared-square hypothesis, the whitened Q2 must be diagonal in that same eigenbasis. With distinct first generalized eigenvalues this is equivalent to [R Q1 R, R Q2 R]=0. A nonzero commutator rules out simultaneous orthogonal diagonalization in the whitened coordinates; zero commutator gives a compatible representation, not proof of the physical computation graph.

Compare genuinely shared third outputs against independently rotated bases, with noise-aware residual thresholds and held-out input/intervention prediction. Do not refit the basis using the third output before scoring this test. This is the next useful mechanism experiment; adding more two-output seeds would not resolve the structural ambiguity.

## Verification

Literal hidden-feature and quadratic-form outputs agree within3.56e-15; noiseless polarization within4.00e-15; reconstructed forms within2.58e-14. All noiseless predictions and axis checks passed. The script records all240 cases,80 ambiguity controls, source hashes and NumPy version in results/unknown_quadratic_basis/results.json. Numerical identities use common generated fixtures; they complement the algebra above and do not establish noise bounds or computational graph recovery. Existing h128 training and evaluation sources were untouched.
