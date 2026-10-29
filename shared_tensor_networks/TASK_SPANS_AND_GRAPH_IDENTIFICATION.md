# What task diversity and interventions can identify

These are direct derivations for the toy studies, not claims of novelty. They distinguish function-subspace identification from physical graph recovery.

## Identifying a reusable function span from multiple tasks

Let K be a P×r full-column-rank matrix whose columns are polynomial feature coefficients. Let task coefficients be `C=K Wᵀ`, where W has rank r. Then `range(C)=range(K)`: inclusion is immediate and both spaces have dimension r. Thus sufficiently diverse **known task functions** identify the reusable feature-function span, even though K and W remain nonunique under invertible basis changes. This does not uniquely identify quadratic factors, individual units, or the execution graph.

If a held-out task is `c_new=K b`, it lies in that span. With exact source task functions as features, at least r generic noiseless scalar evaluations identify its coefficients in a rank-r feature basis. For linearly independent polynomials evaluated under a full-support Gaussian, the evaluation matrix has full rank almost surely once enough evaluations are supplied. Poor conditioning and noise can still make the problem difficult; the singular-readout results remain relevant.

This statement assumes the source functions or their coefficient span are known. Merely observing many task labels at the same small set of inputs does not remove the nullspace of the input-evaluation operator. A low-rank polynomial baseline can remain ambiguous outside the observed inputs. Learning the additional bilinear factor structure is a different inductive assumption.

In the new shared teacher family, h has four quadratic coordinates and φ(h) contains their ten unordered products. Twelve diverse source task heads have coefficient rank ten, as do all sixteen tasks together. The four held-out tasks are therefore in the source span. The oracle minimum-norm transfer tests recover all held-out tasks with 32 or 128 labels to maximum population MSE 2.06e-29. Ridge intentionally retains a nonzero bias.

For independent teacher tasks, source coefficients have rank twelve and all sixteen tasks have rank sixteen. Each new task lies outside the source span; even exact source-task predictions do not suffice for zero error. Measured Gaussian projection floors range from 0.647 to 0.957 across held-out tasks. This is a structural contrast, not just a label saying that a teacher was constructed as “independent.”

## An explicit misspecification lower bound

Every four-hidden-feature shared student produces all tasks in a space of dimension at most ten. Let `Cw=Lᵀ C`, with H=LLᵀ the Gaussian monomial moment matrix, and let its singular values be s₁≥… . The average population MSE of any rank-at-most-ten task coefficient matrix is at least

`(Σ_{j>10} s_j²) / T`.

Proof: the Gaussian loss is the squared Frobenius distance in these white coordinates divided by T. Project onto any candidate ten-dimensional column space. The discarded energy is minimized by the leading ten left singular vectors, yielding the stated tail energy. The bilinear shared architecture is a subset of these rank-ten matrices, so the same lower bound applies to it.

For the three independent teachers, these irreducible average source-task MSE bounds are 0.08284, 0.07501, and 0.09043. Observed shared-student errors are substantially larger; the bound does not explain all training error. In contrast, the shared teachers have zero rank-tail bound and are realizable by the chosen shared architecture. This gives a certified cost of an incorrect sharing assumption for this particular family.

## Execution reuse versus identical recomputation

The shared multi-task model can compute h and φ(h) once and apply all task heads, or recompute them separately with the same parameter storage for every head. The two executions have the same function and derivatives. A direct test verifies gradients within 3.56e-15, and verifies that the tied parameter gradient equals the sum of separate occurrence gradients at equal weights.

That equivalence does not make the execution costs equal. Nor does it make interventions on physical intermediate states equivalent: changing one shared state affects all consumers, while changing one separately computed occurrence need not do so. Resource counts and intervention semantics must be specified separately from input-output predictions.

## A finite graph family with identifiable sharing bits

The instrumented experiment uses eight templates with three binary choices:

1. Whether two scalar parameter occurrences share storage.
2. Whether the two branches' selected input leaves represent the same variables.
3. Whether the intermediate `h=u₀u₁` is computed once and reused or computed twice.

Outputs are degree four:

`y_left = h_left θ_left v₀v₁`, `y_right = h_right θ_right w₀w₁`.

Observational training inputs satisfy v=w, and both numerical parameter values are equal. All eight graphs therefore fit those observations exactly. This is **distributional equivalence on the observed diagonal** for the input-leaf bit; it is not a claim that tied versus independent leaves agree on arbitrary off-diagonal inputs. Parameter tying versus numerical equality and state reuse versus identical recomputation are observationally indistinguishable more broadly in this construction.

The three addressed probes modify the left parameter storage, the left leaf variable, or the left h-state. Each probe changes the right output if and only if its corresponding sharing bit is present, for generic nondegenerate inputs. Hence m distinct probe types reduce the compatible set from eight to `2^(3−m)` candidates. Three types identify the template exactly. This is a small identifiability theorem under known intervention access, not a blind structure-learning algorithm.

Across eight templates and twenty numerical/data seeds, a minimum-description choice based on observations alone picks the fully shared graph and is correct only for 20/160 teachers. It is not valid to equate that compression preference with recovery of the true graph. All three probe types identify 160/160, and predictions on fresh inputs with previously unobserved combinations of interventions agree to 2.67e-15.

An independent graph interpreter builds explicit multiplication nodes and storage/input references, evaluates them with memoization and targeted substitutions, and verifies the analytic simulator in 1,280 cases. It also checks distinct parameter, variable, operation, and operand-edge counts. Each of the 1,280 compatible sets agrees with the independent combinatorial criterion.

## Limits of the result

The graph templates and operation types are known, the target slots are given, the system is noiseless, and interventions have exact semantics. This does not locate unknown nodes inside a trained neural network or distinguish arbitrary observationally equivalent circuits. Off-diagonal passive inputs could reveal the input-leaf bit without an internal intervention; they still would not identify identical recomputation versus physical state reuse in this family. The positive recovery result is conditional on explicitly stronger observations.
