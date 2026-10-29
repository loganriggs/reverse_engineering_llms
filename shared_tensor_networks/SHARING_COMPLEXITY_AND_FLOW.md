# Reuse-aware complexity and gradient flow

Status: elementary propositions proved below; exact/small numerical checks in check_sharing_complexity_flow.py. These build on classical arithmetic circuit complexity and catalecticants, not claims of novel rank theory. Existing E2 tested20trajectories (five initializations, four update rules); multiplicity-corrected SGD and projected occurrence SGD agreed to4.663e-15. See RESULTS.md and results/e2_metric.

## A. Define the object and permitted operations first

For a polynomial f over the reals, fix a grammar of scalar input leaves, real constants, binary addition and binary multiplication. Fan-out is permitted; exponentiation is not a primitive. Let C_DAG(f) be the minimum number of internal gates computing f exactly, and C_formula(f) the analogous minimum with no reused internal node. These are standard circuit/formula size measures, useful here as a reuse-aware complexity rather than a replacement for all notions of tensor rank.

For vector-valued bilinear gates, specify all dimensions and charge scalar arithmetic or tensor storage: otherwise one enormous gate trivializes node count. Track a resource profile (distinct parameter slots, gate count, scalar operation cost, peak live state), or a declared weighted cost/Pareto frontier. Fixed coefficients and trainable parameter slots are different: a fixed polynomial can have zero trainable parameters and substantial evaluation complexity. Number of distinct real values is not a robust parameter-count definition.

For a symmetric order-m tensor T use f_T(x)=<T,x^tensor(m)>. This correspondence is injective over characteristic zero. For a general occurrence tensor, restricting all inputs to x identifies many tensors; a function-complexity measure then lives on that equivalence class, not on the full occurrence tensor. Exact tensor equality, diagonal polynomial equality and approximate distributional agreement are distinct objectives.

An approximate version C_DAG(f;D,epsilon) minimizes the same cost subject to E_D[(g-f)^2]<=epsilon^2. Fix D and absolute/relative tolerance before comparing. Its guarantees concern D; it need not preserve off-support behavior or mechanisms. RMSNorm requires a larger operation grammar and falls outside polynomial rank arguments below.

## A1. A certified separation from formula size and tensor rank

Let q(x)=sum_i x_i^2 in d variables, k=2^L, and f=q^k. A DAG first computes q using d multiplications and d-1 additions, then performs L repeated squarings. Thus

C_DAG(f) <= 2d-1+L.

Any binary arithmetic formula computing a polynomial of degree 2k needs at least 2k variable-leaf occurrences: by induction, each subtree's degree is at most its number of variable leaves (addition takes a maximum; multiplication adds degrees). A binary tree with at least 2k leaves has at least 2k-1 internal gates. Hence C_formula(f)>=2k-1. This is an exponential separation in L for fixed d; degree also grows exponentially, so it is not a fixed-degree circuit lower bound.

Let T be the unique symmetric coefficient tensor of f. For g~N(0,I_d), E[(g dot x)^(2k)]=(2k-1)!! q(x)^k, so

T = E[g^tensor(2k)]/(2k-1)!!.

Flatten T into its first and last k indices. For any nonzero symmetric k-tensor U, its quadratic form is

<U,Flatten(T)U> = E[<U,g^tensor(k)>^2]/(2k-1)!! > 0.

The polynomial inside the square is nonzero; a full-support Gaussian gives strictly positive integral. The flattening annihilates the antisymmetric complement and is positive definite on Sym^k(R^d), giving exact rank binomial(d+k-1,k). Each ordinary CP term contributes a rank-one matrix to this flattening, so ordinary CP rank (and symmetric CP rank) is at least this value. Every k-versus-k split gives this flattening rank by symmetry. In particular, a hierarchical Tucker tree with this balanced cut needs at least this bond dimension; this is not a claim about every possible cut or tree topology.

Example: d=3,L=4 gives a circuit of at most9gates, formula size at least31, and CP rank at least153. Thus a genuinely small reusable circuit can have large tensor flattening rank. Neither bound gives the exact minimal circuit, identifies the physical generating DAG, or proves easier learning.

Classical context: repeated-squaring circuit/formula distinctions appear in https://www.math.ias.edu/~avi/PUBLICATIONS/MYPAPERS/CKW11/CKW11.pdf . Full-rank catalecticants of powers of quadrics are established mathematics; see https://arxiv.org/abs/2208.07921 . The Gaussian-moment argument above is a self-contained proof for positive-definite real q.

## B. Separate metric acceleration from newly available directions

At a tied state z=S theta, let S have full column rank, M=S^T S, P=S M^-1 S^T. Let F(z) be the vector of predictions on a fixed probe set and J=dF/dz at this state. Write e=F-y for loss ||e||^2/2; other loss gradients can replace e in the first-order identities.

Three output tangent kernels are:

- ordinary tied Euclidean gradient flow: K_tied=J S S^T J^T;
- occurrence-metric-corrected tied flow: K_projected=J P J^T;
- free occurrence flow: K_free=J J^T.

Proof: chain rule gives grad_theta=S^T J^T e. Multiply by J S after applying either identity or M^-1 preconditioning. Free flow is dot(z)=-J^T e. Each gives dot(F)=-K e and dot(loss)=-e^T K e.

Since I-P is an orthogonal projector,

K_free - K_projected = J(I-P)J^T is positive semidefinite,

and the extra instantaneous loss decrease available to free flow is exactly

|| (I-P)J^T e ||^2.

This measures a disagreement gradient: directions that force formerly tied occurrences to differ. It is zero iff the occurrence gradient lies in im(S). Zero at one point does not guarantee the trajectories remain tied. The ordering is instantaneous at a common parameter state, not a global convergence or generalization guarantee.

For pure coordinate replication, M is diagonal with multiplicities m_i. Put Q=S M^-1/2, so Q^T Q=I. Then

m_min K_projected <= K_tied <= m_max K_projected

in PSD order, because K_tied=(JQ)M(JQ)^T and K_projected=(JQ)(JQ)^T. If multiplicities differ, a single learning-rate rescaling generally cannot match both kernels. Ordinary tied and free kernels have no universal ordering: for S=(1,1)^T and J=(1,1), tied kernel4 exceeds free2; for J=(1,-1), tied0 is less than free2.

In a depth-D binary occurrence expansion, shared level l (bottom l=0) has multiplicity 2^(D-1-l); the readout has multiplicity1. At depth4 the level multiplicities are8,4,2,1. This applies also to differentiable RMSNorm networks, because the proof requires differentiability, not polynomiality. For projected-equivalent SGD divide each shared layer's summed gradient by its multiplicity. Adam does not inherit this equivalence automatically; our Fashion-MNIST Adam runs therefore do not isolate the Euclidean metric effect.

## Concrete queue

A-next: controlled q^k and perturbed-target teachers, report certified circuit upper/formula lower/flattening rank alongside learning curves. Manipulate reuse without attributing degree changes to reuse. For graph recovery use identifiable restricted templates and existing intervention tests; a compact fit alone is insufficient.

B-next: at identical stack/tree initial states on a fixed image probe set, measure disagreement-gradient energy and multiplicity-corrected versus ordinary tied SGD. Compare free occurrence SGD with both. Repeat at multiple depths/seeds and log training curves; do not substitute Adam as a proof check. At tiny sizes independently compute J and the three kernels. This tests whether optimization differences arise from the induced metric or exploitable untied directions.

Both directions are active theory work alongside image scaling, not completed research programs. No new large theory-training sweep is currently running.
