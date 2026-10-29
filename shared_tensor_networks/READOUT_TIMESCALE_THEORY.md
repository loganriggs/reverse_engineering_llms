# Fast readout learning and reduced gradient flow

Let L(theta,U)=1/2||Phi(theta)U-Y||_F^2 + lambda/2||U||_F^2 + R(theta), where the sample normalization is included in Phi and Y. Assume lambda>0 and smooth Phi,R on the region considered. Define H=Phi^T Phi+lambda I and U*=H^{-1}Phi^T Y. Then H>=lambda I and U* is smooth. These assumptions apply locally to nonzero normalized bilinear factor rows; crossing a zero row is excluded.

Consider dot(theta)=-partial_theta L(theta,U) and epsilon dot(U)=-partial_U L(theta,U), epsilon>0. Write e=U-U*(theta). Exact identities are

epsilon dot(e)=-H e-epsilon DU*(theta)[dot(theta)],

H DU*[v]=(D Phi[v])^T(Y-Phi U*)-Phi^T(D Phi[v])U*.

The second follows by differentiating H U*=Phi^T Y. Consequently, whenever ||DU*[dot(theta)]||_F<=K along the path,

||e(t)||_F <= exp(-lambda t/epsilon)||e(0)||_F + (epsilon K/lambda)(1-exp(-lambda t/epsilon)).

Proof: take the Frobenius inner product of the first identity with e, use coercivity and Cauchy–Schwarz, and apply scalar comparison to its norm (by continuity at zero). An initially optimal readout generally does not remain optimal: the optimum moves as theta changes. The equilibrium tracking scale is epsilon/ lambda times that motion, not epsilon alone.

For the reduced loss ell(theta)=L(theta,U*(theta)), the envelope identity gives grad ell=partial_theta L(theta,U*). Suppose all compared paths remain in a common region on a fixed time interval, the reduced vector field is Lipschitz with constant L0, and ||partial_theta L(theta,U)-partial_theta L(theta,U*)||<=C||e|| there. Start reduced and full flows at the same theta. Gronwall then gives

||theta_epsilon(t)-theta_reduced(t)|| <= C exp(L0 t) [epsilon ||e(0)||/lambda + epsilon K t/lambda].

This loose bound suffices for finite-time convergence as epsilon tends to zero when C,K,L0 and the region are uniform in epsilon and the initial mismatch is bounded. It neither proves those compactness assumptions globally nor guarantees convergence to a good minimum. If lambda tends to zero or the unregularized feature Gram loses rank, this bound is not uniform. A large lambda also changes the statistical objective; it is not a free optimization remedy.

The joint objective decreases exactly as dot(L)=-||partial_theta L||^2-(1/epsilon)||partial_U L||_F^2. This does not order different epsilon trajectories or their transfer performance. Finite-dwell alternating flow is a separate switched system, not the same ODE at finite dwell time.

These are elementary tracking and Gronwall arguments, not a novelty claim. The accompanying CPU experiment checks identities and the limit on actual bilinear feature maps; it does not certify the constants over whole trajectories.
