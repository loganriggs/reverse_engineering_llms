"""M79 independent local-chart identities and boundary failure counterexample."""
from pathlib import Path
import hashlib,json
import numpy as np
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    rng=np.random.default_rng(7900);errors=[]
    for _ in range(100):
        L=rng.normal(size=(7,4));R=rng.normal(size=(7,4));D=rng.normal(size=(3,7))
        def F(x):return D@((L@x)*(R@x))
        def J(x):return D@((L@x)[:,None]*R+(R@x)[:,None]*L)
        def T(c,x):return F(c)+J(c)@(x-c)
        c,z,r,d=rng.normal(size=(4,4));v=rng.normal(size=4);h=1e-5
        rem=np.linalg.norm(F(z)-T(c,z)-F(z-c))
        deriv=np.linalg.norm((F(c+h*v)-F(c-h*v))/(2*h)-J(c)@v)/max(1,np.linalg.norm(J(c)@v))
        cr,cd=rng.normal(size=(2,4))
        decomposition=np.linalg.norm((T(cd,d)-T(cr,r))-((T(cr,d)-T(cr,r))+(T(cd,d)-T(cr,d))))
        bound=np.linalg.norm(D,2)*np.linalg.norm(L,2)*np.linalg.norm(R,2)*np.linalg.norm(z-c)**2
        assert np.linalg.norm(F(z-c))<=bound+1e-10
        assert rem<1e-10 and deriv<1e-8 and decomposition<1e-10
        errors.append(dict(remainder=rem,finite_difference_relative=deriv,decomposition=decomposition))
    # F(x,y)=(x*x,x*y). Anchors (+/-1,0); nearest routing boundary x=0.
    # T_(a,0)(x,y)=(2*a*x-1,a*y). At y=1, limits differ by2.
    def f(z):x,y=z;return np.array([x*x,x*y])
    def routed(z):
        anchors=np.array([[-1.,0.],[1.,0.]])
        a=anchors[np.argmin(((anchors-z)**2).sum(1)),0];x,y=z
        return np.array([2*a*x-1,a*y])
    boundary=[]
    for eps in [1e-2,1e-4,1e-6,1e-8]:
        left=np.array([-eps,1.]);right=np.array([eps,1.])
        native=float(np.linalg.norm(f(right)-f(left)));chart=float(np.linalg.norm(routed(right)-routed(left)))
        assert abs(native-2*eps)<1e-12 and chart==2
        boundary.append(dict(epsilon=eps,native_difference=native,chart_difference=chart))
    result=dict(status='passed',source_sha256=sha(__file__),protocol_sha256=sha(V/'local_affine_chart_protocol.json'),random_cases=100,max_errors={k:max(e[k] for e in errors) for k in errors[0]},boundary_cases=boundary,interpretation='Exact native quadratic function is continuous while nearest local charts need not be. This establishes a possible mechanism, not its prevalence or behavioral significance in the native model. Euclidean covering bounds local remainder but does not guarantee causal fidelity.')
    out=V/'local_affine_chart_toys.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
