"""Independent exhaustive interpolation and bilinear XOR-convolution checks."""
import json
import numpy as np
from repetition_cube_algebra import H,SIGNS,coefficients,reconstruct,order_energies,bilinear_coefficients
from fit_pca_encoder import V,sha


def main():
    assert np.array_equal(H.T@H,8*np.eye(8));examples={'first':SIGNS[:,0],'last':SIGNS[:,2],'additive_majority_margin':SIGNS.sum(1),'majority_decision':np.sign(SIGNS.sum(1)),'pair_product':SIGNS[:,0]*SIGNS[:,1]};rows=[]
    for name,y in examples.items():
        c=coefficients(y);oracle=np.linalg.solve(H,y);assert np.allclose(c,oracle) and np.allclose(reconstruct(c),y);assert np.isclose(np.mean(y*y),sum(order_energies(c)));rows.append(dict(name=name,coefficients=c.tolist(),order_energies=order_energies(c)))
    assert np.allclose(coefficients(examples['majority_decision'])[[1,2,4,7]],[.5,.5,.5,-.5]);assert order_energies(coefficients(examples['additive_majority_margin']))[2:]==[0.,0.]
    rng=np.random.default_rng(10800);errors=[]
    for case in range(12):
        L=rng.normal(size=(7,5));R=rng.normal(size=(7,5));D=rng.normal(size=(3,7));z=rng.normal(size=(8,5));y=((z@L.T)*(z@R.T))@D.T;expected=np.linalg.solve(H,y);actual,carry,mixed=bilinear_coefficients(L,R,D,coefficients(z));assert np.allclose(actual,expected,atol=1e-10,rtol=1e-10);assert np.allclose(carry+mixed,actual[7],atol=1e-10);errors.append(float(np.linalg.norm(actual-expected)))
        zc=coefficients(z);zc[[3,5,6,7]]=0;additive=reconstruct(zc);yy=((additive@L.T)*(additive@R.T))@D.T;assert np.linalg.norm(coefficients(yy)[7])<1e-10
    out=dict(status='passed',source_sha256=sha(__file__),helper_sha256=sha(V.parent/'repetition_cube_algebra.py'),examples=rows,random_cases=12,max_coefficient_error=max(errors),scope='Exact uniform binarycube coefficients; bilinear native map combines input coefficients byXOR. Cubicoutput requires cubicinput×mean orlinear×pair components. Additive majority margin has nocubicterm althoughits thresholded choice does. No attribution toindividual nativechannels.')
    f=V/'repetition_cube_algebra_toys.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print('PASSED',max(errors))
if __name__=='__main__':main()
