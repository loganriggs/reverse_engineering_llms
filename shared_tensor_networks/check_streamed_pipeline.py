"""Independent low-dimensional checks of the generalized training/storage path."""
import hashlib,itertools,json,tempfile
from pathlib import Path
import numpy as np
from numpy.polynomial.hermite import hermgauss
from regenerated_coverage import blocks,digest
from coverage_evaluation import features,population_errors,adaptation
from coverage_audit import phi,svd_refit
from quartic_cross_gram import product_factors,cross_gram
from chunked_checkpoint import load_checkpoint
from streamed_training_case import train_case

def main():
 rng=np.random.default_rng(265501);A=rng.normal(size=(2,4))/2;B=rng.normal(size=(2,4))/2
 F=product_factors(A,B,np.ones(2));G=cross_gram(F,F);W=rng.normal(size=(7,3));W/=np.sqrt(np.sum((W@G)*W,axis=1))[:,None]
 source=list(blocks(A,B,W[:3],265502,521));x=np.concatenate([b['x'] for b in source]);y=np.concatenate([b['y'] for b in source])
 for b in source:
  i=b['index'];n=len(b['x']);xx=np.random.default_rng(np.random.SeedSequence([265502,0,i,0])).normal(size=(256,4));noise=np.random.default_rng(np.random.SeedSequence([265502,0,i,1])).normal(size=(256,3));np.testing.assert_array_equal(xx[:n],b['x']);np.testing.assert_allclose(phi(A,B,xx[:n])@W[:3].T+.05*noise[:n],b['y'],rtol=1e-13,atol=1e-13)
 IA=rng.normal(size=(2,4))/2;IB=rng.normal(size=(2,4))/2;IW=rng.normal(size=(3,3))
 with tempfile.TemporaryDirectory() as tmp:
  case=Path(tmp)/'case';record=train_case(IA,IB,IW,x,y[:,:2],case,steps=5,device='cpu',chunk_size=128)
  a,b,w=load_checkpoint(case/'checkpoint');np.testing.assert_array_equal(w[2:],IW.astype(np.float32)[2:]);a,b,w=[v.astype(float) for v in [a,b,w]]
  FF=product_factors(a,b,np.ones(2));GG=cross_gram(FF,FF);C=cross_gram(FF,F)
  error=population_errors(w[:2].T,GG,C,W[:2],np.ones(2))
  nodes,weights=hermgauss(5);ix=np.array(list(itertools.product(range(5),repeat=4)));gx=nodes[ix]*np.sqrt(2);gw=np.prod(weights[ix]/np.sqrt(np.pi),axis=1)
  literal=phi(a,b,gx)@w[:2].T-phi(A,B,gx)@W[:2].T;quad=np.sum(gw[:,None]*literal**2,axis=0);np.testing.assert_allclose(error,quad,rtol=1e-10,atol=1e-10)
  ad=next(blocks(A,B,W[3:],265502,64,stream=1));coef,metrics=adaptation(a,b,ad['x'],ad['y'],GG,C,W[3:],np.ones(4),64);rank=svd_refit(phi(a,b,ad['x']),ad['y'],coef);assert rank==metrics['rank']
  delta=phi(a,b,gx)@coef-phi(A,B,gx)@W[3:].T;np.testing.assert_allclose(metrics['errors'],np.sum(gw[:,None]*delta**2,axis=0),rtol=1e-10,atol=1e-10)
 out=Path('results/streamed_pipeline_check');out.mkdir(exist_ok=False)
 result=dict(status='passed',scope='Small CPU smoke only; no h128 recovery claim',steps=record['steps'],source_blocks=len(source),partial_final_block=len(source[-1]['x']),source_hashes=[digest(q) for q in source],gauss_hermite_points=len(gx),source_error_max_difference=float(np.max(abs(error-quad))),adaptation_rank=rank,checkpoint_inactive_unchanged=True,sources={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ['check_streamed_pipeline.py','streamed_training_case.py','regenerated_coverage.py','coverage_evaluation.py','coverage_audit.py','chunked_checkpoint.py']})
 (out/'checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
