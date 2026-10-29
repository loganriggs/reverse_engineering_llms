"""M113 constructive witnesses for global hard-gate discontinuity and restricted-domain exceptions."""
import json
import numpy as np
import torch
from fit_pca_encoder import V,sha
from continuous_block_shrinkage import block_shrinkage


def hard(a,k):
    e=np.square(a).sum(-1);idx=np.argsort(e)[-k:];out=np.zeros_like(a);out[idx]=a[idx];return out


def main():
    torch.set_num_threads(1)
    # Equal decoder directions alone do not ensure continuity on unrestricted signed codes.
    D=np.array([[1.,1.]])
    vals=[]
    for h in [.01,.001,.0001]:
      left=np.array([[1+h],[-1+h]]);right=np.array([[1-h],[-1-h]])
      gap=float(np.linalg.norm(D@(hard(left,1)-hard(right,1)).reshape(-1)));assert np.isclose(gap,2*(1+h));vals.append(gap)
    # On nonnegative codes with the same output direction, top1 is the continuous max.
    rng=np.random.default_rng(11301);positive=rng.uniform(0,3,(1000,2));errors=[]
    for row in positive:errors.append(abs(float((D@hard(row[:,None],1).reshape(-1))[0])-max(row)))
    assert max(errors)==0
    # Shared dictionary membership preserves common atoms but does not remove exclusive-atom jumps.
    a0=np.array([2.,1.,1.]);M=np.array([[1,1,0],[1,0,1]])
    def overlap(a):
      g=int(np.argmax(M@np.square(a)));return a*(M[g]>0)
    overlap_gaps=[]
    for h in [.01,.001,.0001]:
      l=overlap(a0+np.array([0,h,-h]));r=overlap(a0+np.array([0,-h,h]));assert l[0]==r[0]==2
      gap=float(np.linalg.norm(l-r));assert np.isclose(gap,np.sqrt(2)*(1+h));overlap_gaps.append(gap)
    receipt=json.loads((V/'continuous_training_results.json').read_text());rows=[]
    for entry in receipt['models']:
      assert sha(entry['path'])==entry['sha256'];state=torch.load(entry['path'],weights_only=True)['state'];E=state['encoder'].double().numpy();bias=state['encoder_bias'].double().numpy();D=state['dictionary'].double().numpy();D/=np.linalg.norm(D,axis=0)
      singular=np.linalg.svd(E,compute_uv=False);independent=torch.linalg.svdvals(torch.from_numpy(E)).numpy();assert np.allclose(singular,independent,rtol=1e-10,atol=1e-10)
      rank=int(np.sum(singular>singular[0]*1e-8));assert rank==128
      witnesses=[]
      for sign in [1,-1]:
        a=np.zeros((32,4));a[2:9,0]=3;a[0,0]=sign;a[1,0]=1
        # An explicit right inverse; reconstruct requested coordinates without optimizing a witness.
        x=np.linalg.lstsq(E,a.reshape(-1)-bias,rcond=None)[0];error=float(np.linalg.norm(E@x+bias-a.reshape(-1)));assert error<1e-10
        analytic_jump=D[:,0]*sign-D[:,4]
        trials=[]
        for h in [.01,.001,.0001]:
          left=a.copy();right=a.copy();left[0,0]*=1+h;left[1,0]*=1-h;right[0,0]*=1-h;right[1,0]*=1+h
          xl=np.linalg.lstsq(E,left.reshape(-1)-bias,rcond=None)[0];xr=np.linalg.lstsq(E,right.reshape(-1)-bias,rcond=None)[0]
          al=(E@xl+bias).reshape(32,4);ar=(E@xr+bias).reshape(32,4)
          hardgap=D@(hard(al,8)-hard(ar,8)).reshape(-1);expected=(1+h)*analytic_jump
          assert np.allclose(hardgap,expected,rtol=1e-9,atol=1e-9)
          softgap=D@(block_shrinkage(torch.from_numpy(al),8)-block_shrinkage(torch.from_numpy(ar),8)).numpy().reshape(-1)
          trials.append(dict(h=h,input_distance=float(np.linalg.norm(xl-xr)),hard_output_gap=float(np.linalg.norm(hardgap)),continuous_output_gap=float(np.linalg.norm(softgap))))
        witnesses.append(dict(sign=sign,input_norm=float(np.linalg.norm(x)),coordinate_reconstruction_error=error,predicted_hard_limit_norm=float(np.linalg.norm(analytic_jump)),trials=trials))
      assert max(w['predicted_hard_limit_norm'] for w in witnesses)>1
      rows.append(dict(seed=entry['seed'],trained_mode=entry['mode'],state_sha256=entry['sha256'],encoder_numerical_rank=rank,rank_relative_tolerance=1e-8,min_singular=float(singular[-1]),max_singular=float(singular[0]),condition_number=float(singular[0]/singular[-1]),decoder_column_norm_range=[float(np.linalg.norm(D,axis=0).min()),float(np.linalg.norm(D,axis=0).max())],witnesses=witnesses))
    result=dict(status='passed',source_sha256=sha(__file__),models_sha256=sha(V/'continuous_training_results.json'),map_sha256=sha(V.parent/'continuous_block_shrinkage.py'),toy_equal_decoder_signed_gaps=vals,positive_domain_max_error=max(errors),overlapping_group_gaps=overlap_gaps,models=rows,scope='Constructed standardized-write input space. Continuous-trained states are also evaluated with a hypothetical hard gate to test the theorem assumptions; their actual continuous map does not jump. No claim these witnesses lie on the native activation manifold or occur on text.')
    dest=V/'sparse_continuity_constraints.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n');print('TOYS PASSED')
    for row in rows:print(row['seed'],row['trained_mode'],'RANK',row['encoder_numerical_rank'],'SINGULAR_RANGE',row['min_singular'],row['max_singular'],'JUMPS',[w['predicted_hard_limit_norm'] for w in row['witnesses']])
if __name__=='__main__':main()
