"""Freeze decoder-geometry assignments and null permutations without task effects."""
import json
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from fit_pca_encoder import V,sha
from block_change_components import grouping


def main():
    torch.set_num_threads(1)
    pp=V/'block_change_roles_protocol.json';p=json.loads(pp.read_text());pre=json.loads((V/'block_change_roles_preflight.json').read_text());assert pre['status']=='passed' and pre['protocol_sha256']==sha(pp)
    models=p['models'];weights={}
    for e in models:
        assert sha(e['path'])==e['sha256'];s=torch.load(e['path'],weights_only=True);weights[e['seed'],e['arm']]=s['state']['dictionary'].double().numpy()
    rows=[];maximum=0.;permutations=np.stack([np.random.default_rng(10520+k).permutation(32) for k in range(100)])
    for arm in sorted({x['arm'] for x in models}):
        for shuffled in [False,True]:
            bases={};tbases={}
            for seed in [71,72]:
                D=weights[seed,arm];groups=grouping(seed,shuffled);bases[seed]=[];tbases[seed]=[]
                for atoms in groups:
                    block=D[:,atoms];assert np.linalg.matrix_rank(block)==4
                    q=np.linalg.qr(block)[0];assert np.allclose(q.T@q,np.eye(4),atol=1e-12);bases[seed].append(q)
                    tbases[seed].append(torch.linalg.svd(torch.from_numpy(block),full_matrices=False).U)
            overlap=np.array([[np.square(a.T@b).sum()/4 for b in bases[72]] for a in bases[71]])
            oracle=np.array([[float((a.T@b).square().sum()/4) for b in tbases[72]] for a in tbases[71]])
            maximum=max(maximum,float(np.max(np.abs(overlap-oracle))));assert np.allclose(overlap,oracle,atol=1e-12,rtol=1e-12)
            i,j=linear_sum_assignment(overlap,maximize=True);assert np.array_equal(i,np.arange(32)) and len(set(j))==32
            rows.append(dict(arm=arm,grouping='shuffled' if shuffled else 'trained',source_seed=71,target_seed=72,target_group_for_source=j.tolist(),overlap=overlap.tolist(),matched_mean_overlap=float(overlap[i,j].mean()),random_permutations=permutations.tolist(),random_mean_overlap=float(np.mean([overlap[i,perm].mean() for perm in permutations]))))
    out=dict(status='frozen',source_sha256=sha(__file__),protocol_sha256=sha(pp),preflight_sha256=sha(V/'block_change_roles_preflight.json'),rows=rows,max_independent_overlap_error=maximum,scope='Weight-only one-to-one assignment. NumPy QR versus independent Torch SVD subspace overlaps, all4096scores. Fixed100paired random permutations. No task responses read, no gains fitted; geometric overlap is not causal or semantic validation.')
    f=V/'block_change_role_matching.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print([(r['arm'],r['grouping'],r['matched_mean_overlap'],r['random_mean_overlap']) for r in rows]);print('MAX_ERROR',maximum)
if __name__=='__main__':main()
