"""Exact two-head attention toy: reduction measure x coordinate prior."""

import json
from pathlib import Path

import numpy as np

V=12
GRID=721
PHYSICAL_ANGLES=(.21,.53,.91)
DECODER_GAIN=.005


def build(angle):
    # Each head is a scalar query-key score times a key-dependent value.
    q0=np.zeros((V,V));v0=np.zeros((V,2))
    q0[:4,4:8]=1.;q0[8,9]=15.
    v0[4:8,0]=1.;v0[9,1]=1.
    # A frequent contextual coactivation gives activation sparsity a
    # competing mixed direction without changing the labeled rule pairs.
    q0[:4,:4]=2.;v0[:4,:]=1.
    z0=q0[:,:,None]*v0[None,:,:]
    q1=np.zeros((V,V));v1=np.zeros(V)
    q1[10,11]=20.;v1[11]=1.
    z1=q1*v1[None,:]
    semantic=np.concatenate([z0,z1[:,:,None]],axis=-1)
    c,s=np.cos(angle),np.sin(angle)
    r=np.array([[c,-s],[s,c]])
    native=semantic.copy();native[:,:,:2]=semantic[:,:,:2]@r
    decoder=np.zeros((3,3));decoder[:2,:2]=r.T;decoder[2,2]=DECODER_GAIN
    out=native@decoder
    assert np.max(np.abs(out-semantic*np.array([1.,1.,DECODER_GAIN])))<1e-12
    return native,decoder,out,r


def measures():
    uniform=np.full((V,V),1/V**2)
    p=np.array([.105]*8+[.01,.01,.07,.07]);assert abs(p.sum()-1)<1e-12
    natural=np.outer(p,p)
    return {'uniform':uniform,'usage':natural,
            **{f'floor_{a:g}':a*uniform+(1-a)*natural for a in (.1,.25,.5)}}


def relerror(original,approx,measure):
    delta=original-approx
    return float(np.sum(measure[:,:,None]*delta*delta)/
                 np.sum(measure[:,:,None]*original*original))


def candidates(native,decoder,out,measure):
    angles=np.linspace(0,np.pi,GRID,endpoint=False)
    axes=[np.array([np.cos(a),np.sin(a),0.]) for a in angles]
    axes.append(np.array([0.,0.,1.]))
    rows=[]
    for i,u in enumerate(axes):
        projected=(native@u)[:,:,None]*(u@decoder)[None,None,:]
        hidden=(native@u)[:,:,None]*u[None,None,:]
        rows.append(dict(candidate='decoy_head' if i==len(angles) else 'rule_head',
            angle=None if i==len(angles) else float(angles[i]),
            output_mse=relerror(out,projected,measure),
            hidden_mse=relerror(native,hidden,measure)))
    return min(rows,key=lambda r:r['output_mse']),min(rows,key=lambda r:r['hidden_mse'])


def candidate_output(native,decoder,row):
    u=(np.array([0.,0.,1.]) if row['candidate']=='decoy_head' else
       np.array([np.cos(row['angle']),np.sin(row['angle']),0.]))
    return (native@u)[:,:,None]*(u@decoder)[None,None,:]


def stratum_errors(out,approx):
    masks={'common_rule':(slice(0,4),slice(4,8)),
           'rare_rule':(8,9),
           'coactive_context':(slice(0,4),slice(0,4)),
           'decoy_pair':(10,11)}
    result={}
    for name,ix in masks.items():
        reference=out[ix];candidate=approx[ix]
        result[name]=float(np.sum((reference-candidate)**2)/
                           max(np.sum(reference**2),1e-30))
    return result


def prior_scan(native,decoder,measure):
    # Exact orthogonal rotations inside head 0: sum of both axis
    # contributions equals the same pair function for every angle.
    angles=np.linspace(0,np.pi/2,GRID,endpoint=False)
    rows=[]
    for a in angles:
        c,s=np.cos(a),np.sin(a)
        b=np.array([[c,-s],[s,c]])
        coordinates=native[:,:,:2]@b
        reconstructed=coordinates@(b.T@decoder[:2,:])
        assert np.max(np.abs(reconstructed-native[:,:,:2]@decoder[:2,:]))<1e-11
        # Native coefficient sparsity is coordinate-dependent. The two
        # function-level response scores need not have the same optimum.
        native_basis_l1=float(np.sum(np.abs(b)))
        response_l1=float(np.sum(measure[:,:,None]*np.abs(coordinates)))
        contrib=np.stack([coordinates[:,:,j,None]*(b[:,j]@decoder[:2,:])[None,None,:]
                          for j in range(2)],axis=-2)
        output_l1=float(np.sum(measure[:,:,None]*np.linalg.norm(contrib,axis=-1)))
        # Edit ensemble: isolated common and rare rule pairs, equally
        # weighted; no answer labels are supplied to the score.
        edit=np.stack([native[0,4,:2],native[8,9,:2]])@b
        edit_l1=float(np.mean(np.sum(np.abs(edit),axis=-1)))
        rows.append(dict(angle=float(a),native_basis_l1=native_basis_l1,
                         pair_response_l1=response_l1,
                         output_contribution_l1=output_l1,
                         edit_response_l1=edit_l1))
    selected={name:min(rows,key=lambda r:r[name]) for name in
              ('native_basis_l1','pair_response_l1','output_contribution_l1',
               'edit_response_l1')}
    return selected


def run():
    records=[]
    for seed,physical_angle in enumerate(PHYSICAL_ANGLES):
        native,decoder,out,r=build(physical_angle)
        for name,measure in measures().items():
            output_best,hidden_best=candidates(native,decoder,out,measure)
            for winner in (output_best,hidden_best):
                approx=candidate_output(native,decoder,winner)
                winner['cross_metric_output_mse']={m:relerror(out,approx,weight)
                    for m,weight in measures().items()}
                winner['stratum_output_mse']=stratum_errors(out,approx)
            priors=prior_scan(native,decoder,measure)
            # Geometric alignment with the two ground-truth rule axes in
            # the native coordinates. It is evaluation only.
            for prior,row in priors.items():
                a=row['angle'];c,s=np.cos(a),np.sin(a)
                b=np.array([[c,-s],[s,c]])
                overlap=np.abs(r@b)
                row['mean_rule_alignment']=float(max(overlap[0,0]+overlap[1,1],
                                                      overlap[0,1]+overlap[1,0])/2)
            records.append(dict(seed=seed,physical_angle=physical_angle,
                measure=name,output_best=output_best,hidden_best=hidden_best,
                priors=priors))
    result=dict(description='Exact two-head, 12-token, 3-latent pair function with common/rare rules and high-response low-output decoy; data-free uniform and declared usage measures',
        vocab=V,physical_angles=PHYSICAL_ANGLES,decoder_gain=DECODER_GAIN,
        grid=GRID,records=records)
    Path(__file__).with_name('toy_causal_decoy_matrix.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in records:
        print(r['seed'],r['measure'],'rank1 output',r['output_best']['candidate'],
              'hidden',r['hidden_best']['candidate'],
              'prior angles',{k:round(v['angle'],3) for k,v in r['priors'].items()},flush=True)


if __name__=='__main__':run()
