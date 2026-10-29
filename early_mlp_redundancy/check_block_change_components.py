"""M105 native-background group-change arithmetic and execution preflight."""
import json
import numpy as np
import torch
from fit_pca_encoder import V,A,sha
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer
from block_change_components import grouping,changes


def main():
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
    pp=V/'block_change_roles_protocol.json';protocol=json.loads(pp.read_text());assert protocol['preflight_source_sha256']==sha(__file__);assert protocol['helper_sha256']==sha(V.parent/'block_change_components.py')
    for name,h in protocol['inputs'].items():assert sha(V/name)==h
    native=json.loads((V/'training_causal_coverage.json').read_text());assert sha(native['samples_path'])==native['samples_sha256'];saved=torch.load(native['samples_path'],weights_only=True)
    pairs=json.loads((V/'training_causal_coverage_protocol.json').read_text())['pairs'];pair=next(p for p in pairs if p['id']==protocol['preflight_pair_id'])
    model,receipt=load_model();mlp=model.transformer.h[1].mlp;captures=[];bases=[];calls=0
    for ep in [0,1]:
        ids=torch.tensor([pair['token_ids'][ep]],device='cuda');out,cap=trace(model,ids);calls+=1;assert torch.equal(out[0,-1].cpu(),saved['baselines'][pair['id']+f'|{ep}']['logits']);captures.append(mlp.Down(cap[1]['products']));bases.append(out[0,-1])
    maxima={};rows=[];samples=[]
    def check(name,x,y,tol=1e-5):
        err=float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1.));maxima[name]=max(maxima.get(name,0.),err);assert err<tol,(name,err)
    for entry in protocol['models']:
        assert sha(entry['path'])==entry['sha256'];s=torch.load(entry['path'],weights_only=True);f=Featurizer('block',entry['seed']).cuda();f.load_state_dict(s['state']);mu=s['mean'].cuda();scale=s['scale'].cuda();D=torch.nn.functional.normalize(f.dictionary,dim=0);codes=[];recs=[]
        for w in captures:
            rec,support,code=f((w-mu)/scale);codes.append(code);recs.append(rec*scale+mu)
            # Independent encoder/gate/decode in NumPy float64 on the same native input.
            st=s['state'];x=(w.cpu().double().numpy()-s['mean'].double().numpy())/float(s['scale']);a=x@st['encoder'].double().numpy().T+st['encoder_bias'].double().numpy();M=st['membership'].double().numpy();scores=a*a@M.T;chosen=np.argsort(scores,axis=-1)[...,-8:];gates=np.zeros_like(scores);np.put_along_axis(gates,chosen,1,axis=-1);q=(gates@M>0);assert np.array_equal(q,support.cpu().numpy());check('independent_selected_code',code.cpu().numpy(),a*q)
        for ep in [0,1]:
            ids=torch.tensor([pair['token_ids'][ep]],device='cuda')
            for region in ['changed_input_positions','final_position']:
                target=next(x for x in native['records'] if x['pair_id']==pair['id'] and x['endpoint']==ep and x['layer']==1 and x['region']==region);assert target['status']=='evaluated';pos=target['recipient_positions'];dp=target['donor_positions'];full=recs[1-ep][:,dp]-recs[ep][:,pos]
                for shuffled in [False,True]:
                    groups=grouping(entry['seed'],shuffled);part=changes(codes[ep],codes[1-ep],D,scale,pos,dp,groups);check('sum_groups',part.sum(0).cpu().numpy(),full.cpu().numpy())
                    # Same selected coefficients, independent float64 decoder arithmetic.
                    DD=s['state']['dictionary'].double().numpy();DD/=np.linalg.norm(DD,axis=0);dc=codes[1-ep].cpu().double().numpy()[:,dp]-codes[ep].cpu().double().numpy()[:,pos]
                    for group in range(32):
                        atoms=groups[group];oracle=(dc[:,:,atoms]@DD[:,atoms].T)*float(s['scale']);check('independent_group_decode',part[group].cpu().numpy(),oracle)
                    for group in [0,31]:
                        delta=torch.zeros_like(captures[ep]);delta[:,pos]=part[group];out,_=trace(model,ids,replace=lambda layer,z,w:w+delta if layer==1 else w);calls+=1
                        # Explicit full MLP override verifies patching the native background, bias once.
                        original=mlp.forward
                        try:
                            mlp.forward=lambda z:original(z)+delta
                            override,_=trace(model,ids);calls+=1
                        finally:mlp.forward=original
                        assert torch.equal(out,override)
                        row=dict(seed=entry['seed'],arm=entry['arm'],endpoint=ep,region=region,grouping='shuffled' if shuffled else 'trained',group=group,noop=not bool(delta.count_nonzero()))
                        if row['noop']:assert torch.equal(out[0,-1],bases[ep])
                        rows.append(row);samples.append(dict(row=row,base=bases[ep].cpu(),output=out[0,-1].cpu(),delta=delta.cpu()))
        print('CHECKED',entry['seed'],entry['arm'],flush=True)
    dest=A/'emlp_m105_preflight_samples.pt';assert not dest.exists();torch.save(samples,dest)
    out=dict(status='passed',protocol_sha256=sha(pp),source_sha256=sha(__file__),forward_count=calls,conditions=len(rows),rows=rows,max_relative_errors=maxima,samples_path=str(dest),samples_sha256=sha(dest),scope='Four frozen block states, independently checked NumPy gates and all disjoint group decodes, exact group-sum withinfloat32tolerance. Two sampled groups pergrouping/site/endpoint/model checked via callback and native-module override. Native background retained; these are external encoded-change interventions, not native-neuron deletions. Preflight only; no role/transfer conclusion.')
    path=V/'block_change_roles_preflight.json';assert not path.exists();path.write_text(json.dumps(out,indent=2)+'\n');print('PASSED',calls,len(rows),maxima)
if __name__=='__main__':main()
