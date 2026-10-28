"""H268 exact centered residual-polynomial response accounting, including cancellation."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program,forward_normalized,B
from run_bilin18_l8_execution_alignment import angle_matched_payload
STEM=ART/'bilin18_l17_residual_term_accounting'

def residual_bilinear(a,b,L,R,D,p):
    ma=torch.einsum('nk,okr->nor',a@p['reads'],p['eigenvectors']);mb=torch.einsum('nk,okr->nor',b@p['reads'],p['eigenvectors'])
    return B(a,b,L,R,D)-(ma*mb*p['eigenvalues'][None]).sum(-1)@p['outputs']

def mean_linear(delta,mean,L,R,D,p):
    z=mean@p['reads']-p['offset'];m=torch.einsum('k,okr->or',z,p['eigenvectors']);deriv=2*torch.einsum('or,or,okr->ok',m,p['eigenvalues'],p['eigenvectors'])+p['linear']
    return 2*B(mean[None].expand_as(delta),delta,L,R,D)-((delta@p['reads'])@deriv.T)@p['outputs']

def response_terms(x,z,mean,core,partner,linear,L,R,D,p):
    a=x-mean;b=z-mean;ca=(a@core)@core.T;cb=(b@core)@core.T;pa=(a@partner)@partner.T;pb=(b@partner)@partner.T;ra=a-ca-pa;rb=b-cb-pb;dx=z-x;kept=(dx@linear)@linear.T
    bil=lambda a,b:residual_bilinear(a,b,L,R,D,p)
    return torch.stack([mean_linear(kept,mean,L,R,D,p),mean_linear(dx-kept,mean,L,R,D,p),bil(ca+cb,cb-ca),bil(ca+cb,pb-pa)+bil(cb-ca,pa+pb),bil(ca+cb,rb-ra)+bil(cb-ca,ra+rb),bil(pa+ra+pb+rb,pb+rb-pa-ra)],1)

@torch.inference_mode()
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float();block=model.blocks[17];L,R,D=[m.weight.double() for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda',torch.float64);frozen=load_program(ART/'bilin18_l17_residual_tensor_partners_frozen.npz','cuda',torch.float64);mean=frozen['input_mean'];core=frozen['core'];partner=frozen['tensor_partner'];linear=frozen['linear_basis'];pre,heads,replay=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,)).double()
    def error(z):return B(z,z,L,R,D)-forward_normalized(z,p)
    e0=error(x);models={'linear624':[0],'full_linear':[0,1],'model_shape':[0,2,3],'model_shape_full_linear':[0,1,2,3],'core_all_partners':[0,1,2,3,4],'outside_plus_linear':[0,1,5],'all_quadratic_linear624':[0,2,3,4,5],'full':[0,1,2,3,4,5]};rows={};gen=torch.Generator(device='cuda').manual_seed(cfg['seed']);replay_max=0
    for h in cfg['heads']:
        actual=heads[:,h];source=torch.linalg.qr(block.attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;angle=angle_matched_payload(pre,actual,source,torch.randn(len(pre),128,generator=gen,device='cuda'))
        for payload,pay in [('actual',actual),('angle',angle)]:
            for scale in cfg['scales']:
                z=F.rms_norm(pre-scale*pay,(1152,)).double();target=error(z)-e0;t=response_terms(x,z,mean,core,partner,linear,L,R,D,p);prefix=f'{payload}_s{scale}_h{h}';energy=target.square().sum(-1);replay_max=max(replay_max,float((t.sum(1)-target).norm()/target.norm()));rows[prefix+'_energy']=energy.reshape(128,4).sum(1).cpu().numpy();rows[prefix+'_gram']=torch.einsum('ntd,nud->ntu',t,t).reshape(128,4,6,6).sum(1).cpu().numpy();rows[prefix+'_alignment']=(t*target[:,None]).sum(-1).reshape(128,4,6).sum(1).cpu().numpy()
                for name,ids in models.items():rows[prefix+'_'+name]=(t[:,ids].sum(1)-target).square().sum(-1).reshape(128,4).sum(1).cpu().numpy()
        print('head',h,'complete',flush=True)
    ix=np.random.default_rng(cfg['seed']+1).integers(128,size=(2000,128));results={};contrasts={}
    for payload in cfg['payloads']:
        for scale in cfg['scales']:
            prefix=f'{payload}_s{scale}';total=lambda m:sum(rows[f'{prefix}_h{h}_{m}'] for h in cfg['heads']);energy=total('energy');gram=total('gram').sum(0)/energy.sum();alignment=total('alignment').sum(0)/energy.sum();res={'term_energy_ratios':dict(zip(cfg['terms'],np.diag(gram).tolist())),'signed_target_alignments':dict(zip(cfg['terms'],alignment.tolist())),'normalized_gram':gram.tolist(),'sum_term_energy_over_response':float(np.trace(gram)),'normalized_sum_gram':float(gram.sum()),'models':{}}
            for name in models:
                er=total(name);res['models'][name]={'fidelity':float(1-er.sum()/energy.sum()),'document_bootstrap95':np.quantile(1-er[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist()}
            pairs={'omitted_partner_gain':('core_all_partners','model_shape_full_linear'),'outside_core_needed':('full','core_all_partners'),'linear_restriction':('model_shape_full_linear','model_shape')};contrasts[prefix]={}
            for name,(a,b) in pairs.items():
                d=total(b)-total(a);contrasts[prefix][name]={'gain':float(d.sum()/energy.sum()),'paired_document_bootstrap95':np.quantile(d[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist()}
            results[prefix]=res
    gates={name:all(contrasts[f'actual_s{s}'][name]['gain']>=threshold and contrasts[f'actual_s{s}'][name]['paired_document_bootstrap95'][0]>0 for s in cfg['scales']) for name,threshold in [('omitted_partner_gain',.2),('outside_core_needed',.2),('linear_restriction',.1)]};np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'models':models,'results':results,'contrasts':contrasts,'gates':gates,'relative_replay_error_max':replay_max,'head_replay':replay},indent=2));print('GATES',gates,'replay',replay_max,'RESULTS',results,flush=True)
if __name__=='__main__':main()
