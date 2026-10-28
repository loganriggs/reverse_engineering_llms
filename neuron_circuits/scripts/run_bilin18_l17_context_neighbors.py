"""H272 directly optimize fixed-rank partner projection, frozen native coefficients."""
import json,hashlib
import sys
from types import SimpleNamespace
def minimum_norm(gram):
    w=np.ones(len(gram))/len(gram)
    for iteration in range(20000):
        gw=gram@w;j=int(np.argmin(gw));gap=float(w@gw-gw[j])
        if gap<1e-10:break
        d=-w.copy();d[j]+=1;alpha=min(1.,max(0.,gap/float(d@gram@d)));w+=alpha*d
    return SimpleNamespace(x=w,success=gap<1e-8,message=f"Frank-Wolfe iteration {iteration}, gap {gap}")
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program,forward_normalized,B
from run_bilin18_l17_partner_compression import cross_response
from run_bilin18_l17_core_interaction_factorial import quadratic_terms
from run_bilin18_l17_residual_term_accounting import mean_linear
from run_bilin18_l8_execution_alignment import angle_matched_payload
STEM=ART/'bilin18_l17_direct_partner_fit'
def partner_q(w,core):return torch.linalg.qr(w-core@(core.T@w),mode='reduced').Q

@torch.inference_mode()
def main():
    global STEM
    if len(sys.argv)>1:STEM=ART/sys.argv[1]
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float()
    L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');f=load_program(ART/'bilin18_l17_conditional_partner_rank_bases.npz','cuda');c=torch.linalg.qr(f['core']).Q;mu=f['mean'];base={n:partner_q(f[n][:,:cfg.get('rank',32)],c) for n in ['weighted','random0']}
    with torch.no_grad():
        pre,heads,_=collect_sources(model,np.load(ROOT/cfg['discovery'],mmap_mode='r'),{**cfg,'rows':1024});x=F.rms_norm(pre,(1152,));zs_actual=[F.rms_norm(pre-heads[:,h],(1152,)) for h in cfg['heads']];zs_angle=[];rng=torch.Generator(device='cuda').manual_seed(cfg['payload_seed'])
        for h in cfg['heads']:
            source=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;pay=angle_matched_payload(pre,heads[:,h],source,torch.randn(len(pre),128,generator=rng,device='cuda'));zs_angle.append(F.rms_norm(pre-pay,(1152,)))
    path=ART/'bilin18_l17_document_repartition_responses.npy';assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['response_sha256'];responses=torch.tensor(np.load(path),device='cuda').reshape(18,4096,1152);zs=zs_actual+zs_angle;cut=3072;context=F.normalize(x-x[:cut].mean(0),dim=-1);ctx=context[cut:]@context[:cut].T;mask=torch.arange(1024,device='cuda')[:,None]%4==torch.arange(3072,device='cuda')[None,:]%4;gen=torch.Generator(device='cuda').manual_seed(cfg['neighbor_seed']);random=torch.rand(1024,3072,device='cuda',generator=gen).masked_fill(~mask,-torch.inf).topk(16,dim=1).indices;rows={};indices={};neighborsim={}
    # Replay a small response slice to verify cached-target/seed/geometry alignment.
    frozen=load_program(ART/'bilin18_l17_partner_common_descent_rank64_frozen.npz','cuda');q=partner_q(frozen['optimized'],c);replay=quadratic_terms(x[:256],zs[0][:256],mu,c,L,R,D,p)[:,1]-cross_response(x[:256],zs[0][:256],mu,c,q,L,R,D,p);relative=float((replay-responses[0,:256]).norm()/responses[0,:256].norm());print('replay relative',relative,flush=True);assert relative<1e-5
    for g,z in enumerate(zs):
        delta=F.normalize(z-x,dim=-1);dscore=delta[cut:]@delta[:cut].T;scores={'context':ctx,'delta':dscore,'joint':(ctx+dscore)/2};Y=F.normalize(responses[g],dim=-1);ref=Y[:cut];target=Y[cut:];chosen={'random':random}
        for name,sim in scores.items():chosen[name]=sim.masked_fill(~mask,-torch.inf).topk(16,dim=1).indices
        for name,ix in chosen.items():
            indices[f'g{g}_{name}']=ix.cpu().numpy()
            for k in cfg['neighbor_counts']:
                pred=F.normalize(ref[ix[:,:k]].mean(1),dim=-1);cos=(pred*target).sum(-1);rows[f'g{g}_{name}{k}']=cos.reshape(256,4).mean(1).cpu().numpy()
        print('group',g,'complete',flush=True)
    rng=np.random.default_rng(cfg['neighbor_seed']+1);boot=rng.integers(256,size=(2000,256));summary={};contrasts={}
    for family,groups in [('actual',range(9)),('angle',range(9,18))]:
        summary[family]={};contrasts[family]={}
        for name in ['random','context','delta','joint']:
            for k in cfg['neighbor_counts']:summary[family][name+str(k)]=float(np.mean([rows[f'g{g}_{name}{k}'] for g in groups]))
        for label,new,old in [('joint_direction_gain','joint16','random16'),('joint_over_context','joint16','context16')]:
            diff=np.mean([rows[f'g{g}_{new}']-rows[f'g{g}_{old}'] for g in groups],axis=0);contrasts[family][label]={'gain':float(diff.mean()),'paired95':np.quantile(diff[boot].mean(1),[.025,.975]).tolist()}
    gates={label:all(v[label]['gain']>=threshold and v[label]['paired95'][0]>0 for v in contrasts.values()) for label,threshold in [('joint_direction_gain',.05),('joint_over_context',.02)]};np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);np.savez_compressed(STEM.with_name(STEM.name+'_indices.npz'),**indices);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'contrasts':contrasts,'gates':gates,'response_replay_relative':relative},indent=2)+'\n');print('FINISHED',summary,contrasts,gates,flush=True)
if __name__=='__main__':main()
