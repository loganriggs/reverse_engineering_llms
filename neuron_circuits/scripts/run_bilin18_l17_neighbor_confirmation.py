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
    path=ART/'bilin18_l17_document_repartition_responses.npy';assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['response_sha256'];Y=torch.tensor(np.load(path),device='cuda').reshape(18,4096,1152)[:,:3072];xr=x[:3072];zr=[z[:3072] for z in zs_actual+zs_angle];center=xr.mean(0)
    pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),{**cfg,'rows':128});x=F.rms_norm(pre,(1152,));zs=[];gen=torch.Generator(device='cuda').manual_seed(cfg['fresh_payload_seed']);actual=[F.rms_norm(pre-heads[:,h],(1152,)) for h in range(9)]
    for h in range(9):
        source=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;pay=angle_matched_payload(pre,heads[:,h],source,torch.randn(len(pre),128,generator=gen,device='cuda'));zs.append(F.rms_norm(pre-pay,(1152,)))
    zs=actual+zs;ctx=F.normalize(x-center,dim=-1)@F.normalize(xr-center,dim=-1).T;mask=torch.arange(512,device='cuda')[:,None]%4==torch.arange(3072,device='cuda')[None,:]%4;gen.manual_seed(cfg['neighbor_seed']);random=torch.rand(512,3072,device='cuda',generator=gen).masked_fill(~mask,-torch.inf).topk(16,dim=1).indices;frozen=load_program(ART/'bilin18_l17_partner_common_descent_rank64_frozen.npz','cuda');q=partner_q(frozen['optimized'],c);rows={};indices={}
    for g,z in enumerate(zs):
        delta=F.normalize(z-x,dim=-1)@F.normalize(zr[g]-xr,dim=-1).T;choices={'random':random}
        for name,sim in [('context',ctx),('delta',delta),('joint',(ctx+delta)/2)]:choices[name]=sim.masked_fill(~mask,-torch.inf).topk(16,dim=1).indices
        target=torch.cat([quadratic_terms(x[i:i+256],z[i:i+256],mu,c,L,R,D,p)[:,1]-cross_response(x[i:i+256],z[i:i+256],mu,c,q,L,R,D,p) for i in range(0,512,256)]);unit=F.normalize(Y[g],dim=-1);preds={};directions={}
        for name,ix in choices.items():preds[name]=Y[g][ix].mean(1);directions[name]=unit[ix].mean(1);indices[f'g{g}_{name}']=ix.cpu().numpy()
        preds['additive']=(preds['context']+preds['delta'])/2;directions['additive']=(directions['context']+directions['delta'])/2;rows[f'g{g}_energy']=target.double().square().sum(-1).reshape(128,4).sum(1).cpu().numpy()
        for name,pred in preds.items():
            rows[f'g{g}_{name}_cos']=(F.normalize(directions[name],dim=-1)*F.normalize(target,dim=-1)).sum(-1).reshape(128,4).mean(1).cpu().numpy();rows[f'g{g}_{name}_error']=(pred.double()-target.double()).square().sum(-1).reshape(128,4).sum(1).cpu().numpy()
        print('group',g,'complete',flush=True)
    boot=np.random.default_rng(cfg['neighbor_seed']+1).integers(128,size=(2000,128));summary={};contrasts={}
    for family,groups in [('actual',range(9)),('angle',range(9,18))]:
        energy=sum(rows[f'g{g}_energy'] for g in groups);summary[family]={};contrasts[family]={}
        for name in ['random','context','delta','joint','additive']:
            err=sum(rows[f'g{g}_{name}_error'] for g in groups);summary[family][name]={'cosine':float(np.mean([rows[f'g{g}_{name}_cos'] for g in groups])),'raw_fidelity':float(1-err.sum()/energy.sum()),'raw_fidelity95':np.quantile(1-err[boot].sum(1)/energy[boot].sum(1),[.025,.975]).tolist()}
        for name,old in [('joint_random','random'),('joint_additive','additive')]:
            diff=np.mean([rows[f'g{g}_joint_cos']-rows[f'g{g}_{old}_cos'] for g in groups],axis=0);contrasts[family][name]={'gain':float(diff.mean()),'paired95':np.quantile(diff[boot].mean(1),[.025,.975]).tolist()}
    gates={name:all(v[name]['gain']>=threshold and v[name]['paired95'][0]>0 for v in contrasts.values()) for name,threshold in [('joint_random',.05),('joint_additive',.02)]};gates['positive_reconstruction']=all(v['joint']['raw_fidelity']>=.05 and v['joint']['raw_fidelity95'][0]>0 for v in summary.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);np.savez_compressed(STEM.with_name(STEM.name+'_indices.npz'),**indices);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'contrasts':contrasts,'gates':gates},indent=2)+'\n');print('FINISHED',summary,contrasts,gates,flush=True)
if __name__=='__main__':main()
