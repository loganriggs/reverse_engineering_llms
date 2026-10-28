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
    path=ART/'bilin18_l17_document_repartition_responses.npy';assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['response_sha256'];Y=torch.tensor(np.load(path),device='cuda').reshape(18,4096,1152).double();frozen=load_program(ART/'bilin18_l17_partner_common_descent_rank64_frozen.npz','cuda');q=partner_q(frozen['optimized'],c).double();cc=c.double();xx=x.double()-mu.double();ux=xx@cc;vx=xx-ux@cc.T-(xx@q)@q.T;cut=3072;rows={};audit=[]
    def kernel(a,b):
        ua,va,wa,ta=a;ub,vb,wb,tb=b
        return (wa@wb.T)*(ta@tb.T)+(ua@ub.T)*(va@vb.T)-(wa@ub.T)*(ta@vb.T)-(ua@wb.T)*(va@tb.T)
    # Direct feature-product replay on a small deterministic slice.
    for g,z in enumerate(zs_actual+zs_angle):
        zz=z.double()-mu.double();uz=zz@cc;vz=zz-uz@cc.T-(zz@q)@q.T;features=(ux,vx,uz,vz);tr=tuple(v[:cut] for v in features);te=tuple(v[cut:] for v in features);K=kernel(tr,tr);Kt=kernel(te,tr)
        if g==0:
            a=tuple(v[:7] for v in tr);b=tuple(v[:5] for v in te);pa=(a[2][:,:,None]*a[3][:,None,:]-a[0][:,:,None]*a[1][:,None,:]).flatten(1);pb=(b[2][:,:,None]*b[3][:,None,:]-b[0][:,:,None]*b[1][:,None,:]).flatten(1);rel=float((kernel(a,b)-pa@pb.T).abs().max()/(pa@pb.T).abs().max());assert rel<1e-10
        blocks=[(ux+uz)/2,(vx+vz)/2,uz-ux,vz-vx];add=torch.zeros_like(K);addt=torch.zeros_like(Kt)
        for b in blocks:
            scale=b[:cut].square().sum(1).mean();add+=(b[:cut]@b[:cut].T)/scale;addt+=(b[cut:]@b[:cut].T)/scale
        target=Y[g,cut:];rows[f'g{g}_energy']=target.square().sum(-1).reshape(256,4).sum(1).cpu().numpy()
        for name,kk,kt in [('interaction',K,Kt),('additive',add,addt)]:
            scale=kk.diag().mean();kk=kk/scale;kt=kt/scale;ev,U=torch.linalg.eigh((kk+kk.T)/2);audit.append({'group':g,'kernel':name,'min_eigenvalue':float(ev.min()),'mean_diagonal':float(kk.diag().mean())});ev=ev.clamp_min(0);rhs=U.T@Y[g,:cut];left=kt@U
            for alpha in cfg['ridge']:
                pred=(left/(ev+alpha)[None,:])@rhs;rows[f'g{g}_{name}_a{alpha}_error']=(pred-target).square().sum(-1).reshape(256,4).sum(1).cpu().numpy()
        print('group',g,'complete',flush=True)
    summary={};selected={}
    for family,groups in [('actual',range(9)),('angle',range(9,18))]:
        energy=sum(rows[f'g{g}_energy'] for g in groups);summary[family]={};selected[family]={}
        for name in ['interaction','additive']:
            vals={str(alpha):sum(rows[f'g{g}_{name}_a{alpha}_error'] for g in groups) for alpha in cfg['ridge']};alpha=min(vals,key=lambda k:vals[k][:128].sum());selected[family][name]=alpha;summary[family][name]={'alpha':alpha,'calibration_fidelity':float(1-vals[alpha][:128].sum()/energy[:128].sum()),'evaluation_fidelity':float(1-vals[alpha][128:].sum()/energy[128:].sum()),'grid':{k:{'calibration':float(1-v[:128].sum()/energy[:128].sum()),'evaluation':float(1-v[128:].sum()/energy[128:].sum())} for k,v in vals.items()}}
    gates={'interaction_gain':all(v['interaction']['evaluation_fidelity']-v['additive']['evaluation_fidelity']>=.05 for v in summary.values()),'interaction20':all(v['interaction']['evaluation_fidelity']>=.2 for v in summary.values())};np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'selected':selected,'gates':gates,'kernel_audits':audit,'explicit_feature_kernel_relative_error':rel},indent=2)+'\n');print('FINISHED',summary,gates,flush=True)
if __name__=='__main__':main()
