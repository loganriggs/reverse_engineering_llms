"""H275 compile exact centered residual corrections and evaluate conditional behavior."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program,forward_normalized,B
from run_bilin18_l17_core_interaction_factorial import quadratic_terms
from run_bilin18_l17_residual_term_accounting import mean_linear
from run_bilin18_l17_partner_compression import cross_response
from run_bilin18_l17_direct_partner_fit import partner_q
from run_bilin18_l8_execution_alignment import angle_matched_payload
from bilin18_residual_cross_program import compile_program,forward,counts,change
STEM=ART/'bilin18_l17_cross_program_behavior'
@torch.inference_mode()
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();block=model.blocks[17];L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');pd={k:v.double() for k,v in p.items()};Ld,Rd,Dd=[v.double() for v in [L,R,D]];f=load_program(ART/'bilin18_l17_partner_objective_mixture_frozen.npz','cuda',torch.float64);c=torch.linalg.qr(f['core']).Q;mu=f['mean'];qs={n:partner_q(f[n],c) for n in ['actual','mixture']};programs64={n:compile_program(mu,c,q,Ld,Rd,Dd,pd) for n,q in qs.items()};programs={n:{k:v.float() for k,v in pr.items()} for n,pr in programs64.items()};costs={}
    for n,pr in programs.items():
        path=ART/f'bilin18_l17_residual_cross_{n}_program.npz';np.savez_compressed(path,**{k:v.cpu().numpy() for k,v in pr.items()});costs[n]={'array_scalars':counts(pr),'total_scalars':sum(counts(pr).values()),'uncompressed_float32_bytes':4*sum(counts(pr).values()),'saved_bytes':path.stat().st_size,'native_mlp_weight_scalars':L.numel()+R.numel()+D.numel(),'original_compiler_scalars':sum(v.numel() for v in p.values())}
    corpus=np.load(ROOT/cfg['corpus'],mmap_mode='r');pre,heads,replay=collect_sources(model,corpus,cfg);x=F.rms_norm(pre,(1152,));native=B(x,x,L,R,D);compiled=forward_normalized(x,p);intact={n:forward(x,pr) for n,pr in programs.items()};intact['common']=forward(x,programs['actual'],False);rows={};gen=torch.Generator(device='cuda').manual_seed(cfg['seed']);audit={'float64_relative_max':0.,'float32_relative_max':0.,'angle_norm_relative_max':0.,'angle_dot_relative_max':0.}
    def logits(state):return 30*torch.tanh(model.lm_head(F.rms_norm(state,(1152,)))/30)
    def save(key,v):rows[key]=v.reshape(128,4).sum(1).cpu().numpy()
    for h in cfg['heads']:
        actual=heads[:,h];source=torch.linalg.qr(block.attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;angle=angle_matched_payload(pre,actual,source,torch.randn(len(pre),128,generator=gen,device='cuda'));audit['angle_norm_relative_max']=max(audit['angle_norm_relative_max'],float(((angle.norm(dim=-1)-actual.norm(dim=-1)).abs()/actual.norm(dim=-1).clamp_min(1e-20)).max()));audit['angle_dot_relative_max']=max(audit['angle_dot_relative_max'],float(((pre*angle).sum(-1)-(pre*actual).sum(-1)).abs().max()/(pre.norm(dim=-1)*actual.norm(dim=-1)).max()))
        for label,pay in [('actual',actual),('angle',angle)]:
            for scale in cfg['scales']:
                changed=pre-scale*pay;z=F.rms_norm(changed,(1152,));gold=B(z,z,L,R,D);cdelta=forward_normalized(z,p)-compiled;target=(gold-native)-cdelta;prefix=f'{label}_s{scale}_h{h}';deltas={n:change(x,z,pr) for n,pr in programs.items()};deltas['common']=change(x,z,programs['actual'],False);deltas['compiled']=torch.zeros_like(target);deltas['oracle']=target;save(prefix+'_energy',target.square().sum(-1))
                xd=x.double();zd=z.double();base=mean_linear(zd-xd,mu,Ld,Rd,Dd,pd)+quadratic_terms(xd,zd,mu,c,Ld,Rd,Dd,pd)[:,0]
                for n,q in qs.items():
                    exact=base+cross_response(xd,zd,mu,c,q,Ld,Rd,Dd,pd);reconstructed=change(xd,zd,programs64[n]);audit['float64_relative_max']=max(audit['float64_relative_max'],float((reconstructed-exact).norm()/exact.norm()));audit['float32_relative_max']=max(audit['float32_relative_max'],float((deltas[n].double()-exact).norm()/exact.norm()))
                for n,v in deltas.items():save(prefix+'_'+n+'_error',(v-target).square().sum(-1))
                metrics={n:[] for n in deltas};top={n:[] for n in deltas}
                for i in range(0,len(pre),16):
                    sl=slice(i,i+16);truth=logits(changed[sl]+gold[sl]+block.mlp.Down_bias);lp=truth.double().log_softmax(-1);prob=lp.exp()
                    for n,delta in deltas.items():
                        out=logits(changed[sl]+native[sl]+cdelta[sl]+delta[sl]+block.mlp.Down_bias);kl=(prob*(lp-out.double().log_softmax(-1))).sum(-1);metrics[n].append(kl);top[n].append((out.argmax(-1)==truth.argmax(-1)).float())
                for n in deltas:save(prefix+'_'+n+'_kl',torch.cat(metrics[n]));save(prefix+'_'+n+'_agreement',torch.cat(top[n]))
        print('head',h,'complete',flush=True)
    ix=np.random.default_rng(cfg['seed']+1).integers(128,size=(2000,128));results={};contrasts={}
    for label in cfg['payloads']:
        for scale in cfg['scales']:
            prefix=f'{label}_s{scale}';total=lambda m:sum(rows[f'{prefix}_h{h}_{m}'] for h in cfg['heads']);energy=total('energy');klbase=total('compiled_kl');results[prefix]={};contrasts[prefix]={}
            for n in ['compiled','common','actual','mixture','oracle']:
                kl=total(n+'_kl');er=total(n+'_error');results[prefix][n]={'residual_fidelity':float(1-er.sum()/energy.sum()),'mean_conditional_kl':float(kl.sum()/(128*4*9)),'kl_recovery':float(1-kl.sum()/klbase.sum()),'kl_recovery_bootstrap95':np.quantile(1-kl[ix].sum(1)/klbase[ix].sum(1),[.025,.975]).tolist(),'top1_agreement':float(total(n+'_agreement').sum()/(128*4*9))}
            for key,n,b in [('cross_behavior','actual','common'),('mixture_advantage','mixture','actual')]:
                baseline=total(b+'_kl');gain=baseline-total(n+'_kl');contrasts[prefix][key]={'relative_kl_reduction':float(gain.sum()/baseline.sum()),'paired95':np.quantile(gain[ix].sum(1)/baseline[ix].sum(1),[.025,.975]).tolist()}
    gates={'cross_behavior':all(contrasts[f'actual_s{s}']['cross_behavior']['relative_kl_reduction']>=.1 and contrasts[f'actual_s{s}']['cross_behavior']['paired95'][0]>0 for s in cfg['scales']),'mixture_angle_advantage':all(contrasts[f'angle_s{s}']['mixture_advantage']['relative_kl_reduction']>=.05 and contrasts[f'angle_s{s}']['mixture_advantage']['paired95'][0]>0 for s in cfg['scales']),'mixture_actual_retention':all(contrasts[f'actual_s{s}']['mixture_advantage']['paired95'][0]>-.03 for s in cfg['scales']),'half_kl':all(results[f'actual_s{s}']['actual']['kl_recovery']>=.5 for s in cfg['scales'])};np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'costs':costs,'audits':audit,'results':results,'contrasts':contrasts,'gates':gates,'head_replay':replay},indent=2));print('GATES',gates,'AUDIT',audit,flush=True)
if __name__=='__main__':main()
