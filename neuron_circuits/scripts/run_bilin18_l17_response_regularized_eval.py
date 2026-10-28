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
from bilin18_readout_feature_families import features as family_features
from run_bilin18_l17_fixed_readout import read_features
from bilin18_residual_cross_program import intact_forward
STEM=ART/'bilin18_l17_response_regularized_eval'
@torch.inference_mode()
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();block=model.blocks[17];L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');pr=load_program(ART/'bilin18_l17_residual_cross_actual_refined_program.npz','cuda');selection=json.loads((ART/'bilin18_l17_response_regularized_selection.json').read_text());f=load_program(ART/'bilin18_l17_response_regularized_frozen.npz','cuda');control_key=min((k for k in selection['records'] if k.startswith('lambda0_')),key=lambda k:selection['records'][k]['kl']['validation']);f['optimization_control']=f[control_key];ws={n:f[n] for n in ['base','h280','selected','optimization_control']}
    corpus=np.load(ROOT/cfg['corpus'],mmap_mode='r');pre,heads,replay=collect_sources(model,corpus,cfg);x=F.rms_norm(pre,(1152,));native=B(x,x,L,R,D);compiled=forward_normalized(x,p);features=read_features(x,pr,f['feature_mean'],f['feature_std']);rows={};gen=torch.Generator(device='cuda').manual_seed(cfg['seed']);audit={'angle_norm_relative_max':0.,'angle_dot_relative_max':0.}
    def logits(state):return 30*torch.tanh(model.lm_head(F.rms_norm(state,(1152,)))/30)
    def save(key,v):rows[key]=v.reshape(128,4).sum(1).cpu().numpy()
    intact_base=compiled+intact_forward(x,pr);intact_metrics={n:[] for n in ws}
    for i in range(0,len(pre),16):
        sl=slice(i,i+16);lp=logits(pre[sl]+native[sl]+block.mlp.Down_bias).double().log_softmax(-1)
        for n,w in ws.items():
            q=logits(pre[sl]+intact_base[sl]+features[sl]@w+block.mlp.Down_bias).double().log_softmax(-1);intact_metrics[n].append((lp.exp()*(lp-q)).sum(-1))
    for n,v in intact_metrics.items():save('intact_'+n+'_kl',torch.cat(v))
    for h in cfg['heads']:
        actual=heads[:,h];source=torch.linalg.qr(block.attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;angle=angle_matched_payload(pre,actual,source,torch.randn(len(pre),128,generator=gen,device='cuda'));audit['angle_norm_relative_max']=max(audit['angle_norm_relative_max'],float(((angle.norm(dim=-1)-actual.norm(dim=-1)).abs()/actual.norm(dim=-1).clamp_min(1e-20)).max()));audit['angle_dot_relative_max']=max(audit['angle_dot_relative_max'],float(((pre*angle).sum(-1)-(pre*actual).sum(-1)).abs().max()/(pre.norm(dim=-1)*actual.norm(dim=-1)).max()))
        for label,pay in [('actual',actual),('angle',angle)]:
            for scale in cfg['scales']:
                changed=pre-scale*pay;z=F.rms_norm(changed,(1152,));gold=B(z,z,L,R,D);cdelta=forward_normalized(z,p)-compiled;target=(gold-native)-cdelta;prefix=f'{label}_s{scale}_h{h}';base_delta=change(x,z,pr);df=((z-x)@torch.cat([pr['core'],pr['partner']],1))/f['feature_std'];deltas={'compiled':torch.zeros_like(target),'oracle':target,**{n:base_delta+df@w for n,w in ws.items()}}
                save(prefix+'_energy',target.square().sum(-1))
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
            for n in ['compiled','base','h280','selected','optimization_control','oracle']:
                kl=total(n+'_kl');er=total(n+'_error');results[prefix][n]={'residual_fidelity':float(1-er.sum()/energy.sum()),'mean_conditional_kl':float(kl.sum()/(128*4*9)),'kl_recovery':float(1-kl.sum()/klbase.sum()),'kl_recovery_bootstrap95':np.quantile(1-kl[ix].sum(1)/klbase[ix].sum(1),[.025,.975]).tolist(),'top1_agreement':float(total(n+'_agreement').sum()/(128*4*9))}
            for key,n,b in [('selected_vs_base','selected','base'),('selected_vs_h280','selected','h280')]:
                baseline=total(b+'_kl');gain=baseline-total(n+'_kl');contrasts[prefix][key]={'relative_kl_reduction':float(gain.sum()/baseline.sum()),'paired95':np.quantile(gain[ix].sum(1)/baseline[ix].sum(1),[.025,.975]).tolist()}
    def contrast(n,b):
        gain=b-n;return {'relative_reduction':float(gain.sum()/b.sum()),'paired95':np.quantile(gain[ix].sum(1)/b[ix].sum(1),[.025,.975]).tolist()}
    intact={n:float(rows['intact_'+n+'_kl'].sum()/512) for n in ws};state=contrast(rows['intact_selected_kl'],rows['intact_h280_kl']);groups={}
    for label in cfg['payloads']:
        for group,hs in [('fit',cfg['fit_heads']),('heldout',cfg['heldout_heads'])]:
            group_results={}
            for metric in ['error','kl']:
                vals={n:sum(rows[f'{label}_s1.0_h{h}_{n}_{metric}'] for h in hs) for n in ws}
                group_results[metric]={b:contrast(vals['selected'],vals[b]) for b in ['base','h280','optimization_control']}
            groups[label+'_'+group]=group_results
    gates={'state_retention':state['paired95'][0]>=-.02,'response_gain':all(groups[label+'_fit']['error']['h280']['relative_reduction']>=.05 and groups[label+'_fit']['error']['h280']['paired95'][0]>0 for label in cfg['payloads']),'base_response_retention':all(groups[label+'_fit']['error']['base']['paired95'][0]>=-.02 for label in cfg['payloads']),'heldout_transfer':all(groups[label+'_heldout']['error']['h280']['relative_reduction']>=.05 and groups[label+'_heldout']['error']['h280']['paired95'][0]>0 for label in cfg['payloads'])}
    per_head={}
    for label in cfg['payloads']:
        for scale in cfg['scales']:
            for h in cfg['heads']:
                prefix=f'{label}_s{scale}_h{h}';per_head[prefix]={n:{'mean_conditional_kl':float(rows[prefix+'_'+n+'_kl'].sum()/512),'residual_fidelity':float(1-rows[prefix+'_'+n+'_error'].sum()/rows[prefix+'_energy'].sum())} for n in deltas}
    np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'selection':selection,'intact_kl':intact,'state_contrast':state,'secondary_control':{'key':control_key,'scope':'Post-primary exploratory audit; choice uses validation only; same fresh panel reused','intact_contrast':contrast(rows['intact_selected_kl'],rows['intact_optimization_control_kl'])},'group_contrasts':groups,'audits':audit,'results':results,'per_head':per_head,'contrasts':contrasts,'gates':gates,'head_replay':replay},indent=2));print('GATES',gates,'CONTRASTS',contrasts,flush=True)
if __name__=='__main__':main()
