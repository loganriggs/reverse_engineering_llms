"""H260 natural-state residual predictors tested on frozen causal source changes."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program,forward_normalized,B
from run_bilin18_l17_residual_feature_objective import raw_features
from run_bilin18_l8_execution_alignment import angle_matched_payload
STEM=ART/'bilin18_l17_residual_causal_transfer'

def mediation_state(changed,native_intact,compiled_change,predicted_change,bias):
    return changed+native_intact+compiled_change+predicted_change+bias

@torch.inference_mode()
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();block=model.blocks[17];L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]]
    program=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');frozen=load_program(ART/(cfg.get('frozen_stem','bilin18_l17_residual_feature_objective')+'_frozen.npz'),'cuda');selection=json.loads((ART/(cfg.get('selection_stem','bilin18_l17_residual_feature_objective')+'_selection.json')).read_text())['selected']
    corpus=np.load(ROOT/cfg['corpus'],mmap_mode='r');pre,heads,replay=collect_sources(model,corpus,cfg);x=F.rms_norm(pre,(1152,));native=B(x,x,L,R,D);compiled=forward_normalized(x,program);labels=torch.from_numpy(np.array(corpus[:,np.array(cfg['positions'])+1],dtype=np.int64).flatten()).cuda()
    def predict(z):
        outputs={}
        for family in ['affine1152','quadratic1152']:
            f=(raw_features(z,frozen['basis'],frozen['input_mean'],family)-frozen[family+'_mean'])/frozen[family+'_std'];f=torch.cat([f,torch.ones(len(f),1,device='cuda')],1)
            for name in cfg['models']:
                if name.startswith(family+'_'):outputs[name]=f@frozen[selection[name]]
        return outputs
    intact=predict(x);rows={};angle_audit=[];gen=torch.Generator(device='cuda').manual_seed(cfg['seed'])
    def logits(s):return 30*torch.tanh(model.lm_head(F.rms_norm(s,(1152,)))/30)
    def save(key,v):rows[key]=v.reshape(cfg['rows'],-1).sum(1).cpu().numpy()
    for h in cfg['heads']:
        source=torch.linalg.qr(block.attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;actual=heads[:,h];control=angle_matched_payload(pre,actual,source,torch.randn(len(pre),128,generator=gen,device='cuda'));angle_audit.append({'norm_relative_max':float(((control.norm(dim=-1)-actual.norm(dim=-1)).abs()/actual.norm(dim=-1).clamp_min(1e-20)).max()),'dot_relative_max':float(((pre*control).sum(-1)-(pre*actual).sum(-1)).abs().max()/(pre.norm(dim=-1)*actual.norm(dim=-1)).max())})
        for payload,p in [('actual',actual),('angle',control)]:
            for scale in cfg['scales']:
                key=f'{payload}_s{scale}_h{h}';changed=pre-scale*p;z=F.rms_norm(changed,(1152,));gold=B(z,z,L,R,D);cdelta=forward_normalized(z,program)-compiled;residual_delta=(gold-native)-cdelta;pred=predict(z);deltas={name:pred[name]-intact[name] for name in cfg['models']};deltas['compiled']=torch.zeros_like(residual_delta);deltas['oracle']=residual_delta
                save(key+'_residual_energy',residual_delta.square().sum(-1));save(key+'_write_energy',(gold-native).square().sum(-1))
                for name,delta in deltas.items():save(key+'_'+name+'_residual_error',(delta-residual_delta).square().sum(-1))
                metrics={}
                for i in range(0,len(pre),16):
                    sl=slice(i,i+16);truth=logits(changed[sl]+gold[sl]+block.mlp.Down_bias);baseline=logits(changed[sl]+native[sl]+block.mlp.Down_bias);ce_truth=F.cross_entropy(truth,labels[sl],reduction='none');cebase=F.cross_entropy(baseline,labels[sl],reduction='none')
                    vals={'logit_effect_energy':(truth-baseline).square().sum(-1),'ce_effect_energy':(ce_truth-cebase).square()}
                    for name,delta in deltas.items():
                        q=logits(mediation_state(changed[sl],native[sl],cdelta[sl],delta[sl],block.mlp.Down_bias));vals[name+'_kl']=(truth.double().softmax(-1)*(truth.double().log_softmax(-1)-q.double().log_softmax(-1))).sum(-1);vals[name+'_logit_error']=(q-truth).square().sum(-1);vals[name+'_ce_error']=(F.cross_entropy(q,labels[sl],reduction='none')-ce_truth).square()
                    for name,v in vals.items():metrics.setdefault(name,[]);metrics[name].append(v)
                for name,v in metrics.items():save(key+'_'+name,torch.cat(v))
        print('head',h,'complete',flush=True)
    results={};contrasts={};ix=np.random.default_rng(cfg['seed']+1).integers(cfg['rows'],size=(2000,cfg['rows']))
    for payload in cfg['payloads']:
        for scale in cfg['scales']:
            for group,hs in [('pooled',cfg['heads'])]+[(f'h{h}',[h]) for h in cfg['heads']]:
                key=f'{payload}_s{scale}_{group}'
                def total(metric):return sum(rows[f'{payload}_s{scale}_h{h}_{metric}'] for h in hs)
                energy=total('residual_energy');klbase=total('compiled_kl');results[key]={}
                for name in cfg['models']+['compiled','oracle']:
                    err=total(name+'_residual_error');kl=total(name+'_kl');results[key][name]={'residual_change_fidelity':float(1-err.sum()/energy.sum()),'residual_fidelity_document_bootstrap95':np.quantile(1-err[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist(),'mean_conditional_kl':float(kl.sum()/(len(pre)*len(hs))),'relative_kl_improvement':float(1-kl.sum()/klbase.sum()),'logit_effect_fidelity':float(1-total(name+'_logit_error').sum()/total('logit_effect_energy').sum()),'ce_effect_fidelity':float(1-total(name+'_ce_error').sum()/total('ce_effect_energy').sum())}
                if group=='pooled' and 'quadratic1152_sensitivity' in cfg['models']:
                    qe=total('quadratic1152_sensitivity_residual_error');ae=total('affine1152_sensitivity_residual_error');qkl=total('quadratic1152_sensitivity_kl');contrasts[key]={'quadratic_residual_gain':float((ae-qe).sum()/energy.sum()),'quadratic_gain_bootstrap95':np.quantile((ae-qe)[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist(),'behavior_gain':float((klbase-qkl).sum()/klbase.sum()),'behavior_gain_bootstrap95':np.quantile((klbase-qkl)[ix].sum(1)/klbase[ix].sum(1),[.025,.975]).tolist()}
    actual=[f'actual_s{s}_pooled' for s in cfg['scales']] if 'quadratic1152_sensitivity' in cfg['models'] else [];gates={'residual_response_transfer':all(results[k]['quadratic1152_sensitivity']['residual_change_fidelity']>=.2 and results[k]['quadratic1152_sensitivity']['residual_fidelity_document_bootstrap95'][0]>0 for k in actual),'behavioral_transfer':all(contrasts[k]['behavior_gain']>=.1 and contrasts[k]['behavior_gain_bootstrap95'][0]>0 for k in actual),'quadratic_advantage':all(contrasts[k]['quadratic_residual_gain']>=.05 and contrasts[k]['quadratic_gain_bootstrap95'][0]>0 for k in actual)}
    if not actual:gates={}
    np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'selection':selection,'results':results,'contrasts':contrasts,'gates':gates,'angle_audit':angle_audit,'head_replay':replay},indent=2));print('GATES',gates,'CONTRASTS',contrasts,flush=True)
if __name__=='__main__':main()
