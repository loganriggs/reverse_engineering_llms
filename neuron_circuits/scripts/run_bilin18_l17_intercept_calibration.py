"""H277 discovery-only output intercept calibration, frozen response polynomials."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program,forward_normalized,B
from bilin18_residual_cross_program import forward,intact_forward,counts,change
STEM=ART/'bilin18_l17_intercept_calibration'
def fit_output_constant(target,prediction):return (target.double()-prediction.double()).mean(0)
@torch.inference_mode()
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();block=model.blocks[17];L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');ps={n:load_program(ART/f'bilin18_l17_residual_cross_{n}_program.npz','cuda') for n in ['actual','mixture']};mu=ps['actual']['mean'];pd={k:v.double() for k,v in p.items()};anchor64=B(mu[None].double(),mu[None].double(),L.double(),R.double(),D.double())-forward_normalized(mu[None].double(),pd);anchor=anchor64[0].float();costs={}
    discovery=np.load(ROOT/cfg['discovery'],mmap_mode='r');pre_d,_,_=collect_sources(model,discovery,{**cfg,'rows':768});xd=F.rms_norm(pre_d,(1152,));rd=B(xd,xd,L,R,D)-forward_normalized(xd,p);kd={'compiled':torch.zeros_like(rd),'common':forward(xd,ps['actual'],False),**{n:forward(xd,pr) for n,pr in ps.items()}};cal={n:fit_output_constant(rd,k).float() for n,k in kd.items()};fit_audit={}
    for n,k in kd.items():
        error=rd.double()-k.double();fit_audit[n]={'training_mse_exact_anchor':float((error-anchor.double()).square().sum(1).mean()),'training_mse_calibrated':float((error-cal[n].double()).square().sum(1).mean()),'calibrated_norm':float(cal[n].norm()),'delta_from_exact_norm':float((cal[n]-anchor).norm()),'half_sample_constant_difference_norm':float((error[:1536].mean(0)-error[1536:].mean(0)).norm())}
    response_equal={}
    for n,pr in ps.items():
        before=change(xd[:16],xd[-16:],pr);pr['anchor']=cal[n];response_equal[n]=torch.equal(before,change(xd[:16],xd[-16:],pr));path=ART/f'bilin18_l17_residual_cross_{n}_calibrated_program.npz';np.savez_compressed(path,**{k:v.cpu().numpy() for k,v in pr.items()});costs[n]={'correction_scalars':sum(counts(pr).values()),'compiler_scalars':sum(v.numel() for v in p.values()),'saved_bytes':path.stat().st_size}
    np.savez_compressed(STEM.with_name(STEM.name+'_constants.npz'),exact=anchor.cpu().numpy(),**{n:v.cpu().numpy() for n,v in cal.items()});STEM.with_name(STEM.name+'_calibration.json').write_text(json.dumps({'config':cfg,'fit_audit':fit_audit,'stable_response_bitwise_equal':response_equal},indent=2));print('CONSTANTS FROZEN',fit_audit,flush=True)
    corpus=np.load(ROOT/cfg['corpus'],mmap_mode='r');pre,_,replay=collect_sources(model,corpus,cfg);x=F.rms_norm(pre,(1152,));native=B(x,x,L,R,D);compiled=forward_normalized(x,p);kcommon=forward(x,ps['actual'],False);pred={'compiled':compiled,'constant_exact':compiled+anchor,'compiled_mean':compiled+cal['compiled'],'common_exact':compiled+anchor+kcommon,'common_cal':compiled+cal['common']+kcommon,'native':native}
    for n,pr in ps.items():
        k=forward(x,pr);pred[n+'_zero']=compiled+k;pred[n+'_exact']=compiled+anchor+k;pred[n+'_cal']=compiled+cal[n]+k
    rows={};labels=torch.from_numpy(np.array(corpus[:,np.array(cfg['positions'])+1],dtype=np.int64).flatten()).cuda()
    def save(k,v):rows[k]=v.reshape(128,4).sum(1).cpu().numpy()
    save('residual_energy',(native-compiled).square().sum(-1));save('native_energy',native.square().sum(-1))
    for n,v in pred.items():save(n+'_error',(v-native).square().sum(-1))
    def logits(v):return 30*torch.tanh(model.lm_head(F.rms_norm(v,(1152,)))/30)
    metrics={n:{} for n in pred}
    for i in range(0,len(pre),16):
        sl=slice(i,i+16);truth=logits(pre[sl]+native[sl]+block.mlp.Down_bias);lp=truth.double().log_softmax(-1);prob=lp.exp();ce=F.cross_entropy(truth.double(),labels[sl],reduction='none')
        for n,v in pred.items():
            q=logits(pre[sl]+v[sl]+block.mlp.Down_bias);lq=q.double().log_softmax(-1);vals={'kl':(prob*(lp-lq)).sum(-1),'ce_increase':F.cross_entropy(q.double(),labels[sl],reduction='none')-ce,'top1':(q.argmax(-1)==truth.argmax(-1)).float(),'label_accuracy':(q.argmax(-1)==labels[sl]).float()}
            for k,val in vals.items():metrics[n].setdefault(k,[]);metrics[n][k].append(val)
    for n,vs in metrics.items():
        for k,v in vs.items():save(n+'_'+k,torch.cat(v))
    ix=np.random.default_rng(20261226).integers(128,size=(4000,128));results={};contrasts={};energy=rows['residual_energy'];klbase=rows['compiled_kl']
    for n in cfg['models']:
        kl=rows[n+'_kl'];ce=rows[n+'_ce_increase'];top=rows[n+'_top1'];results[n]={'residual_fidelity':float(1-rows[n+'_error'].sum()/energy.sum()),'native_write_fidelity':float(1-rows[n+'_error'].sum()/rows['native_energy'].sum()),'mean_kl':float(kl.sum()/512),'kl_recovery':float(1-kl.sum()/klbase.sum()),'kl_recovery_bootstrap95':np.quantile(1-kl[ix].sum(1)/klbase[ix].sum(1),[.025,.975]).tolist(),'mean_ce_increase':float(ce.sum()/512),'ce_increase_bootstrap95':np.quantile(ce[ix].sum(1)/512,[.025,.975]).tolist(),'top1_agreement':float(top.sum()/512),'top1_bootstrap95':np.quantile(top[ix].sum(1)/512,[.025,.975]).tolist(),'next_token_accuracy':float(rows[n+'_label_accuracy'].sum()/512)}
    for n in ['actual','mixture']:
        baseline=rows[n+'_exact_kl'];d=baseline-rows[n+'_cal_kl'];contrasts[n]={'relative_kl_reduction':float(d.sum()/baseline.sum()),'paired95':np.quantile(d[ix].sum(1)/baseline[ix].sum(1),[.025,.975]).tolist()}
    gates={'calibration_material':all(contrasts[n]['relative_kl_reduction']>=.2 and contrasts[n]['paired95'][0]>0 for n in ['actual','mixture']),'bias_dominant':all(contrasts[n]['relative_kl_reduction']>=.5 and contrasts[n]['paired95'][0]>0 for n in ['actual','mixture'])}
    for n in ['actual','mixture']:
        v=results[n+'_cal'];gates['repair_'+n]=v['mean_ce_increase']<=.02 and v['top1_agreement']>=.98 and v['kl_recovery']>=.8
    audit={'fit_audit':fit_audit,'stable_response_bitwise_equal':response_equal,'anchor_float32_relative_error':float((anchor.double()-anchor64[0]).norm()/anchor64.norm()),'anchor_norm':float(anchor.norm()),'common_at_mean_norm':float(forward(mu[None],ps['actual']).norm())};np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'results':results,'contrasts':contrasts,'gates':gates,'costs':costs,'audits':audit,'head_replay':replay},indent=2));print('GATES',gates,'RESULTS',results,flush=True)
if __name__=='__main__':main()
