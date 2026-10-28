"""H278 optimize only an output constant for downstream native-distribution KL."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program,forward_normalized,B
from bilin18_residual_cross_program import forward,change
STEM=ART/'bilin18_l17_kl_intercept'
def distillation_kl(logits,teacher_log_probs):
    return (teacher_log_probs.exp()*(teacher_log_probs-logits.double().log_softmax(-1))).sum(-1)

def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float()
    for v in model.lm_head.parameters():v.requires_grad_(False)
    block=model.blocks[17];L,R,D=[m.weight.detach() for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');ps={n:load_program(ART/f'bilin18_l17_residual_cross_{n}_calibrated_program.npz','cuda') for n in ['actual','mixture']}
    def logits(state):return 30*torch.tanh(model.lm_head(F.rms_norm(state,(1152,)))/30)
    with torch.no_grad():
        pre,_,_=collect_sources(model,np.load(ROOT/cfg['discovery'],mmap_mode='r'),{**cfg,'rows':1024});x=F.rms_norm(pre,(1152,));native=B(x,x,L,R,D);compiled=forward_normalized(x,p);base={n:(pre+compiled+forward(x,pr)+block.mlp.Down_bias).detach() for n,pr in ps.items()};teacher=torch.cat([logits(pre[i:i+16]+native[i:i+16]+block.mlp.Down_bias).double().log_softmax(-1) for i in range(0,len(pre),16)]);cut=768*4
    @torch.no_grad()
    def score(n,b,start,end):
        total=0.
        for i in range(start,end,16):total+=distillation_kl(logits(base[n][i:i+16]+b),teacher[i:i+16]).sum()
        return float(total/(end-start))
    curves={};selected={};candidates={};constants={};winners={};audits={};gen=torch.Generator(device='cuda')
    for n,pr in ps.items():
        for lr in cfg['learning_rates']:
            key=f'{n}_lr{lr}';b=torch.nn.Parameter(pr['anchor'].clone());opt=torch.optim.Adam([b],lr=lr);gen.manual_seed(cfg['seed']);best=float('inf');curves[key]=[]
            for step in range(cfg['steps']+1):
                if step in cfg['checkpoints']:
                    val=score(n,b,cut,len(pre));curves[key].append({'step':step,'validation_kl':val});print(key,step,val,flush=True)
                    if val<best:best=val;candidates[key]=b.detach().clone();selected[key]={'step':step,'validation_kl':val}
                if step==cfg['steps']:break
                ix=torch.randint(cut,(cfg['train_batch'],),generator=gen,device='cuda');loss=distillation_kl(logits(base[n][ix]+b),teacher[ix]).mean();opt.zero_grad();loss.backward();opt.step()
        winner=min([k for k in selected if k.startswith(n+'_')],key=lambda k:selected[k]['validation_kl']);winners[n]=winner;constants[n]=candidates[winner];audits[n]={}
        for label,b in [('mse',pr['anchor']),('kl',constants[n])]:audits[n][label]={'train_kl':score(n,b,0,cut),'validation_kl':score(n,b,cut,len(pre)),'constant_norm':float(b.norm()),'delta_from_mse_norm':float((b-pr['anchor']).norm())}
    invariant={}
    for n,pr in ps.items():
        before=change(x[:16],x[-16:],pr);copy={**pr,'anchor':constants[n]};invariant[n]=torch.equal(before,change(x[:16],x[-16:],copy));np.savez_compressed(ART/f'bilin18_l17_residual_cross_{n}_kl_program.npz',**{k:v.detach().cpu().numpy() for k,v in copy.items()})
    np.savez_compressed(STEM.with_name(STEM.name+'_constants.npz'),**{k:v.cpu().numpy() for k,v in {**constants,**candidates}.items()});STEM.with_name(STEM.name+'_selection.json').write_text(json.dumps({'config':cfg,'curves':curves,'selected':selected,'winners':winners,'audits':audits,'stable_response_bitwise_equal':invariant},indent=2));print('CONSTANTS FROZEN',winners,audits,flush=True)
    del teacher,base
    with torch.no_grad():
        corpus=np.load(ROOT/cfg['corpus'],mmap_mode='r');pre,_,replay=collect_sources(model,corpus,cfg);x=F.rms_norm(pre,(1152,));native=B(x,x,L,R,D);compiled=forward_normalized(x,p);pred={'compiled':compiled,'native':native}
        for n,pr in ps.items():
            k=forward(x,pr);pred[n+'_mse']=compiled+k+pr['anchor'];pred[n+'_kl']=compiled+k+constants[n]
        labels=torch.from_numpy(np.array(corpus[:,np.array(cfg['positions'])+1],dtype=np.int64).flatten()).cuda();rows={}
        def save(k,v):rows[k]=v.reshape(128,4).sum(1).cpu().numpy()
        save('residual_energy',(native-compiled).square().sum(-1))
        for n,v in pred.items():save(n+'_error',(v-native).square().sum(-1))
        metrics={n:{} for n in pred}
        for i in range(0,len(pre),16):
            sl=slice(i,i+16);truth=logits(pre[sl]+native[sl]+block.mlp.Down_bias);lp=truth.double().log_softmax(-1);ce=F.cross_entropy(truth.double(),labels[sl],reduction='none')
            for n,v in pred.items():
                q=logits(pre[sl]+v[sl]+block.mlp.Down_bias);vals={'kl':distillation_kl(q,lp),'ce_increase':F.cross_entropy(q.double(),labels[sl],reduction='none')-ce,'top1':(q.argmax(-1)==truth.argmax(-1)).float()}
                for k,val in vals.items():metrics[n].setdefault(k,[]);metrics[n][k].append(val)
        for n,vs in metrics.items():
            for k,v in vs.items():save(n+'_'+k,torch.cat(v))
    ix=np.random.default_rng(cfg['seed']+1).integers(128,size=(4000,128));results={};contrasts={};energy=rows['residual_energy'];klbase=rows['compiled_kl']
    for n in pred:
        kl=rows[n+'_kl'];ce=rows[n+'_ce_increase'];top=rows[n+'_top1'];results[n]={'residual_fidelity':float(1-rows[n+'_error'].sum()/energy.sum()),'mean_kl':float(kl.sum()/512),'kl_recovery':float(1-kl.sum()/klbase.sum()),'kl_recovery_bootstrap95':np.quantile(1-kl[ix].sum(1)/klbase[ix].sum(1),[.025,.975]).tolist(),'mean_ce_increase':float(ce.sum()/512),'ce_bootstrap95':np.quantile(ce[ix].sum(1)/512,[.025,.975]).tolist(),'top1_agreement':float(top.sum()/512),'top1_bootstrap95':np.quantile(top[ix].sum(1)/512,[.025,.975]).tolist()}
    for n in ps:
        baseline=rows[n+'_mse_kl'];d=baseline-rows[n+'_kl_kl'];contrasts[n]={'relative_kl_reduction':float(d.sum()/baseline.sum()),'paired95':np.quantile(d[ix].sum(1)/baseline[ix].sum(1),[.025,.975]).tolist()}
    gates={'behavior_constant_gain':all(contrasts[n]['relative_kl_reduction']>=.2 and contrasts[n]['paired95'][0]>0 for n in ps),'bias_dominant':all(contrasts[n]['relative_kl_reduction']>=.5 and contrasts[n]['paired95'][0]>0 for n in ps)}
    for n in ps:
        v=results[n+'_kl'];gates['repair_'+n]=v['kl_recovery']>=.8 and v['mean_ce_increase']<=.02 and v['top1_agreement']>=.98
    np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'winners':winners,'audits':audits,'results':results,'contrasts':contrasts,'gates':gates,'stable_response_bitwise_equal':invariant,'head_replay':replay},indent=2));print('GATES',gates,'RESULTS',results,flush=True)
if __name__=='__main__':main()
