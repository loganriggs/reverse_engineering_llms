"""H258 remove oracle coefficients using discovery-fitted affine residual prediction."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,local_chart,local_project,load_full_model,collect_sources,load_program,forward_normalized,B
STEM=ART/'bilin18_l17_predicted_residual'

def fit_affine(x,y,strength):
    x=x.double();y=y.double();xm=x.mean(0);ym=y.mean(0);xc=x-xm;yc=y-ym;g=xc.T@xc
    w=torch.linalg.solve(g+strength*torch.trace(g)/len(g)*torch.eye(len(g),device=x.device,dtype=x.dtype),xc.T@yc)
    return w.float(),(ym-xm@w).float()

@torch.inference_mode()
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();block=model.blocks[17];L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];program=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda')
    def panel(path,rows):
        corpus=np.load(ROOT/path,mmap_mode='r');pre,_,_=collect_sources(model,corpus,{**cfg,'layer':17,'rows':rows});x=F.rms_norm(pre,(1152,));pred=forward_normalized(x,program);state=pre+pred+block.mlp.Down_bias;labels=torch.from_numpy(np.array(corpus[:rows,np.array(cfg['positions'])+1],dtype=np.int64).flatten()).cuda();return x,state,labels
    def charts(state):return torch.cat([local_chart(state[i:i+8],model.lm_head.weight,256,64)[0] for i in range(0,len(state),8)])
    def logits(state):return 30*torch.tanh(model.lm_head(F.rms_norm(state,(1152,)))/30)
    def score(state,error,labels,c):
        vals={'kl':[],'ce_delta':[],'top1':[]}
        for i in range(0,len(state),16):
            sl=slice(i,i+16);p=logits(state[sl]+error[sl]);q=logits(state[sl]+c[sl]);vals['kl'].append((p.softmax(-1)*(p.log_softmax(-1)-q.log_softmax(-1))).sum(-1));vals['ce_delta'].append(F.cross_entropy(q,labels[sl],reduction='none')-F.cross_entropy(p,labels[sl],reduction='none'));vals['top1'].append((p.argmax(-1)==q.argmax(-1)).float())
        return {k:torch.cat(v).reshape(-1,4).sum(1).cpu().numpy() for k,v in vals.items()}
    x,state,labels=panel(cfg['discovery'],1024);cut=cfg['train_documents']*4;vchart=charts(state[cut:]);error=B(x,x,L,R,D)-forward_normalized(x,program);validation={}
    for alpha in cfg['ridge_strengths']:
        w,b=fit_affine(x[:cut],error[:cut],alpha);pred=x[cut:]@w+b
        for mode,c in [('full',pred),('chart16',local_project(pred,vchart[:,:16]))]:validation[f'{mode}_{alpha}']=float(score(state[cut:],error[cut:],labels[cut:],c)['kl'].mean()/4)
    selected={mode:min(cfg['ridge_strengths'],key=lambda alpha:validation[f'{mode}_{alpha}']) for mode in ['full','chart16']};fitted={alpha:fit_affine(x,error,alpha) for alpha in cfg['ridge_strengths']};constant=error.mean(0);frozen={'constant':constant.cpu().numpy()}
    for alpha,(w,b) in fitted.items():frozen[f'w_{alpha}']=w.cpu().numpy();frozen[f'b_{alpha}']=b.cpu().numpy()
    np.savez_compressed(STEM.with_name(STEM.name+'_frozen.npz'),**frozen);STEM.with_name(STEM.name+'_selection.json').write_text(json.dumps({'validation_kl':validation,'selected':selected,'config':cfg},indent=2));print('FROZEN',selected,validation,flush=True)
    del x,state,labels,error,vchart
    x,state,labels=panel(cfg['corpus'],cfg['test_documents']);chart=charts(state)
    # No test-native residual used to construct these executable corrections.
    corrections={'compiled':torch.zeros_like(x),'constant':constant.expand_as(x),'constant_chart16':local_project(constant.expand_as(x),chart[:,:16])}
    for alpha,(w,b) in fitted.items():
        pred=x@w+b;corrections[f'full_{alpha}']=pred;corrections[f'chart16_{alpha}']=local_project(pred,chart[:,:16]);corrections[f'chart64_{alpha}']=local_project(pred,chart[:,:64])
    error=B(x,x,L,R,D)-forward_normalized(x,program);corrections['oracle16']=local_project(error,chart[:,:16]);corrections['oracle64']=local_project(error,chart);corrections['oracle_full']=error
    rows={};results={};ix=np.random.default_rng(cfg['seed']).integers(cfg['test_documents'],size=(2000,cfg['test_documents']))
    for name,c in corrections.items():
        metrics=score(state,error,labels,c)
        for k,v in metrics.items():rows[f'{name}_{k}']=v
        results[name]={'mean_kl':float(metrics['kl'].mean()/4),'ce_delta':float(metrics['ce_delta'].mean()/4),'top1':float(metrics['top1'].mean()/4),'residual_energy_recovery':float(1-(error-c).square().sum()/error.square().sum())};print(name,results[name],flush=True)
    for name,v in results.items():v['kl_recovery']=1-v['mean_kl']/results['compiled']['mean_kl'];v['ce_delta_document_bootstrap95']=np.quantile(rows[f'{name}_ce_delta'][ix].mean(1)/4,[.025,.975]).tolist()
    chosen=f"chart16_{selected['chart16']}";full=f"full_{selected['full']}";delta=rows[chosen+'_kl']-rows['constant_chart16_kl'];contrast={'selected_minus_constant_kl':float(delta.mean()/4),'paired_document_bootstrap95':np.quantile(delta[ix].mean(1)/4,[.025,.975]).tolist()};fraction=results[chosen]['kl_recovery']/results['oracle16']['kl_recovery']
    gates={'predictive_signal':results[chosen]['kl_recovery']>=.2 and contrast['paired_document_bootstrap95'][1]<0,'oracle_gap_closed':fraction>=.8,'behavioral_repair':any(results[n]['kl_recovery']>=.8 and results[n]['ce_delta']<=.02 and results[n]['top1']>=.98 for n in [chosen,full])}
    np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'validation_kl':validation,'selected':selected,'results':results,'contrast':contrast,'oracle_reduction_fraction':fraction,'gates':gates,'affine_added_scalars':1152*1152+1152},indent=2));print('GATES',gates,'oracle_fraction',fraction,flush=True)
if __name__=='__main__':main()
