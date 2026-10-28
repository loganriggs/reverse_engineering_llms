#!/usr/bin/env python3
"""Discovery-frozen oracle residual restoration under energy and Fisher metrics."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from bilin18_compiled_program import load_program,forward_normalized
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_l8_source_support_factorial import collect_sources
from run_bilin18_l8_context_role_compression import B,state_rules
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';STEM=ART/'bilin18_l17_residual_restoration'

def score_gradients(state,weight,indices):
    dim=state.shape[-1];r=(state.square().mean(-1,keepdim=True)+torch.finfo(state.dtype).eps).sqrt();z=state/r
    raw=z@weight.T;logits=30*torch.tanh(raw/30);p=logits.softmax(-1);slope=1-torch.tanh(raw/30).square()
    centered=weight[indices]*slope.gather(1,indices)[:,:,None]-((p*slope)@weight)[:,None,:]
    return centered/r[:,:,None]-state[:,None,:]*(centered*state[:,None,:]).sum(-1,keepdim=True)/(dim*r[:,:,None].pow(3)),p

def fisher_maps(residual,gradients,ranks):
    _,s,vh=torch.linalg.svd(residual.double(),full_matrices=False);active=s>s[0]*1e-8;s=s[active];vh=vh[active];root=vh.T*(s/(len(residual)**.5))[None];inv=(vh/(s/(len(residual)**.5))[:,None]);g=gradients.double()@root;metric=g.T@g/len(g);_,v=torch.linalg.eigh(metric);v=v.flip(1)
    return {r:((v[:,:r].T@inv).float(),(root@v[:,:r]).float()) for r in ranks},vh.float()

@torch.inference_mode()
def main():
    global STEM
    if '--scaling' in sys.argv:STEM=ART/'bilin18_l17_residual_scaling'
    torch.set_num_threads(6);spec=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());cfg=json.loads((ART/'bilin18_l17_compiled_partners.json').read_text())['parent_config'];model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();block=model.blocks[17];L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];program=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');panels={}
    for split,path in [('discovery',spec.get('discovery',cfg['discovery'])),('test',cfg['corpus'])]:
        corpus=np.load(ROOT/path,mmap_mode='r');pre,_,_=collect_sources(model,corpus,{**cfg,'layer':17,'rows':spec.get('discovery_rows',cfg['rows']) if split=='discovery' else cfg['rows']});x=F.rms_norm(pre,(1152,));native=B(x,x,L,R,D);pred=forward_normalized(x,program);labels=torch.from_numpy(np.array(corpus[:spec.get('discovery_rows',cfg['rows']) if split=='discovery' else cfg['rows'],np.array(cfg['positions'])+1],dtype=np.int64).flatten()).cuda();panels[split]={'pre':pre,'x':x,'native':native,'compiled':pred,'error':native-pred,'labels':labels}
    disc=panels['discovery'];grad=[];gen=torch.Generator(device='cuda').manual_seed(spec['seed']);weight=model.lm_head.weight
    for off in range(0,len(disc['pre']),spec['logit_batch']):
        state=disc['pre'][off:off+spec['logit_batch']]+disc['native'][off:off+spec['logit_batch']]+block.mlp.Down_bias;r=F.rms_norm(state,(1152,));p=(30*torch.tanh((r@weight.T)/30)).softmax(-1);idx=torch.multinomial(p,max(spec['score_draws']),replacement=True,generator=gen);g,_=score_gradients(state,weight,idx);grad.append(g)
    grad=torch.cat(grad);maps={};frozen={};_,_,vh=torch.linalg.svd(disc['error'].double(),full_matrices=False);vh=vh.float()
    for r in spec['ranks']:maps[f'energy{r}']=(vh[:r],vh[:r].T)
    for count in spec.get('nested_discovery_rows',[]):
        _,_,vsmall=torch.linalg.svd(disc['error'][:count*len(cfg['positions'])].double(),full_matrices=False);vsmall=vsmall.float()
        for r in spec['ranks']:
            if r<=len(vsmall):maps[f'energy{count}docs_r{r}']=(vsmall[:r],vsmall[:r].T)
    for draws in spec['score_draws']:
        fitted,_=fisher_maps(disc['error'],grad[:,:draws].flatten(0,1),spec['ranks'])
        for r,pair in fitted.items():maps[f'fisher{draws}_r{r}']=pair
    gen=torch.Generator(device='cuda').manual_seed(spec['seed']+1)
    for rep in range(spec['random_replicates']):
        rotate=torch.linalg.qr(torch.randn(len(vh),len(vh),device='cuda',generator=gen)).Q;random=rotate.T@vh
        for r in spec['ranks']:maps[f'random{rep}_r{r}']=(random[:r],random[:r].T)
    fit={}
    for n,(e,a) in maps.items():
        frozen[n+'_encoder']=e.cpu().numpy();frozen[n+'_decoder']=a.cpu().numpy();res=disc['error']-(disc['error']@e.T)@a.T
        fit[n]={'discovery_residual_energy_recovery':float(1-res.square().sum()/disc['error'].square().sum()),'idempotence_error':float((e@a-torch.eye(len(e),device='cuda')).abs().max())}
    np.savez_compressed(STEM.with_name(STEM.name+'_frozen.npz'),**frozen);print('All restoration maps frozen',flush=True)
    # Diagnose overlap of the two analytic truncations; no fitting on test.
    f=np.load(ART/'bilin18_depth_linear_context_frozen.npz');mu=torch.from_numpy(f['L17_mean']).cuda();context=torch.from_numpy(f['L17_pca'][:,:16]).cuda();v=torch.from_numpy(np.load(ART/'bilin18_depth_source_geometry_frozen.npz')['L17_global'][:16]).cuda();calibration_sites=cfg['rows']*len(cfg['positions']);meanwrite=disc['native'][:calibration_sites].mean(0);crossbias=meanwrite-state_rules(disc['x'][:calibration_sites],mu,context.T,context,L,R,D)['cross'].mean(0);test=panels['test'];cross=state_rules(test['x'],mu,context.T,context,L,R,D)['cross']+crossbias;ec=test['native']-cross;eo=test['native']-(meanwrite+((test['native']-meanwrite)@v.T)@v)
    overlap={'context_output_error_cosine':float((ec*eo).sum()/(ec.norm()*eo.norm())),'context_error_fraction_outside_output16':float((ec-(ec@v.T)@v).square().sum()/ec.square().sum())}
    corrections={'compiled':torch.zeros_like(test['error']),'full_oracle':test['error'],**{n:(test['error']@e.T)@a.T for n,(e,a) in maps.items()}};rows={};results={};native_logits=[]
    def logits(state):return 30*torch.tanh(model.lm_head(F.rms_norm(state,(1152,)))/30)
    for off in range(0,len(test['pre']),spec['logit_batch']):native_logits.append(logits(test['pre'][off:off+spec['logit_batch']]+test['native'][off:off+spec['logit_batch']]+block.mlp.Down_bias))
    native_logits=torch.cat(native_logits);native_ce=F.cross_entropy(native_logits,test['labels'],reduction='none');top=native_logits.topk(2,dim=-1).indices;row_count=cfg['rows']
    for name,correction in corrections.items():
        metric={}
        for off in range(0,len(test['pre']),spec['logit_batch']):
            sl=slice(off,off+spec['logit_batch']);pred=logits(test['pre'][sl]+test['compiled'][sl]+correction[sl]+block.mlp.Down_bias);truth=native_logits[sl];ce=F.cross_entropy(pred,test['labels'][sl],reduction='none');margin=pred.gather(1,top[sl,:1]).squeeze(1)-pred.gather(1,top[sl,1:]).squeeze(1);native_margin=truth.gather(1,top[sl,:1]).squeeze(1)-truth.gather(1,top[sl,1:]).squeeze(1)
            values={'kl':(truth.softmax(-1)*(truth.log_softmax(-1)-pred.log_softmax(-1))).sum(-1),'ce_delta':ce-native_ce[sl],'top1':(pred.argmax(-1)==truth.argmax(-1)).float(),'margin_error':(margin-native_margin).square(),'logit_error':(pred-truth).square().sum(-1)}
            for k,val in values.items():metric.setdefault(k,[]);metric[k].append(val)
        error=test['error']-correction;metric['residual_error']=[error.square().sum(-1)]
        for k,vs in metric.items():rows[name+'_'+k]=torch.cat(vs).reshape(row_count,-1).sum(1).cpu().numpy()
        results[name]={'mean_native_kl':float(rows[name+'_kl'].sum()/len(native_logits)),'mean_ce_delta':float(rows[name+'_ce_delta'].sum()/len(native_logits)),'native_top1_agreement':float(rows[name+'_top1'].sum()/len(native_logits)),'residual_energy_recovery':float(1-error.square().sum()/test['error'].square().sum())}
        print(name,results[name],flush=True)
    ix=np.random.default_rng(spec['seed']).integers(row_count,size=(2000,row_count));base=results['compiled'];contrasts={}
    for name,vv in results.items():
        vv['kl_recovery']=1-vv['mean_native_kl']/base['mean_native_kl'];vv['logit_error_recovery']=float(1-rows[name+'_logit_error'].sum()/rows['compiled_logit_error'].sum());vv['margin_error_recovery']=float(1-rows[name+'_margin_error'].sum()/rows['compiled_margin_error'].sum());vv['ce_delta_document_bootstrap95']=np.quantile(rows[name+'_ce_delta'][ix].sum(1)/len(native_logits),[.025,.975]).tolist()
    for draws in spec['score_draws']:
        n=f'fisher{draws}_r16';difference=(rows[n+'_kl']-rows['energy16_kl']);contrasts[n+'_minus_energy16_kl']={'value':float(difference.sum()/len(native_logits)),'paired_document_bootstrap95':np.quantile(difference[ix].sum(1)/len(native_logits),[.025,.975]).tolist()}
    candidates=[f'energy{r}' for r in spec['ranks'] if r<=16]+[f'fisher{d}_r{r}' for d in spec['score_draws'] for r in spec['ranks'] if r<=16]
    gates={'small_residual_repairs':any(results[n]['kl_recovery']>=.8 and results[n]['mean_ce_delta']<=.02 and results[n]['native_top1_agreement']>=.98 for n in candidates),'fisher_advantage':all(results[f'fisher{d}_r16']['mean_native_kl']<=.75*results['energy16']['mean_native_kl'] for d in spec['score_draws'])}
    if spec.get('nested_discovery_rows'):
        gates['discovery_scaling_material']=results['energy128']['residual_energy_recovery']-results['energy64docs_r128']['residual_energy_recovery']>=.15
    np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':spec,'gates':gates,'results':results,'fit':fit,'overlap':overlap,'contrasts':contrasts,'native_ce':float(native_ce.mean())},indent=2));print('gates',gates,flush=True)
if __name__=='__main__':main()
