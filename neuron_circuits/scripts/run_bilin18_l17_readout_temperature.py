"""H257: intervene on the behavioral metric with fixed internal computation."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,local_project,load_full_model,collect_sources,load_program,forward_normalized,B
STEM=ART/'bilin18_l17_readout_temperature'

def temperature_scores(state,weight,indices,temperature):
    dim=state.shape[-1];r=(state.square().mean(-1,keepdim=True)+torch.finfo(state.dtype).eps).sqrt();raw=(state/r)@weight.T
    p=(30*torch.tanh(raw/30)/temperature).softmax(-1);slope=(1-torch.tanh(raw/30).square())/temperature
    centered=weight[indices]*slope.gather(1,indices)[:,:,None]-((p*slope)@weight)[:,None,:]
    grad=centered/r[:,:,None]-state[:,None,:]*(centered*state[:,None,:]).sum(-1,keepdim=True)/(dim*r[:,:,None].pow(3))
    return grad,p

def temperature_chart(state,weight,temperature,candidates,rank):
    raw=F.rms_norm(state,(state.shape[-1],))@weight.T;ids=raw.topk(candidates,dim=-1).indices
    grad,p=temperature_scores(state,weight,ids,temperature);_,s,vh=torch.linalg.svd(grad*p.gather(1,ids).sqrt()[:,:,None],full_matrices=False)
    chart=torch.linalg.qr(vh[:,:rank].transpose(1,2)).Q.transpose(1,2)
    return chart,p.gather(1,ids).sum(-1),-(p*p.clamp_min(1e-38).log()).sum(-1)

@torch.inference_mode()
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();block=model.blocks[17];L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]]
    pre,_,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),{**cfg,'layer':17});x=F.rms_norm(pre,(1152,));pred=forward_normalized(x,load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda'));state=pre+pred+block.mlp.Down_bias
    charts={};selection={}
    for t in cfg['temperatures']:
        cs=[];ms=[];es=[]
        for i in range(0,len(state),4):
            c,m,e=temperature_chart(state[i:i+4],model.lm_head.weight,t,cfg['candidate_tokens'],64);cs.append(c);ms.append(m);es.append(e)
        charts[t]=torch.cat(cs);mass=torch.cat(ms);ent=torch.cat(es);selection[str(t)]={'probability_mass_mean':float(mass.mean()),'probability_mass_min':float(mass.min()),'entropy_mean':float(ent.mean())};print('chart',t,selection[str(t)],flush=True)
    error=B(x,x,L,R,D)-pred
    def logits(s):return 30*torch.tanh(model.lm_head(F.rms_norm(s,(1152,)))/30)
    base_logits=torch.cat([logits(state[i:i+16]) for i in range(0,len(state),16)]);rows={};results={}
    for scale in cfg['scales']:
        target=torch.cat([logits(state[i:i+16]+scale*error[i:i+16]) for i in range(0,len(state),16)])
        corrections={'none':torch.zeros_like(error)}
        for tc,q in charts.items():
            for rank in cfg['ranks']:corrections[f'chart{tc}_r{rank}']=local_project(error,q[:,:rank])
        for name,c in corrections.items():
            z=torch.cat([logits(state[i:i+16]+scale*c[i:i+16]) for i in range(0,len(state),16)]) if name!='none' else base_logits
            for te in cfg['temperatures']:
                p=(target/te).log_softmax(-1);q=(z/te).log_softmax(-1);kl=(p.exp()*(p-q)).sum(-1);key=f'scale{scale}_eval{te}_{name}';rows[key]=kl.reshape(cfg['rows'],-1).sum(1).cpu().numpy()
        print('scored scale',scale,flush=True)
    ix=np.random.default_rng(cfg['seed']).integers(cfg['rows'],size=(2000,cfg['rows']));boots={}
    for key,v in rows.items():
        prefix=key.split('_chart')[0] if '_chart' in key else key.removesuffix('_none');den=rows[prefix+'_none'];b=1-v[ix].sum(1)/den[ix].sum(1);boots[key]=b;results[key]={'mean_kl':float(v.sum()/len(state)),'recovery':float(1-v.sum()/den.sum()),'document_bootstrap95':np.quantile(b,[.025,.975]).tolist()}
    contrasts={}
    for scale in cfg['scales']:
        ka=f'scale{scale}_eval0.5_chart1.0_r16';kb=f'scale{scale}_eval2.0_chart1.0_r16';contrasts[f'metric_s{scale}']={'gap':results[ka]['recovery']-results[kb]['recovery'],'paired_document_bootstrap95':np.quantile(boots[ka]-boots[kb],[.025,.975]).tolist()}
    for te in [.5,2.]:
        ka=f'scale1.0_eval{te}_chart{te}_r16';kb=f'scale1.0_eval{te}_chart1.0_r16';contrasts[f'adapt_t{te}']={'gap':results[ka]['recovery']-results[kb]['recovery'],'paired_document_bootstrap95':np.quantile(boots[ka]-boots[kb],[.025,.975]).tolist()}
    gates={name:all(v['gap']>=cut and v['paired_document_bootstrap95'][0]>0 for k,v in contrasts.items() if k.startswith(prefix)) for name,prefix,cut in [('metric_dependence','metric',.15),('chart_adaptation','adapt',.02)]}
    np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'selection':selection,'results':results,'contrasts':contrasts,'gates':gates},indent=2));print('contrasts',contrasts,'gates',gates,flush=True)
if __name__=='__main__':main()
