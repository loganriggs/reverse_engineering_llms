"""H280 fit a linear adapter on frozen existing core/partner reads."""
import json
import sys
from bilin18_readout_feature_families import fit_features
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program,forward_normalized,B
from bilin18_residual_cross_program import intact_forward
from run_bilin18_l17_kl_intercept import distillation_kl
STEM=ART/'bilin18_l17_fixed_readout'
def read_features(x,p,center,scale):return (((x-p['mean'])@torch.cat([p['core'],p['partner']],1))-center)/scale
def ridge_fit(x,y,alpha):
    x=x.double();y=y.double();g=x.T@x;rhs=x.T@y
    if alpha==0:return torch.linalg.lstsq(x,y).solution
    return torch.linalg.solve(g+alpha*g.diag().mean()*torch.eye(g.shape[0],device=g.device,dtype=g.dtype),rhs)
def main():
    global STEM
    if len(sys.argv)>1:STEM=ART/sys.argv[1]
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float()
    for v in model.lm_head.parameters():v.requires_grad_(False)
    block=model.blocks[17];L,R,D=[m.weight.detach() for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');pr=load_program(ART/'bilin18_l17_residual_cross_actual_refined_program.npz','cuda')
    def logits(state):return 30*torch.tanh(model.lm_head(F.rms_norm(state,(1152,)))/30)
    with torch.no_grad():
        pre,_,_=collect_sources(model,np.load(ROOT/cfg['discovery'],mmap_mode='r'),{**cfg,'rows':1024});x=F.rms_norm(pre,(1152,));native=B(x,x,L,R,D);compiled=forward_normalized(x,p);write=compiled+intact_forward(x,pr);base=(pre+write+block.mlp.Down_bias).detach();teacher=torch.cat([logits(pre[i:i+16]+native[i:i+16]+block.mlp.Down_bias).double().log_softmax(-1) for i in range(0,len(pre),16)]);cut=cfg.get('train_documents',768)*4;valcut=768*4;assert cut<=valcut;raw=(x-pr['mean'])@torch.cat([pr['core'],pr['partner']],1);fm=raw[:cut].mean(0);fs=raw[:cut].std(0,unbiased=False).clamp_min(1e-6)
        if cfg.get('fixed_statistics'):
            stats=load_program(ART/'bilin18_l17_fixed_readout_frozen.npz','cuda');fm=stats['feature_mean'];fs=stats['feature_std']
        features,feature_map=fit_features(x,pr,cfg['family'],cut);fm=feature_map['feature_mean'];fs=feature_map['feature_std'];target=native-write;energy=target[:cut].square().sum(1).mean()
    @torch.no_grad()
    def score(w,start,end):
        kl=0.;mse=0.
        for i in range(start,end,16):
            correction=features[i:i+16]@w;kl+=distillation_kl(logits(base[i:i+16]+correction),teacher[i:i+16]).sum();mse+=(correction-target[i:i+16]).square().sum().double()
        return {'kl':float(kl/(end-start)),'mse':float(mse/(end-start))}
    zeros=torch.zeros(64,1152,device='cuda');fits={};mse_validation={};solver={}
    for alpha in cfg['ridge_strengths']:
        w=ridge_fit(features[:cut],target[:cut],alpha);key=f'mse_a{alpha}';fits[key]=w.float();mse_validation[key]=score(fits[key],valcut,len(pre));xx=features[:cut].double();yy=target[:cut].double();g=xx.T@xx;rhs=xx.T@yy;solver[key]=float(((g+alpha*g.diag().mean()*torch.eye(64,device='cuda'))@w-rhs).norm()/rhs.norm())
    mse_winner=min(mse_validation,key=lambda k:mse_validation[k]['kl']);print('MSE SELECTED',mse_winner,mse_validation,flush=True);curves={};selected={};gen=torch.Generator(device='cuda')
    for initial in cfg['initializations']:
        for lr in cfg['learning_rates']:
            for penalty in cfg['penalties']:
                key=f'{initial}_lr{lr}_p{penalty}';w=torch.nn.Parameter((zeros if initial=='zero' else fits[mse_winner]).clone());opt=torch.optim.Adam([w],lr=lr);gen.manual_seed(cfg['seed']);best=float('inf');curves[key]=[]
                for step in range(cfg['steps']+1):
                    if step in cfg['checkpoints']:
                        val=score(w,valcut,len(pre));curves[key].append({'step':step,**val});print(key,step,val['kl'],flush=True)
                        if val['kl']<best:best=val['kl'];fits[key]=w.detach().clone();selected[key]={'step':step,**val}
                    if step==cfg['steps']:break
                    ix=torch.randint(cut,(cfg['train_batch'],),generator=gen,device='cuda');correction=features[ix]@w;loss=distillation_kl(logits(base[ix]+correction),teacher[ix]).mean()+penalty*correction.square().sum(1).mean()/energy;opt.zero_grad();loss.backward();opt.step()
    kl_winner=min(selected,key=lambda k:selected[k]['kl']);ws={'constant':zeros,'mse':fits[mse_winner],'kl':fits[kl_winner]};audits={n:{'train':score(w,0,cut),'validation':score(w,valcut,len(pre))} for n,w in ws.items()};np.savez_compressed(STEM.with_name(STEM.name+'_frozen.npz'),**{k:v.cpu().numpy() for k,v in feature_map.items()},**{n:w.cpu().numpy() for n,w in {**ws,**fits}.items()});STEM.with_name(STEM.name+'_selection.json').write_text(json.dumps({'config':cfg,'mse_winner':mse_winner,'kl_winner':kl_winner,'mse_validation':mse_validation,'solver_relative_residuals':solver,'curves':curves,'selected':selected,'audits':audits},indent=2));print('FROZEN',kl_winner,audits,flush=True)
    return
if __name__=='__main__':main()
