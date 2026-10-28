"""H264 empirical response capacity of fixed linear/quadratic feature maps."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program,forward_normalized,B
from run_bilin18_l17_residual_feature_objective import raw_features
STEM=ART/'bilin18_l17_response_capacity'

def response_solutions(df,dy,alphas,cutoff=1e-12):
    f=df.double();y=dy.double();gram=f.T@f;rhs=f.T@y;ev,v=torch.linalg.eigh(gram);keep=ev>ev.max()*cutoff;projected=v.T@rhs;models={};audit={}
    for alpha in alphas:
        lam=alpha*gram.diagonal().mean();den=ev+lam
        inv=torch.where(keep,1/den.clamp_min(1e-300),torch.zeros_like(den)) if alpha==0 else 1/den
        w=v@(inv[:,None]*projected);res=(gram@w+lam*w-rhs).norm()/rhs.norm();models[alpha]=w
        audit[str(alpha)]={'normal_relative_residual':float(res),'training_fidelity':float(1-(f@w-y).square().sum()/y.square().sum())}
    return models,{'rank':int(keep.sum()),'feature_count':len(ev),'min_eigenvalue':float(ev.min()),'max_eigenvalue':float(ev.max()),'retained_condition':float(ev.max()/ev[keep].min()),'fits':audit}

@torch.inference_mode()
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float();block=model.blocks[17];L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');fixed=load_program(ART/'bilin18_l17_residual_feature_objective_frozen.npz','cuda');pre,heads,_=collect_sources(model,np.load(ROOT/cfg['discovery'],mmap_mode='r'),{**cfg,'rows':1024});x=F.rms_norm(pre,(1152,))
    def error(z):return B(z,z,L,R,D)-forward_normalized(z,p)
    y=error(x);cut=cfg['train_documents']*4;changed={h:F.rms_norm(pre-heads[:,h],(1152,)) for h in cfg['heads']};deltas={h:error(z)-y for h,z in changed.items()};train_y=torch.cat([v[:cut] for v in deltas.values()]);val_y=torch.cat([v[cut:] for v in deltas.values()]);frozen={k:fixed[k].cpu().numpy() for k in ['basis','input_mean']};selection={};diagnostics={};validation={}
    for family in ['affine1152','quadratic1152']:
        fm=fixed[family+'_mean'];fs=fixed[family+'_std'];frozen[family+'_mean']=fm.cpu().numpy();frozen[family+'_std']=fs.cpu().numpy()
        def features(z):return (raw_features(z,fixed['basis'],fixed['input_mean'],family)-fm)/fs
        f=features(x);df={h:features(z)-f for h,z in changed.items()};train_f=torch.cat([v[:cut] for v in df.values()]);val_f=torch.cat([v[cut:] for v in df.values()]);solutions,audit=response_solutions(train_f,train_y,cfg['ridge_strengths'],cfg['eigenvalue_cutoff_relative']);diagnostics[family]=audit
        for alpha,w64 in solutions.items():
            w=w64.float();b=y[:cut].mean(0)-f[:cut].mean(0)@w;name=f'{family}_alpha{alpha}';frozen[name]=torch.cat([w,b[None]],0).cpu().numpy();ve=val_f@w-val_y;vf=float(1-ve.square().sum()/val_y.square().sum());validation[name]={'response_fidelity':vf,'natural_residual_fidelity':float(1-(f[cut:]@w+b-y[cut:]).square().sum()/y[cut:].square().sum())};audit['fits'][str(alpha)]['float32_training_fidelity']=float(1-(train_f@w-train_y).square().sum()/train_y.square().sum());print(name,audit['fits'][str(alpha)],validation[name],flush=True)
        selection[family+'_response']=max([f'{family}_alpha{alpha}' for alpha in cfg['ridge_strengths']],key=lambda k:validation[k]['response_fidelity']);selection[family+'_unregularized']=f'{family}_alpha0'
    np.savez_compressed(STEM.with_name(STEM.name+'_frozen.npz'),**frozen);STEM.with_name(STEM.name+'_selection.json').write_text(json.dumps({'config':cfg,'selected':selection,'diagnostics':diagnostics,'validation':validation},indent=2));print('FROZEN',selection,flush=True)
if __name__=='__main__':main()
