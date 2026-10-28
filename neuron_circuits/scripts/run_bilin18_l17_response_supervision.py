"""H261 fixed feature maps with heldout-source response supervision."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program,forward_normalized,B
from run_bilin18_l17_residual_feature_objective import raw_features
from run_bilin18_l8_execution_alignment import angle_matched_payload
STEM=ART/'bilin18_l17_response_supervision'

def joint_fit(f,y,df,dy,beta,alpha):
    f=f.double();y=y.double();df=df.double();dy=dy.double();x=torch.cat([f,torch.ones(len(f),1,device=f.device,dtype=f.dtype)],1);dx=torch.cat([df,torch.zeros(len(df),1,device=f.device,dtype=f.dtype)],1)
    en=y.square().sum(-1).mean();er=dy.square().sum(-1).mean();a=x.T@x/(len(x)*en)+beta*(dx.T@dx)/(len(dx)*er);b=x.T@y/(len(x)*en)+beta*(dx.T@dy)/(len(dx)*er);lam=alpha*a[:-1,:-1].diagonal().mean();pen=torch.eye(len(a),device=a.device,dtype=a.dtype);pen[-1,-1]=0;w=torch.linalg.solve(a+lam*pen,b)
    return w.float(),float(en),float(er),float(((a+lam*pen)@w-b).norm()/b.norm())

@torch.inference_mode()
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float();block=model.blocks[17];L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];program=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');fixed=load_program(ART/'bilin18_l17_residual_feature_objective_frozen.npz','cuda')
    pre,heads,_=collect_sources(model,np.load(ROOT/cfg['discovery'],mmap_mode='r'),{**cfg,'rows':1024});x=F.rms_norm(pre,(1152,))
    def error(z):return B(z,z,L,R,D)-forward_normalized(z,program)
    y=error(x);cut=cfg['train_documents']*4;changed={};payload_audit={};gen=torch.Generator(device='cuda').manual_seed(cfg.get('training_payload_seed',20261206))
    for h in cfg['supervised_heads']:
        actual=heads[:,h];kind=cfg.get('training_payload','actual')
        if kind=='actual':payload=actual
        else:
            source=torch.linalg.qr(block.attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q if kind=='angle' else torch.eye(1152,device='cuda')
            noise=torch.randn(len(pre),source.shape[1],generator=gen,device='cuda');payload=angle_matched_payload(pre,actual,source,noise)
        changed[h]=F.rms_norm(pre-payload,(1152,));payload_audit[h]={'norm_relative_max':float(((payload.norm(dim=-1)-actual.norm(dim=-1)).abs()/actual.norm(dim=-1).clamp_min(1e-20)).max()),'dot_global_relative':float(((pre*payload).sum(-1)-(pre*actual).sum(-1)).abs().max()/(pre.norm(dim=-1)*actual.norm(dim=-1)).max())}
    dy={h:error(z)-y for h,z in changed.items()};selected={};validation={};diagnostics={};weights={};frozen={k:fixed[k].cpu().numpy() for k in ['basis','input_mean']}
    for family in cfg['families']:
        fm= fixed[family+'_mean'];fs=fixed[family+'_std'];frozen[family+'_mean']=fm.cpu().numpy();frozen[family+'_std']=fs.cpu().numpy()
        def features(z):return (raw_features(z,fixed['basis'],fixed['input_mean'],family)-fm)/fs
        f=features(x);df={h:features(z)-f for h,z in changed.items()};train_df=torch.cat([v[:cut] for v in df.values()]);train_dy=torch.cat([v[:cut] for v in dy.values()]);val_df=torch.cat([v[cut:] for v in df.values()]);val_dy=torch.cat([v[cut:] for v in dy.values()]);vf=torch.cat([f[cut:],torch.ones(len(f)-cut,1,device='cuda')],1);vd=torch.cat([val_df,torch.zeros(len(val_df),1,device='cuda')],1)
        for beta in cfg['response_weights']:
            prefix=f'{family}_beta{beta}'
            for alpha in cfg['ridge_strengths']:
                name=f'{prefix}_alpha{alpha}';w,en,er,res=joint_fit(f[:cut],y[:cut],train_df,train_dy,beta,alpha);natural=float((vf@w-y[cut:]).square().sum(-1).mean()/en);response=float((vd@w-val_dy).square().sum(-1).mean()/er);validation[name]={'natural_normalized_error':natural,'response_normalized_error':response,'selection_score':natural+beta*response};diagnostics[name]=res;weights[name]=w.cpu().numpy();print(name,validation[name],flush=True)
            selected[prefix]=min([f'{prefix}_alpha{alpha}' for alpha in cfg['ridge_strengths']],key=lambda name:validation[name]['selection_score'])
    frozen.update(weights);np.savez_compressed(STEM.with_name(STEM.name+'_frozen.npz'),**frozen);STEM.with_name(STEM.name+'_selection.json').write_text(json.dumps({'config':cfg,'selected':selected,'validation':validation,'normal_equation_relative_residual':diagnostics,'training_payload_audit':payload_audit},indent=2));print('FROZEN',selected,flush=True)
if __name__=='__main__':main()
