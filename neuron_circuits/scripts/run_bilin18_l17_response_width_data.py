"""H265 feature-width versus labeled-response data factorial."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program,forward_normalized,B
from run_bilin18_l17_response_capacity import response_solutions
from run_bilin18_l8_execution_alignment import angle_matched_payload
STEM=ART/'bilin18_l17_response_width_data'

def width_features(x,basis,mean,width):
    z=(x-mean)@basis;ij=torch.triu_indices(width,width,device=x.device);q=z[:,ij[0]]*z[:,ij[1]]*torch.where(ij[0]==ij[1],1.,2**.5)
    return torch.cat([z[:,:624],q],1)

@torch.inference_mode()
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float();block=model.blocks[17];L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];program=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');fixed=load_program(ART/'bilin18_l17_residual_feature_objective_frozen.npz','cuda')
    def err(x):return B(x,x,L,R,D)-forward_normalized(x,program)
    pre,heads,_=collect_sources(model,np.load(ROOT/cfg['discovery'],mmap_mode='r'),{**cfg,'rows':1024});x=F.rms_norm(pre,(1152,));y=err(x);zs={h:F.rms_norm(pre-heads[:,h],(1152,)) for h in cfg['heads']};dy={h:err(z)-y for h,z in zs.items()};vstart=cfg['validation_start']*4;vy=torch.cat([v[vstart:] for v in dy.values()]);fitted={};selected={};audits={};validation={};normalizers={};frozen={'basis':fixed['basis'].cpu().numpy(),'input_mean':fixed['input_mean'].cpu().numpy()}
    for width in cfg['quadratic_widths']:
        raw=width_features(x,fixed['basis'],fixed['input_mean'],width)
        if width==32:fm=fixed['quadratic1152_mean'];fs=fixed['quadratic1152_std']
        else:fm=raw[:vstart].mean(0);fs=raw[:vstart].std(0,unbiased=False).clamp_min(1e-6)
        normalizers[width]=(fm,fs);frozen[f'w{width}_mean']=fm.cpu().numpy();frozen[f'w{width}_std']=fs.cpu().numpy();features=(raw-fm)/fs;df={h:(width_features(z,fixed['basis'],fixed['input_mean'],width)-fm)/fs-features for h,z in zs.items()};vf=torch.cat([v[vstart:] for v in df.values()])
        for n in cfg['train_documents']:
            cut=n*4;tf=torch.cat([v[:cut] for v in df.values()]);ty=torch.cat([v[:cut] for v in dy.values()]);solutions,audit=response_solutions(tf,ty,cfg['ridge_strengths']);cell=f'w{width}_n{n}';audits[cell]=audit
            for alpha,w64 in solutions.items():
                name=f'{cell}_a{alpha}';w=w64.float();b=y[:cut].mean(0)-features[:cut].mean(0)@w;fitted[name]=w;frozen[name]=torch.cat([w,b[None]],0).cpu().numpy();validation[name]=float(1-(vf@w-vy).square().sum()/vy.square().sum());print(name,'train',audit['fits'][str(alpha)]['training_fidelity'],'val',validation[name],flush=True)
            selected[cell]=max([f'{cell}_a{alpha}' for alpha in cfg['ridge_strengths']],key=lambda k:validation[k]);del tf,ty,solutions
    np.savez_compressed(STEM.with_name(STEM.name+'_frozen.npz'),**frozen);STEM.with_name(STEM.name+'_selection.json').write_text(json.dumps({'config':cfg,'selected':selected,'audits':audits,'validation':validation},indent=2));print('FROZEN',selected,flush=True)
    del pre,heads,x,y,zs,dy,df,features,raw,vf,vy
    pre,heads,replay=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,));y=err(x);intact={width:(width_features(x,fixed['basis'],fixed['input_mean'],width)-normalizers[width][0])/normalizers[width][1] for width in cfg['quadratic_widths']};rows={};gen=torch.Generator(device='cuda').manual_seed(cfg['seed']);names={cell:{'selected':key,'unregularized':cell+'_a0'} for cell,key in selected.items()}
    for h in cfg['heads']:
        actual=heads[:,h];source=torch.linalg.qr(block.attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;angle=angle_matched_payload(pre,actual,source,torch.randn(len(pre),128,device='cuda',generator=gen))
        for payload,p in [('actual',actual),('angle',angle)]:
            for scale in cfg['scales']:
                z=F.rms_norm(pre-scale*p,(1152,));target=err(z)-y;prefix=f'{payload}_s{scale}_h{h}';rows[prefix+'_energy']=target.square().sum(-1).reshape(128,4).sum(1).cpu().numpy()
                for width in cfg['quadratic_widths']:
                    fm,fs=normalizers[width];d=(width_features(z,fixed['basis'],fixed['input_mean'],width)-fm)/fs-intact[width]
                    for n in cfg['train_documents']:
                        cell=f'w{width}_n{n}'
                        for role,key in names[cell].items():rows[f'{prefix}_{cell}_{role}']=(d@fitted[key]-target).square().sum(-1).reshape(128,4).sum(1).cpu().numpy()
        print('test head',h,flush=True)
    ix=np.random.default_rng(cfg['seed']+1).integers(128,size=(2000,128));results={};contrasts={}
    for payload in cfg['payloads']:
        for scale in cfg['scales']:
            prefix=f'{payload}_s{scale}';total=lambda m:sum(rows[f'{prefix}_h{h}_{m}'] for h in cfg['heads']);energy=total('energy');results[prefix]={}
            for cell in selected:
                for role in ['selected','unregularized']:
                    name=cell+'_'+role;e=total(name);results[prefix][name]={'fidelity':float(1-e.sum()/energy.sum()),'document_bootstrap95':np.quantile(1-e[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist()}
            pairs={**{f'width_n{n}':(f'w64_n{n}',f'w32_n{n}') for n in cfg['train_documents']},**{f'data_w{w}':(f'w{w}_n768',f'w{w}_n256') for w in cfg['quadratic_widths']}};contrasts[prefix]={}
            for name,(a,b) in pairs.items():
                delta=total(b+'_selected')-total(a+'_selected');contrasts[prefix][name]={'gain':float(delta.sum()/energy.sum()),'paired_document_bootstrap95':np.quantile(delta[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist()}
    gates={kind+'_transfer':all(v['gain']>=.05 and v['paired_document_bootstrap95'][0]>0 for s in cfg['scales'] for k,v in contrasts[f'actual_s{s}'].items() if k.startswith(kind)) for kind in ['width','data']};gates['half_response']=all(results[f'actual_s{s}']['w64_n768_selected']['fidelity']>=.5 for s in cfg['scales'])
    np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'selected':selected,'audits':audits,'validation':validation,'results':results,'contrasts':contrasts,'gates':gates,'head_replay':replay,'readout_scalars':{str(w):(624+w*(w+1)//2+1)*1152 for w in cfg['quadratic_widths']}},indent=2));print('RESULTS',results,'CONTRASTS',contrasts,'GATES',gates,flush=True)
if __name__=='__main__':main()
