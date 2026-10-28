"""H715: finite-difference downstream response geometry on fresh contexts."""
import json,time,hashlib
from pathlib import Path
import torch
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_group_response_geometry'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time()
 paths={'data':ART/'bilin18_coactivation_replication_captures.pt','old':ART/'bilin18_coactivation_group_swaps_captures.pt','controls':ART/'bilin18_group_neighbor_controls_frozen_controls.json','spectra':ART/'bilin18_early_rotation_spectra_captures.pt','rotations':ART/'bilin18_early_sparse_rotations_captures.pt','model':SNAPSHOT/'pytorch_model.bin','runner':Path(__file__),'loader':ROOT/'scripts/run_bilin18_qkov_causal_edges.py','forward':ROOT/'scripts/run_bilin18_reader_operation_format.py'}
 hashes={k:sha(p) for k,p in paths.items()};data=torch.load(paths['data'],weights_only=False);old=torch.load(paths['old'],weights_only=False);groups=json.loads(paths['controls'].read_text());excluded={p[k]//16 for p in old['pairs'] for k in ['receiver','donor']};docs=sorted([d for d in range(len(data['all_token_ids'])) if d not in excluded],key=lambda d:hashlib.sha256(f'H715:{d}'.encode()).hexdigest())[:16];directions=sorted({i for g in groups for mem in [g['members']]+[c['members'] for c in g['controls']] for i in mem});ix={v:i for i,v in enumerate(directions)};sp=torch.load(paths['spectra'],weights_only=False)['states']['residual1'];Q=torch.load(paths['rotations'],weights_only=False)['rotations']['residual1','learned',7061].cuda();V=sp['eigenvectors'][:,:256].float().cuda();sd=sp['eigenvalues'][:256].float().cuda().sqrt();D=(V*sd)@Q;D/=D.norm(dim=0);D=D[:,directions];decoder_cos=D.T@D
 cfg=dict(experiment='H715',hashes=hashes,docs=docs,position=63,directions=directions,dose=.01,design='Freeze16hash-selected H712documents excludingall64H713pairdocuments; sameposition63, noactivationoroutcomeconditioning. All directions in8groupsand24H714matchedcontrols. Symmetric +/-1%residualnorm decoder-unit-direction edits atblock1output, preservebus. Response=(logitsplus-logitsminus)/(2epsilon). Fisher signature sqrt(p)*(response-Epresponse). Removeprojectiononto sqrt(p)*(native_logits-Epnative_logits) forconfidence-removedsignature. Primarycontrast mean squaredanchor-neighborcosine inrealgroup minusmean3matchedcontrols, averagedacrossall8anchors then16contexts. Absolute/squaredcosines remove arbitrary dictionarysign. Halfdoseaudit all directions in firstcontext. No taskcorrectness inference.',limitations='16exploratorycontexts fixedposition, notconditionalonjointactivation; approximatecontrolmatching; responsealignmentnotproofsharedalgorithm. Confidence removalone localtemperaturedirection. Samecorpusresearchholdout, notmodeltrainingholdout. Originalreservedtestuntouched.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 records=[];captures=[];forwards=0;replay=[];halfaudit={};savepath=ART/'group_response_geometry';savepath.mkdir(exist_ok=True)
 for ci,doc in enumerate(docs):
  ids=data['all_token_ids'][doc:doc+1,:64].cuda();cap={}
  def get(m,args,out):cap['x']=out[0][0,-1].detach()
  hook=model.blocks[1].register_forward_hook(get)
  try:native,_=run(model,ids)
  finally:hook.remove()
  native=native[0].double();p=native.softmax(-1);sqrtp=p.sqrt();temp=sqrtp*(native-(p*native).sum());temp=temp/temp.norm();eps=.01*float(cap['x'].norm());reference=data['activations']['residual1'][doc,7].cuda();replay.append(float((cap['x']-reference).norm()/reference.norm()));forwards+=1
  def responses(epsilon):
   nonlocal forwards
   rows=[]
   for begin in range(0,len(directions),8):
    atoms=D[:,begin:begin+8].T;patch=torch.cat([epsilon*atoms,-epsilon*atoms],dim=0);nn=len(atoms)
    def edit(m,args,out):
     x,bus=out;x=x.clone();x[:,-1]+=patch.to(x);return x,bus
    hook=model.blocks[1].register_forward_hook(edit)
    try:logits,_=run(model,ids.expand(2*nn,-1))
    finally:hook.remove()
    rows.append((logits[:nn].double()-logits[nn:].double())/(2*epsilon));forwards+=2*nn
   response=torch.cat(rows);signature=(response-(response*p).sum(-1,keepdim=True))*sqrtp;clean=signature-(signature@temp)[:,None]*temp;return response,signature,clean
  response,sig,clean=responses(eps);norm=sig.norm(dim=1);cnorm=clean.norm(dim=1);assert bool((norm>1e-10).all()) and bool((cnorm>1e-10).all());unit=sig/norm[:,None];cunit=clean/cnorm[:,None];cos=unit@unit.T;ccos=cunit@cunit.T
  if ci==0:
   _,hs,hc=responses(eps/2);halfaudit=dict(full_signature_relative_error_quantiles=torch.quantile((sig-hs).norm(dim=1)/hs.norm(dim=1),torch.tensor([0.,.5,.95,1.],device='cuda',dtype=torch.double)).cpu().tolist(),clean_signature_cosine_min=float(((clean/cnorm[:,None])*(hc/hc.norm(dim=1)[:,None])).sum(-1).min()),gram_max_abs_change=float((ccos-(hc/hc.norm(dim=1)[:,None])@(hc/hc.norm(dim=1)[:,None]).T).abs().max()))
  for g in groups:
   a=ix[g['anchor']];nei=[ix[i] for i in g['members'][1:]];ctrl=[[ix[i] for i in c['members']] for c in g['controls']];rec=dict(doc=doc,anchor=g['anchor'],native_entropy=float(-(p*p.log()).sum()),anchor_temperature_energy_fraction=float(1-cnorm[a].square()/norm[a].square()))
   for name,mat in [('fisher',cos),('confidence_removed',ccos),('decoder',decoder_cos)]:
    real=float(mat[a,nei].square().mean());sub=[float(mat[a,n].square().mean()) for n in ctrl];rec[name]=dict(real=real,controls=sub,difference=real-sum(sub)/3)
   records.append(rec)
  torch.save(dict(doc=doc,directions=directions,native_logits=native.cpu(),response=response.float().cpu(),epsilon=eps),savepath/f'context_{ci:02d}.pt');captures.append(dict(doc=doc,fisher_cosine=cos.float().cpu(),confidence_removed_cosine=ccos.float().cpu(),signature_norms=norm.cpu(),confidence_removed_norms=cnorm.cpu(),anchor_signatures=unit[[ix[g['anchor']] for g in groups]].float().cpu(),anchor_confidence_removed_signatures=cunit[[ix[g['anchor']] for g in groups]].float().cpu()));(ART/f'{STEM}.partial.json').write_text(json.dumps(dict(status='running',completed_contexts=ci+1,records=records,half_dose_audit=halfaudit),indent=2));print('CONTEXT',ci+1,'/',len(docs),flush=True)
 audits=dict(hashes_unchanged={k:sha(p)==hashes[k] for k,p in paths.items()},finite=all(np.isfinite(r['confidence_removed']['difference']) for r in records),prefix_state_replay_max_relative=max(replay),half_dose=halfaudit,whole_model_forwards=forwards,test_evaluations=0,contexts_disjoint_from_H713=not bool(set(docs)&excluded));assert all(audits['hashes_unchanged'].values()) and audits['finite'] and max(replay)<1e-4
 torch.save(dict(contexts=captures,decoder_cosine=decoder_cos.cpu(),directions=directions),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()
