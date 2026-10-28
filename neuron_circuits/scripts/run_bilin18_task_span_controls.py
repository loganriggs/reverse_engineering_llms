"""H718: full residual, retained span and complement task-transfer controls."""
import json,time
from pathlib import Path
from collections import defaultdict
import torch,numpy as np
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART,groups as answer_groups,measure
from run_bilin18_broad_task_competence import continuation_id
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_task_span_controls'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'old':ART/'bilin18_group_task_transfer.json','captures':ART/'bilin18_group_task_transfer_captures.pt','manifest':ART/'bilin18_group_task_transfer_manifest.json','spectra':ART/'bilin18_early_rotation_spectra_captures.pt','suite':ROOT/'scripts/run_bilin18_diverse_causal_tasks.py','tokenization':ROOT/'scripts/run_bilin18_broad_task_competence.py','model':SNAPSHOT/'pytorch_model.bin','runner':Path(__file__),'forward':ROOT/'scripts/run_bilin18_reader_operation_format.py','loader':ROOT/'scripts/run_bilin18_qkov_causal_edges.py'};hashes={k:sha(p) for k,p in paths.items()};old=json.loads(paths['old'].read_text());s=torch.load(paths['captures'],weights_only=False);rows=json.loads(paths['manifest'].read_text());V=torch.load(paths['spectra'],weights_only=False)['states']['residual1']['eigenvectors'][:,:256].double();tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);examples={}
 for row in rows:
  if not old['competence'][row['family']]['discovery']['passed']:continue
  for v,text in enumerate(row['prompts']+[row['nuisance']]):
   gs=[[continuation_id(tok,text,a) for a in g] for g in answer_groups(row['family'],row['answers'])];gs=[[a for a in g if a is not None] for g in gs];assert all(gs);examples[row['id'],v]=dict(ids=tok.encode(text,add_special_tokens=False),groups=gs)
 cfg=dict(experiment='H718',hashes=hashes,design='All80H717eligible taskpairs, discoveryandtransfer; bidirectionaltasks andnuisance0<-2. Atblock1lastposition comparefullresidualdelta, orthogonalprojection ontoH705top256span andcomplement. Eachnaturalcomponent atfullstrength orcappedat5%receiverresidualnorm; nocontrolamplification. Alsozeroedit forallswapendpoints. Persistentvaluebus andotherpositions unchanged. Predefinedanswercontrast,fullvocabtop1,receiverKL andnuisancespecificity. No groupselection.',limitations='Naturalfulleditscanbelarge/OOD; spancomponentsnotnormmatchedatnaturalstrength. Smallcapcontrols equalmaximumdosebutmaydifferbelowcap. Thisislastpositionresidualtransfer,notentirecontexttransfer; donorpositionsmaydiffer. Nuisancecriterionconservativeandunequaltask/nuisanceinputchanges. Reusedsuite; originalcorpusreservedtestuntouched.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));native={(r['id'],r['variant']):r for r in old['native']};jobs=[];decomp=[];orth=[]
 for row in rows:
  if (row['id'],0) not in examples:continue
  for rv,dv in [(0,1),(1,0),(0,2)]:
   x=s['states'][row['id'],rv].double();delta=s['states'][row['id'],dv].double()-x;proj=V@(V.T@delta);comp=delta-proj;decomp.append(float((proj+comp-delta).norm()));orth.append(float(abs(proj@comp)/(proj.norm()*comp.norm()).clamp_min(1e-20)))
   for mode,v in [('full',delta),('retained',proj),('complement',comp)]:
    for dose in ['cap05','natural']:
     scale=min(1.,.05*float(x.norm())/max(float(v.norm()),1e-20)) if dose=='cap05' else 1.;jobs.append(dict(id=row['id'],family=row['family'],split=row['split'],receiver=rv,donor=dv,mode=mode,dose=dose,patch=(v*scale).float(),relative_edit_norm=float(v.norm()*scale/x.norm())))
   jobs.append(dict(id=row['id'],family=row['family'],split=row['split'],receiver=rv,donor=dv,mode='zero',dose='zero',patch=torch.zeros_like(x).float(),relative_edit_norm=0.))
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 records=[];logitrows=[];forwards=0;zerokl=[];zeromargin=[];postreplay=[]
 for mode in ['full','retained','complement','zero']:
  buckets=defaultdict(list)
  for job in jobs:
   if job['mode']==mode:buckets[len(examples[job['id'],job['receiver']]['ids'])].append(job)
  for length,bbjobs in buckets.items():
   for st in range(0,len(bbjobs),12):
    bb=bbjobs[st:st+12];patch=torch.stack([r['patch'] for r in bb]).cuda();cap={}
    def edit(m,args,out):
     x,bus=out;x=x.clone();x[:,-1]+=patch.to(x);cap['edited']=x[:,-1].cpu();return x,bus
    h=model.blocks[1].register_forward_hook(edit)
    try:logits,_=run(model,torch.tensor([examples[r['id'],r['receiver']]['ids'] for r in bb],device='cuda'))
    finally:h.remove()
    forwards+=len(bb)
    for n,r in enumerate(bb):
     key=(r['id'],r['receiver']);m=measure(logits[n],examples[key]['groups']);base=native[key];change=m['margin']-base['margin'];bp=s['native_logits'][key].double().log_softmax(-1);ep=logits[n].double().cpu().log_softmax(-1);dp=s['native_logits'][r['id'],r['donor']].double().log_softmax(-1);rec={k:v for k,v in r.items() if k!='patch'};rec.update(margin_change=change,donor_directed_margin_change=change*(1 if r['donor'] in [0,2] else -1),receiver_kl=float((bp.exp()*(bp-ep)).sum()),donor_kl_gain=float((dp.exp()*(ep-bp)).sum()),native_correct=base['top1'][r['receiver']],edited_receiver_correct=m['top1'][r['receiver']],edited_donor_correct=m['top1'][0 if r['donor']==2 else r['donor']]);records.append(rec);logitrows.append(logits[n].cpu())
     if mode=='zero':zerokl.append(rec['receiver_kl']);zeromargin.append(abs(change))
     if mode=='full' and r['dose']=='natural':target=s['states'][r['id'],r['donor']];postreplay.append(float((cap['edited'][n]-target).norm()/target.norm()))
  print('DONE',mode,flush=True);(ART/f'{STEM}.partial.json').write_text(json.dumps(dict(status='running',records=records),indent=2))
 audits=dict(hashes_unchanged={k:sha(p)==hashes[k] for k,p in paths.items()},finite=all(np.isfinite(r['receiver_kl']) for r in records),projection_complement_sum_max_abs_norm=max(decomp),projection_complement_cosine_max_abs=max(orth),zero_edit_max_kl=max(zerokl),zero_edit_max_abs_margin=max(zeromargin),full_swap_state_replay_max_relative=max(postreplay),whole_model_forwards=forwards,test_evaluations=0);assert all(audits['hashes_unchanged'].values()) and audits['finite'] and max(zerokl)<1e-7 and max(postreplay)<1e-4 and max(orth)<1e-8;torch.save(dict(edited_logits=torch.stack(logitrows)),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()
