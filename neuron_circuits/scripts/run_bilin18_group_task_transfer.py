"""H717: fixed diverse task competence and early-group answer transfer."""
import json,time
from pathlib import Path
from collections import defaultdict
import numpy as np,torch
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART,suite,groups as answer_groups,measure
from run_bilin18_broad_task_competence import continuation_id
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_group_task_transfer'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'suite':ROOT/'scripts/run_bilin18_diverse_causal_tasks.py','tokenization':ROOT/'scripts/run_bilin18_broad_task_competence.py','controls':ART/'bilin18_group_neighbor_controls_frozen_controls.json','spectra':ART/'bilin18_early_rotation_spectra_captures.pt','rotations':ART/'bilin18_early_sparse_rotations_captures.pt','model':SNAPSHOT/'pytorch_model.bin','runner':Path(__file__),'forward':ROOT/'scripts/run_bilin18_reader_operation_format.py','loader':ROOT/'scripts/run_bilin18_qkov_causal_edges.py'};hashes={k:sha(p) for k,p in paths.items()};rows=suite();cfg=dict(experiment='H717',hashes=hashes,design='Reuse fixed H5338family pairedsuite:8discovery/8transferpairsperfamily, counterfactual andsame-answer nuisance. Nativegate>=4validpairs,>=.75both-endpointbinarychoice,>=.5fullvocabsemanticTop1. Gatecausal eligibilityondiscoveryonly; reporttransferfailure. All8frozenearlygroups: task swaps0<-1and1<-0,nuisance0<-2. Grouprawdualcoordinate contributions cappedat5%receiverresidualnorm andnaturalgroupdelta. Anchoronlyuses samegroup scale(notnormmatched);3substitutegroups includeanchor+7matchedneighbors, rescaled tofullgroupeditnorm. Selectgroup ondiscovery mean taskdonor-directedmarginchange minusmean absolute nuisance marginchange, thentesttransferwithoutreselection. Saveallgroups/controls. Fullvocabcorrectness,contrastmargin andreceiverKL separate.',limitations='Previouslyused fixedsuite andhypothesis-generating labels, notnewbenchmark. Groupselectionexploratory; 8pairsperfamilysplit. Anchoronlynotnormmatched; controlgroupnormmatchednotnatural-delta-matched. Small edits neednotflipanswers. No claimsemanticunitwithoutselectivityandtransfer. Originalcorpusreservedtestuntouched.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));(ART/f'{STEM}_manifest.json').write_text(json.dumps(rows,indent=2));tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);examples={};excluded=[]
 for row in rows:
  ex=[]
  for v,text in enumerate(row['prompts']+[row['nuisance']]):
   gs=[[continuation_id(tok,text,a) for a in g] for g in answer_groups(row['family'],row['answers'])];gs=[[a for a in g if a is not None] for g in gs]
   if any(not g for g in gs) or set(gs[0])&set(gs[1]):break
   ex.append(dict(ids=tok.encode(text,add_special_tokens=False),groups=gs,variant=v))
  if len(ex)!=3:excluded.append(row['id']);continue
  for e in ex:examples[row['id'],e['variant']]=e
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 native={};states={};native_metrics={};buckets=defaultdict(list);forwards=0
 for k,e in examples.items():buckets[len(e['ids'])].append(k)
 for length,keys in buckets.items():
  for begin in range(0,len(keys),12):
   bb=keys[begin:begin+12];cap={}
   def get(m,args,out):cap['x']=out[0][:,-1].cpu()
   h=model.blocks[1].register_forward_hook(get)
   try:logits,_=run(model,torch.tensor([examples[k]['ids'] for k in bb],device='cuda'))
   finally:h.remove()
   forwards+=len(bb)
   for n,k in enumerate(bb):native[k]=logits[n].cpu();states[k]=cap['x'][n];native_metrics[k]=measure(logits[n],examples[k]['groups'])
 competence={}
 for fam in sorted({r['family'] for r in rows}):
  competence[fam]={}
  for split in ['discovery','transfer']:
   valid=[r for r in rows if r['family']==fam and r['split']==split and (r['id'],0) in examples];both=[native_metrics[r['id'],0]['margin']>0 and native_metrics[r['id'],1]['margin']<0 for r in valid];top=[native_metrics[r['id'],v]['top1'][v] for r in valid for v in [0,1]];n=len(valid);choice=float(np.mean(both)) if n else 0;full=float(np.mean(top)) if n else 0;competence[fam][split]=dict(pairs=n,both_endpoint_choice=choice,full_vocab_top1=full,passed=n>=4 and choice>=.75 and full>=.5)
 print('COMPETENCE',json.dumps(competence),flush=True);(ART/f'{STEM}.partial.json').write_text(json.dumps(dict(status='native_complete',competence=competence),indent=2));groups=json.loads(paths['controls'].read_text());sp=torch.load(paths['spectra'],weights_only=False)['states']['residual1'];Q=torch.load(paths['rotations'],weights_only=False)['rotations']['residual1','learned',7061].float();V=sp['eigenvectors'][:,:256].float();sd=sp['eigenvalues'][:256].float().sqrt();D=(V*sd)@Q;norm=D.norm(dim=0);D/=norm;mu=sp['mean'].float();codes={k:(((x-mu)@V)/sd)@Q*norm for k,x in states.items()};records=[];normerrors=[];patchmeta=[];patchvectors=[]
 for g in groups:
  jobs=[]
  for row in rows:
   if not competence[row['family']]['discovery']['passed'] or (row['id'],0) not in examples:continue
   for rv,dv in [(0,1),(1,0),(0,2)]:
    rk,dk=(row['id'],rv),(row['id'],dv);delta=codes[dk]-codes[rk];v=D[:,g['members']]@delta[g['members']];n=float(v.norm());scale=min(1.,.05*float(states[rk].norm())/max(n,1e-20));patches={'group':v*scale,'anchor':D[:,g['anchor']]*delta[g['anchor']]*scale}
    for i,c in enumerate(g['controls']):
     mem=[g['anchor']]+c['members'];w=D[:,mem]@delta[mem];assert w.norm()>1e-12;patches[f'control_{i}']=w*(n*scale/w.norm());normerrors.append(abs(float(patches[f'control_{i}'].norm())-n*scale)/max(n*scale,1e-20))
    for mode,patch in patches.items():jobs.append(dict(id=row['id'],family=row['family'],split=row['split'],receiver=rv,donor=dv,anchor=g['anchor'],mode=mode,patch=patch))
  buckets=defaultdict(list)
  for job in jobs:buckets[len(examples[job['id'],job['receiver']]['ids'])].append(job)
  for length,jobs2 in buckets.items():
   for begin in range(0,len(jobs2),12):
    bb=jobs2[begin:begin+12];patch=torch.stack([v['patch'] for v in bb]).cuda();ids=torch.tensor([examples[v['id'],v['receiver']]['ids'] for v in bb],device='cuda')
    def edit(m,args,out):
     x,bus=out;x=x.clone();x[:,-1]+=patch.to(x);return x,bus
    h=model.blocks[1].register_forward_hook(edit)
    try:logits,_=run(model,ids)
    finally:h.remove()
    forwards+=len(bb)
    for n,job in enumerate(bb):
     k=(job['id'],job['receiver']);m=measure(logits[n],examples[k]['groups']);base=native_metrics[k];marginchange=m['margin']-base['margin'];bp=native[k].double().log_softmax(-1);ep=logits[n].cpu().double().log_softmax(-1);record={a:b for a,b in job.items() if a!='patch'};record.update(margin_change=marginchange,donor_directed_margin_change=marginchange*(1 if job['donor'] in [0,2] else -1),receiver_kl=float((bp.exp()*(bp-ep)).sum()),native_correct=base['top1'][job['receiver']],edited_receiver_correct=m['top1'][job['receiver']],edited_donor_correct=m['top1'][0 if job['donor']==2 else job['donor']],relative_edit_norm=float(job['patch'].norm()/states[k].norm()));records.append(record);patchmeta.append({a:b for a,b in record.items() if a not in ['receiver_kl']});patchvectors.append(job['patch'])
  print('GROUP',g['anchor'],'records',len(records),flush=True);(ART/f'{STEM}.partial.json').write_text(json.dumps(dict(status='running',competence=competence,records=records),indent=2))
 audits=dict(hashes_unchanged={k:sha(p)==hashes[k] for k,p in paths.items()},finite=all(np.isfinite(r['receiver_kl']) and np.isfinite(r['margin_change']) for r in records),control_norm_match_max_relative=max(normerrors,default=0),whole_model_forwards=forwards,native_forwards=len(native),test_evaluations=0);assert all(audits['hashes_unchanged'].values()) and audits['finite'] and audits['control_norm_match_max_relative']<1e-5;torch.save(dict(native_logits=native,states=states,patch_metadata=patchmeta,patch_vectors=torch.stack(patchvectors) if patchvectors else None),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,competence=competence,excluded=excluded,records=records,native=[dict(id=k[0],variant=k[1],**v) for k,v in native_metrics.items()],audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()
