"""H887 frozen number axes on irregular nouns and new contexts; no refit."""
from run_bilin18_reader_recombination_panel import ROOT,ART,sha,SNAPSHOT,load_full_model,continuation_id,final_logits,measure,TOKENIZER
import json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer
S='bilin18_number_transfer_screen'
PAIRS=[('irregular',a,b) for a,b in [('man','men'),('woman','women'),('child','children'),('person','people'),('foot','feet'),('tooth','teeth'),('mouse','mice'),('goose','geese')]]+[('new_regular',a,b) for a,b in [('teacher','teachers'),('worker','workers'),('student','students'),('farmer','farmers'),('artist','artists'),('driver','drivers'),('doctor','doctors'),('player','players')]]
TEMPLATES={'A':'The {noun} near the river','B':'The {noun} in the garden','C':'The {noun} beside the road','D':'The {noun} near the dogs','E':'The {noun} near the dog','F':'The {noun} that the guards noticed'}
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','fits':ART/'bilin18_agreement_partner_number_fits.pt','centroids':ART/'bilin18_agreement_partner_number.json'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H887','hashes':hashes,'pairs':PAIRS,'templates':TEMPLATES,'axes':['A','B'],'design':'Freeze H886 discovery PCA directions, means and number centroid thresholds. Evaluate both old-template axes on every new context; no refit or best-axis selection. Six templates including old A/B and fixed singular/plural distractors, both families and all endpoints.','gate':'Each cohort/family/template: >=4 tokenization-eligible pairs, both endpoints binarycorrect>=.75, fullvocabtop1>=.5. All failures retained. Paired causal eligibility also requires copy control.','scope':'Morphological/lexical/template generalization of frozen number coordinate, not causal necessity or complete manifold validation. No reservedtest.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);examples=[];excluded=[]
 for p,(cohort,a,b) in enumerate(PAIRS):
  source=[tok.encode(' '+v,add_special_tokens=False) for v in [a,b]];rows=[];errors=[]
  if any(len(v)!=1 for v in source):excluded.append({'pair':p,'cohort':cohort,'words':[a,b],'reason':'source_not_single'});continue
  for fmt,template in TEMPLATES.items():
   for family in ['copy','task']:
    rr=[];answers=[a,b] if family=='copy' else ['is','are']
    for order,noun in enumerate([a,b]):
     text=template.format(noun=noun) if family=='task' else (f'word {noun}; word {noun}; word' if fmt in ['A','C','D'] else f'item {noun}, item {noun}, item');ids=tok.encode(text,add_special_tokens=False);aid=[continuation_id(tok,text,' '+v) for v in answers]
     if any(v is None for v in aid):errors.append('answer_not_single');continue
     rr.append(dict(pair=p,cohort=cohort,format=fmt,family=family,order=order,expected=order,text=text,ids=ids,groups=[[v] for v in aid],id=f'{p}-{fmt}-{family}'))
    if len(rr)!=2:continue
    if len(rr[0]['ids'])!=len(rr[1]['ids']):errors.append('length');continue
    changed=[k for k,(x,y) in enumerate(zip(rr[0]['ids'],rr[1]['ids'])) if x!=y]
    if not changed or any(rr[o]['ids'][k]!=source[o][0] for k in changed for o in [0,1]):errors.append('source_alignment');continue
    for r in rr:r['source_positions']=changed
    rows.extend(rr)
  if errors or len(rows)!=24:excluded.append({'pair':p,'cohort':cohort,'words':[a,b],'reason':errors});continue
  examples.extend(rows)
 (ART/f'{S}_manifest.json').write_text(json.dumps({'examples':examples,'excluded':excluded},indent=2)+'\n');fits=torch.load(paths['fits'],weights_only=False);old=json.loads(paths['centroids'].read_text());centroids={(r['family'],r['format'],r['layer']):r['centroids'] for r in old['geometry'] if r['split']=='discovery'};model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 raw={};stash={};handles=[]
 def pre(k):
  def f(mod,args):raw[k]=mod.lambdas[0]*args[0][:,-1]+mod.lambdas[1]*args[2][:,-1]
  return f
 def att(k):
  def f(mod,args,out):raw[k]=raw[k]+out[0][:,-1]
  return f
 for k,b in enumerate(model.blocks):
  if k>=8:handles.extend([b.register_forward_pre_hook(pre(k)),b.attn.register_forward_hook(att(k))])
 attn=model.blocks[8].attn;original=attn.squared_attention
 def capture(q,k,v,q2,k2):
  A=(torch.einsum('bhd,bthd->bht',q[:,-1],k)/128)*(torch.einsum('bhd,bthd->bht',q2[:,-1],k2)/128);stash['routing']=A[:,3].cpu();return original(q,k,v,q2,k2)
 attn.squared_attention=capture
 def buscap(mod,args,out):stash['bus']=out[1].cpu()
 handles.append(model.blocks[0].attn.register_forward_hook(buscap));native={};states={};bus={};routing={};measurements={};geometry=[];actual=0;replay=0.;eps=torch.finfo(torch.float32).eps;buckets=defaultdict(list)
 for i,e in enumerate(examples):buckets[len(e['ids'])].append(i)
 try:
  for length,ii in sorted(buckets.items()):
   for st in range(0,len(ii),12):
    ix=ii[st:st+12];ids=torch.tensor([examples[i]['ids'] for i in ix],device='cuda');y=final_logits(model,ids);actual+=len(ix)
    if not native:yy=final_logits(model,ids);actual+=len(ix);replay=float((y-yy).abs().max())
    for n,i in enumerate(ix):
     e=examples[i];native[i]=y[n].double().cpu();bus[i]=stash['bus'][n].clone();routing[i]=stash['routing'][n].clone();measurements[i]=measure(native[i],e['groups']);measurements[i]['predicted_token']=tok.decode([measurements[i]['top_token']]);states[i]={}
     for k in range(8,18):
      x=raw[k][n].double().cpu();states[i][k]=x;u=x/(x.square().mean()+eps).sqrt()
      for axis in ['A','B']:
       mean,basis,_=fits['agreement',e['family'],axis,k];cc=centroids[e['family'],axis,k];coord=float((u-mean)@basis[0]);prediction=int((coord-np.mean(cc))*(cc[1]-cc[0])>0);geometry.append({key:e[key] for key in ['pair','cohort','format','family','order']}|{'layer':k,'axis':axis,'coordinate':coord,'prediction':prediction,'correct':prediction==e['order']})
   print('DONE',length,actual,flush=True)
 finally:
  attn.squared_attention=original
  for h in handles:h.remove()
 cells=[]
 for cohort in ['irregular','new_regular']:
  for family in ['copy','task']:
   for fmt in TEMPLATES:
    ix=[i for i,e in enumerate(examples) if (e['cohort'],e['family'],e['format'])==(cohort,family,fmt)];ps=sorted({examples[i]['pair'] for i in ix});both=[all((measurements[i]['margin']>0)==(examples[i]['order']==0) for i in ix if examples[i]['pair']==p) for p in ps];top=[measurements[i]['top1'][examples[i]['order']] for i in ix];r=dict(cohort=cohort,family=family,format=fmt,pairs=len(ps),both_correct=float(np.mean(both)) if both else 0.,full_vocab=float(np.mean(top)) if top else 0.);r['pass']=len(ps)>=4 and r['both_correct']>=.75 and r['full_vocab']>=.5;cells.append(r)
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'native_replay_max_abs':replay,'examples':len(examples),'eligible_pairs':len(examples)//24,'whole_model_forwards':actual,'test_evaluations':0,'finite':all(torch.isfinite(v).all().item() for v in native.values())};assert audits['hashes_unchanged'] and replay<1e-4 and audits['finite'];torch.save(dict(examples=examples,native=native,states=states,bus=bus,routing=routing),ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,competence=cells,excluded=excluded,measurements=measurements,geometry=geometry,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
