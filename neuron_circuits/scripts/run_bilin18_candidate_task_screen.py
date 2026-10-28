"""H840: frozen candidate availability and seed agreement on a diverse competence suite."""
import json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART,suite,groups,measure,forward
from run_bilin18_broad_task_competence import continuation_id,final_logits
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_balanced_shared_featurizers import Sparse
from run_bilin18_early_local_geometry import sha
S='bilin18_candidate_task_screen'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
 paths={'candidates':ART/'bilin18_exact_site_stability_phi_candidates.json','fits':ART/'bilin18_balanced_shared_featurizers_captures.pt','means':ART/'bilin18_operator_state_prediction_captures.pt','runner':Path(__file__),'suite':ROOT/'scripts/run_bilin18_diverse_causal_tasks.py','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(v) for k,v in paths.items()}
 cfg={'experiment':'H840','hashes':hashes,'design':'H533 eight-family suite reused, two split-specific formats and lexicons, paired answer-changing endpoints plus same-answer nuisance. Prefix Yesterday, to all entity-binding prompts to align name tokenization, motivated by H820. Not fresh task confirmation. No task-driven candidate replacement.','candidates':'All16 H839 phi-selected scalar48/block8 components at sites1/2, both seeds. Candidate selection uses prior natural-text stability only.','gate':'Per family/split >=4 token-valid pairs, both-endpoint binary correctness >=.75, pooled full-vocab intended-group top1 >=.5. No individual competence filtering.','analysis':'Final-position normalized preMLP state; training-token-mean subtraction and frozen training scale. Report support, decoded contribution agreement, paired task-changing vs nuisance contribution distance. Descriptive availability only, no causal/semantic evidence.','next_lock':'Keep all candidates. Causal swaps should use every valid pair in competent family/split cells, both seeds, task-changing and nuisance donor, plus matched-direction and identity controls. Do not rank candidates by this task screen.'}
 (ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');rows=suite()
 for r in rows:
  if r['family']=='entity_binding':r['prompts']=['Yesterday, '+t for t in r['prompts']];r['nuisance']='Yesterday, '+r['nuisance']
 (ART/f'{S}_manifest.json').write_text(json.dumps(rows,indent=2)+'\n');tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);examples=[];excluded=[]
 for row in rows:
  enc=[]
  for variant,text in enumerate(row['prompts']+[row['nuisance']]):
   gg=[[continuation_id(tok,text,c) for c in g] for g in groups(row['family'],row['answers'])];gg=[[v for v in g if v is not None] for g in gg]
   if any(not g for g in gg) or set(gg[0])&set(gg[1]):break
   enc.append(dict(id=row['id'],family=row['family'],split=row['split'],variant=variant,text=text,ids=tok.encode(text,add_special_tokens=False),groups=gg))
  if len(enc)==3:examples.extend(enc)
  else:excluded.append(row['id'])
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 caps={l:torch.empty(len(examples),1152) for l in [1,2]};native=[None]*len(examples);buckets=defaultdict(list);audit=0
 for i,e in enumerate(examples):buckets[len(e['ids'])].append(i)
 for length,ixs in sorted(buckets.items()):
  for st in range(0,len(ixs),12):
   ix=ixs[st:st+12];captured={};hooks=[]
   for l in [1,2]:
    def h(mod,args,layer=l):captured[layer]=args[0][:,-1].detach().cpu()
    hooks.append(model.blocks[l].mlp.register_forward_pre_hook(h))
   ids=torch.tensor([examples[i]['ids'] for i in ix],device='cuda')
   try:logits,_=forward(model,ids)
   finally:
    for h in hooks:h.remove()
   if st==0 and length==min(buckets):audit=float((logits-final_logits(model,ids)).abs().max())
   for k,i in enumerate(ix):
    native[i]=measure(logits[k],examples[i]['groups']);native[i]['predicted_text']=tok.decode([native[i]['top_token']])
   for l in caps:caps[l][ix]=captured[l]
  print('captured',length,len(ixs),flush=True)
 del model
 index={(e['id'],e['variant']):i for i,e in enumerate(examples)};competence=[]
 for fam in sorted({r['family'] for r in rows}):
  for split in ['discovery','transfer']:
   rr=[r for r in rows if r['family']==fam and r['split']==split and (r['id'],0) in index];n=len(rr)
   both=[native[index[r['id'],0]]['margin']>0 and native[index[r['id'],1]]['margin']<0 for r in rr];top=[native[index[r['id'],v]]['top1'][v] for r in rr for v in [0,1]]
   c={'family':fam,'split':split,'pairs':n,'both_correct':float(np.mean(both)) if n else 0,'full_vocab':float(np.mean(top)) if n else 0};c['pass']=n>=4 and c['both_correct']>=.75 and c['full_vocab']>=.5;competence.append(c)
 states=torch.load(paths['fits'],weights_only=False)['states'];means=torch.load(paths['means'],weights_only=False)['fits'];candidates=json.loads(paths['candidates'].read_text());records=[];saved={};tids=[e['ids'][-1] for e in examples]
 for spec in candidates:
  l,kind=spec['layer'],spec['kind'];rank=8 if kind=='block8' else 1;f=means[l];lookup={int(t):i for i,t in enumerate(f['token_ids'])};tm=torch.stack([f['token_shape_mean'][lookup[t]] if t in lookup else f['shape_mean'] for t in tids]);res=caps[l].double()-tm;decoded={};supports={}
  for seed in [8351,8352]:
   ck=states[l,kind,seed];m=Sparse(rank,48 if kind=='scalar48' else 64).cuda();m.load_state_dict(ck['state_dict']);out=m((res/ck['scale']).float().cuda());units=[u['seed8351_unit' if seed==8351 else 'seed8352_match'] for u in spec['units']];components=[];ss=[]
   for unit in units:
    sl=slice(unit*rank,(unit+1)*rank);components.append((out['code'][:,sl]@m.decoder[sl]*ck['scale']).cpu());ss.append(out['support'][:,unit*rank].bool().cpu())
   decoded[seed]=torch.stack(components,1);supports[seed]=torch.stack(ss,1);del m
  saved[l,kind]={'decoded':decoded,'support':supports}
  for ci,u in enumerate(spec['units']):
   for c in competence:
    inds=[i for i,e in enumerate(examples) if e['family']==c['family'] and e['split']==c['split']];a=supports[8351][inds,ci];b=supports[8352][inds,ci];da=decoded[8351][inds,ci];db=decoded[8352][inds,ci];joint=a&b;uni=a|b
    na,nb=da.norm(dim=1),db.norm(dim=1);cos=(da*db).sum(1)/(na*nb).clamp_min(1e-30);pa,pb=float(a.float().mean()),float(b.float().mean());den=(pa*(1-pa)*pb*(1-pb))**.5;phi=(float(joint.float().mean())-pa*pb)/den if den else None
    contrast=[]
    for seed in [8351,8352]:
     vals=[]
     for r in rows:
      if r['family']==c['family'] and r['split']==c['split'] and (r['id'],0) in index:
       z=decoded[seed][:,ci];i,j,k=[index[r['id'],v] for v in [0,1,2]];vals.append([float((z[i]-z[j]).norm()),float((z[i]-z[k]).norm())])
     contrast.append({'seed':seed,'task_change_mean':float(np.mean([v[0] for v in vals])) if vals else None,'nuisance_change_mean':float(np.mean([v[1] for v in vals])) if vals else None})
    records.append({'layer':l,'kind':kind,'candidate':ci,'unit_a':u['seed8351_unit'],'unit_b':u['seed8352_match'],'family':c['family'],'split':c['split'],'competent':c['pass'],'contexts':len(inds),'active_counts':[int(a.sum()),int(b.sum())],'support_Jaccard':float(joint.sum()/uni.sum()) if uni.any() else None,'support_phi':phi,'joint_active':int(joint.sum()),'decoded_cosine_joint':float(cos[joint].mean()) if joint.any() else None,'paired_distances':contrast})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'native_forward_max_abs':audit,'whole_model_forwards':len(examples)+min(12,len(buckets[min(buckets)])),'test_evaluations':0,'all_finite':all(bool(torch.isfinite(v).all()) for v in caps.values()),'candidates_retained':sum(len(s['units']) for s in candidates)};assert audits['hashes_unchanged'] and audit<1e-4 and audits['all_finite']
 torch.save({'examples':examples,'normalized_premlp':caps,'components':saved,'native':native},ART/f'{S}_captures.pt');result={'config':cfg,'excluded':excluded,'competence':competence,'records':records,'native':[dict(**e,measurement=n) for e,n in zip(examples,native)],'audits':audits,'elapsed_seconds':time.time()-start};(ART/f'{S}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'competence':competence,'audits':audits}),flush=True)
if __name__=='__main__':main()
