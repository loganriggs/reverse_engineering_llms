"""H746: equal-rank operation versus nuisance subspaces; reused-panel diagnostic."""
import json,hashlib
from pathlib import Path
from collections import defaultdict
import torch
from run_bilin18_query_key_edit_subspaces import rotate,projected_edit
from run_bilin18_recall_query_key_cross import coefficient
ROOT=Path('/home/loganriggs/Coding/neuron_circuits');ART=ROOT/'artifacts/bilin18_tensor_decomposition';STEM='bilin18_operation_nuisance_subspaces'
def main():
 torch.set_num_threads(6)
 paths=[ART/'bilin18_frozen_operation_generalization_captures.pt',ART/'bilin18_frozen_operation_generalization.json',ART/'bilin18_query_key_edit_subspaces_captures.pt']
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();hashes={str(p):sha(p) for p in paths}
 cfg={'experiment':'H746','hashes':hashes,'rank':4,'fit_names':'old indices 0..3 only','evaluation_names':'old indices 4..7 and all token-valid new names','design':'Existing H744 panel, retrospective diagnostic. Separate q2/k2 uncentered SVD. Identity nuisance uses endpoint0->1 within operation/format/old wording (16rows); wording nuisance uses old->new wording within operation/format/variant (32rows). Operation new wording uses binding->copy (16rows); pooled wording operation 32rows; union uses two old and two new operation directions with QR. All bases rank4, receiver-norm preserving. No rank or task-success selection. All test cases already inspected; not prospective confirmation.','next_confirmation':'Independent wording-family panel required.'}
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 cap=torch.load(paths[0],weights_only=False);data=json.loads(paths[1].read_text());prior=torch.load(paths[2],weights_only=False)['bases'];states=cap['states'];meta={(r['id'],r['receiver']):r for r in data['records'] if r['scope']=='pooled'};bases={};training=[];orth=[]
 def key(w,fmt,fam,i,v):return (f'h744-old-{w}-{fmt}-{fam}-{i}',v)
 for c,u in [('q2','uq2'),('k2','uk2')]:
  mats=defaultdict(list)
  for i in range(4):
   for fmt in ['narrative','list']:
    for fam in ['entity_binding','sequence_copy']:
     mats['identity'].append(states[key('old',fmt,fam,i,1)][u]-states[key('old',fmt,fam,i,0)][u])
     for v in [0,1]:mats['wording'].append(states[key('new',fmt,fam,i,v)][u]-states[key('old',fmt,fam,i,v)][u])
    for w in ['old','new']:
     for v in [0,1]:mats['operation_'+w].append(states[key(w,fmt,'sequence_copy',i,v)][u]-states[key(w,fmt,'entity_binding',i,v)][u])
  mats['operation_pooled']=mats['operation_old']+mats['operation_new']
  for scope,rows in mats.items():
   _,s,vh=torch.linalg.svd(torch.stack(rows),full_matrices=False);assert s[3]>s[0]*1e-8;bases[scope,c]=vh[:4].T;training.append({'scope':scope,'component':c,'rows':len(rows),'singular_values':s.tolist()})
  bases['union_2plus2',c]=torch.linalg.qr(torch.cat([bases['operation_old',c][:,:2],bases['operation_new',c][:,:2]],dim=1)).Q
  bases['frozen_H742',c]=prior['pooled',4,None,c]
  for seed in [7421,7422,7423]:bases[f'random_{seed}',c]=prior['random',4,seed,c]
 for b in bases.values():orth.append(float((b.T@b-torch.eye(4)).abs().max()))
 fit_name_ids={('old',i) for i in range(4)};records=[];replay=[]
 for key0,donor in cap['pairs']:
  m=meta[key0]
  if m['name_set']=='old' and m['item']<4:continue
  r,d=states[key0],states[donor];e=cap['examples'][key0];target=float(coefficient({**r,'q2':d['q2'],'k2':d['k2']},4.1875));native=float(coefficient(r,4.1875))
  for scope in sorted({s for s,c in bases}):
   q=rotate(projected_edit(r['uq2'],d['uq2'],bases[scope,'q2'],True),len(e['ids'])-1);k=rotate(projected_edit(r['uk2'],d['uk2'],bases[scope,'k2'],True),e['source_position']);pred=float(coefficient({**r,'q2':q,'k2':k},4.1875))
   records.append({**{s:m[s] for s in ['id','receiver','name_set','wording','format','family','item']},'scope':scope,'native':native,'target':target,'pred':pred})
   if scope=='frozen_H742':replay.append(abs(pred-m['predicted_alpha']))
 groups=defaultdict(list)
 for r in records:groups[r['name_set'],r['wording'],r['format'],r['scope']].append(r)
 summary=[]
 for (n,w,f,s),rr in groups.items():
  den=sum((r['target']-r['native'])**2 for r in rr)**.5;eligible=[r for r in rr if r['target']*r['native']<0]
  summary.append({'name_set':n,'wording':w,'format':f,'scope':s,'contexts':len(rr),'competent':all(data['competence'][f'{n}|{w}|{f}|{fam}']['pass'] for fam in ['entity_binding','sequence_copy']),'relative_error':sum((r['pred']-r['target'])**2 for r in rr)**.5/den,'sign_transfer':[sum(r['pred']*r['target']>0 for r in eligible),len(eligible)]})
 audits={'hashes_unchanged':all(sha(p)==hashes[str(p)] for p in paths),'orthogonality_max_abs':max(orth),'H744_replay_max_abs':max(replay),'fit_evaluation_name_overlap':bool(fit_name_ids & {(r['name_set'],r['item']) for r in records}),'finite':all(torch.isfinite(torch.tensor(r['pred'])).item() for r in records),'whole_model_forwards':0,'test_evaluations':0}
 assert not audits['fit_evaluation_name_overlap'] and audits['hashes_unchanged'] and audits['finite'] and max(orth)<1e-10 and max(replay)<1e-10
 torch.save({'bases':bases},ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'training':training,'records':records,'summary':summary,'audits':audits},indent=2)+'\n');print(audits)
 for x in summary:
  if x['scope'] in ['frozen_H742','identity','wording','operation_new','operation_pooled','union_2plus2']:print(x['name_set'],x['wording'],x['format'],x['scope'],round(x['relative_error']*100,1),x['sign_transfer'])
if __name__=='__main__':main()
