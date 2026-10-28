"""H973 source-position finite local/inherited factorial interaction removals."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import forward,measure
from pathlib import Path
from collections import defaultdict
import torch,json,numpy as np,time
S='bilin18_source_component_removals'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'source':ART/'bilin18_source_position_composition_captures.pt','attention':ART/'bilin18_attention5_sources_captures.pt','data':ART/'bilin18_factorial_valence_negation_captures.pt','panel':ART/'bilin18_factorial_valence_negation.json','model':SNAPSHOT/'pytorch_model.bin','forward':ROOT/'scripts/run_bilin18_diverse_causal_tasks.py'};hashes={k:sha(p) for k,p in paths.items()};src=torch.load(paths['source'],weights_only=False);attdata=torch.load(paths['attention'],weights_only=False);data=torch.load(paths['data'],weights_only=False);panel=json.loads(paths['panel'].read_text());ex=data['examples'];qs=data['quartets'];native=data['native'];pos=src['positions']
 cfg={'experiment':'H973','hashes':hashes,'design':'Independently remove a*n times local, inherited or total MLP factorial interaction at adjective-last-subtoken or period at layers0-4. Native prompt replay with targeted output replacement, all later positions/layers recomputed. Identity and paired norm-matched random factorial output edits for each site/component.','downstream':'Record projections of attention5 mixed source values onto their native factorial interaction at adjective/period and final-token attention5 output onto its native interaction. Quartet signed average gives retained projection fraction; may exceed1 or be negative. This is a tracking diagnostic, not full causal mediation.','limits':'Teacher-informed quartet output edits can be off-manifold. Same-panel site scan with onlyfour native-competent familiar formatA quartets; failed cohorts retained, not filtered. No training or reserved test.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 lam=float(model.blocks[5].attn.lamb);targets={}
 for q in qs:
  targets[q['id']]={'attention':attdata['parts'][q['id']]['total'].float()}
  for role in ['adjective','period']:
   c=src['coefficients'][q['id'],role];targets[q['id']][role]=((1-lam)*c[5,'current_value'][3]+lam*c[5,'value_bus'][3]).float()
 current=[];buf={}
 def valuehook(module,args):
  x,bus=args;b,t,c=x.shape;v=(1-module.lamb)*module.c_v(x).reshape(b,t,module.n_head,module.head_dim)+module.lamb*bus.view(b,t,module.n_head,module.head_dim)
  for role in ['adjective','period']:buf[role]=torch.stack([v[k,pos[j['i']][role]] for k,j in enumerate(current)]).cpu()
 def attentionhook(module,args,out):buf['attention']=out[0][:,-1].detach().cpu()
 handles=[model.blocks[5].attn.register_forward_pre_hook(valuehook),model.blocks[5].attn.register_forward_hook(attentionhook)];records=[];actual=0;ident=0.;normerr=0.;site_identity_projection_error=0.
 try:
  for layer in range(5):
   for ri,role in enumerate(['adjective','period']):
    jobs=[]
    for qi,q in enumerate(qs):
     p=src['parts'][q['id'],role,layer];cc={'local':p['local'],'inherited':p['inherited'],'total':p['local']+p['inherited']}
     for state,i in enumerate(q['indices']):
      b=dict(q=q,state=state,i=i);v=ex[i]['valence'];jobs.append(b|dict(component='identity',arm='identity',edit=torch.zeros(1152)))
      for ci,name in enumerate(cc):
       c=cc[name];gen=torch.Generator().manual_seed(973000+qi*100+layer*10+ri*3+ci);z=torch.randn(1152,generator=gen);z=z/z.norm()*c.norm();normerr=max(normerr,float(abs(z.norm()-c.norm())/c.norm().clamp_min(1e-20)));jobs.extend([b|dict(component=name,arm='remove',edit=-v*c),b|dict(component=name,arm='random_plus',edit=v*z),b|dict(component=name,arm='random_minus',edit=-v*z)])
    buckets=defaultdict(list)
    for j in jobs:buckets[len(ex[j['i']]['ids'])].append(j)
    def patch(module,args,out):
     y=out.clone()
     for k,j in enumerate(current):y[k,pos[j['i']][role]]=src['captures'][j['i']][layer,'mlp'][ri].to(y)+j['edit'].to(y)
     return y
    h=model.blocks[layer].mlp.register_forward_hook(patch)
    try:
     for length,jj in sorted(buckets.items()):
      for st in range(0,len(jj),24):
       current=jj[st:st+24];y,_=forward(model,torch.tensor([ex[j['i']]['ids'] for j in current],device='cuda'));actual+=len(current)
       for k,j in enumerate(current):
        out=y[k].double().cpu();base=native[j['i']];lp=out.log_softmax(0);ln=base.log_softmax(0);margin=measure(out,ex[j['i']]['groups'])['margin'];bm=measure(base,ex[j['i']]['groups'])['margin'];v=ex[j['i']]['valence']
        if j['arm']=='identity':ident=max(ident,float((out-base).abs().max()))
        projections={}
        for key in ['adjective','period','attention']:
         target=targets[j['q']['id']][key].double();projections[key]=float((buf[key][k].double()*target).sum()/target.square().sum().clamp_min(1e-20))
        records.append({key:j['q'][key] for key in ['id','format','cohort','source']}|dict(layer=layer,role=role,state=j['state'],arm=j['arm'],component=j['component'],margin=margin,loss=v*(bm-margin),KL=float((ln.exp()*(ln-lp)).sum()),binary_correct=v*margin>0,valence=v,downstream_projection=projections))
    finally:h.remove()
    print('DONE',layer,role,len(records),flush=True)
 finally:
  for h in handles:h.remove()
 summary=[];grouped=defaultdict(list)
 for r in records:grouped[r['format'],r['cohort'],r['layer'],r['role'],r['component'],r['arm']].append(r)
 for key,rr in grouped.items():
  retention={k:float(np.mean([r['valence']*r['downstream_projection'][k] for r in rr])) for k in ['adjective','period','attention']}
  if key[-1]=='identity':site_identity_projection_error=max(site_identity_projection_error,max(abs(v-1) for v in retention.values()))
  summary.append(dict(zip(['format','cohort','layer','role','component','arm'],key))|dict(n=len(rr),mean_loss=float(np.mean([r['loss'] for r in rr])),rms_loss=float(np.mean([r['loss']**2 for r in rr])**.5),mean_KL=float(np.mean([r['KL'] for r in rr])),binary_correct=float(np.mean([r['binary_correct'] for r in rr])),retained_interaction_projection=retention))
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),whole_model_forwards=actual,expected_forwards=len(ex)*10*10,identity_logits_max_abs=ident,random_norm_relative=normerr,identity_downstream_projection_max_abs=site_identity_projection_error,finite=all(np.isfinite(r['KL']) for r in records),no_training=True,test_evaluations=0)
 assert audits['hashes_unchanged'] and actual==audits['expected_forwards'] and ident<1e-4 and normerr<1e-5 and site_identity_projection_error<1e-4 and audits['finite']
 (ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,competence=panel['competence'],records=records,summary=summary,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
