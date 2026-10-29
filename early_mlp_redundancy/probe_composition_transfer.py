"""M116 actual native lexical changes transferred across modifier/format contexts."""
import json
import torch
from fit_pca_encoder import V,A,sha
from inspect_model import load_model,trace


def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
 pp=V/'composition_transfer_protocol.json';p=json.loads(pp.read_text());assert p['source_sha256']==sha(__file__)
 for name,h in p['bindings'].items():assert sha(V/name)==h
 prior=json.loads((V/'sentiment_composition_native.json').read_text());assert sha(prior['path'])==prior['sha256'];old=torch.load(prior['path'],weights_only=True);panel=json.loads((V/'sentiment_composition_protocol.json').read_text());items=[x for x in panel['rows'] if x['cohort']=='factorial'];index={(x['format'],x['lexicon'],x['subject'],x['valence'],x['negated']):x for x in items}
 model,mr=load_model();assert mr.weights_sha256==prior['checkpoint_sha256'];rows=[];saved={};ineligible=[];calls=0;replays=0
 for item in items:
  ident=(item['format'],item['lexicon'],item['subject'],item['valence'],item['negated']);donor=index[ident[:3]+(1-ident[3],ident[4])];ids=torch.tensor([item['token_ids']],device='cuda');baseline,cap=trace(model,ids);calls+=1;assert torch.equal(baseline[0,-1].cpu(),old[item['id']]['logits']);replays+=1
  assert torch.equal(cap[1]['normalized_input'].cpu(),old[item['id']]['layers'][1]['input'])
  contexts={}
  for name,flip_format,flip_modifier in [('native',False,False),('modifier',False,True),('format',True,False),('both',True,True)]:
   fmt=('text' if ident[0]=='review' else 'review') if flip_format else ident[0];neg=1-ident[4] if flip_modifier else ident[4];r=index[(fmt,ident[1],ident[2],ident[3],neg)];d=index[(fmt,ident[1],ident[2],1-ident[3],neg)];assert len(r['token_ids'])==len(d['token_ids']);positions=[i for i,(a,b) in enumerate(zip(r['token_ids'],d['token_ids'])) if a!=b];contexts[name]=(r,d,positions)
  rp=contexts['native'][2]
  for site in ['source','final']:
   pos=rp if site=='source' else [len(item['token_ids'])-1];deltas={}
   for name,(r,d,positions) in contexts.items():
    rpos=positions if site=='source' else [len(r['token_ids'])-1];dpos=positions if site=='source' else [len(d['token_ids'])-1];assert len(pos)==len(rpos)
    deltas[name]=(old[d['id']]['layers'][1]['write'][:,dpos]-old[r['id']]['layers'][1]['write'][:,rpos]).cuda()
   targetnorm=deltas['native'].double().norm()
   for name in ['modifier','format','both']:
    norm=deltas[name].double().norm()
    if float(norm)==0 and float(targetnorm)>0:ineligible.append(dict(id=item['id'],site=site,condition=name+'_norm',reason='zero_transferred_nonzero_native'))
    else:deltas[name+'_norm']=deltas[name]*(targetnorm/norm).to(deltas[name].dtype) if float(norm)>0 else deltas[name]
   for condition,delta in deltas.items():
    full=torch.zeros_like(cap[1]['native_write']);full[:,pos]=delta
    if bool(delta.count_nonzero()):out,_=trace(model,ids,replace=lambda layer,z,w:w+full if layer==1 else w);calls+=1;logits=out[0,-1].cpu()
    else:logits=baseline[0,-1].cpu()
    key=f"{item['id']}|{site}|{condition}";saved[key]=dict(logits=logits,delta=delta.cpu());rows.append(dict(id=item['id'],format=item['format'],lexicon=item['lexicon'],subject=item['subject'],valence=item['valence'],negated=item['negated'],site=site,condition=condition,positions=pos,donor=donor['id'],key=key))
  if replays%16==0:print('RECIPIENTS',replays,'FORWARDS',calls,flush=True)
 path=A/'emlp_m116_composition_transfer.pt';assert not path.exists();torch.save(saved,path);out=dict(status='completed_pending_audit',source_sha256=sha(__file__),protocol_sha256=sha(pp),checkpoint_sha256=mr.weights_sha256,forwards=calls,baseline_replays=replays,rows=rows,ineligible=ineligible,path=str(path),sha256=sha(path));dest=V/'composition_transfer_native.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print('COMPLETE',calls,len(rows),len(ineligible))
if __name__=='__main__':main()
