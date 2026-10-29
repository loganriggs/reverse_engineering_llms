"""M67 all split/layer/site/family/mask comparisons; no candidate selection."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 rp=V/'native_change_subset_responses.json';r=json.loads(rp.read_text());a=json.loads((V/'native_change_subset_audit.json').read_text());assert a['status']=='passed' and a['result_sha256']==sha(rp);summaries=[];comparisons=[]
 for split in ['train','source_eval']:
  fams=sorted({x['family'] for x in r['rows'] if x['split']==split})
  for layer in [0,1,2]:
   for region in ['changed_input_positions','final_position']:
    rr=[x for x in r['rows'] if x['status']=='evaluated' and x['split']==split and x['layer']==layer and x['region']==region]
    for factor in [1,2]:
     for mask in [None,0,1,2,3]:
      for family in [None]+fams:
       rows=[x for x in rr if x['factor']==factor and (mask is None or x['mask']==mask) and (family is None or x['family']==family)];te=sum(x['target_energy'] for x in rows);me=sum(x['margin_target_energy'] for x in rows);ne=sum(x['native_delta_energy'] for x in rows);summaries.append(dict(split=split,layer=layer,region=region,factor=factor,mask=mask,family=family,n=len(rows),target_energy=te,response_error=sum(x['response_error'] for x in rows),relative_response_error=sum(x['response_error'] for x in rows)/te if te else None,relative_margin_error=sum(x['margin_error'] for x in rows)/me if me else None,relative_local_error=sum(x['local_error'] for x in rows)/ne if ne else None))
    comp=dict(split=split,layer=layer,region=region)
    for factor in [1,2]:
     rows=[x for x in summaries if x['split']==split and x['layer']==layer and x['region']==region and x['factor']==factor and x['mask'] is None];pool=next(x for x in rows if x['family'] is None);families=[x for x in rows if x['family'] is not None and x['relative_response_error'] is not None];comp[str(factor)+'_pooled']=pool['relative_response_error'];comp[str(factor)+'_family_mean']=sum(x['relative_response_error'] for x in families)/len(families);comp[str(factor)+'_margin_pooled']=pool['relative_margin_error'];comp[str(factor)+'_local']=pool['relative_local_error']
    comparisons.append(comp)
 op=V/'native_change_subset_comparison.json';assert not op.exists();op.write_text(json.dumps(dict(status='completed',source_sha256=sha(Path(__file__)),result_sha256=sha(rp),summaries=summaries,comparisons=comparisons,scope='Four masks include complements, reused across conditions; not independent random trials. Family means average family energy ratios, pooling averages by native target energy. Training and familiar source-group evaluation separate.'),indent=2)+'\n');print(json.dumps(comparisons,indent=2))
if __name__=='__main__':main()
