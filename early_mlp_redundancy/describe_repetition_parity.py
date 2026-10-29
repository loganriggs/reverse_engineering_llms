"""Explicit post-result parity-like preference description; not a circuit claim."""
import json,hashlib
from fit_pca_encoder import V,sha


def main():
    rp=V/'repetition_cube_native.json';r=json.loads(rp.read_text());audit=json.loads((V/'repetition_cube_audit.json').read_text());assert audit['status']=='passed' and audit['native_sha256']==sha(rp);rows=[x for x in r['rows'] if x['cohort']=='cube'];conditions=[]
    for x in rows:
        pred=int(x['canonical_margin']>0);parity=sum(x['bits'])%2;majority=int(sum(x['bits'])>=2);mixed=x['vertex'] not in [0,7];assert not mixed or parity==1-majority;assert pred==x['group_choice'];conditions.append(dict(id=x['id'],mixed=mixed,native_choice=pred,parity_choice=parity,agrees=pred==parity))
    out=dict(status='completed',source_sha256=sha(__file__),native_sha256=sha(rp),rows=conditions,agreement_all=sum(x['agrees'] for x in conditions),n_all=64,agreement_mixed=sum(x['agrees'] for x in conditions if x['mixed']),n_mixed=48,top1_in_either_value=sum(x['top1_value_group'] is not None for x in rows),scope='Post-result hypothesis only: on3binaryoccurrences, oddparity agreeswithminority onmixed vertices andwithcommonvalue onuniform vertices. Describes preferencebetweentwocandidates, not fullvocabularygeneration orproof ofXOR circuit. Allcases retained, no model/promptfit. Must not replaceprespecified first/last/majority comparisons.')
    p=V/'repetition_cube_parity_description.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='rows'})
if __name__=='__main__':main()
