"""M115 post-result descriptive factorial accounting; no label/gate changes."""
import json
from collections import Counter
import numpy as np
import tiktoken
from fit_pca_encoder import V,sha


def main():
 rp=V/'sentiment_composition_native.json';r=json.loads(rp.read_text());audit=json.loads((V/'sentiment_composition_audit.json').read_text());assert audit['status']=='passed' and audit['native_sha256']==sha(rp);tok=tiktoken.get_encoding('gpt2');rows=[];tops=[]
 for fmt in ['review','text']:
  for lex in range(4):
   for subject in ['movie','book','meal','game']:
    rr=sorted([x for x in r['rows'] if x.get('format')==fmt and x['lexicon']==lex and x['subject']==subject],key=lambda x:(x['valence'],x['negated']));assert len(rr)==4
    X=np.array([[1,1-2*x['valence'],1-2*x['negated'],(1-2*x['valence'])*(1-2*x['negated'])] for x in rr]);y=np.array([x['positive_minus_negative_margin'] for x in rr]);c=np.linalg.solve(X,y);assert np.allclose(X@c,y)
    direct=X.T@y/4;assert np.allclose(direct,c)
    rows.append(dict(format=fmt,lexicon=lex,subject=subject,offset=float(c[0]),valence_main=float(c[1]),modifier_main=float(c[2]),interaction=float(c[3]),coefficients=c.tolist(),margins=y.tolist()))
  for valence in [0,1]:
   for negated in [0,1]:
    rr=[x for x in r['rows'] if x.get('format')==fmt and x['valence']==valence and x['negated']==negated];counts=Counter(tok.decode([x['top1_token']]) for x in rr);tops.append(dict(format=fmt,valence=valence,negated=negated,counts=dict(counts)))
 summaries=[]
 for fmt in ['review','text']:
  rr=[x for x in rows if x['format']==fmt];summaries.append(dict(format=fmt,mean_coefficients=np.mean([x['coefficients'] for x in rr],0).tolist(),positive_interactions=sum(x['interaction']>0 for x in rr),interaction_range=[min(x['interaction'] for x in rr),max(x['interaction'] for x in rr)]))
 out=dict(status='descriptive_completed',source_sha256=sha(__file__),native_sha256=sha(rp),audit_sha256=sha(V/'sentiment_composition_audit.json'),cubes=rows,summary=summaries,top1_tokens=tops,scope='Post-result two-factor decomposition of fixed semantic-group logsumexp margin, not raw neural-computation attribution. Positive interaction is consistent with negation-sensitive valence; additive offsets can prevent choice flips. No gate changes, answer-set expansion or causal localization.')
 dest=V/'sentiment_composition_description.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print('COEFFICIENTS',summaries);print('TOP1',tops)
if __name__=='__main__':main()
