"""H256 document-stratified analyses fixed in config before evaluation."""
import json
import numpy as np
from run_bilin18_l17_local_residual_charts import ART
from run_bilin18_l17_probability_support import summarize
STEM=ART/'bilin18_l17_local_chart_replication'
def main():
    j=json.loads(STEM.with_suffix('.json').read_text());d=np.load(STEM.with_name(STEM.name+'_sites.npz'));cfg=j['config'];ix=np.random.default_rng(20261129).integers(128,size=(2000,128));entropy=d['entropy'];masks={'low':entropy<2,'medium':(entropy>=2)&(entropy<4),'high':entropy>=4};results={}
    for group,mask in masks.items():
        results[group]={}
        for family in ['native','isotropic','shuffled']:
            names=['native'] if family=='native' else [f'{family}{s}' for s in cfg['seeds']]
            base=np.mean([d[f'{n}_s1.0_none'] for n in names],axis=0)
            results[group][family]={}
            for correction in ['local16','wrong16','local64']:
                kl=np.mean([d[f'{n}_s1.0_{correction}'] for n in names],axis=0);results[group][family][correction]=summarize(kl,base,mask,ix)
    def boot(mask):
        a=(d['native_s1.0_local16']*mask).reshape(128,4).sum(1);b=(d['native_s1.0_none']*mask).reshape(128,4).sum(1)
        return 1-a[ix].sum(1)/b[ix].sum(1)
    lo=results['low']['native']['local16'];hi=results['high']['native']['local16'];gap=lo['recovery']-hi['recovery'];ci=np.nanquantile(boot(masks['low'])-boot(masks['high']),[.025,.975]).tolist()
    j['entropy_strata']=results;j['entropy_contrast']={'low_minus_high_recovery':gap,'document_bootstrap95':ci};j['gates']['entropy_dependence']=min(lo['sites'],hi['sites'])>=30 and gap>=.15 and ci[0]>0
    j['gates']['directional_native_alignment']=all(v['paired_document_bootstrap95'][0]>0 for c in j['contrasts'].values() for v in c.values())
    STEM.with_suffix('.json').write_text(json.dumps(j,indent=2));print(json.dumps({'gates':j['gates'],'entropy_contrast':j['entropy_contrast'],'strata':results},indent=2))
if __name__=='__main__':main()
