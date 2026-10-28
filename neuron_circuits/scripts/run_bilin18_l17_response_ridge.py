"""H290 ridge selection using already saved response sufficient statistics."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';STEM=ART/'bilin18_l17_response_ridge'
def main():
 cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());data=np.load(ART/'bilin18_l17_response_span_statistics.npz');keys=[k[:-5] for k in data.files if k.endswith('_gram')];records={};weights={}
 def aggregates(split,width):return [sum(data[k+'_'+suffix] for k in keys if k.startswith(split)) for suffix in ['gram','rhs','energy']]
 for width in cfg['widths']:
  gg,rr,e=aggregates('train',width);g=gg[:width,:width];r=rr[:width]
  for alpha in cfg['alphas']:
   regularized=g+alpha*np.trace(g)/width*np.eye(width);w=np.linalg.solve(regularized,r);name=f'n{width}_a{alpha}';full=np.zeros((128,1152));full[:width]=w;weights[name]=full.astype('float32');record={'width':width,'alpha':alpha,'normal_equation_relative_residual':float(np.linalg.norm(regularized@w-r)/np.linalg.norm(r)),'groups':{}}
   for key in keys:
    g1=data[key+'_gram'];r1=data[key+'_rhs'];e1=data[key+'_energy'];record['groups'][key]=float((np.sum(full*(g1@full))-2*np.sum(full*r1)+e1)/e1)
   for split in ['train','validation']:
    energy=sum(float(data[k+'_energy']) for k in keys if k.startswith(split));record[split]=sum(record['groups'][k]*float(data[k+'_energy']) for k in keys if k.startswith(split))/energy
   records[name]=record
 winners={str(w):min((k for k,v in records.items() if v['width']==w),key=lambda k:records[k]['validation']) for w in cfg['widths']};best64=winners['64'];expanded=min([winners['96'],winners['128']],key=lambda k:records[k]['validation']);same0=next(k for k,v in records.items() if v['width']==records[expanded]['width'] and v['alpha']==0);gain=1-records[expanded]['validation']/records[best64]['validation'];ridge=1-records[expanded]['validation']/records[same0]['validation'];gates={'expanded_gain':gain>=.02,'ridge_gain':ridge>=.02};np.savez_compressed(STEM.with_name(STEM.name+'_frozen.npz'),**weights,best64=weights[best64],expanded=weights[expanded]);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'records':records,'winners':winners,'best64':best64,'expanded':expanded,'expanded_relative_gain':gain,'ridge_relative_gain':ridge,'gates':gates},indent=2)+'\n');print('WINNERS',winners,'GAIN',gain,ridge,'GATES',gates)
if __name__=='__main__':main()
