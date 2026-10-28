"""H340 localize head6 sensitivity-energy amplification and document concentration."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';STEM=ART/'bilin18_l17_head6_energy_localization'
def components(G):return np.stack([G[...,1,1],G[...,2,2],2*G[...,1,2]],-1)
def main():
 cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());path=ART/'bilin18_l17_head_pairing_confirmation_rows.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['rows_sha256'];rows=np.load(path);actual=np.stack([rows[f'actual_h{h}_gram'] for h in range(9)]);controls={k:np.stack([np.stack([rows[f'{k}{i}_h{h}_gram'] for h in range(9)]) for i in range(6)]) for k in ['moment','donor']};old=json.loads((ART/'bilin18_l17_head_pairing_confirmation.json').read_text());summary={};draws={};rng=np.random.default_rng(cfg['seed']);replay=0.
 A=components(actual[6]);a=A.sum(0);totalactual=components(actual).sum((1,2));assert np.all(totalactual>0)
 for name,arr in controls.items():
  G=arr.mean(0);C=components(G[6]);c=C.sum(0);e=C.sum(1);ea=A.sum(1);alls=components(G).sum((1,2));share=float(e.sum()/alls.sum());replay=max(replay,abs(share-old['summary'][name]['head6_mlp_energy_share']));excess=c-a;order=np.argsort(e)[::-1];keep=np.ones(128,dtype=bool);keep[order[:7]]=False;ratios=[];fractions=[]
  for _ in range(4000):
   ids=rng.integers(128,size=128);seeds=rng.integers(6,size=6);cc=components(arr[seeds,6].mean(0)[ids]).sum(0);aa=A[ids].sum(0);ratios.append(cc.sum()/aa.sum());fractions.append((cc[0]-aa[0])/(cc.sum()-aa.sum()))
  summary[name]={'head6_energy_ratio':float(c.sum()/a.sum()),'head6_energy_ratio95':np.quantile(ratios,[.025,.975]).tolist(),'all_heads_energy_ratio':float(alls.sum()/totalactual.sum()),'other_heads_energy_ratio':float((alls.sum()-alls[6])/(totalactual.sum()-totalactual[6])),'head6_energy_share':share,'actual_head6_components':a.tolist(),'control_head6_components':c.tolist(),'excess_components':excess.tolist(),'compiled_fraction_of_excess':float(excess[0]/excess.sum()),'compiled_fraction95':np.quantile(fractions,[.025,.975]).tolist(),'top1_doc_energy_fraction':float(e[order[0]]/e.sum()),'top7_docs_energy_fraction':float(e[order[:7]].sum()/e.sum()),'energy_kish_count':float(e.sum()**2/(e*e).sum()),'drop_top7_paired_energy_ratio':float(e[keep].sum()/ea[keep].sum()),'documents_control_exceeds_actual':int((e>ea).sum()),'median_document_energy_ratio':float(np.median(e/ea))};draws[name+'_head6_document_components']=C
 assert replay<1e-12
 gates={'absolute_boost2':all(v['head6_energy_ratio']>=2 and v['head6_energy_ratio95'][0]>1 for v in summary.values()),'compiled_majority':all(v['compiled_fraction_of_excess']>=.5 and v['compiled_fraction95'][0]>.5 for v in summary.values()),'tail_robust2':all(v['drop_top7_paired_energy_ratio']>=2 for v in summary.values())};np.savez_compressed(STEM.with_name(STEM.name+'_document_components.npz'),actual=A,**draws);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'share_replay_error':replay,'gates':gates,'component_order':['compiled_squared','remaining_squared','twice_cross']},indent=2)+'\n');print('FINISHED',summary,gates)
if __name__=='__main__':main()
