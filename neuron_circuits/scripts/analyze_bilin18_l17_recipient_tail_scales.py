from pathlib import Path
import numpy as np,json
A=Path(__file__).resolve().parents[1]/'artifacts/bilin18_tensor_decomposition';z=np.load(A/'bilin18_l17_recipient_tail_diagnostic_rows.npz');result={}
def ranks(v):
 _,inv,cnt=np.unique(v,return_inverse=True,return_counts=True);return (np.cumsum(cnt)-(cnt+1)/2)[inv]
for label in ['current','previous']:
 stats=z[label+'_stats'];q=z[label+'_quartic'];si=797 if label=='current' else int(q[1].argmax());inp=np.sqrt(stats[:,0]**2+np.finfo(np.float32).eps);out=np.sqrt(stats[:,1]**2+np.finfo(np.float32).eps);scales={'native':np.ones(len(inp)),'input_scalar_equalized':(inp/np.median(inp))**4,'output_scalar_equalized':(out/np.median(out))**2,'both_scalar_equalized':(inp/np.median(inp))**4*(out/np.median(out))**2};cells={}
 for name,sc in scales.items():
  v=q*sc;cells[name]={'selected_site_share':(v[:,si]/v.sum(1)).tolist(),'maximum_site_share':(v.max(1)/v.sum(1)).tolist(),'maximum_document_share':(v.reshape(2,-1,4).sum(2).max(1)/v.sum(1)).tolist()}
 result[label]={'selected_site':si,'selected_scale_multiplier':float(scales['both_scalar_equalized'][si]),'cells':cells,'rank_correlation_quartic_vs_inverse_input4_output2':[float(np.corrcoef(ranks(v),ranks(1/(inp**4*out**2)))[0,1]) for v in q]}
j={'scope':'Post-selection scalar-gain diagnostic, not native forward intervention. Hold each RMS derivative radial projector and all directions fixed; replace input and output inverse scalar RMS gains by panel medians. Quartic energy multiplier is (input_scale/median)^4*(output_scale/median)^2. Native float32 epsilon retained. No exclusions, significance claims, or changes to H429 gates.','panels':result};(A/'bilin18_l17_recipient_tail_scale_diagnostic.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(j,indent=2))
