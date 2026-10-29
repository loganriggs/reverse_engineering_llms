"""M110 finite-limit geometry and all first-crossing downstream effects."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fit_pca_encoder import V

r=json.loads((V/'gate_boundary_native.json').read_text());a=json.loads((V/'gate_boundary_analysis.json').read_text())
rows=[x for x in r['rows'] if x['status']=='evaluated']
x=np.array([.1,.01,.001]);native=np.array([[y['native_change_norm']/z['native_path_norm'] for y in z['finite_limits']] for z in rows]);decoded=np.array([[y['decoded_change_norm']/z['native_path_norm'] for y in z['finite_limits']] for z in rows])
fig,ax=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
for values,label,color in [(native,'Underlying write change','#2878b5'),(decoded,'Decoded sparse change','#d85f20')]:
    q=np.quantile(values,[.25,.5,.75],axis=0);ax[0].plot(x,q[1],marker='o',label=label,color=color);ax[0].fill_between(x,q[0],q[2],alpha=.15,color=color)
ax[0].set(xscale='log',yscale='log',xlabel='Distance fraction around boundary (smaller →)',ylabel='Change norm / full endpoint-change norm',title='Finite decoded jumps on continuous paths');ax[0].invert_xaxis();ax[0].legend(fontsize=9)
families=sorted({z['family'] for z in a['rows']})
for i,f in enumerate(families):
    values=[z['KL_left_right'] for z in a['rows'] if z['family']==f];jitter=np.linspace(-.18,.18,len(values));ax[1].scatter(i+jitter,values,s=18,alpha=.7)
ax[1].set(yscale='log',ylabel='KL(left-limit output || right-limit output)',title='117 first-crossing limits; no top1 changes');ax[1].set_xticks(range(len(families)),[f.replace('_','\n') for f in families],fontsize=8)
fig.suptitle('Hard selection can introduce structure absent from the underlying write path',fontsize=12)
for ext in ['png','svg']:fig.savefig(V.parent/'figures'/('gate_boundary_limits.'+ext),dpi=180)
