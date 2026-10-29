"""M114 all-case finite-limit behavior, no selected example."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fit_pca_encoder import V
r=json.loads((V/'native_path_boundaries.json').read_text());rows=[x for x in r['rows'] if x['status']=='evaluated'];x=np.array([.1,.01,.001])
fig,ax=plt.subplots(figsize=(7.5,4.8),layout='constrained')
for key,label,color in [('native_change_norm','Native bilinear write','#2878b5'),('hard_change_norm','Hard-block reconstruction','#d85f20'),('continuous_change_norm','Continuous map, same weights','#329650')]:
    values=np.array([[y[key]/row['native_endpoint_change_norm'] for y in row['finite_limits']] for row in rows]);q=np.quantile(values,[.25,.5,.75],axis=0);ax.plot(x,q[1],marker='o',label=label,color=color);ax.fill_between(x,q[0],q[2],color=color,alpha=.13)
ax.set(xscale='log',yscale='log',xlabel='Fraction of safe boundary radius (smaller →)',ylabel='Change norm / full native endpoint change',title='Extraction jumps persist on native MLP input paths');ax.invert_xaxis();ax.legend(fontsize=9);fig.suptitle('16 fixed input paths × 2 hard-trained encoders; median and interquartile range',fontsize=10)
for ext in ['png','svg']:fig.savefig(V.parent/'figures'/('native_path_boundary_limits.'+ext),dpi=180)
