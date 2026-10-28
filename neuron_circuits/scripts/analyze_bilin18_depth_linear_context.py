#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ART=Path(__file__).resolve().parents[1]/'artifacts/bilin18_tensor_decomposition'
def main():
    a=json.loads((ART/'bilin18_depth_linear_context.json').read_text());b=json.loads((ART/'bilin18_depth_compact_input.json').read_text());ar=np.load(ART/'bilin18_depth_linear_context_rows.npz');br=np.load(ART/'bilin18_depth_compact_input_rows.npz');ix=np.random.default_rng(20261115).integers(64,size=(2000,64));contrasts={}
    for l in [1,8,17]:
        prefix=f'L{l}_intact_actual_s1';energy=ar[prefix+'_energy'];assert np.allclose(energy,br[prefix+'_energy'],rtol=1e-6)
        for n in ['strict16','both_full_linear16']:
            difference=br[prefix+'_'+n+'_error']-ar[prefix+'_cross16_error'];contrasts[f'L{l}_cross16_minus_{n}']={'fidelity_gain':float(difference.sum()/energy.sum()),'paired_document_bootstrap95':np.quantile(difference[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist()}
    (ART/'bilin18_depth_context_partner_contrasts.json').write_text(json.dumps(contrasts,indent=2))
    fig,axs=plt.subplots(1,3,figsize=(15,4.8));methods=[('mean_linear','Fixed mean',a),('strict16','16 input\ncoordinates',b),('both_full_linear16','16 coordinates +\nfull linear term',b),('cross16','16-context ×\nbroad partners',a)]
    for i,(fam,color,label) in enumerate([('actual','#28649c','Actual source'),('angle','#b87529','Angle control')]):
        vals=[p['results'][f'L17_intact_{fam}_s1_{n}']['pooled'] for n,_,p in methods];axs[0].bar(np.arange(4)+(i-.5)*.35,vals,.35,color=color,label=label)
    axs[0].set_xticks(range(4),[v[1] for v in methods],rotation=15);axs[0].set_title('L17 finite source removal');axs[0].set_ylabel('Response fidelity (1 − SSE / energy)');axs[0].legend(fontsize=9)
    for n,label,p in methods:
        vals=[p['natural_r2'][str(l)][n] for l in [1,8,17]];axs[1].plot(range(3),vals,'o-',label=label.replace('\n',' '))
    axs[1].set_xticks(range(3),['L1','L8','L17']);axs[1].set_title('Natural-state forward prediction');axs[1].set_ylabel('Natural-write R²');axs[1].legend(fontsize=8)
    for n,label,p in [('strict16','16 input coordinates',b),('cross16','16-context × broad partners',a)]:
        vals=[p['results'][f'L{l}_intact_actual_s1_{n}']['pooled'] for l in [1,8,17]];axs[2].plot(range(3),vals,'o-',label=label)
    axs[2].set_xticks(range(3),['L1','L8','L17']);axs[2].set_title('Actual response prediction across depth');axs[2].set_ylabel('Response fidelity');axs[2].legend(fontsize=8)
    for ax in axs:ax.grid(axis='y',alpha=.2);ax.set_ylim(-.08,1.02)
    fig.suptitle('A small context subspace still needs interactions with other input directions')
    fig.text(.5,.015,'Discovery-frozen PCA; 64 test documents. Exact full RMS normalization retained. Cross rules are not16-total-read models.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.07,1,.94])
    for ext in ['png','svg']:fig.savefig(ART/f'BILIN18_DEPTH_LINEAR_CONTEXT.{ext}',dpi=170)
    print(json.dumps(contrasts,indent=2))
if __name__=='__main__':main()
