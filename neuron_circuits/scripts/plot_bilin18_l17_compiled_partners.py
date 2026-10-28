#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ART=Path(__file__).resolve().parents[1]/'artifacts/bilin18_tensor_decomposition'
def main():
    p=json.loads((ART/'bilin18_l17_compiled_partners.json').read_text());f=json.loads((ART/'bilin18_l17_compiled_partners_factorization.json').read_text());b=json.loads((ART/'bilin18_l17_compiled_behavior.json').read_text());q=json.loads((ART/'bilin18_l17_compilation_error_factorial.json').read_text())
    fig,axs=plt.subplots(1,3,figsize=(15,4.6));x=np.arange(2)
    for i,(label,values,color) in enumerate([('Natural-write R²',[p['programs'][m]['natural_write_r2'] for m in ['4','16']],'#28649c'),('Actual response',[p['results'][f'intact_actual_s1_m{m}']['pooled'] for m in ['4','16']],'#56836a'),('Angle response',[p['results'][f'intact_angle_s1_m{m}']['pooled'] for m in ['4','16']],'#b87529')]):axs[0].bar(x+(i-1)*.24,values,.24,label=label,color=color)
    axs[0].set_xticks(x,['84 reads / 4 outputs','288 reads / 16 outputs']);axs[0].set_ylim(0,1);axs[0].set_ylabel('Local fidelity');axs[0].set_title('Fixed quadratic programs');axs[0].legend(fontsize=8)
    vals=[100*f[m]['storage_fraction'] for m in ['4','16']];axs[1].bar(x,vals,color=['#56836a','#28649c']);axs[1].set_xticks(x,['4 outputs','16 outputs']);axs[1].set_ylabel('% of native MLP scalar storage');axs[1].set_title('Factorized inference storage')
    for i,v in enumerate(vals):axs[1].text(i,v+.04,f'{v:.2f}%',ha='center')
    axs[1].set_ylim(0,4)
    vals=[q['results']['intact'][n]['mean_ce_delta'] for n in ['mcontext','moutput','mboth']];axs[2].bar(range(3),vals,color=['#b87529','#8b5999','#28649c']);axs[2].axhline(.02,ls='--',color='black',label='Predeclared loss threshold');axs[2].set_xticks(range(3),['Context\ntruncation','Output\ntruncation','Both\n(compiled)']);axs[2].set_ylabel('Next-token CE increase (nats)');axs[2].set_title('Behavior loss remains');axs[2].legend(fontsize=8);axs[2].set_ylim(0,.14)
    for ax in axs:ax.grid(axis='y',alpha=.2)
    fig.suptitle('A compact executable program preserves most local energy, but misses behavior-relevant detail')
    fig.text(.5,.012,'Same64 test documents; full RMS retained. Storage includes all inference coefficients. Local fidelity is not behavioral equivalence.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.07,1,.94])
    for ext in ['png','svg']:fig.savefig(ART/f'BILIN18_L17_COMPILED_PARTNERS.{ext}',dpi=170)
if __name__=='__main__':main()
