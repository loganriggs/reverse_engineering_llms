#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ART=Path(__file__).resolve().parents[1]/'artifacts/bilin18_tensor_decomposition'
def main():
    a=json.loads((ART/'bilin18_l17_residual_restoration.json').read_text());b=json.loads((ART/'bilin18_l17_residual_scaling.json').read_text());c=json.loads((ART/'bilin18_l17_residual_replication.json').read_text());fig,axs=plt.subplots(1,3,figsize=(15,4.6))
    ranks=[16,128,256]
    for family,color,label in [('energy','#28649c','Energy basis'),('fisher64_r','#b87529','Downstream-sensitive basis')]:
        vals=[c['results'][f'{family}{r}']['kl_recovery'] for r in ranks];axs[0].plot(range(3),vals,'o-',label=label,color=color)
        vals=[c['results'][f'{family}{r}']['native_top1_agreement'] for r in ranks];axs[1].plot(range(3),vals,'o-',label=label,color=color)
    for ax in axs[:2]:ax.set_xticks(range(3),ranks);ax.set_xlabel('Oracle restoration rank');ax.grid(alpha=.2)
    axs[0].set_ylim(0,1);axs[0].set_ylabel('Fraction of compiled KL error removed');axs[0].set_title('New 128-document test');axs[0].legend(fontsize=8)
    axs[1].axhline(.98,ls='--',color='black',label='Acceptance threshold');axs[1].set_ylim(.8,1);axs[1].set_ylabel('Agreement with native top prediction');axs[1].set_title('Even broad restoration is incomplete');axs[1].legend(fontsize=8)
    for r in [16,128]:
        vals=[b['results'][n]['residual_energy_recovery'] for n in [f'energy64docs_r{r}',f'energy256docs_r{r}',f'energy{r}']];axs[2].plot(range(3),vals,'o-',label=f'Energy rank{r}')
    axs[2].set_xticks(range(3),[64,256,1024]);axs[2].set_xlabel('Distinct discovery documents');axs[2].set_ylabel('Test residual-energy recovery');axs[2].set_title('Discovery scaling on original test');axs[2].set_ylim(0,1);axs[2].grid(alpha=.2);axs[2].legend(fontsize=8)
    fig.suptitle('Behavior-sensitive residuals transfer, but the tested low-rank repairs remain incomplete')
    fig.text(.5,.012,'All coefficients are ORACLE native residual reads; this is a diagnostic, not a forward solution. Compiled model stays frozen.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.07,1,.94])
    for ext in ['png','svg']:fig.savefig(ART/f'BILIN18_L17_RESIDUAL_RESTORATION.{ext}',dpi=170)
    lines=['# Oracle residual restoration and discovery scaling','','New 128-document test, all maps frozen from1024 discovery documents.','','| Restoration | Rank | KL error recovered | Mean CE increase | Native top1 agreement | Residual energy recovered |','|---|---:|---:|---:|---:|---:|']
    for fam,label in [('energy','Energy'),('fisher64_r','Downstream sensitive')]:
        for r in ranks:
            v=c['results'][f'{fam}{r}'];lines.append(f"| {label} | {r} | {v['kl_recovery']:.3f} | {v['mean_ce_delta']:.4f} | {v['native_top1_agreement']:.3f} | {v['residual_energy_recovery']:.3f} |")
    lines+=['',f"Replication gates: {c['gates']}.",'',c['config']['scope']];(ART/'BILIN18_L17_RESIDUAL_RESTORATION.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
