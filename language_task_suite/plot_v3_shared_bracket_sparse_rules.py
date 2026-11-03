"""Ground-truth, prediction, and exact sparse-weight-path graphs for shared v3."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_original_v3_shared_bracket import make_batch
from model import BilinearGPT, Config
from original_v3_shared_bracket import LanguageV3SharedBracket
from original_v3_shared_bracket import bracket_close_distribution
from plot_original_v3_sparse_rule_graphs import events, exact_edge_contrasts


def load(model, path, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(payload['model'])
    model.eval()
    return payload


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sparse-run', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=97901)
    p.add_argument('--streams', type=int, default=512)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    torch.set_num_threads(1)
    device = torch.device(a.device)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {'dense':a.sparse_run/'dense.pt',
             '40% zero':a.sparse_run/'pruned_40.pt',
             '60% zero + no L1':a.sparse_run/'final_no_l1.pt'}
    payload = torch.load(paths['dense'], map_location=device, weights_only=True)
    model = BilinearGPT(Config(**payload['config'])).to(device)
    gen = LanguageV3SharedBracket(seed=a.seed)
    x,y,labels,_ = make_batch(gen,a.streams,129,device=device)
    found = events(x,y,labels,gen,256)
    nouns,verbs = gen.categories['NOUN'],gen.categories['VERB_T']
    noun_ids = [gen.token2id[t] for t in nouns]
    verb_ids = [gen.token2id[t] for t in verbs]
    noun_truth = np.zeros((len(nouns),len(verbs)))
    for i,noun in enumerate(nouns):
        for verb,prob in gen.noun_bigram_dists[noun]:
            noun_truth[i,verbs.index(verb)] = prob
    noun_sites = [(b,t) for b,row in enumerate(labels) for t,lab in enumerate(row)
                  if lab=='noun_bigram' and int(x[b,t]) in noun_ids]
    noun_predictions = {}
    direct = {}
    skip_outputs = list(dict.fromkeys(list(gen.skip_bigrams.values())+[')',']']))
    output_ids = [gen.token2id[t] for t in skip_outputs]
    specs = []
    for (anchor,trigger),target in gen.skip_bigrams.items():
        records = [r for r in found['skip_bigram']
                   if r[5]==anchor and r[6]==trigger]
        specs.append((f'{anchor} … {trigger}',target,records))
    for opener,target in [('(',')'),('[',']')]:
        specs.append((f'{opener} … close',target,
                      [r for r in found['skip_trigram'] if r[5]==opener]))
    skip_predictions = {}
    edge_means = {}
    checks = {}
    mask_stats = {}
    for name,path in paths.items():
        checkpoint = load(model,path,device)
        probs = model(x)[0].softmax(-1).cpu().numpy()
        noun = np.full_like(noun_truth,np.nan)
        for i,token_id in enumerate(noun_ids):
            sites = [(b,t) for b,t in noun_sites if int(x[b,t])==token_id]
            if sites:
                noun[i] = np.mean([probs[b,t,verb_ids] for b,t in sites],axis=0)
        noun_predictions[name] = noun
        direct[name] = (model.embed.weight[noun_ids] @
                        model.lm_head.weight[verb_ids].T).softmax(-1).cpu().numpy()
        skip = np.full((len(specs),len(skip_outputs)),np.nan)
        for i,(_,_,records) in enumerate(specs):
            if records:
                skip[i] = np.mean([probs[b,q,output_ids] for b,q,*_ in records],axis=0)
        skip_predictions[name] = skip
        paths_data,checks[name] = exact_edge_contrasts(model,x,found)
        edge_means[name] = {rule:np.stack(paths_data[rule],axis=1).mean(0)
                            for rule in ('skip_bigram','skip_trigram')}
        mask = checkpoint['mask']
        total = sum(m.numel() for m in mask.values())
        live = sum(int(m.sum()) for m in mask.values())
        mask_stats[name] = dict(total=total,live=live,zero_fraction=1-live/total)
        for parameter_name,m in mask.items():
            parameter = dict(model.named_parameters())[parameter_name]
            assert torch.count_nonzero(parameter[~m])==0

    fig,axes = plt.subplots(2,4,figsize=(15,7),layout='constrained')
    for ax,title,mat in zip(axes[0],['Planted P(verb | noun)']+list(paths),
                            [noun_truth]+list(noun_predictions.values())):
        im=ax.imshow(mat,vmin=0,vmax=1,cmap='viridis')
        ax.set_title(title)
        ax.set_xticks(range(len(verbs)),verbs,rotation=70,fontsize=7)
        ax.set_yticks(range(len(nouns)),nouns,fontsize=7)
    axes[1,0].axis('off')
    axes[1,0].text(.02,.9,'Bottom: E → unembedding direct path\n'
                   '(conditional softmax over verbs).\n'
                   'These two matrices remain dense.',va='top',fontsize=10)
    for ax,(name,mat) in zip(axes[1,1:],direct.items()):
        ax.imshow(mat,vmin=0,vmax=1,cmap='viridis')
        ax.set_title(name+' direct path')
        ax.set_xticks(range(len(verbs)),verbs,rotation=70,fontsize=7)
        ax.set_yticks(range(len(nouns)),nouns,fontsize=7)
    fig.colorbar(im,ax=axes[0],fraction=.018,pad=.01,label='Probability')
    fig.savefig(a.output_dir/'noun_bigram_truth_predictions_weights.png',dpi=180)
    plt.close(fig)

    truth = np.zeros((len(specs),len(skip_outputs)))
    for i,(_,target,records) in enumerate(specs):
        if target in (')',']'):
            truth[i,skip_outputs.index(target)] = np.mean([
                next(iter(bracket_close_distribution(
                    '(' if target==')' else '[',q-k).values()))
                for _,q,k,*_ in records])
        else:
            truth[i,skip_outputs.index(target)] = 1
    fig,axes = plt.subplots(1,4,figsize=(16,5),layout='constrained')
    for ax,title,mat in zip(axes,['Planted target']+list(paths),
                            [truth]+list(skip_predictions.values())):
        im=ax.imshow(mat,vmin=0,vmax=1,cmap='viridis',aspect='auto')
        ax.set_title(title)
        ax.set_xticks(range(len(skip_outputs)),skip_outputs)
        ax.set_yticks(range(len(specs)),
                      [f'{label} (n={len(records)})' for label,_,records in specs],fontsize=8)
    fig.colorbar(im,ax=axes,fraction=.018,pad=.01,label='Full-vocabulary probability')
    fig.suptitle('Fresh sampled rule events; planted bracket mass averages the full oracle over those prefixes')
    fig.savefig(a.output_dir/'skip_rules_truth_predictions.png',dpi=180)
    plt.close(fig)

    fig,axes = plt.subplots(2,3,figsize=(13,6),layout='constrained')
    for r,rule in enumerate(('skip_bigram','skip_trigram')):
        vmax = max(np.abs(edge_means[name][rule]).max() for name in paths)
        for c,name in enumerate(paths):
            ax=axes[r,c]
            im=ax.imshow(edge_means[name][rule],vmin=-vmax,vmax=vmax,
                         cmap='coolwarm',aspect='auto')
            ax.set_title(name)
            ax.set_xticks(range(4),[f'H{i}' for i in range(4)])
            ax.set_yticks(range(2),[f'L{i}' for i in range(2)])
            if c==0:ax.set_ylabel(rule+'\nLayer')
        fig.colorbar(im,ax=axes[r],fraction=.02,pad=.015,
                     label='Logit contrast')
    fig.suptitle('Exact QK₁×QK₂×OV→unembedding path through selected earlier token')
    fig.savefig(a.output_dir/'sparse_attention_edge_paths.png',dpi=180)
    plt.close(fig)

    training = json.loads((a.sparse_run/'results.json').read_text())
    stage_names = ['dense','pruned_40','pruned_60','final_no_l1']
    rule_names = ['noun_bigram','skip_bigram','skip_trigram','induction']
    stage_rows = {row['stage']:row for row in training['rows']}
    values = np.asarray([[stage_rows[stage]['per_rule_ce'][rule]
                          for stage in stage_names] for rule in rule_names])
    fig,ax = plt.subplots(figsize=(9,4.6),layout='constrained')
    positions = np.arange(len(rule_names))
    for i,stage in enumerate(stage_names):
        ax.bar(positions+(i-1.5)*.19,values[:,i],.18,
               label=stage.replace('_',' '))
    ax.set_xticks(positions,['Noun bigram','Skip bigram','Bracket close','Induction'])
    ax.set_ylabel('Fresh-stream CE (nats; lower is better)')
    ax.set_title('Attention pruning, then final fine-tune without L1')
    ax.grid(axis='y',alpha=.2)
    ax.legend(frameon=False,ncol=4,fontsize=8)
    fig.savefig(a.output_dir/'rule_competence.png',dpi=180)
    plt.close(fig)

    summary=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        mask_stats=mask_stats, noun_event_count=len(noun_sites),
        event_count={rule:len(rows) for rule,rows in found.items()},
        edge_reconstruction_max_error=checks,
        skip_rows=[dict(label=label,target=target,n=len(records))
                   for label,target,records in specs],
        skip_outputs=skip_outputs,
        edge_means={name:{rule:matrix.tolist() for rule,matrix in by_rule.items()}
                    for name,by_rule in edge_means.items()},
        caveat='Sparse masks cover attention projection matrices only. Noun '
               'direct path uses dense embedding/unembedding matrices. '
               'Selected-edge contributions are exact local paths, not '
               'end-to-end causal effects. Bracket events condition on observed '
               'closure, but the planted probability is the mean oracle mass '
               'given each visible prefix, rather than a unit event label.')
    (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('mask',mask_stats,'events',summary['event_count'],
          'replay',checks,flush=True)


if __name__=='__main__':
    main()
