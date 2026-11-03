"""Plot planted v3 rules, held-out competence, and exact attention-edge weight paths."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from sparsify_original_v3_for_rule_graphs import make_data, model_and_generator


def load_model(model, path, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(payload['model'])
    model.eval()
    return model


def events(x, y, labels, gen, max_per_rule=256):
    """Find realized rule events and the relevant earlier token in the same stream."""
    found = {'skip_bigram': [], 'skip_trigram': []}
    words = gen.id2token
    skip = gen.skip_bigrams
    for b, row in enumerate(labels):
        for q, label in enumerate(row):
            if label not in found or len(found[label]) >= max_per_rule:
                continue
            target = words[int(y[b, q])]
            if label == 'skip_bigram':
                trigger = words[int(x[b, q])]
                matches = [(k, anchor) for k in range(max(0, q-8), q)
                           for anchor in [words[int(x[b, k])]]
                           if skip.get((anchor, trigger)) == target]
                if len(matches) != 1:
                    continue
                k, opener = matches[0]
                alternatives = [gen.token2id[t] for t in sorted(set(skip.values()))
                                if t != target]
            else:
                opener_token = '(' if target == ')' else '['
                matches = [k for k in range(q-1, -1, -1)
                           if words[int(x[b, k])] in ('(', '[', ')', ']')]
                if not matches or words[int(x[b, matches[0]])] != opener_token:
                    continue
                k, opener = matches[0], opener_token
                alternatives = [gen.token2id[']' if target == ')' else ')']]
            found[label].append((b, q, k, int(y[b, q]), alternatives,
                                 opener, words[int(x[b, q])]))
    return found


@torch.no_grad()
def exact_edge_contrasts(model, x, found):
    """Single (query,key,head) contribution through QK1*QK2*OV*unembedding.

    Layer-0 contributions are direct residual readout, not total downstream effects.
    """
    from model import apply_rotary_emb
    output = {kind: [] for kind in found}
    residual = model.embed(x)
    unembed = model.lm_head.weight
    checks = []
    for layer, block in enumerate(model.blocks):
        attn = block.attn
        B, T, C = residual.shape
        H, D = attn.n_head, attn.head_dim
        def shaped(linear):
            return linear(residual).reshape(B, T, H, D)
        q1, k1, q2, k2, v = [shaped(w) for w in
                              (attn.q1, attn.k1, attn.q2, attn.k2, attn.v)]
        cos, sin = attn.rotary(q1)
        q1 = apply_rotary_emb(q1, cos, sin)
        k1 = apply_rotary_emb(k1, cos, sin)
        q2 = apply_rotary_emb(q2, cos, sin)
        k2 = apply_rotary_emb(k2, cos, sin)
        assert not attn.qk_norm
        # Verify the selected query's sum of individual edges reconstructs
        # the actual attention contribution before using the edge attribution.
        b0, q0 = 0, min(20, T-1)
        s1 = torch.einsum('hd,khd->hk', q1[b0,q0], k1[b0,:q0+1]) / D
        s2 = torch.einsum('hd,khd->hk', q2[b0,q0], k2[b0,:q0+1]) / D
        reconstructed = attn.out.weight @ torch.einsum(
            'hk,khd->hd', s1*s2, v[b0,:q0+1]).reshape(C)
        actual = attn(residual[b0:b0+1])[:,q0,:].squeeze(0)
        checks.append(float((reconstructed-actual).abs().max()))
        out_parts = attn.out.weight.reshape(C,H,D).permute(1,2,0)
        for kind, records in found.items():
            rows = []
            for b,q,k,target,alts,*_ in records:
                s1 = (q1[b,q]*k1[b,k]).sum(-1)/D
                s2 = (q2[b,q]*k2[b,k]).sum(-1)/D
                vecs = torch.einsum('hd,hdc->hc', v[b,k], out_parts)*(s1*s2)[:,None]
                diff = unembed[target]-unembed[alts].mean(0)
                rows.append((vecs @ diff).cpu().numpy())
            output[kind].append(np.asarray(rows))
        residual = block(residual)
    assert max(checks) < 1e-3, checks
    return output, checks


def plot_competence(paths, out):
    names = ['Dense', '40% zero', '60% zero', '85% zero']
    source = [paths['dense'], paths['s40'], paths['s60'], paths['s85']]
    rules = [('noun_bigram','Noun bigram'),('skip_bigram','Skip-bigram'),
             ('skip_trigram','Bracket close (skip-trigram)')]
    vals = np.array([[json.loads((p.parent/'results.json').read_text())['rows'][-1]
                      ['test']['rule_ce'][rule] for p in source]
                     for rule,_ in rules])
    # Dense file appears in a sparse run where final row differs.
    vals[:,0] = [json.loads((paths['s40'].parent/'results.json').read_text())
                 ['rows'][0]['test']['rule_ce'][r] for r,_ in rules]
    fig, ax = plt.subplots(figsize=(8,4.7))
    xx = np.arange(len(rules))
    for j,name in enumerate(names):
        ax.bar(xx+(j-1.5)*.19, vals[:,j], .18, label=name)
    ax.set_xticks(xx, [x[1] for x in rules])
    ax.set_ylabel('Held-out cross-entropy (nats; lower is better)')
    ax.set_title('Rule competence after iterative L1, pruning, and final no-L1 recovery')
    ax.legend(frameon=False, ncol=4, fontsize=8)
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(out/'rule_competence.png', dpi=180)
    plt.close(fig)
    return vals.tolist()


@torch.no_grad()
def plot_noun(paths, model, gen, x, labels, out, device):
    nouns, verbs = gen.categories['NOUN'], gen.categories['VERB_T']
    ni = [gen.token2id[t] for t in nouns]
    vi = [gen.token2id[t] for t in verbs]
    truth = np.zeros((len(nouns),len(verbs)))
    for i,noun in enumerate(nouns):
        for verb,p in gen.noun_bigram_dists[noun]:
            truth[i,verbs.index(verb)] = p
    fig, axes = plt.subplots(2,4,figsize=(12,6.6),layout='constrained')
    im = axes[0,0].imshow(truth,vmin=0,vmax=1,cmap='viridis')
    axes[0,0].set_title('Planted conditional distribution')
    axes[1,0].axis('off')
    axes[1,0].text(.02,.86,'Top: full-model P(verb | sampled noun context)\n'
               'Bottom: direct embedding → unembedding path\n'
               '(conditional softmax over transitive verbs).\n'
               'The direct path is only one circuit, not the full model.',va='top',fontsize=9)
    positions = [(b,t) for b,row in enumerate(labels) for t,lab in enumerate(row)
                 if lab=='noun_bigram' and int(x[b,t]) in ni]
    for col,key in enumerate(('dense','s40','s60'),1):
        load_model(model,paths[key],device)
        logits=model(x)[0]
        full=np.full_like(truth,np.nan)
        probs=logits.softmax(-1).cpu().numpy()
        for i,nid in enumerate(ni):
            pairs=[(b,t) for b,t in positions if int(x[b,t])==nid]
            if pairs:
                full[i]=np.mean([probs[b,t,vi] for b,t in pairs],axis=0)
        E=model.embed.weight.detach(); U=model.lm_head.weight.detach()
        direct=(E[ni] @ U[vi].T).softmax(-1).cpu().numpy()
        axes[0,col].imshow(full,vmin=0,vmax=1,cmap='viridis')
        axes[1,col].imshow(direct,vmin=0,vmax=1,cmap='viridis')
        axes[0,col].set_title(f'{key}: full model')
        axes[1,col].set_title(f'{key}: direct weight path')
    for ax in axes.ravel():
        if ax is axes[1,0]: continue
        ax.set_xticks(range(len(verbs)),verbs,rotation=70,fontsize=8)
        ax.set_yticks(range(len(nouns)),nouns,fontsize=8)
    fig.colorbar(im,ax=axes[0,:],fraction=.02,pad=.02,label='Probability')
    fig.savefig(out/'noun_bigram_truth_predictions_weights.png',dpi=180)
    plt.close(fig)
    return {'noun_examples':len(positions),'per_noun':{n:sum(int(x[b,t])==nid
              for b,t in positions) for n,nid in zip(nouns,ni)}}


def plot_edges(paths,model,x,found,out,device):
    arrays={}
    checks={}
    for key in ('dense','s40','s60','s85'):
        load_model(model,paths[key],device)
        arrays[key],checks[key]=exact_edge_contrasts(model,x,found)
    fig, axes = plt.subplots(2,4,figsize=(14,6),layout='constrained',sharey='row')
    kinds=['skip_bigram','skip_trigram']
    for r,kind in enumerate(kinds):
        for c,key in enumerate(('dense','s40','s60','s85')):
            ax=axes[r,c]
            data=np.stack(arrays[key][kind],axis=1) # event, layer, head
            mean=data.mean(0)
            vmax=max(1e-5,np.quantile(np.abs(mean),.98))
            im=ax.imshow(mean,aspect='auto',cmap='coolwarm',vmin=-vmax,vmax=vmax)
            ax.set_xticks(range(mean.shape[1]),[f'H{i}' for i in range(mean.shape[1])])
            ax.set_yticks(range(mean.shape[0]),[f'L{i}' for i in range(mean.shape[0])])
            ax.set_title(f'{key} ({len(data)} events)')
            if c==0:
                ax.set_ylabel(('Place…adjective → function word' if r==0
                               else 'Opener…content → correct closer')+'\nLayer')
            fig.colorbar(im,ax=ax,fraction=.045,pad=.02)
    fig.suptitle('Exact selected attention-edge QK₁×QK₂×OV→logit contrast',fontsize=13)
    fig.savefig(out/'skip_rule_sparse_weight_paths.png',dpi=180)
    plt.close(fig)
    # Also show the rule identities and event counts, to make planted mapping readable.
    return {'edge_reconstruction_max_error':checks,
            'n_events':{k:len(v) for k,v in found.items()},
            'edge_means':{k:{rule:np.stack(arrays[k][rule],axis=1).mean(0).tolist()
                             for rule in kinds} for k in arrays}}


@torch.no_grad()
def plot_skip_predictions(paths,model,gen,x,found,out,device):
    specs=[]
    for (anchor,trigger),target in gen.skip_bigrams.items():
        specs.append((f'{anchor} … {trigger}',target,
                     [v for v in found['skip_bigram']
                      if v[5]==anchor and v[6]==trigger]))
    for opener,target in [('(',')'),('[',']')]:
        specs.append((f'{opener} …',target,
                     [v for v in found['skip_trigram'] if v[5]==opener]))
    outputs=list(dict.fromkeys(list(gen.skip_bigrams.values())+[')',']']))
    oid=[gen.token2id[t] for t in outputs]
    truth=np.zeros((len(specs),len(outputs)))
    for i,(_,target,_) in enumerate(specs):
        truth[i,outputs.index(target)]=1
    matrices=[truth]
    for key in ('dense','s40','s60'):
        load_model(model,paths[key],device)
        probs=model(x)[0].softmax(-1).cpu().numpy()
        table=np.full_like(truth,np.nan)
        for i,(_,_,records) in enumerate(specs):
            if records:
                table[i]=np.mean([probs[b,q,oid] for b,q,*_ in records],axis=0)
        matrices.append(table)
    fig,axes=plt.subplots(1,4,figsize=(13,4.7),layout='constrained')
    for ax,title,mat in zip(axes,('Planted target','Dense','40% zero','60% zero'),matrices):
        im=ax.imshow(mat,vmin=0,vmax=1,cmap='viridis',aspect='auto')
        ax.set_title(title)
        ax.set_xticks(range(len(outputs)),outputs)
        ax.set_yticks(range(len(specs)),
                      [f'{name} (n={len(records)})' for name,_,records in specs],fontsize=8)
    fig.colorbar(im,ax=axes,fraction=.025,pad=.02,label='Full-vocabulary probability')
    fig.suptitle('Skip-rule target probabilities on fresh jointly sampled streams')
    fig.savefig(out/'skip_rule_truth_full_model_predictions.png',dpi=180)
    plt.close(fig)
    return {'rows':[s[0] for s in specs], 'n':[len(s[2]) for s in specs],
            'columns':outputs,'probabilities':
            {k:m.tolist() for k,m in zip(('truth','dense','s40','s60'),matrices)}}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',default='/home/loganriggs/Coding/reverse_engineering_llms')
    p.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--streams',type=int,default=256)
    p.add_argument('--length',type=int,default=129)
    p.add_argument('--seed',type=int,default=95001)
    a=p.parse_args()
    torch.set_num_threads(1)
    root=Path(a.repo)/'language_task_suite'/'results'
    paths={
       'dense':root/'original_v3_sparse40_recovery_seed94001'/'dense.pt',
       's40':root/'original_v3_sparse40_recovery_seed94001'/'final_no_l1.pt',
       's60':root/'original_v3_sparse60_recovery_seed94001'/'final_no_l1.pt',
       's85':root/'original_v3_sparse_rule_graphs_seed94001'/'final_no_l1.pt'}
    out=root/'original_v3_sparse_rule_graphs'
    out.mkdir(exist_ok=True)
    device=torch.device(a.device)
    model,Generator,_=model_and_generator(Path(a.source),device)
    gen=Generator(seed=a.seed)
    x,y,labels,_=make_data(gen,a.streams,a.length,device)
    found=events(x,y,labels,gen)
    competence=plot_competence(paths,out)
    noun=plot_noun(paths,model,gen,x,labels,out,device)
    edge=plot_edges(paths,model,x,found,out,device)
    skip_predictions=plot_skip_predictions(paths,model,gen,x,found,out,device)
    summary={'config':vars(a),'competence_rows':
             ['noun_bigram','skip_bigram','skip_trigram'],
             'competence_columns':['dense','40pct_zero','60pct_zero','85pct_zero'],
             'heldout_ce':competence,'noun':noun,'edge':edge,
             'skip_predictions':skip_predictions,
             'planted_skip_bigrams':{f'{a} ... {b}':c for (a,b),c in gen.skip_bigrams.items()},
             'caveat':'Attention edge path is exact for a selected edge and layer input; layer-0 direct logit contribution is not total downstream causal effect. Sparse masks apply only to attention matrices. Full-model noun probabilities are averaged over sampled contexts.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('heldout_ce','noun','planted_skip_bigrams')},indent=2))
    print('events',edge['n_events'],'checks',edge['edge_reconstruction_max_error'])

if __name__=='__main__': main()
