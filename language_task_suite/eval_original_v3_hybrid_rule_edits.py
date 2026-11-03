"""Test native and reduced v3 models on supported lexical rule edits.

Skip edits switch between two simultaneously available place...adjective
rules. Noun edits replace a novel noun with another novel noun. The bracket
control changes only its last content token within the same category, leaving
the oracle closer fixed. V3 cannot support an opener-only bracket swap because
the two bracket types have disjoint content vocabularies.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from eval_original_v3_multirule_bond_bases import activation_grams, bases_from_grams
from sparsify_original_v3_for_rule_graphs import model_and_generator


def distribution(gen, noun):
    return {token:float(p) for token,p in gen.noun_bigram_dists[noun]}


def normal_next_distribution(gen, rule_distribution):
    """Exact next-token mixture when the sampled prefix is in normal state."""
    paren=gen.paren_prob if gen.use_brackets else 0.
    quote=gen.quote_prob if gen.use_brackets else 0.
    normal=1-paren-quote
    result={token:normal*p for token,p in rule_distribution.items()}
    if paren:result['(']=paren
    if quote:result['[']=quote
    assert abs(sum(result.values())-1)<1e-9
    return result


def choose_pairs(gen, nstreams, length, limit):
    pairs = defaultdict(list)
    rules = gen.skip_bigrams
    by_trigger = {trigger:(anchor,target) for (anchor,trigger),target in rules.items()}
    adj = set(gen.categories['ADJ'])
    nouns = gen.categories['NOUN']
    for _ in range(nstreams):
        ids, labels = gen.sample_sequence(length)
        words = [gen.id2token[i] for i in ids]
        for t in range(2,length):
            q=t-1
            if labels[t]=='skip_bigram' and labels[q]=='cat_bigram' and len(pairs['skip_switch'])<limit:
                # The adjective remains supported after substitution by the
                # same ordinary category transition, independent of boosts.
                prev_cat=gen.token2cat.get(words[q-1], 'PUNCT')
                row=gen.cat_index.get(prev_cat, gen.cat_index['PUNCT'])
                if gen.transition_matrix[row,gen.cat_index['ADJ']]<=0:
                    continue
                current=words[q]
                if current not in by_trigger:
                    continue
                old_anchor,old_target=by_trigger[current]
                if old_target!=words[t]:
                    continue
                window=words[max(0,q-8):q]
                if old_anchor not in window:
                    continue
                for trigger,(anchor,new_target) in by_trigger.items():
                    if trigger==current or new_target==old_target or anchor not in window:
                        continue
                    changed=words[:t].copy();changed[q]=trigger
                    sampled,label=gen._sample_normal_token(changed,{})
                    if label!='skip_bigram' or sampled!=new_target:
                        continue
                    edited=ids[:-1].copy();edited[q]=gen.token2id[trigger]
                    pairs['skip_switch'].append(dict(original=ids[:-1],edited=edited,
                        query=q,
                        old=normal_next_distribution(gen,{old_target:1.}),
                        new=normal_next_distribution(gen,{new_target:1.}),
                        detail=f'{old_anchor}...{current}->{old_target} | '
                               f'{anchor}...{trigger}->{new_target}'))
                    break
            elif labels[t]=='noun_bigram' and labels[q]=='cat_bigram' and len(pairs['noun_switch'])<limit:
                current=words[q]
                if current not in nouns or words[q-1]=='the':
                    continue
                prev_cat=gen.token2cat.get(words[q-1], 'PUNCT')
                row=gen.cat_index.get(prev_cat, gen.cat_index['PUNCT'])
                if gen.transition_matrix[row,gen.cat_index['NOUN']]<=0:
                    continue
                options=[noun for noun in nouns if noun!=current and noun not in words[:q]]
                if not options:
                    continue
                old=distribution(gen,current)
                alt=max(options,key=lambda noun:sum(abs(old.get(tok,0)-
                    distribution(gen,noun).get(tok,0)) for tok in gen.categories['VERB_T']))
                # A new noun cannot trigger induction. The sampled original
                # was a noun-bigram site, and the previous token is not the
                # token-trigram trigger 'the'; same-category substitution
                # therefore leaves the rule dispatch at noun_bigram.
                edited=ids[:-1].copy();edited[q]=gen.token2id[alt]
                pairs['noun_switch'].append(dict(original=ids[:-1],edited=edited,
                    query=q,old=normal_next_distribution(gen,old),
                    new=normal_next_distribution(gen,distribution(gen,alt)),
                    detail=f'{current}->{alt}'))
            elif labels[t]=='skip_trigram' and len(pairs['bracket_content_control'])<limit:
                # Last content word belongs to a bracket span. Replace it by
                # a different word of the same category, excluding its
                # immediate predecessor so generation support is maintained.
                original=words[q]
                cat=gen.token2cat.get(original)
                if cat not in gen.categories:
                    continue
                alternative=next((token for token in gen.categories[cat]
                    if token!=original and token!=words[q-1]),None)
                if alternative is None:
                    continue
                opener=next((words[k] for k in range(q-1,-1,-1)
                    if words[k] in ('(','[',')',']')),None)
                if opener not in ('(','['):
                    continue
                closer=')' if opener=='(' else ']'
                edited=ids[:-1].copy();edited[q]=gen.token2id[alternative]
                pairs['bracket_content_control'].append(dict(original=ids[:-1],
                    edited=edited,query=q,old={closer:1.},new={closer:1.},
                    detail=f'{opener}...{original}->{alternative}'))
        if all(len(pairs[k])>=limit for k in ('skip_switch','noun_switch',
                                             'bracket_content_control')):
            break
    return dict(pairs)


@torch.no_grad()
def probabilities(model, pairs, basis, rank, layer, batch_size, device):
    records=[p for group in pairs.values() for p in group]
    U=None if basis is None else basis[:, :rank]
    output=[]
    handle=None
    if U is not None:
        def project(module,inputs):
            z=inputs[0]
            return ((z@U)@U.T,)
        handle=model.blocks[layer].attn.out.register_forward_pre_hook(project)
    try:
        for start in range(0,len(records),batch_size):
            part=records[start:start+batch_size]
            source=torch.tensor([p['original'] for p in part],device=device)
            edited=torch.tensor([p['edited'] for p in part],device=device)
            q=torch.tensor([p['query'] for p in part],device=device)
            both=torch.cat((source,edited))
            logits=model(both)[0].float()
            probs=F.softmax(logits[torch.arange(len(both),device=device),
                                   torch.cat((q,q))],-1)
            output.extend(zip(probs[:len(part)].cpu().tolist(),
                              probs[len(part):].cpu().tolist()))
    finally:
        if handle is not None:handle.remove()
    return output


def summarize(gen,pairs,native,experiment):
    result={}
    offset=0
    for kind,group in pairs.items():
        rows=[]
        for i,pair in enumerate(group):
            b,a=experiment[offset+i]
            nb,na=native[offset+i]
            q0=[pair['old'].get(token,0.) for token in gen.vocab]
            q1=[pair['new'].get(token,0.) for token in gen.vocab]
            effect=[new-old for old,new in zip(b,a)]
            oracle=[new-old for old,new in zip(q0,q1)]
            native_effect=[new-old for old,new in zip(nb,na)]
            def l1(x,y):return sum(abs(v-w) for v,w in zip(x,y))
            rows.append(dict(oracle_effect_l1=l1(effect,oracle),
                             native_effect_l1=l1(effect,native_effect),
                             edit_size=l1(b,a),
                             native_edit_size=l1(nb,na),
                             original_target_mass=sum(q*v for q,v in zip(q0,b)),
                             edited_target_mass=sum(q*v for q,v in zip(q1,a))))
        result[kind]=dict(n=len(rows),**{name:sum(r[name] for r in rows)/len(rows)
                          for name in rows[0]}) if rows else dict(n=0)
        offset+=len(group)
    return result


@torch.no_grad()
def prefix_replay_check(model,pairs,device):
    """Verify that ignored full-row suffixes do not change queried logits."""
    errors=[]
    for group in pairs.values():
        for pair in group[:2]:
            q=pair['query']
            for key in ('original','edited'):
                row=torch.tensor(pair[key],device=device)[None,:]
                full=model(row)[0][0,q].float()
                prefix=model(row[:,:q+1])[0][0,-1].float()
                errors.append(float((full-prefix).abs().max()))
    worst=max(errors)
    assert worst<1e-5,worst
    return worst


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=Path('/home/loganriggs/Coding/toy_models_of_tensor_networks'))
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--fit-streams',type=int,default=256)
    p.add_argument('--panel-streams',type=int,default=1024)
    p.add_argument('--length',type=int,default=129)
    p.add_argument('--limit',type=int,default=128)
    p.add_argument('--fit-seed',type=int,default=96001)
    p.add_argument('--panel-seed',type=int,default=96002)
    p.add_argument('--rank',type=int,default=16)
    p.add_argument('--layer',type=int,default=1)
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    device=torch.device(a.device)
    model,Generator,_=model_and_generator(a.source,device)
    payload=torch.load(a.checkpoint,map_location=device,weights_only=True)
    model.load_state_dict(payload['model'])
    model.eval()
    grams,fit_counts=activation_grams(model,Generator(seed=a.fit_seed),
        a.fit_streams,16,a.length,device,a.fit_seed)
    bases=bases_from_grams(model,grams)
    gen=Generator(seed=a.panel_seed)
    pairs=choose_pairs(gen,a.panel_streams,a.length,a.limit)
    assert all(pairs.get(kind) for kind in ('skip_switch','noun_switch',
                                          'bracket_content_control'))
    replay_error=prefix_replay_check(model,pairs,device)
    native=probabilities(model,pairs,None,a.rank,a.layer,16,device)
    methods=('weight_only','all','hybrid_0.1','hybrid_1')
    scores={'native':summarize(gen,pairs,native,native)}
    for method in methods:
        trial=probabilities(model,pairs,bases[a.layer][method],a.rank,a.layer,16,device)
        scores[method]=summarize(gen,pairs,native,trial)
        print(method,{k:round(v['native_effect_l1'],3)
              for k,v in scores[method].items()},flush=True)
    mask=payload['mask']
    total=sum(x.numel() for x in mask.values())
    live=sum(int(x.sum()) for x in mask.values())
    result=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                attention_zero_fraction=1-live/total,
                prefix_replay_max_abs_logit_error=replay_error,
                fit_counts=fit_counts,panel_counts={k:len(v) for k,v in pairs.items()},
                examples={k:[p['detail'] for p in v[:4]] for k,v in pairs.items()},
                scores=scores,
                caveat='Skip and noun edits change supported normal-context '
                       'tokens; bracket control changes one supported same-category '
                       'content token and should not alter the closer. The v3 '
                       'bracket types have disjoint content vocabularies, so an '
                       'opener-only swap is not supported. Normal-state oracle '
                       'includes the generator bracket-opening probability and '
                       'the selected planted rule; selected labels establish '
                       'normal state, but this does not infer arbitrary hidden '
                       'state from a visible prefix. The model was trained on '
                       'all rule families.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('counts',result['panel_counts'])


if __name__=='__main__':main()
