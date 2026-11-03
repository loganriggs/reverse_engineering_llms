"""Audit crossed skip-rule coverage and supported two-cue counterfactuals."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_pre=argparse.ArgumentParser(add_help=False)
_pre.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
_source,_=_pre.parse_known_args()
sys.path.insert(0,_source.source)

from original_v3_crossed_skip import LanguageV3CrossedSkip, nearest_skip_anchor


def audit(gen,nstreams,length,edit_limit):
    anchors=gen.crossed_skip_anchors
    triggers=gen.crossed_skip_triggers
    outputs=gen.crossed_skip_outputs
    table=[[gen.skip_bigrams[a,t] for t in triggers] for a in anchors]
    assert all(set(row)==set(outputs) for row in table)
    assert all(set(row[j] for row in table)==set(outputs) for j in range(4))
    counts=Counter()
    labels=Counter()
    distances=Counter()
    joint_streams=0
    valid_edits=0
    examples=[]
    place_to_adj=float(gen.transition_matrix[
        gen.cat_index['PLACE']][gen.cat_index['ADJ']])
    assert 0<gen.trigger_boost_prob<1
    for _ in range(nstreams):
        ids,labs=gen.sample_sequence(length)
        words=[gen.id2token[i] for i in ids]
        labels.update(labs)
        joint_streams+=all(rule in labs for rule in
                           ('noun_bigram','skip_bigram','skip_trigram'))
        for q,label in enumerate(labs):
            if label!='skip_bigram':continue
            found=nearest_skip_anchor(gen,words[:q])
            assert found is not None
            k,anchor=found
            trigger=words[q-1]
            target=gen.skip_bigrams[anchor,trigger]
            assert words[q]==target,(q,words[max(0,k-2):q+1])
            counts[anchor,trigger]+=1
            distances[q-k]+=1
            # The anchor was sampled from the fallback PLACE category and
            # the immediately following ADJ was sampled from a probabilistic
            # boost (PLACE→ADJ fallback can have zero mass). Every crossed
            # anchor boosts every trigger with nonzero probability, so a
            # same-category anchor swap preserves prefix support up to q.
            # No intervening tokens exist and the queried rule remains exact.
            if (q-k!=2 or labs[k]!='cat_bigram'
                    or labs[q-1]!='cat_bigram'):
                continue
            for alternative in anchors:
                if alternative==anchor:continue
                edited=words[:q].copy();edited[k]=alternative
                new_target=gen.skip_bigrams[alternative,trigger]
                assert new_target!=target
                replay,replay_label=gen._sample_normal_token(edited,{})
                assert (replay,replay_label)==(new_target,'skip_bigram')
                valid_edits+=1
                if len(examples)<8:
                    examples.append(dict(prefix=' '.join(words[max(0,k-2):q]),
                        original_anchor=anchor,edited_anchor=alternative,
                        trigger=trigger,old_target=target,new_target=new_target,
                        anchor_distance=q-k))
                # Continue counting the full sampled panel even after the
                # requested minimum number of edits has been reached.
    return dict(table=dict(anchors=anchors,triggers=triggers,outputs=outputs,
                           targets=table),
                sampled_streams=nstreams,processed_streams=_+1,
                labels=dict(labels),joint_streams=joint_streams,
                pair_counts={f'{a}|{t}':counts[a,t] for a in anchors for t in triggers},
                distances=dict(sorted(distances.items())),
                supported_dist2_anchor_edits=valid_edits,examples=examples,
                place_to_adj_transition_probability=place_to_adj,
                caveat='The exact deterministic skip-site target and dist-2 '
                       'anchor-edit support are audited. This does not certify '
                       'a full visible-prefix oracle for every other v3 rule '
                       'or realistic language form.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--nstreams',type=int,default=4000)
    p.add_argument('--length',type=int,default=129)
    p.add_argument('--seed',type=int,default=99101)
    p.add_argument('--edit-limit',type=int,default=256)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    gen=LanguageV3CrossedSkip(seed=a.seed)
    result=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                audit=audit(gen,a.nstreams,a.length,a.edit_limit))
    assert min(result['audit']['pair_counts'].values())>0
    assert result['audit']['supported_dist2_anchor_edits']>=a.edit_limit
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('pair counts',result['audit']['pair_counts'])
    print('supported edits',result['audit']['supported_dist2_anchor_edits'],
          'joint streams',result['audit']['joint_streams'],flush=True)


if __name__=='__main__':main()
