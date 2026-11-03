"""Audit joint-rule coverage, exact mixtures, and supported heading edits."""

import argparse
import json
from collections import Counter
from pathlib import Path

from surface_mixture_v2 import _split
from surface_mixture_v4 import _content
from surface_mixture_v7 import TOPICS
from surface_mixture_v10_linked_events import SurfaceMixtureV10LinkedEvents
from surface_mixture_v11_section_topic import (SurfaceMixtureV11SectionTopic,
                                                active_section_topic)


REFERENCE = dict(counterbalanced_queries=True,question_rate=.05,count_rate=0.,
                 quote_rate=.08,source_rate=.15,section_rate=.06,
                 detail_rate=.8,scope_question_rate=.08,
                 source_object_mass=.15,source_subject_mass=.15,
                 bracket_rate=.06,linked_subject_mass=.65,
                 linked_object_mass=.65,linked_question_mass=.30,
                 linked_query_key_mass=.65)


def latest_heading_topic_index(words):
    positions=[i+3 for i,word in enumerate(words) if word=='Section'
               and i+4<len(words) and words[i+2]==':' and words[i+4]=='.']
    return positions[-1] if positions else None


def panel(cls,seed,streams,length,edit_limit,kwargs=None):
    kwargs=dict(REFERENCE if kwargs is None else kwargs)
    gen=cls(seed=seed,**kwargs)
    labels_count=Counter()
    joint=Counter()
    examples=[]
    edits=[]
    site_count=0
    component_error=0.
    topic_match=0
    predicted_topic_mass=0.
    baseline_topic_mass=0.
    for _ in range(streams):
        ids,labels,qs=gen.sample_sequence(length)
        words=[gen.id2token[i] for i in ids]
        for i,(word,q) in enumerate(zip(words,qs)):
            if q.get(word,0)<=0 or abs(sum(q.values())-1)>1e-10:
                raise AssertionError(('sampled support',i,word,q.get(word)))
        counts=Counter(labels)
        labels_count.update(counts)
        for name in ('mixed_verb','mixed_object','source_answer',
                     'scope_answer','bracket_close','section_topic_place'):
            if counts[name]:joint[name]+=1
        if all(counts[k] for k in ('mixed_verb','mixed_object',
                                   'section_topic_place','bracket_close')):
            joint['verb_object_topic_bracket']+=1
        if not isinstance(gen,SurfaceMixtureV11SectionTopic):
            continue
        for t,label in enumerate(labels):
            if label!='section_topic_place':continue
            site_count+=1
            prefix=words[:t]
            parts=gen.place_rule_components(prefix)
            topic=active_section_topic(_split(prefix)[0])
            topic_match+=words[t]==topic
            predicted_topic_mass+=qs[t][topic]
            baseline_topic_mass+=parts['base_place']['distribution'][topic]
            replay=Counter()
            for part in parts.values():
                for token,prob in part['distribution'].items():
                    replay[token]+=part['mass']*prob
            component_error=max(component_error,max(abs(replay[token]-qs[t].get(token,0.))
                                 for token in set(replay)|set(qs[t])))
            if len(edits)>=edit_limit:continue
            heading=latest_heading_topic_index(prefix)
            assert heading is not None
            old=prefix[heading]
            assert old==active_section_topic(_split(prefix)[0])
            base=parts['base_place']['distribution']
            alternatives=[topic for topic in TOPICS if topic!=old and topic in base]
            if not alternatives:continue
            new=alternatives[0]
            edited=prefix.copy();edited[heading]=new
            for pos in range(heading+1,len(prefix)):
                assert gen.next_distribution(edited[:pos]).get(edited[pos],0)>0
            original=gen.next_distribution(prefix)
            changed=gen.next_distribution(edited)
            assert abs(changed[new]-original[new]-gen.section_topic_mass)<1e-10
            assert abs(changed[old]-original[old]+gen.section_topic_mass)<1e-10
            assert all(abs(changed.get(token,0.)-original.get(token,0.))<1e-10
                       for token in set(changed)|set(original) if token not in (old,new))
            edits.append(dict(old_topic=old,new_topic=new,
                              observed_place=words[t],
                              heading_to_site_distance=t-heading,
                              changed_new_topic_probability=changed[new]-original[new]))
            if len(examples)<5:
                examples.append(dict(prefix=' '.join(prefix[-25:]),
                                     old_topic=old,new_topic=new,
                                     observed_place=words[t],
                                     distance=t-heading))
    return dict(config=dict(seed=seed,streams=streams,length=length,
                            generator=cls.__name__,kwargs=kwargs),
                labels=dict(labels_count),streams_with=dict(joint),
                section_topic_sites=site_count,
                section_topic_empirical_match_fraction=(topic_match/site_count
                    if site_count else None),
                section_topic_predicted_match_fraction=(predicted_topic_mass/site_count
                    if site_count else None),
                section_topic_base_match_fraction=(baseline_topic_mass/site_count
                    if site_count else None),
                component_replay_max_error=component_error,
                n_supported_topic_edits=len(edits),
                heading_distance=dict(min=min((e['heading_to_site_distance'] for e in edits),default=None),
                                      max=max((e['heading_to_site_distance'] for e in edits),default=None),
                                      mean=(sum(e['heading_to_site_distance'] for e in edits)/len(edits)
                                            if edits else None)),
                edit_examples=examples,
                caveat='The reference-candidate rates and topic mass are '
                       'research choices, not FineWeb-calibrated. Sections '
                       'alter place probabilities but do not make the surface '
                       'naturalistic. Oracle support is checked on every '
                       'sampled token; edited prefixes checked through the '
                       'selected place site. No model competence claim.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=98401)
    p.add_argument('--streams',type=int,default=400)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--edit-limit',type=int,default=256)
    p.add_argument('--section-rate',type=float,default=REFERENCE['section_rate'])
    p.add_argument('--section-topic-mass',type=float,default=.35)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    kwargs=dict(REFERENCE,section_rate=a.section_rate)
    old=panel(SurfaceMixtureV10LinkedEvents,a.seed,a.streams,a.length,
              a.edit_limit,kwargs)
    new=panel(SurfaceMixtureV11SectionTopic,a.seed,a.streams,a.length,
              a.edit_limit,dict(kwargs,section_topic_mass=a.section_topic_mass))
    old_zero=SurfaceMixtureV10LinkedEvents(seed=a.seed+100000,**kwargs)
    new_zero=SurfaceMixtureV11SectionTopic(seed=a.seed+100000,**kwargs,
                                           section_topic_mass=0.)
    zero_mass_max_error=0.
    for _ in range(min(a.streams,32)):
        old_ids,_,old_q=old_zero.sample_sequence(a.length)
        new_ids,_,new_q=new_zero.sample_sequence(a.length)
        assert old_ids==new_ids
        for left,right in zip(old_q,new_q):
            zero_mass_max_error=max(zero_mass_max_error,
                max(abs(left.get(token,0.)-right.get(token,0.))
                    for token in set(left)|set(right)))
    assert zero_mass_max_error<1e-10
    assert new['section_topic_sites']>0 and new['n_supported_topic_edits']>0
    assert new['component_replay_max_error']<1e-10
    output=dict(v10=old,v11=new,
                zero_mass_v10_replay_max_error=zero_mass_max_error)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,indent=2)+'\n')
    print('V10',old['streams_with'],'V11',new['streams_with'])
    print('topic sites',new['section_topic_sites'],'supported edits',
          new['n_supported_topic_edits'],'replay error',
          new['component_replay_max_error'])


if __name__=='__main__':main()
