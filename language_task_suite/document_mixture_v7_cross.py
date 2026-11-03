"""Joint task-rich document pilot with one cross-section composition task.

V14 narrative rules, variable-clause event reports, scoped records, scores,
and an event-source -> reviewer -> ship -> category chain coexist under one
document-kind sampler. The cross kind is intentionally oversampled and is not
claimed to match natural task prevalence. Its answer parser reads text only.
"""

from __future__ import annotations

import copy
import random

from document_mixture_v5 import (Document,Site,draw_narrative,
                                 realize_record,realize_score)
from document_mixture_v6 import draw_record_v6,draw_score_v6
from document_world import NAMES
from document_world_v5_clause_realizations import (
    answer_from_visible_prefix as event_answer,draw_varied,realize,
)


KINDS=('narrative','event_report','record_listing','score_report',
       'cross_report')
REFERENCE_WEIGHTS=(.25,.45,.12,.13,.05)
TASK_RICH_WEIGHTS=(.25,.30,.12,.13,.20)
EVENT_KWARGS=dict(style_entropy=.7,aside_rate=.3,paragraph_rate=.22,
                  header_rate=.5,source_delay_rate=1.)


def _section(tokens,heading,content):
    tokens.extend((heading,':','\n'))
    tokens.extend(content)
    tokens.append('\n')


def _record_facts(world):
    tokens=[]
    for i in world['order']:
        tokens.extend(('The',world['ships'][i],'listed','room',world['room'],
                       'as','category',world['categories'][i],'.'))
    return tokens


def _latest_source(world,obj,attr):
    j=next(j for j in range(len(world['events'])-1,-1,-1)
           if world['events'][j]['object']==obj and
           world['events'][j]['attribute']==attr)
    return world['sources'][j],j


def draw_cross(rng,split='train',event_kwargs=None):
    event_kwargs=event_kwargs or EVENT_KWARGS
    event=draw_varied(rng,**event_kwargs)
    world=copy.deepcopy(event.latent)
    # A source answer printed in the event section would leak the cross answer.
    world['recap_plan']=[row for row in world['recap_plan']
                         if row[1]!='source_attribution']
    event=realize(world,**event_kwargs)
    obj=world['events'][0]['object']
    attr=world['events'][0]['attribute']
    target_source,target_j=_latest_source(world,obj,attr)
    # The last event in draw_world has a different object: recency is wrong.
    assert target_j<len(world['events'])-1
    other_source=next((s for s in reversed(world['sources'])
                       if s!=target_source),None)
    if other_source is None:
        other_source=next(s for s in NAMES if s!=target_source)
    record=draw_record_v6(rng,split)
    target_ship=record['query_ship']
    other_ship=next(ship for ship in record['ships'] if ship!=target_ship)
    assignments=((target_source,target_ship),(other_source,other_ship))
    assignments=tuple(rng.sample(assignments,len(assignments)))
    return dict(event=world,record=record,assignments=assignments,
                query_object=obj,query_attribute=attr,
                section_order=tuple(rng.sample(
                    ('Events','Reviewers','Records'),3)),
                topic=world['topic'],event_kwargs=dict(event_kwargs))


def realize_cross(world,event_kwargs=None):
    event_kwargs=event_kwargs or world.get('event_kwargs',EVENT_KWARGS)
    event=realize(world['event'],**event_kwargs)
    record=world['record']
    sections=dict(Events=event.tokens,
                  Reviewers=tuple(token for person,ship in
                                  world['assignments'] for token in
                                  (person,'reviewed','the',ship,'.')),
                  Records=tuple(_record_facts(record)))
    tokens=['Bulletin',':',world['topic'],'.','\n']
    sites=[]
    for section in world['section_order']:
        begin=len(tokens)+3
        _section(tokens,section,sections[section])
        if section=='Events':
            sites.extend(Site(begin+s.index,s.family,s.answer,
                              {s.answer:1.}) for s in event.sites)
    obj=world['query_object'];attr=world['query_attribute']
    room=record['room']
    tokens.extend(('Question',':','The','category','for','room',room,
                   'on','the','ship','reviewed','by','the','source','of',
                   'the','latest',attr,'report','about','the',obj,'was'))
    index=len(tokens)
    source,_=_latest_source(world['event'],obj,attr)
    ship=dict(world['assignments'])[source]
    answer=dict(zip(record['ships'],record['categories'],strict=True))[ship]
    tokens.extend((answer,'.'))
    sites.append(Site(index,'cross_source_record',answer,{answer:1.}))
    return Document(tuple(tokens),tuple(sites),'cross_report',world)


def _split_sections(prefix):
    headings=('Events','Reviewers','Records','Question')
    assert len(prefix)>=4 and prefix[:2]==('Bulletin',':') and prefix[3]=='.'
    sections={}
    cursor=4
    while cursor<len(prefix):
        while cursor<len(prefix) and prefix[cursor]=='\n':
            cursor+=1
        if cursor==len(prefix):
            break
        heading=prefix[cursor]
        assert heading in headings and prefix[cursor+1]==':'
        cursor+=2
        if cursor<len(prefix) and prefix[cursor]=='\n':
            cursor+=1
        start=cursor
        while cursor<len(prefix):
            if (prefix[cursor] in headings and cursor+1<len(prefix)
                    and prefix[cursor+1]==':'):
                break
            cursor+=1
        section=tuple(t for t in prefix[start:cursor] if t!='\n')
        assert heading not in sections
        sections[heading]=section
    return sections


def cross_answer_from_visible(prefix):
    """Derive the cross answer from complete visible sections and query."""
    sections=_split_sections(tuple(prefix))
    assert set(sections)=={'Events','Reviewers','Records','Question'}
    question=sections['Question']
    assert len(question)==21 and question[:4]==('The','category','for','room')
    room=question[4]
    assert question[5:16]==('on','the','ship','reviewed','by','the',
                            'source','of','the','latest',question[15])
    # The exact fixed question is checked without reading sampled state.
    attr=question[15]
    assert question[16:19]==('report','about','the')
    obj=question[19]
    assert question[20]=='was'
    source=event_answer(sections['Events']+(
        'The','latest','report','about','the',attr,'of','the',obj,
        'came','from'))
    reviewer=sections['Reviewers']
    assert len(reviewer)==10
    mapping={}
    for i in (0,5):
        assert reviewer[i+1:i+4]==('reviewed','the',reviewer[i+3])
        assert reviewer[i+4]=='.'
        mapping[reviewer[i]]=reviewer[i+3]
    ship=mapping[source]
    records=sections['Records']
    assert len(records)==18
    facts={}
    for i in (0,9):
        assert records[i:i+1]==('The',)
        assert records[i+2:i+4]==('listed','room')
        assert records[i+5:i+7]==('as','category') and records[i+8]=='.'
        facts[(records[i+1],records[i+4])]=records[i+7]
    return facts[(ship,room)]


def sample_document(rng,kind=None,split='train',weights=TASK_RICH_WEIGHTS,
                    event_kwargs=None):
    kind=kind or rng.choices(KINDS,weights)[0]
    if kind=='narrative':
        return draw_narrative(rng)
    if kind=='event_report':
        event=draw_varied(rng,**(event_kwargs or EVENT_KWARGS))
        sites=tuple(Site(s.index,s.family,s.answer,{s.answer:1.})
                    for s in event.sites)
        return Document(event.tokens,sites,kind,event.latent)
    if kind=='record_listing':
        return realize_record(draw_record_v6(rng,split))
    if kind=='score_report':
        return realize_score(draw_score_v6(rng,split))
    assert kind=='cross_report'
    return realize_cross(draw_cross(rng,split,event_kwargs),event_kwargs)


def site_oracle(doc,site):
    if site.family=='cross_source_record':
        return {cross_answer_from_visible(doc.tokens[:site.index]):1.}
    if doc.kind=='event_report':
        assert event_answer(doc.tokens[:site.index])==site.target
        return {site.target:1.}
    if doc.kind=='cross_report':
        # World sites are local to the Events section. The cross answer uses
        # the full visible document parser above.
        sections=_split_sections(doc.tokens[:site.index])
        assert event_answer(sections['Events'])==site.target
        return {site.target:1.}
    from document_mixture_v5 import site_oracle as old_oracle
    return old_oracle(doc,site)
