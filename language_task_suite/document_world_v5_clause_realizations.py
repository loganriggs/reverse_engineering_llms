"""Variable-clause rendering of the visible world-state document pilot.

The event ledger and interleaved recap plan come from V4. Unlike V4's one
form per genre, each introduction, update, and recap samples a surface form.
An independent transducer recovers the canonical V4 visible text, so scored
answers need not read sampled latent state. This is a structural DGP pilot;
full next-token probabilities and natural-language realism are not claimed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from document_world import NAMES, PLACES, VALUES, _preposition, replay
from document_world_v3_multisite import (ALIASES, GENRES, AnswerSite,
                                         _action, article, parse_visible)
from document_world_v4_interleaved import (answer_from_visible_prefix as
                                           canonical_answer,
                                           draw_interleaved)
from surface_mixture_v2 import ADJECTIVES, EVENTS, OBJECTS


INCIDENTAL_NOUNS = tuple(w for w in OBJECTS if w not in VALUES['place'])
INTRO_FORMS = 3
EVENT_FORMS = 4
RECAP_FORMS = 4


@dataclass(frozen=True)
class VariedDocument:
    tokens: tuple[str, ...]
    sites: tuple[AnswerSite, ...]
    genre: str
    latent: dict

    def text(self):
        return (' '.join(self.tokens).replace(' .', '.')
                .replace(' ,', ',').replace(' :', ':')
                .replace(' ;', ';').replace(' \n ', '\n'))


def _choose_form(rng, genre, count, entropy):
    assert 0 <= entropy <= 1
    preferred = {'story': 0, 'summary': 1, 'log': 2}[genre]
    weights = [entropy/count]*count
    weights[preferred] += 1-entropy
    return rng.choices(range(count), weights)[0]


def _intro(obj, state, alias, form):
    owner, color, place = (state[k] for k in ('owner','color','place'))
    prep = _preposition(place)
    if form == 0:
        return [owner,'kept',article(color),color,obj,prep,'the',place,
                ',','labeled',alias,'.']
    if form == 1:
        return ['The',color,obj,prep,'the',place,'belonged','to',owner,
                'and','carried','the','label',alias,'.']
    assert form == 2
    return ['Entry',':',obj,';','color',':',color,';','owner',':',owner,
            ';','location',':',place,';','label',':',alias,'.']


def _event_action(event, previous, ref, form):
    actor = previous[event['object']]['owner']
    value = event['value']
    attr = event['attribute']
    if form in (0,3):
        return _action(event, previous, ref)
    if form == 1:
        if attr == 'owner':
            return [actor,'gave','the',ref,'to',value]
        if attr == 'color':
            return [actor,'changed','the',ref,'to',value]
        return [actor,'placed','the',ref,_preposition(value),'the',value]
    assert form == 2
    if attr == 'owner':
        return [actor,'transferred','the',ref,'to',value]
    if attr == 'color':
        return [actor,'recolored','the',ref,value]
    return [actor,'relocated','the',ref,'to','the',value]


def _event(event, previous, source, ref, form):
    action = _event_action(event, previous, ref, form)
    if form == 0:
        return [source,'reported','that']+action+['.']
    if form == 1:
        return ['According','to',source,',']+action+['.']
    if form == 2:
        return ['In',source+"'s",'report',',']+action+['.']
    return [source,'wrote',',','"']+action+['"','.']


def _aside(rng, obj, place, topic, genre):
    noun = rng.choice(INCIDENTAL_NOUNS)
    adjective = rng.choice(ADJECTIVES)
    event = rng.choice(EVENTS)
    form = _choose_form(rng, genre, 3, .8)
    if form == 0:
        return ['Several','visitors','discussed','the',obj,'near','the',place,'.']
    if form == 1:
        return ['A','short','note','about','the',obj,'circulated','after',
                'the',event,'.']
    return ['During','the',event,',','staff','reviewed',article(adjective),
            adjective,noun,'for','the',topic,'.']


def _recap_prefix(family, obj, attr, state, aliases, form):
    if family == 'alias_identity':
        alias = aliases[obj]
        prefixes = (["The",'label',alias,'identified','the'],
                    [alias,'was','the','code','for','the'],
                    ['For','the','label',alias,',','the','item','was','the'],
                    ['Which','item','had','the','label',alias,'?','Answer',':'])
        return prefixes[form],obj,tuple(state)
    if family == 'source_attribution':
        prefixes = (['The','latest','report','about','the',attr,'of','the',obj,
                     'came','from'],
                    ['For',obj,',','the','latest',attr,'report','was','filed','by'],
                    ['The','source','of','the','most','recent',attr,'change',
                     'to',obj,'was'],
                    ['Who','reported','the','latest',attr,'change','for',obj,
                     '?','Answer',':'])
        return prefixes[form],None,NAMES
    value = state[obj][family.removeprefix('world_')]
    if family == 'world_owner':
        prefixes = (['By','then',',','the',obj,'belonged','to'],
                    ['At','this','point',',',obj,'belonged','to'],
                    ['Current','owner','of',obj,':'],
                    ['Who','owned',obj,'then','?','Answer',':'])
    elif family == 'world_color':
        prefixes = (['By','then',',','the',obj,'was'],
                    ['At','this','point',',',obj,'was'],
                    ['Current','color','of',obj,':'],
                    ['What','color','was',obj,'then','?','Answer',':'])
    else:
        assert family == 'world_place'
        prefixes = (['By','then',',','the',obj,'was',_preposition(value),'the'],
                    ['At','this','point',',',obj,'was','located','at'],
                    ['Current','location','of',obj,':'],
                    ['Where','was',obj,'then','?','Answer',':'])
    return prefixes[form],value,VALUES[family.removeprefix('world_')]


def realize(world, style_entropy=.7, aside_rate=.3, paragraph_rate=.22,
            header_rate=.5, source_delay_rate=0.):
    assert (world['genre'] in GENRES and 0 <= aside_rate <= 1
            and 0 <= paragraph_rate <= 1 and 0 <= header_rate <= 1)
    rng=random.Random(world['realization_seed'])
    genre=world['genre']
    tokens,sites=[],[]
    current=replay(world['initial'],[])
    sources={}

    def emit(sentence):
        if tokens and rng.random()<paragraph_rate:
            tokens.append('\n')
        tokens.extend(sentence)

    if rng.random()<header_rate:
        kind={'story':'Story','summary':'Report','log':'File'}[genre]
        digits=list(str(rng.randrange(100,1000)))
        emit([kind]+digits+[':',world['topic'],'notes','.'])

    for obj,state in world['initial'].items():
        form=_choose_form(rng,genre,INTRO_FORMS,style_entropy)
        emit(_intro(obj,state,world['aliases'][obj],form))
        if rng.random()<aside_rate:
            emit(_aside(rng,obj,state['place'],world['topic'],genre))

    def append_recap(family,obj,attr):
        form=_choose_form(rng,genre,RECAP_FORMS,style_entropy)
        prefix,answer,candidates=_recap_prefix(
            family,obj,attr,current,world['aliases'],form)
        if family == 'source_attribution':
            answer=sources[(obj,attr)]
        if tokens and rng.random()<paragraph_rate:
            tokens.append('\n')
        tokens.extend(prefix)
        index=len(tokens)
        tokens.extend((answer,'.'))
        sites.append(AnswerSite(index,family,answer,tuple(candidates),obj))

    for after,family,obj,attr in world['recap_plan']:
        if after == -1:
            append_recap(family,obj,attr)
    for j,(event,source,use_alias) in enumerate(zip(
            world['events'],world['sources'],world['use_alias'],strict=True)):
        ref=world['aliases'][event['object']] if use_alias else event['object']
        form=_choose_form(rng,genre,EVENT_FORMS,style_entropy)
        emit(_event(event,current,source,ref,form))
        current[event['object']][event['attribute']]=event['value']
        sources[(event['object'],event['attribute'])]=source
        if rng.random()<aside_rate:
            emit(_aside(rng,event['object'],
                        current[event['object']]['place'],
                        world['topic'],genre))
        for after,family,obj,attr in world['recap_plan']:
            if after == j:
                append_recap(family,obj,attr)
    return VariedDocument(tuple(tokens),tuple(sites),genre,world)


def draw_varied(rng, genre=None, style_entropy=.7, aside_rate=.3,
                paragraph_rate=.22, header_rate=.5, source_delay_rate=0.):
    world=draw_interleaved(rng,genre)
    assert 0 <= source_delay_rate <= 1
    delayed=[]
    for after,family,obj,attr in world['recap_plan']:
        if (family=='source_attribution' and source_delay_rate
                and rng.random()<source_delay_rate):
            # Require an intervening report about a *different* key. The
            # source answer is then no longer the most recent source name.
            candidates=[k for k in range(after+1,len(world['events']))
                        if (world['events'][k]['object'],
                            world['events'][k]['attribute'])!=(obj,attr)]
            if candidates:
                after=rng.choice(candidates)
        delayed.append((after,family,obj,attr))
    world['recap_plan']=sorted(delayed,key=lambda item:item[0])
    world['realization_seed']=rng.randrange(2**63)
    return realize(world,style_entropy,aside_rate,paragraph_rate,header_rate,
                   source_delay_rate)


def sample_documents(n,seed,style_entropy=.7,aside_rate=.3,
                     paragraph_rate=.22,header_rate=.5,
                     source_delay_rate=0.):
    rng=random.Random(seed)
    return [draw_varied(rng,style_entropy=style_entropy,aside_rate=aside_rate,
                        paragraph_rate=paragraph_rate,header_rate=header_rate,
                        source_delay_rate=source_delay_rate)
            for _ in range(n)]


def _sentences(prefix):
    complete=[];raw=[]
    for word in prefix:
        raw.append(word)
        if word=='.':
            complete.append(raw[:-1]);raw=[]
    return complete,raw


def _canonical_intro(sentence):
    if sentence[0] in NAMES and sentence[1]=='kept':
        owner,_,det,color,obj,prep,the,place,comma,_,alias=sentence
        assert det==article(color) and prep==_preposition(place)
        assert (the,comma)==('the',',')
    elif sentence[0]=='The' and sentence[6:8]==['belonged','to']:
        _,color,obj,prep,the,place,_,_,owner,conj,carried,the2,label,alias=sentence
        assert (prep,the,conj,carried,the2,label)==(
            _preposition(place),'the','and','carried','the','label')
    else:
        assert sentence[0:2]==['Entry',':']
        assert sentence[3:6]==[';','color',':']
        assert sentence[7:10]==[';','owner',':']
        assert sentence[11:14]==[';','location',':']
        assert sentence[15:18]==[';','label',':']
        obj,color,owner,place,alias=(sentence[i] for i in (2,6,10,14,18))
    return ([owner,'kept',article(color),color,obj,_preposition(place),
             'the',place,'.'],
            ['The',obj,'was','labeled',alias,'.'])


def _canonical_action(action):
    actor,verb,det,ref,*tail=action
    assert actor in NAMES and det=='the'
    if verb in ('handed','gave','transferred'):
        assert tail[0]=='to' and tail[1] in NAMES
        return [actor,'handed','the',ref,'to',tail[1]]
    if verb in ('painted','changed','recolored'):
        value=tail[-1]
        assert value in VALUES['color']
        return [actor,'painted','the',ref,value]
    assert verb in ('moved','placed','relocated')
    value=tail[-1]
    assert value in PLACES
    if verb=='placed':
        assert tail==[_preposition(value),'the',value]
    else:
        assert tail==['to','the',value]
    return [actor,'moved','the',ref,'to','the',value]


def _canonical_event(sentence):
    if sentence[1:3]==['reported','that']:
        source=sentence[0];action=sentence[3:]
    elif sentence[:2]==['According','to']:
        source=sentence[2]
        assert sentence[3]==','
        action=sentence[4:]
    elif sentence[1:4]==['wrote',',','"']:
        source=sentence[0]
        assert sentence[-1]=='"'
        action=sentence[4:-1]
    else:
        assert sentence[0]=='In' and sentence[1].endswith("'s")
        assert sentence[2:4]==['report',',']
        source=sentence[1][:-2];action=sentence[4:]
    assert source in NAMES
    return [source,'reported','that']+_canonical_action(action)+['.']


def _canonical_query(raw, canonical_history):
    if raw[:2]==['Which','item']:
        assert raw[:5]==['Which','item','had','the','label']
        assert raw[6:]==['?','Answer',':']
        return ['The','item','labeled',raw[5],'was','the']
    if raw[:2]==['Who','reported']:
        assert raw[:4]==['Who','reported','the','latest']
        assert raw[5:7]==['change','for'] and raw[8:]==['?','Answer',':']
        return ['The','latest','report','about','the',raw[4],'of','the',
                raw[7],'came','from']
    if raw[:2]==['Who','owned']:
        assert raw[3:]==['then','?','Answer',':']
        return ['By','then',',','the',raw[2],'belonged','to']
    if raw[:2]==['What','color']:
        assert raw[2]=='was' and raw[4:]==['then','?','Answer',':']
        return ['By','then',',','the',raw[3],'was']
    if raw[:2]==['Where','was']:
        assert raw[3:]==['then','?','Answer',':']
        obj=raw[2]
        state,_,_,_=parse_visible(canonical_history)
        return ['By','then',',','the',obj,'was',
                _preposition(state[obj]['place']),'the']
    if raw[:2]==['The','label']:
        alias=raw[2]
        assert raw[3:]==['identified','the']
        return ['The','item','labeled',alias,'was','the']
    if raw[:1]==['For'] and raw[1:3]==['the','label']:
        alias=raw[3]
        assert raw[4:]==[',','the','item','was','the']
        return ['The','item','labeled',alias,'was','the']
    if raw and raw[0] in ALIASES:
        assert raw[1:]==['was','the','code','for','the']
        return ['The','item','labeled',raw[0],'was','the']
    if raw[:3]==['The','latest','report']:
        return raw
    if raw[:1]==['For']:
        _,obj,comma,the,latest,attr,report,was,filed,by=raw
        assert (comma,the,latest,report,was,filed,by)==(
            ',','the','latest','report','was','filed','by')
        return ['The','latest','report','about','the',attr,'of','the',obj,
                'came','from']
    if raw[:3]==['The','source','of']:
        _,_,_,the,most,recent,attr,change,to,obj,was=raw
        assert (the,most,recent,change,to,was)==(
            'the','most','recent','change','to','was')
        return ['The','latest','report','about','the',attr,'of','the',obj,
                'came','from']
    if raw[:2]==['By','then']:
        return raw
    if raw[:4]==['At','this','point',',']:
        obj=raw[4]
        if raw[5:]==['belonged','to']:
            return ['By','then',',','the',obj,'belonged','to']
        if raw[5:]==['was']:
            return ['By','then',',','the',obj,'was']
        assert raw[5:]==['was','located','at']
        state,_,_,_=parse_visible(canonical_history)
        return ['By','then',',','the',obj,'was',
                _preposition(state[obj]['place']),'the']
    assert raw[:1]==['Current'] and raw[2]=='of' and raw[-1]==':'
    attr={'owner':'owner','color':'color','location':'place'}[raw[1]]
    obj=raw[3]
    if attr=='owner':
        return ['By','then',',','the',obj,'belonged','to']
    if attr=='color':
        return ['By','then',',','the',obj,'was']
    state,_,_,_=parse_visible(canonical_history)
    return ['By','then',',','the',obj,'was',
            _preposition(state[obj]['place']),'the']


def canonical_visible_prefix(prefix):
    """Parse only visible text into V4's independently checked syntax."""
    prefix=[word for word in prefix if word!='\n']
    complete,raw=_sentences(prefix)
    out=[]
    for sentence in complete:
        if not sentence or sentence[0] in (
                'Several','A','During','Story','Report','File'):
            continue  # Incidental sentences assert no state change.
        if sentence[0]=='Entry' or (sentence[0] in NAMES and
                                    sentence[1]=='kept') or (
                                        sentence[0]=='The' and len(sentence)>7
                                        and sentence[6:8]==['belonged','to']):
            for canonical in _canonical_intro(sentence):
                out.extend(canonical)
        elif (sentence[1:3]==['reported','that'] or
              sentence[:2]==['According','to'] or
              sentence[1:4]==['wrote',',','"'] or
              (sentence[0]=='In' and sentence[1].endswith("'s"))):
            out.extend(_canonical_event(sentence))
        else:
            # Completed recaps are observations, never state updates.
            assert sentence[0] in ('The','For','By','At','Current','Which',
                                   'Who','What','Where') or (
                sentence[0] in ALIASES),sentence
    out.extend(_canonical_query(raw,out))
    return out


def answer_from_visible_prefix(prefix):
    return canonical_answer(canonical_visible_prefix(prefix))
