"""Freeze typed lexical candidates and aggregate FineWeb POS frequencies.

Optional build-time dependencies: datasets and NLTK with the English averaged
perceptron tagger. The runtime generator only needs the resulting JSON.
The candidate list is manually reviewed; POS counts set a sampling prior,
not a claim that all uses of a word in FineWeb have the listed sense.
"""

import argparse
import gc
import json
import os
import re
import sys
from collections import Counter
from itertools import islice
from pathlib import Path

from surface_mixture_v2 import (ADJECTIVES as BASE_ADJECTIVES,
                                EVENTS as BASE_EVENTS,
                                INTERIOR_PLACES as BASE_INTERIOR,
                                OBJECTS as BASE_OBJECTS,
                                PEOPLE as BASE_PEOPLE,
                                PLACES as BASE_PLACES,unique)


# Curated single-word English role inventories. The semantic grouping was
# checked against WordNet noun categories and reviewed for plausible usage in
# the V2 sentence frames. FineWeb frequencies are measured separately.
EXTRA_PEOPLE=tuple('Alice Andrew Benjamin Charlotte David Evelyn Frank Grace '
    'Henry Isabella James Katherine Lucas Maria Nathan Olivia Peter Rachel '
    'Sofia Thomas Victoria William Zoe Adrian Amelia Arthur Beatrice Caleb '
    'Camila Chloe Edward Eleanor Eli Elizabeth Emily Ethan Fiona Gabriel '
    'Harper Jack Jacob Jasmine Joseph Julia Leah Lily Logan Mason Mia Michael '
    'Miles Nora Patrick Rebecca Robert Sarah Simon Sophie Stella Susan Theo '
    'Violet Walter'.split())
EXTRA_OBJECTS=tuple('camera keyboard monitor printer scanner tablet router '
    'modem speaker radio television flashlight lantern compass map passport '
    'wallet purse suitcase backpack briefcase envelope manuscript ledger '
    'invoice receipt ticket certificate calendar poster banner brochure '
    'newspaper magazine journal notebook pencil pen marker ruler eraser '
    'folder binder clipboard board canvas easel sculpture statue trophy '
    'medal badge uniform helmet glove coat jacket shirt shoe boot bicycle '
    'scooter motorcycle car truck van wagon boat drone engine battery cable '
    'circuit sensor motor valve pump filter switch button wrench hammer saw '
    'drill shovel ladder chair table desk shelf cabinet drawer bottle jar '
    'basket bucket bowl plate cup kettle pan clock watch guitar piano drum '
    'violin microphone headset headphone toy puzzle ball net racket mask '
    'thermometer bandage specimen container box parcel package'.split())
EXTRA_PLACES=tuple('office classroom library laboratory workshop factory '
    'warehouse hospital clinic pharmacy school university campus museum '
    'gallery theater cinema studio restaurant cafe hotel motel airport '
    'station terminal platform port harbor marina garage kitchen pantry '
    'hallway lobby basement attic bedroom bathroom apartment house cottage '
    'cabin farmhouse barn market store shop bank courthouse church temple '
    'palace stadium arena gym park garden plaza square street alley bridge '
    'riverbank valley forest beach coast campsite meadow field mountain '
    'hill island village town city'.split())
EXTRA_INTERIOR=tuple('office classroom library laboratory workshop factory '
    'warehouse hospital clinic pharmacy school university museum gallery '
    'theater cinema studio restaurant cafe hotel motel airport station '
    'terminal garage kitchen pantry hallway lobby basement attic bedroom '
    'bathroom apartment house cottage cabin farmhouse barn market store '
    'shop bank courthouse church temple palace stadium arena gym'.split())
EXTRA_EVENTS=tuple('meeting conference seminar workshop lecture presentation '
    'debate discussion interview inspection review hearing trial ceremony '
    'wedding funeral festival concert performance rehearsal exhibition '
    'auction sale launch opening election campaign contest competition '
    'tournament match game race parade protest strike accident storm flood '
    'fire repair delivery trip journey tour visit dinner celebration '
    'graduation training exercise experiment'.split())
EXTRA_ADJECTIVES=tuple('bright dark polished rusty broken clean dusty sealed '
    'painted small large old new wooden metallic heavy light delicate '
    'valuable rare common recent historic urgent local public private '
    'annual temporary permanent handmade imported recycled portable '
    'digital analog printed handwritten colorful plain damaged repaired '
    'open closed empty full'.split())


def candidate_lists():
    groups=dict(people=unique(BASE_PEOPLE,EXTRA_PEOPLE),
                objects=unique(BASE_OBJECTS,EXTRA_OBJECTS),
                places=unique(BASE_PLACES,EXTRA_PLACES),
                interior_places=unique(BASE_INTERIOR,EXTRA_INTERIOR),
                events=unique(BASE_EVENTS,EXTRA_EVENTS),
                adjectives=unique(BASE_ADJECTIVES,EXTRA_ADJECTIVES))
    assert set(groups['interior_places'])<=set(groups['places'])
    assert all(len(v)==len(set(v)) for v in groups.values())
    return groups


def count_pos(skip,n,chars):
    import nltk
    from datasets import load_dataset
    ds=load_dataset('HuggingFaceFW/fineweb',name='sample-10BT',
                    split='train',streaming=True)
    stream=iter(ds)
    try:
        rows=list(islice(stream,skip,skip+n))
    finally:
        if hasattr(stream,'close'):stream.close()
    del stream,ds
    gc.collect()
    assert len(rows)==n
    word=re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
    counts={kind:Counter() for kind in ('noun','proper','adjective')}
    for row in rows:
        for token,tag in nltk.pos_tag(word.findall(row['text'][:chars])):
            if tag in ('NN','NNS'):
                counts['noun'][token.lower()]+=1
            elif tag in ('NNP','NNPS'):
                counts['proper'][token]+=1
            elif tag.startswith('JJ'):
                counts['adjective'][token.lower()]+=1
    return counts


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--skip',type=int,default=32000)
    p.add_argument('--n',type=int,default=512)
    p.add_argument('--chars',type=int,default=5000)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    groups=candidate_lists()
    counts=count_pos(a.skip,a.n,a.chars)
    table={}
    for kind,words in groups.items():
        source=('proper' if kind=='people' else
                'adjective' if kind=='adjectives' else 'noun')
        table[kind]=[{"word":word,"dev_pos_count":int(
            counts[source].get(word,0))} for word in words]
    result=dict(source=dict(dataset='HuggingFaceFW/fineweb sample-10BT',
                            split='train',offset=a.skip,n_documents=a.n,
                            chars_per_document=a.chars,
                            measure='NLTK POS-tagged alphabetic-word counts'),
                categories=table,
                caveat='Candidate lists were manually reviewed by role. '
                       'POS tag counts are noisy and only provide a heavy-tail '
                       'prior; no source passages are stored. A separate '
                       'FineWeb offset must be used for evaluation.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print({kind:len(rows) for kind,rows in table.items()},flush=True)
    sys.stdout.flush()
    # Streaming dataset worker may otherwise wait at interpreter shutdown.
    os._exit(0)


if __name__=='__main__':main()
