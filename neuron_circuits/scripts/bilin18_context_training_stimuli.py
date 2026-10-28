"""Eight fixed contexts/family, source and context splits declared before screening."""
from run_bilin18_expanded_task_bank import suite as old_suite
from bilin18_broader_task_stimuli import suite as broader_suite,BANK
FAMILIES=['agreement','sentiment','negation','entity_binding','pronoun_gender','capital_recall','antonym','country_language']
def suite():
 base=[r for r in old_suite()+broader_suite() if r['family'] in FAMILIES and r['format']=='A'];assert len(base)==64
 countries=[('France','Paris'),('Germany','Berlin'),('Italy','Rome'),('Japan','Tokyo'),('Spain','Madrid'),('China','Beijing'),('Russia','Moscow'),('England','London'),('Portugal','Lisbon'),('Greece','Athens'),('Austria','Vienna'),('Poland','Warsaw'),('Egypt','Cairo'),('Kenya','Nairobi'),('Thailand','Bangkok'),('Norway','Oslo')]
 cap_templates=['The capital city of {x} is','The national capital of {x} is','The city serving as the capital of {x} is','In {x}, the capital city is','The country {x} has its capital in','The government of {x} is based in its capital,','The capital of the country of {x} is','When visiting the capital of {x}, one visits']
 lang_templates=['The primary language spoken in {x} is','The principal language of {x} is','In {x}, most people speak','The official language of {x} is','The people of {x} generally speak','The dominant language in {x} is','The language most widely spoken in {x} is','Residents of {x} commonly speak']
 opposite_templates=['The usual opposite of {x} is','The direct opposite of {x} is','A word meaning the opposite of {x} is','In meaning, the opposite of {x} is','The word opposite in meaning to {x} is','The opposite meaning of {x} is expressed by','An opposite word for {x} is','An antonym for the word {x} is']
 rows=[]
 for r in base:
  f=r['family'];i=r['source']
  for c in range(8):
   ans=r['answers'];ps=[]
   if f in ['capital_recall','country_language','antonym']:
    items=countries if f=='capital_recall' else BANK[f][0];template={'capital_recall':cap_templates,'country_language':lang_templates,'antonym':opposite_templates}[f][c];pair=items[2*i:2*i+2];ps=[template.format(x=x) for x,y in pair];ans=[y for x,y in pair]
   else:
    for p in r['prompts']:
     if f=='agreement':p=p.replace('near the window',['near the bridge','behind the fence','beside the tree','by the river','inside the house','outside the building','under the roof','across the street'][c])
     elif f in ['sentiment','negation']:p=p.replace('film',['movie','book','story','play','novel','meal','concert','game'][c])
     elif f=='pronoun_gender':p=p.replace('near the',['behind the','beside the','close to the','in front of the','far from the','next to the','opposite the','across from the'][c]).replace('said that',['said that','said that','mentioned that','mentioned that','reported that','reported that','thought that','thought that'][c])
     elif f=='entity_binding':p=p.replace('store',['library','school','cafe','office','station','museum','garden','market'][c]).replace('gave the book to',['gave the letter to','gave the bag to','gave the gift to','gave the key to','handed the letter to','handed the bag to','handed the gift to','handed the key to'][c])
     ps.append(p)
   rows.append(dict(id=f'{f}-context{c}-{i}',family=f,format=f'context{c}',context=c,source=i,split=('discovery' if i<4 else 'lexical') if c<4 else ('structural' if i<4 else 'joint'),whole_task_holdout=False,prompts=ps,answers=ans,intervention_variable=r['intervention_variable']))
 # Permit within-training overlaps between task labels, never cross the training boundary.
 train={p for r in rows if r['split']=='discovery' for p in r['prompts']};validation={p for r in rows if r['split']!='discovery' for p in r['prompts']};assert not train&validation
 return rows
