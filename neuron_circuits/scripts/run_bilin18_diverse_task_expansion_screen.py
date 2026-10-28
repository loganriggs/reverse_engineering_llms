"""H1039 prospective fixed 20-family competence and alignment screen."""
from pathlib import Path
import sys,json,hashlib,ast,numpy as np,torch
from transformers import AutoTokenizer
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_diverse_task_expansion_screen';out=A/'diverse_task_expansion_screen';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import bilin18_observed_model_facade as facade
import run_subject_number_sparse_graph_token_extraction_v1 as runner
F=torch.nn.functional;torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
tok=AutoTokenizer.from_pretrained('/home/loganriggs/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e',local_files_only=True)
specs=[]
def add(family,split,template,items):
 assert len(items)==8
 endpoints=[(template.format(x=x),y) for x,y in items]
 specs.append((family,split,endpoints))
add('antonym','train','The opposite of {x} is',[('hot','cold'),('cold','hot'),('big','small'),('small','big'),('up','down'),('down','up'),('good','bad'),('bad','good')])
add('plural_noun','train','One {x}, two',[('cat','cats'),('dog','dogs'),('book','books'),('car','cars'),('bird','birds'),('tree','trees'),('house','houses'),('shoe','shoes')])
add('past_tense','train','Today I {x}. Yesterday I',[('walk','walked'),('play','played'),('jump','jumped'),('talk','talked'),('work','worked'),('look','looked'),('go','went'),('eat','ate')])
add('comparative','train','The comparative form of {x} is',[('big','bigger'),('small','smaller'),('tall','taller'),('short','shorter'),('old','older'),('young','younger'),('fast','faster'),('slow','slower')])
add('animal_sound','train','The sound made by a {x} is',[('dog','bark'),('cat','meow'),('cow','moo'),('sheep','baa'),('duck','quack'),('lion','roar'),('snake','hiss'),('bee','buzz')])
add('object_color','train','The usual color of {x} is',[('snow','white'),('coal','black'),('grass','green'),('blood','red'),('a banana','yellow'),('an orange','orange'),('the sky','blue'),('milk','white')])
add('country_language','train','The main language spoken in {x} is',[('Italy','Italian'),('Japan','Japanese'),('France','French'),('China','Chinese'),('Germany','German'),('Spain','Spanish'),('Russia','Russian'),('Greece','Greek')])
add('number_successor','train','{x},',[('1, 2, 3','4'),('2, 3, 4','5'),('3, 4, 5','6'),('4, 5, 6','7'),('5, 6, 7','8'),('6, 7, 8','9'),('7, 8, 9','10'),('8, 9, 10','11')])
add('addition','train','{x} =',[('2 + 3','5'),('3 + 3','6'),('4 + 2','6'),('5 + 2','7'),('1 + 7','8'),('2 + 7','9'),('6 + 4','10'),('7 + 4','11')])
add('subtraction','train','{x} =',[('8 - 3','5'),('9 - 3','6'),('7 - 2','5'),('6 - 2','4'),('10 - 1','9'),('10 - 2','8'),('9 - 5','4'),('8 - 5','3')])
add('boolean_negation','train','{x}',[('The opposite of true is','false'),('The opposite of false is','true'),('Not true means','false'),('Not false means','true'),('Boolean NOT true equals','false'),('Boolean NOT false equals','true'),('The logical negation of true is','false'),('The logical negation of false is','true')])
add('month_successor','train','The month after {x} is',[('January','February'),('February','March'),('March','April'),('April','May'),('May','June'),('June','July'),('July','August'),('August','September')])
add('weekday_successor','development','The day after {x} is',[('Monday','Tuesday'),('Tuesday','Wednesday'),('Wednesday','Thursday'),('Thursday','Friday'),('Friday','Saturday'),('Saturday','Sunday'),('Sunday','Monday'),('Sunday night','Monday')])
add('ordinal_word','development','The ordinal form of {x} is',[('one','first'),('two','second'),('three','third'),('four','fourth'),('five','fifth'),('six','sixth'),('seven','seventh'),('eight','eighth')])
add('animal_category','development','A {x} is a type of',[('robin','bird'),('salmon','fish'),('eagle','bird'),('trout','fish'),('sparrow','bird'),('shark','fish'),('pigeon','bird'),('tuna','fish')])
add('synonym','development','A synonym for {x} is',[('large','big'),('small','little'),('quick','fast'),('silent','quiet'),('happy','glad'),('angry','mad'),('clever','smart'),('ill','sick')])
add('superlative','evaluation','The superlative form of {x} is',[('big','biggest'),('small','smallest'),('tall','tallest'),('short','shortest'),('old','oldest'),('young','youngest'),('fast','fastest'),('slow','slowest')])
add('multiplication','evaluation','{x} =',[('2 * 3','6'),('3 * 3','9'),('4 * 2','8'),('5 * 2','10'),('3 * 4','12'),('4 * 4','16'),('5 * 3','15'),('6 * 3','18')])
add('past_participle','evaluation','The past participle of {x} is',[('go','gone'),('eat','eaten'),('write','written'),('speak','spoken'),('take','taken'),('give','given'),('see','seen'),('do','done')])
add('unit_completion','evaluation','{x}',[('Sixty seconds make one','minute'),('Sixty minutes make one','hour'),('Twenty-four hours make one','day'),('Seven days make one','week'),('Twelve months make one','year'),('One hundred years make one','century'),('One thousand meters make one','kilometer'),('One hundred centimeters make one','meter')])
assert len(specs)==20
manifest=[];examples={};pairs=[];invalid=[]
for family,split,endpoints in specs:
 for n in range(4):
  pp=endpoints[2*n:2*n+2];row=dict(id=f'{family}-{n}',family=family,split=split,prompts=[x[0] for x in pp],answers=[x[1] for x in pp]);manifest.append(row);ee=[]
  for variant,(text,answer) in enumerate(pp):
   ids=tok.encode(text,add_special_tokens=False);groups=[]
   for _,a in pp:
    gg=[]
    for form in dict.fromkeys([a,a.capitalize(),a.upper()]):
     comb=tok.encode(text+' '+form,add_special_tokens=False)
     if len(comb)==len(ids)+1 and comb[:len(ids)]==ids:gg.append(comb[-1])
    groups.append(sorted(set(gg)))
   if any(not g for g in groups) or set(groups[0])&set(groups[1]):break
   ee.append(dict(id=row['id'],variant=variant,text=text,ids=ids,groups=groups))
  if len(ee)!=2:invalid.append(row|dict(reason='No disjoint single-token answer groups'));continue
  ii=[len(examples),len(examples)+1];examples.update(zip(ii,ee));pairs.append(row|dict(indices=ii,equal_length=len(ee[0]['ids'])==len(ee[1]['ids']),same_final_token=ee[0]['ids'][-1]==ee[1]['ids'][-1]))
assert len({e['text'] for e in examples.values()})==len(examples)
config=dict(experiment='H1039',families=20,declared_pairs=80,split_assignment='12 train,4development,4evaluation whole families fixed before forward. Related domains cross splits; not a claim of unseen primitives.',gate='Per family at least4 token-valid equal-length pairs, >=75% both-endpoint binary accuracy, >=50% intended full-vocabulary top1. Training eligibility determined only by train cohort.',answer_policy='Single-token exact continuation groups for original/capitalized/uppercase answer. Alternatives not listed can make open-vocabulary accuracy conservative.',historical_scope='New candidate prompts for current five-family training expansion; some task types/lexical items were studied previously. No claim of globally novel tasks.',evaluation_scope='Native competence screened on all splits; evaluation prompts thus opened for competence, not untouched. No representation fitted or selected.',alignment='Equal token lengths permit positionwise patching but do not prove semantic alignment.',historical_exclusions=['sequence_copy','spatial_inverse'])
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n');(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');mh=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest()
model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True)
tree=ast.parse((R/'scripts/run_bilin18_projection_stream_factorial.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['suffix','KL','margin']];exec(compile(ast.Module(body=nodes,type_ignores=[]),'<audited-functions>','exec'),globals())
states={};native={}
with torch.no_grad():
 for i,e in examples.items():
  initial=F.rms_norm(model.transformer.wte(torch.tensor([e['ids']],device='cuda')),(1152,)).float();raw,x0,bus,_,_=runner.component._capture(model,initial,torch,F);st=(raw,x0,bus);native[i]=suffix(st).cpu();states[i]=tuple(v.cpu() for v in st)
records=[]
for p in pairs:
 i,d=p['indices'];correct=[margin(native[k],examples[k]['groups'])*(1 if v==0 else -1)>0 for v,k in enumerate([i,d])];top=[int(native[k].argmax()) in examples[k]['groups'][v] for v,k in enumerate([i,d])];records.append(p|dict(both_correct=all(correct),full_vocab=float(np.mean(top)),margins=[margin(native[k],examples[k]['groups']) for k in [i,d]],top_tokens=[tok.decode([int(native[k].argmax())]) for k in [i,d]]))
summary=[]
for f,sp,_ in specs:
 rr=[r for r in records if r['family']==f and r['equal_length']];both=float(np.mean([r['both_correct'] for r in rr])) if rr else 0;top=float(np.mean([r['full_vocab'] for r in rr])) if rr else 0;passed=len(rr)>=4 and both>=.75 and top>=.5
 summary.append(dict(family=f,split=sp,valid_aligned_pairs=len(rr),both_endpoint_accuracy=both,top1=top,pass_gate=passed,training_eligible=sp=='train' and passed))
# Independent direct full forward at the same precision for one exemplar per family.
replay=0.
with torch.no_grad():
 for f,_,_ in specs:
  matches=[i for i,e in examples.items() if e['id']==f+'-0']
  if not matches:continue
  i=matches[0];x0=F.rms_norm(model.transformer.wte(torch.tensor([examples[i]['ids']],device='cuda')),(1152,)).float();x=x0;bus=None
  for b in model.transformer.h:x,bus=b(x,bus,x0)
  y=(30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1])/30))[0].double().cpu();replay=max(replay,float((y-native[i]).abs().max()))
audit=dict(manifest_unchanged=mh==hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(),manifest_sha256=mh,native_direct_replay_max=replay,prefix_sequences=len(examples),suffix_sequences=len(examples),fit_rows=0,all_split_prompts_unique=True,finite=all(bool(torch.isfinite(v).all()) for v in native.values()))
assert replay<1e-5 and audit['manifest_unchanged'] and audit['finite']
torch.save(dict(states=states,native=native,examples=examples),out/'states.pt');(A/f'{S}.json').write_text(json.dumps(dict(config=config,summary=summary,records=records,invalid=invalid,audits=audit),indent=2)+'\n');print(json.dumps(dict(summary=summary,invalid_pairs=len(invalid),audits=audit),indent=2))
