"""Short same-vocabulary curriculum for v8 section-key retrieval.

Each grammar-valid prefix has two numbered sections with distinct badge
colors; both section queries are supervised. This is a capability diagnostic,
not a realistic sample from the full v8 language distribution.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from surface_mixture_v7 import COLORS, SECTION_IDS, TOPICS
from surface_mixture_v8 import SurfaceMixtureV8


def generator(seed):
    return SurfaceMixtureV8(seed=seed, section_rate=.4, detail_rate=.8,
                            scope_question_rate=.5, question_rate=.05,
                            quote_rate=.08, source_rate=.15,
                            counterbalanced_queries=True)


def add_supported(gen, tokens, word):
    if gen.next_distribution(tokens).get(word, 0) <= 0:
        raise AssertionError(f'Unsupported {word!r} after {tokens!r}')
    tokens.append(word)


def make_world(gen, rng):
    """Return two supported answer prefixes sharing one sampled document."""
    for _ in range(100):
        base=[]
        for _ in range(100):
            q=gen.next_distribution(base)
            word=rng.choices(list(q),weights=list(q.values()))[0]
            base.append(word)
            if word=='.':
                break
        if base and base[-1]=='.' and len(base)<100:
            break
    else:
        raise RuntimeError('Could not sample a short starter clause')
    add_supported(gen,base,'\n')
    section_ids=rng.sample(SECTION_IDS,2)
    colors=rng.sample(COLORS,2)
    topics=[rng.choice(TOPICS) for _ in range(2)]
    for i,(section,color,topic) in enumerate(zip(section_ids,colors,topics)):
        if i:
            add_supported(gen,base,'\n')
        for word in ('Section',section,':',topic,'.','Detail',':',
                     'the','badge','was',color,'.'):
            add_supported(gen,base,word)
    answer_prefixes=[]
    for section,color in zip(section_ids,colors):
        prefix=base.copy()
        for word in ('What','color','was','the','badge','in','section',
                     section,'?','Answer',':'):
            add_supported(gen,prefix,word)
        assert gen.next_distribution(prefix)=={color:1.}
        answer_prefixes.append((prefix,color))
    return answer_prefixes


def make_batch(gen,rng,worlds,device):
    examples=[example for _ in range(worlds) for example in make_world(gen,rng)]
    lengths=torch.tensor([len(tokens) for tokens,_ in examples],
                         dtype=torch.long,device=device)
    maximum=int(lengths.max())
    x=torch.full((len(examples),maximum),gen.token2id['.'],
                 dtype=torch.long,device=device)
    y=torch.tensor([gen.token2id[color] for _,color in examples],
                   dtype=torch.long,device=device)
    for row,(tokens,_) in enumerate(examples):
        x[row,:len(tokens)]=torch.tensor([gen.token2id[w] for w in tokens],
                                         dtype=torch.long,device=device)
    return x,lengths,y,examples


@torch.no_grad()
def evaluate(model,gen,rng,nworlds,device,batch_worlds=32):
    model.eval()
    correct=0
    total=0
    ce_total=0.
    key_tv_total=0.
    examples=[]
    for start in range(0,nworlds,batch_worlds):
        x,lengths,y,raw=make_batch(gen,rng,min(batch_worlds,nworlds-start),device)
        logits=model(x)[0][torch.arange(len(x),device=device),lengths-1]
        probs=logits.softmax(-1)
        correct+=int((probs.argmax(-1)==y).sum())
        ce_total+=float(F.cross_entropy(logits,y,reduction='sum'))
        key_tv_total+=float((probs[0::2]-probs[1::2]).abs().sum(-1).sum()/2)
        total+=len(x)
        if len(examples)<2:
            examples.append(dict(prefix=' '.join(raw[0][0]).replace(' \n ','\n'),
                                 answer=raw[0][1]))
    model.train()
    return dict(n_answers=total,accuracy=correct/total,
                answer_ce=ce_total/total,query_key_edit_tv=key_tv_total/(total/2),
                query_blind_accuracy=.5,query_blind_ce=.6931471805599453,
                examples=examples)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-config-dir',default='language_task_suite/v8_scope_depth3_seed1')
    p.add_argument('--steps',type=int,default=1500)
    p.add_argument('--batch-worlds',type=int,default=32)
    p.add_argument('--eval-worlds',type=int,default=512)
    p.add_argument('--eval-every',type=int,default=250)
    p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--device',default='cpu')
    p.add_argument('--output-dir',required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0,a.source)
    from model import BilinearGPT,Config
    saved=json.loads((Path(a.model_config_dir)/'results.json').read_text())
    config=saved['records'][0]['model_config']
    assert config['n_layer']==3 and config['vocab_size']==len(generator(0).vocab)
    torch.manual_seed(a.seed)
    model=BilinearGPT(Config(**config)).to(a.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=a.lr)
    train_gen=generator(41000+a.seed)
    train_rng=random.Random(42000+a.seed)
    output_dir=Path(a.output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    records=[]
    best=(float('inf'),-1)
    for step in range(1,a.steps+1):
        model.train()
        x,lengths,y,_=make_batch(train_gen,train_rng,a.batch_worlds,a.device)
        optimizer.zero_grad()
        logits=model(x)[0][torch.arange(len(x),device=a.device),lengths-1]
        loss=F.cross_entropy(logits,y)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite short-scope loss at {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        optimizer.step()
        if step%a.eval_every==0 or step==a.steps:
            dev=evaluate(model,generator(43000+a.seed),
                         random.Random(44000+a.seed),a.eval_worlds,a.device)
            row=dict(step=step,train_ce=float(loss.detach()),dev=dev)
            records.append(row)
            print(step,'train',round(row['train_ce'],4),
                  'dev',round(dev['answer_ce'],4),
                  'accuracy',round(dev['accuracy'],4),
                  'key_tv',round(dev['query_key_edit_tv'],4),flush=True)
            if dev['answer_ce']<best[0]:
                best=(dev['answer_ce'],step)
                torch.save(model.state_dict(),output_dir/'model_depth3.pt')
    # Reopen the independent test only after the development-selected step.
    selected=BilinearGPT(Config(**config)).to(a.device)
    selected.load_state_dict(torch.load(output_dir/'model_depth3.pt',
                                        map_location=a.device,weights_only=True))
    test=evaluate(selected,generator(45000+a.seed),
                  random.Random(46000+a.seed),a.eval_worlds,a.device)
    result=dict(config=vars(a),model_config=config,
                generator_kwargs=saved['generator_kwargs'],
                records=records,best_dev_ce=best[0],selected_step=best[1],
                test=test,
                caveat='Short two-section answer-only curriculum. Both query '
                       'keys are supervised for every sampled world. Section '
                       'and color pairings are not held out from training. '
                       'Generalization to long mixed v8 streams is untested.')
    (output_dir/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('selected',best[1],'test',test['accuracy'],test['query_key_edit_tv'],flush=True)


if __name__=='__main__':
    main()
