"""M83 descriptive instrumentation summaries, no causal attribution."""
import json,math
from fit_pca_encoder import sha,V

def main():
 rp=V/'bilinear_input_optimization_replay.json';r=json.loads(rp.read_text());assert r['status']=='completed_exact_replays';summaries=[]
 for m in r['models']:
  assert m['history_exact'] and m['final_state_exact'] and m['optimizer_state_exact'];rows=m['rows'];assert len(rows)==1000
  growth=[x for x in rows if x['after']['error']>x['before']['error']];rare=[x for x in rows if x['rare_count']];largest=sorted(rows,key=lambda x:x['after']['error']-x['before']['error'],reverse=True)[:10]
  summaries.append(dict(kind=m['kind'],seed=m['seed'],rare_batches=len(rare),rare_total_draws=sum(x['rare_count'] for x in rows),support_changes=sum(x['support_changed'] for x in rows),initial_rare_error=rows[0]['before']['error'],final_rare_error=rows[-1]['after']['error'],maximum_rare_error=max(x['after']['error'] for x in rows),growing_steps=len(growth),growing_steps_without_rare=sum(x['rare_count']==0 for x in growth),rare_appearance_error_reductions=sum(x['after']['error']<x['before']['error'] for x in rare),top10_growth_without_rare=sum(x['rare_count']==0 for x in largest),top10_growth_support_changes=sum(x['support_changed'] for x in largest),largest_growth=[dict(step=x['step'],rare_count=x['rare_count'],before=x['before']['error'],after=x['after']['error'],support_changed=x['support_changed'],gradient_norm=math.sqrt(sum(v*v for v in x['gradient_norms'].values())),update_norm=math.sqrt(sum(v*v for v in x['update_norms'].values()))) for x in largest]))
 out=dict(status='completed',source_sha256=sha(__file__),replay_sha256=sha(rp),summaries=summaries,interpretation='Dense transient explosions show hard gating is not necessary. Rare examples are intermittently corrected while other updates increase their error; shared support changes accompany some jumps. Momentum carryover is a candidate explanation, not yet isolated. No general superiority or optimized architecture claim.')
 p=V/'bilinear_input_optimization_replay_analysis.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
