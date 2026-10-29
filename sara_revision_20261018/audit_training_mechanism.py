from __future__ import annotations
import sys, json, csv, re, hashlib
from pathlib import Path
sys.path.insert(0, 'D:/信/sara_revision_20261014/lib')
import numpy as np
import pandas as pd

BASE = Path('D:/信/gpu8')
OUT = Path('D:/信/sara_revision_20261018/audit/training_mechanism')
OUT.mkdir(parents=True, exist_ok=True)
R = BASE/'results_v8'

def write_json(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

# Checkpoint records and actual joint-stage histories. No fitting is performed.
teachers=[]
for seed in (42,52,62):
    for j in range(5):
        for c in [f'f{j}_inner{b}' for b in range(3)] + [f'f{j}_outer']:
            rd=BASE/'runs_v8'/f'V8T__main__{c}__background__s{seed}'
            res=json.loads((rd/'result.json').read_text(encoding='utf-8'))
            hist=json.loads((rd/'history.json').read_text(encoding='utf-8'))
            joint=[h for h in hist if h['stage']=='joint']
            picked=next(h for h in joint if h['epoch']==res['fit']['best_epoch'])
            teachers.append({'context':c,'fold':j,'seed':seed,'kind':res['context_id'].split('_')[1],
                             'run':res['run_id'],'checkpoint_metric':res['config']['checkpoint'],
                             'fit_disease_hash':res['fit_disease_hash'],**res['fit'],
                             'first_sF_recall10':joint[0]['sF_recall10'],
                             'picked_sF_recall10':picked['sF_recall10'],
                             'first_r_recall10':joint[0]['r_recall10'],
                             'picked_r_recall10':picked['r_recall10'],
                             'max_r_recall10':max(h['r_recall10'] for h in joint),
                             'sha256_result':hashlib.sha256((rd/'result.json').read_bytes()).hexdigest()})
pd.DataFrame(teachers).to_csv(OUT/'teacher_checkpoint_records.csv',index=False,encoding='utf-8-sig')

query=pd.read_parquet(R/'pub/query_context.parquet')
bg=pd.read_parquet(R/'pub/genes.parquet')
family=pd.read_parquet(R/'pub/diseases.parquet')[['disease_id','family_group']]
bud=pd.read_parquet(R/'pub/budget_ranks.parquet')
query=query.merge(bg[['gene_row','has_go','gtype']],on='gene_row',validate='many_to_one')
query=query.merge(family,on='disease_id',validate='many_to_one')
query=query.merge(bud[['fold','disease_id','gene_row','m','r_aware','r_plain']],
                  on=['fold','disease_id','gene_row'],validate='one_to_one')

specs=[('retrieval','formal/main','channel_lgb_v6'),('aware','formal/main','deployed'),
       ('plain','variants/plain_rrf','deployed')]
for short,sub,meth in specs:
    q_ranks={};q_pool={};counts=[]
    for j in range(5):
        p=R/sub/'s42'/f'fold{j}'/f'pred_{meth}__test.npz'
        z=np.load(p)
        ids=[str(x) for x in z['ids']]
        for i,d in enumerate(ids):
            genes=z['genes'][i].astype(np.int64); scores=z['score'][i].astype(float)
            valid=np.isfinite(scores)&(genes>=0)
            order=np.lexsort((genes[valid],-scores[valid]))
            sorted_genes=genes[valid][order]
            rank={int(g):r+1 for r,g in enumerate(sorted_genes)}
            counts.append(len(sorted_genes))
            for g in query.loc[query['disease_id']==d,'gene_row']:
                q_ranks[(d,int(g))]=rank.get(int(g),0)
    query[f'{short}_final_rank']=[q_ranks[(d,int(g))] for d,g in zip(query.disease_id,query.gene_row)]
    query[f'{short}_in500']=query[f'{short}_final_rank']>0
    query[f'{short}_hit10']=query[f'{short}_final_rank'].between(1,10)
    assert set(counts)=={500},(short,set(counts))

assert np.array_equal(query['rB'].le(500),query['retrieval_in500'])
assert np.array_equal(query['r_aware'].le(500),query['aware_in500'])
assert np.array_equal(query['r_plain'].le(500),query['plain_in500'])
query.to_csv(OUT/'positive_relation_fates_seed42.csv',index=False,encoding='utf-8-sig')

masks={
    'all':np.ones(len(query),bool),
    'semantic_unavailable':~query['m'],
    'natural_no_GO':~query['has_go'],
    'ncRNA':query['gtype'].eq('ncrna'),
    'natural_no_GO_protein':(~query['has_go'])&query['gtype'].eq('protein'),
    'reference_path_cold':query['cold'],
}
summary=[]; fate=[]
for group,mask in masks.items():
    q=query[mask]; n=len(q)
    for method in ('retrieval','plain','aware'):
        rank=q[f'{method}_final_rank']; inc=q[f'{method}_in500']; hit=q[f'{method}_hit10']
        old=q['retrieval_in500']; retained=old&inc; rescued=~old&inc; extruded=old&~inc
        summary.append({'group':group,'method':method,'seed':42,'n_relations':n,
                        'n_diseases':q.disease_id.nunique(),'n_families':q.family_group.nunique(),
                        'candidate_positive_count':int(inc.sum()),'candidate_coverage_micro':float(inc.mean()),
                        'top10_positive_count':int(hit.sum()),'top10_recovery_micro':float(hit.mean()),
                        'top10_given_candidate':float(hit.sum()/inc.sum()) if inc.sum() else None,
                        'candidate_positive_final_rank_median':float(rank[inc].median()),
                        'candidate_positive_final_rank_q1':float(rank[inc].quantile(.25)),
                        'candidate_positive_final_rank_q3':float(rank[inc].quantile(.75)),
                        'retained_from_retrieval500':int(retained.sum()),
                        'rescued_from_outside_retrieval500':int(rescued.sum()),
                        'extruded_from_retrieval500':int(extruded.sum()),
                        'retained_top10':int((retained&hit).sum()),'rescued_top10':int((rescued&hit).sum())})
        for state,state_mask in [('retained',retained),('rescued',rescued),('extruded',extruded),('neither',~old&~inc)]:
            fate.append({'group':group,'method':method,'state_vs_retrieval500':state,'seed':42,
                         'n_relations':int(state_mask.sum()),'top10_positive_count':int((state_mask&hit).sum())})
pd.DataFrame(summary).to_csv(OUT/'missing_subgroup_candidate_to_top10_seed42.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(fate).to_csv(OUT/'missing_subgroup_fates_seed42.csv',index=False,encoding='utf-8-sig')

# Actual comparable method IDs, not inferred from rounded presentation.
ev=pd.read_parquet(R/'eval/eval_test_main.parquet')
budgetev=pd.read_parquet(R/'eval/eval_test_v6pool.parquet')
identity=[]
for location,df,method,seeds in [('Fig3e',ev,'v71_direct_v6',[42,52,62]),
                               ('Table4',budgetev,'deployed',[42]),
                               ('Table4_direct_fusion_reference',ev,'v71_direct_v6',[42])]:
    s=df[df.method.eq(method)&df.seed.isin(seeds)].groupby('disease_id').recall10.mean()
    identity.append({'location':location,'method_id':method,'seeds':seeds,'n_diseases':len(s),
                     'recall10':float(s.mean()),'source':str((R/'eval'/('eval_test_v6pool.parquet' if df is budgetev else 'eval_test_main.parquet'))),
                     'candidate_rule':'retrieval top 500','features':49 if method=='deployed' else 44,
                     'missing_encoding':'NaN' if method=='deployed' else 'zero'})
write_json('figure3e_table4_identity.json',identity)

# Archived ranker headers prove the early stopping metric passed to LightGBM.
metric=[]
for j in range(5):
    p=R/'dev'/f'fold{j}'/'selection.json'
    sel=json.loads(p.read_text(encoding='utf-8'))
    name=sel['ab']['best_A']
    modelp=R/'dev'/f'fold{j}'/f'A_{name}_model.txt'
    lines=modelp.read_text(encoding='utf-8').splitlines()
    match=[{'line':k+1,'text':s} for k,s in enumerate(lines) if s.startswith(('[metric:', '[objective:', '[eval_at:'))]
    metric.append({'fold':j,'selected_name':name,'model':str(modelp),'headers':match})
write_json('ranker_actual_metric.json',metric)

# E6 actual training exclusion counts in saved run evidence.
e6=[]
for arm in ['main','e5_testmask','e6_cold','e6_cold_gomask']:
    for j in range(5):
        rd=BASE/'runs_v8'/f'V8T__{arm}__f{j}_outer__background__s42'
        if rd.exists():
            v=json.loads((rd/'result.json').read_text(encoding='utf-8'))
            e6.append({'arm':arm,'fold':j,**v['fit']})
pd.DataFrame(e6).to_csv(OUT/'withdrawal_saved_training_counts.csv',index=False,encoding='utf-8-sig')

print(json.dumps({'teacher_records':len(teachers),'subgroup_rows':summary,
                  'figure_identity':identity,'ranker_metric':metric[:1],
                  'withdrawal_saved_counts':e6[:4]},ensure_ascii=False,indent=2))
