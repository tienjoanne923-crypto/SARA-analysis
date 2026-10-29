exec((__import__('pathlib').Path(__file__).parent/'external_family_audit.py').read_text(encoding='utf-8').split('# Upstream and frozen external truth funnel')[0])
cross=pd.read_csv(OUT/'strict_equivalence_cross_roles.csv')
excluded=sorted(set(cross.disease_id)&set(fold_of))
q=pd.read_parquet(ROOT/'results_v8/pub/query_ranks.parquet')
methods=['deployed','channel_lgb_v6','v71_direct_strict','B']
q=q[q.method.isin(methods)].copy();q['hit10']=(q['rank']>0)&(q['rank']<=10)
perseed=q.groupby(['method','seed','disease_id']).agg(n_query=('gene_row','size'),hits10=('hit10','sum')).reset_index()
perseed['recall10']=perseed.hits10/perseed.n_query
per=perseed.groupby(['method','disease_id']).agg(n_query=('n_query','first'),recall10=('recall10','mean')).reset_index()
saved=pd.read_parquet(ROOT/'results_v8/eval/eval_test_main.parquet')
saved=saved[saved.method.isin(methods)].groupby(['method','disease_id']).recall10.mean().reset_index()
cmp=per.merge(saved,on=['method','disease_id'],suffixes=('_rank','_saved'))
node=next(n for n in ast.parse((ROOT/'src/gpu8v71/evaluate71.py').read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='fast_family_bootstrap')
scope={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'fast_family_bootstrap','exec'),scope);boot=scope['fast_family_bootstrap']
fam=dict(zip(ds.disease_id,ds.family_group));tables=[];contrasts=[]
for name,pdsub in [('original',per),('exclude_cross_role_equivalence_groups',per[~per.disease_id.isin(excluded)])]:
    for m,g in pdsub.groupby('method'):tables.append({'analysis':name,'method':m,'n_diseases':g.disease_id.nunique(),'n_pairs':int(g.n_query.sum()),'n_families':len({fam[d] for d in g.disease_id}),'recall10':float(g.recall10.mean())})
    a=pdsub[pdsub.method=='deployed'].set_index('disease_id').recall10
    for m in ['channel_lgb_v6','v71_direct_strict','B']:
        b=pdsub[pdsub.method==m].set_index('disease_id').recall10;ix=a.index.intersection(b.index);delta=(a[ix]-b[ix]).to_numpy();lo,hi=boot(delta,np.array([fam[d] for d in ix]))
        contrasts.append({'analysis':name,'a':'deployed','b':m,'n_diseases':len(ix),'n_families':len({fam[d] for d in ix}),'delta':float(delta.mean()),'ci_low':lo,'ci_high':hi})
per[per.disease_id.isin(excluded)].merge(ds[['disease_id','disease_name','equivalence_group','family_group']],on='disease_id').to_csv(OUT/'cross_role_equivalence_excluded_main_diseases.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(tables).to_csv(OUT/'main_equivalence_exclusion_sensitivity_table.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(contrasts).to_csv(OUT/'main_equivalence_exclusion_sensitivity_contrasts.csv',index=False,encoding='utf-8-sig')
out={'status':'post hoc denominator sensitivity on existing saved predictions, no retraining; does not correct training-context overlap','excluded_test_ids':excluded,'n_excluded_test_diseases':len(excluded),'n_excluded_test_positive_pairs':int(per[(per.method=='deployed')&per.disease_id.isin(excluded)].n_query.sum()),'max_abs_reproduction_error_rank_vs_archived_eval':float((cmp.recall10_rank-cmp.recall10_saved).abs().max()),'table':tables,'contrasts':contrasts}
(OUT/'main_equivalence_exclusion_sensitivity.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
