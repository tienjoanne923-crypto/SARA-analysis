exec((__import__('pathlib').Path(__file__).parent/'external_family_audit.py').read_text(encoding='utf-8').split('# Upstream and frozen external truth funnel')[0])
from collections import Counter
cv=pd.read_csv(OUT/'clinvar_frozen_confirmation_record_provenance.csv',dtype=str).fillna('')
exact=cv.clinical_significance.isin(['Pathogenic','Likely pathogenic','Pathogenic/Likely pathogenic'])
pure=cv.origin=='germline'
gc=pairs[pairs.source=='GenCC'][['disease_id','gene_id']].drop_duplicates()
cvkeys={n:set(zip(x.disease_id,x.gene_id)) for n,x in [('frozen_original',cv),('exact_PLP',cv[exact]),('pure_germline_exact_PLP',cv[exact&pure])]}
truths={}
funnel=[]
for name,keys in cvkeys.items():
    for subset in ['ClinVar','union']:
        kk=keys if subset=='ClinVar' else keys|set(zip(gc.disease_id,gc.gene_id.astype(str)))
        by=defaultdict(set)
        for d,g in kk: by[d].add(gix[g])
        truths[(name,subset)]=by
        funnel.append({'definition':name,'source':subset,'n_pairs':len(kk),'n_diseases':len(by)})
source=ROOT/'src/gpu8v71/evaluate71.py'
tree=ast.parse(source.read_text(encoding='utf-8'))
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fast_family_bootstrap')
scope={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
boot=scope['fast_family_bootstrap'];rows=[]
for method in ['deployed','B','channel_lgb_v6','v71_direct_strict','F7_v6']:
  for seed in [42,52,62]:
    for j in range(5):
      with np.load(ROOT/f'results_v8/formal/main/s{seed}/fold{j}/pred_{method}__test.npz') as z:
        ix={str(x):i for i,x in enumerate(z['ids'])};gg=z['genes'];ss=z['score'];vv=z['valid'] if 'valid' in z.files else np.isfinite(ss)
        for (name,subset),by in truths.items():
          for d,q in by.items():
            if fold_of[d]!=j:continue
            i=ix[d];g=gg[i][vv[i]].astype(int);s=ss[i][vv[i]].astype(float)
            o=g[np.lexsort((g,-s))];o=o[~np.isin(o,list(pos_rows.get(d,set())))]
            rows.append({'definition':name,'source':subset,'method':method,'seed':seed,'disease_id':d,'n':len(q),'recall10':len(q&set(o[:10]))/len(q)})
raw=pd.DataFrame(rows)
per=raw.groupby(['definition','source','method','disease_id']).agg(n=('n','first'),recall10=('recall10','mean')).reset_index()
per.to_csv(OUT/'external_clinvar_definition_sensitivity_per_disease.csv',index=False,encoding='utf-8-sig')
tab=per.groupby(['definition','source','method']).agg(n_diseases=('disease_id','nunique'),n_pairs=('n','sum'),recall10=('recall10','mean')).reset_index()
tab.to_csv(OUT/'external_clinvar_definition_sensitivity_table.csv',index=False,encoding='utf-8-sig')
family=dict(zip(ds.disease_id,ds.family_group));contrasts=[]
for name,subset in truths:
    for b in ['channel_lgb_v6','v71_direct_strict','B']:
        a=per[(per.definition==name)&(per.source==subset)&(per.method=='deployed')].set_index('disease_id').recall10
        bb=per[(per.definition==name)&(per.source==subset)&(per.method==b)].set_index('disease_id').recall10
        ix=a.index.intersection(bb.index);delta=(a[ix]-bb[ix]).to_numpy();lo,hi=boot(delta,np.array([family[d] for d in ix]))
        contrasts.append({'definition':name,'source':subset,'a':'deployed','b':b,'n_diseases':len(ix),'n_families':len({family[d] for d in ix}),'delta':float(delta.mean()),'ci_low':lo,'ci_high':hi})
pd.DataFrame(contrasts).to_csv(OUT/'external_clinvar_definition_sensitivity_contrasts.csv',index=False,encoding='utf-8-sig')
# Demonstrate literal reproduction of the archived full original external readout.
saved=pd.read_parquet(ROOT/'results_v8/external/external618/external_per_disease.parquet')
orig=per[per.definition=='frozen_original'].merge(saved,left_on=['source','method','disease_id'],right_on=['subset','method','disease_id'],suffixes=('_new','_saved'))
summary={'definitions':funnel,'max_absolute_reproduction_error_original':float((orig.recall10_new-orig.recall10_saved).abs().max()),'table':tab.to_dict('records'),'contrasts':contrasts,'status':'post hoc sensitivity re-scoring saved predictions, no retraining or selection, source set frozen and not expanded'}
(OUT/'external_clinvar_definition_sensitivity.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
