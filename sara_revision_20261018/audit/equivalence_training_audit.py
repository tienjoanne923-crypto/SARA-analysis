exec((__import__('pathlib').Path(__file__).parent/'external_family_audit.py').read_text(encoding='utf-8').split('# Upstream and frozen external truth funnel')[0])
eq_groups=ds.groupby('equivalence_group').disease_id.apply(list).to_dict()
cross=[]
for eq,ids in eq_groups.items():
    labelled_ids=[d for d in ids if d in role]
    if len({role[d] for d in labelled_ids})>1:
        for d in labelled_ids:
            rr=ds[ds.disease_id==d].iloc[0]
            cross.append({'equivalence_id':eq,'disease_id':d,'disease_name':rr.disease_name,'family_group':rr.family_group,'role':role[d],'n_positive':len(pos.get(d,set())),'positives':'|'.join(sorted(pos.get(d,set())))})
pd.DataFrame(cross).to_csv(OUT/'strict_equivalence_cross_roles.csv',index=False,encoding='utf-8-sig')
rows=[]
equiv=dict(zip(ds.disease_id,ds.equivalence_group))
for j in range(5):
  tr={d for k,dd in spec['folds'].items() if int(k[1:]) not in [j,(j+1)%5] for d in dd}
  test=set(spec['folds'][f'F{j}']);val=set(spec['folds'][f'F{(j+1)%5}'])
  held=test|val|set(spec['dev_diseases'])|set(spec['external_diseases'])
  forbidden={equiv.get(d,d) for d in held}
  ref={d for d in tr if equiv.get(d,d) not in forbidden}
  for targetrole,targets in [('test',test),('validation',val),('development',set(spec['dev_diseases']))]:
    for d in targets:
      for td in tr:
        if equiv.get(d,d)!=equiv.get(td,td):continue
        shared=pos.get(d,set())&pos.get(td,set())
        base={'outer_fold':j,'target_role':targetrole,'target_disease':d,'training_disease':td,'equivalence_id':equiv[d],'teacher_gradient_reference_contains_equiv':td in ref,'reranker_OOF_train_disease':True,'n_shared_positive_genes':len(shared),'shared_positive_genes':'|'.join(sorted(shared))}
        for seed in [42,52,62]:
          for b in range(3):
            p=ROOT/f'results_v8/formal/main/s{seed}/fold{j}/pools_f{j}_inner{b}_oof.npz'
            if not p.exists():continue
            with np.load(p) as z:
              ids={str(x):i for i,x in enumerate(z['ids'])}
              if td not in ids:continue
              gs=set(z['genes'][ids[td]].astype(int))
              candidatepos={g for g in pos.get(td,set()) if gix.get(g,-1) in gs}
              rows.append({**base,'seed':seed,'inner_block':b,'n_train_positive_genes_in_candidates':len(candidatepos),'n_shared_positive_genes_in_train_candidates':len(candidatepos&shared),'shared_candidate_positive_genes':'|'.join(sorted(candidatepos&shared))})
detail=pd.DataFrame(rows)
detail.to_csv(OUT/'strict_equivalence_reranker_training_overlap.csv',index=False,encoding='utf-8-sig')
summary={'cross_role_groups':len({r['equivalence_id'] for r in cross}),'cross_role_diseases':len(cross),'teacher_gradient_excludes_equivalent_heldout_concepts':not bool(detail.teacher_gradient_reference_contains_equiv.any()) if len(detail) else None,'reranker_equivalent_train_targets':detail[['outer_fold','target_role','target_disease','training_disease']].drop_duplicates().to_dict('records'),'target_counts':detail.groupby('target_role').target_disease.nunique().to_dict() if len(detail) else {},'positive_candidate_overlap_seed42':detail[detail.seed==42].to_dict('records') if len(detail) else []}
(OUT/'strict_equivalence_training_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
