exec((__import__('pathlib').Path(__file__).parent/'external_family_audit.py').read_text(encoding='utf-8').split('# Upstream and frozen external truth funnel')[0])
equiv=dict(zip(ds.disease_id,ds.equivalence_group))
dids=sorted(ds.disease_id);dix={d:i for i,d in enumerate(dids)}
with np.load(ROOT/'caches_v71/featurepack_v71.npz') as z:ok=z['d_text_mask']|z['d_hpo_mask']
rows=[]
for p in sorted((ROOT/'runs_v8').glob('V8T__main__*/result.json')):
    r=json.loads(p.read_text(encoding='utf-8'));cid=r['context_id'];j=int(r['fold']);c=next(c for c in ctxs if c['context_id']==cid)
    held=set(spec['folds'][f'F{j}'])|set(spec['folds'][f'F{(j+1)%5}'])|set(spec['dev_diseases'])|set(spec['external_diseases'])
    if c['kind']=='inner':held|=set(c['held_out_block'])
    forbidden={equiv.get(d,d) for d in held}
    ref=[d for d in c['reference_diseases'] if equiv.get(d,d) not in forbidden]
    train=sorted(d for d in ref if pos_rows.get(d) and ok[dix[d]])
    h=hashlib.sha256()
    for d in train:h.update(d.encode('utf-8'));h.update(b'\x00')
    rows.append({'run_id':r['run_id'],'context_id':cid,'seed':r['seed'],'controls':r['config']['train']['controls'],'n_reconstructed_train_diseases':len(train),'n_logged_train_diseases':r['fit']['n_train_diseases'],'logged_fit_disease_hash':r['fit_disease_hash'],'reconstructed_hash':h.hexdigest(),'hash_matches':h.hexdigest()==r['fit_disease_hash'],'equivalent_heldout_train_diseases':sum(equiv.get(d,d) in forbidden for d in train)})
pd.DataFrame(rows).to_csv(OUT/'teacher_equivalence_fit_hash_verification.csv',index=False,encoding='utf-8-sig')
out={'n_main_teacher_runs_checked':len(rows),'all_hashes_match':bool(rows) and all(r['hash_matches'] for r in rows),'n_unique_contexts':len({r['context_id'] for r in rows}),'all_have_no_heldout_equivalent_training_disease':all(r['equivalent_heldout_train_diseases']==0 for r in rows)}
(OUT/'teacher_equivalence_fit_hash_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(out)
