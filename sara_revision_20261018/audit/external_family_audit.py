from pathlib import Path
import sys, json, re, hashlib, ast
from collections import defaultdict, deque
sys.path.insert(0, r'D:\信\sara_revision_20261014\lib')
import pandas as pd
import numpy as np

ROOT = Path(r'D:\信\gpu8')
OUT = Path(__file__).parent
OUT.mkdir(exist_ok=True, parents=True)
J = lambda p: json.loads((ROOT/p).read_text(encoding='utf-8'))
ds = pd.read_parquet(ROOT/'tables/diseases.parquet').fillna('')
genes = pd.read_parquet(ROOT/'tables/genes.parquet')
rel = pd.read_parquet(ROOT/'tables/relations.parquet')
hi = pd.read_parquet(ROOT/'tables/hierarchy.parquet')
ext = pd.read_parquet(ROOT/'tables/external_labels.parquet').fillna('')
conf = pd.read_parquet(ROOT/'tables_v71/confirmation_set_v71.parquet')
pairs = pd.read_parquet(ROOT/'results_v8/external/external618/external_pairs_fields.parquet')
spec = J('manifests/splits_main_5fold.json')
ctxs = J('manifests/contexts.json')['contexts']
cfg = J('configs/frozen_config.json')
primary = rel[rel.relation_type.isin(cfg['data']['primary_evidence'])]
gids = sorted(genes.gene_id.astype(str), key=lambda x: int(x) if x.isdigit() else 0)
gix = {g:i for i,g in enumerate(gids)}
pos = primary.groupby('disease_id').gene_id.apply(lambda s:set(s.astype(str))).to_dict()
pos_pairs = {(d,g) for d,gg in pos.items() for g in gg}
pos_rows = {d:{gix[g] for g in gs if g in gix} for d,gs in pos.items()}
fold_of = {d:int(k[1:]) for k,ids in spec['folds'].items() for d in ids}
role = {d:k for k,ids in spec['folds'].items() for d in ids}
role.update({d:'development' for d in spec['dev_diseases']})
role.update({d:'external_holdout' for d in spec['external_diseases']})

def cnt(df):
    return {'records':int(len(df)), 'unique_pairs':int(df[['disease_id','gene_id']].drop_duplicates().shape[0]), 'diseases':int(df.disease_id.nunique())}

# Upstream and frozen external truth funnel, matching register71.confirmation_set.
funnel=[]
for source in ['GenCC','ClinVar','union']:
    x=ext if source=='union' else ext[ext.source==source]
    stages=[('mapped_external_source_rows',x)]
    x=x[((x.source=='GenCC') & x.classification.isin(['Definitive','Strong'])) | ((x.source=='ClinVar') & (pd.to_numeric(x.review_stars,errors='coerce').fillna(0)>=2))]
    stages.append(('high_tier',x))
    x=x[x.disease_id.isin(fold_of)]
    stages.append(('five_fold_diseases',x))
    x=x[x.layer1_relation_overlap.astype(int)==0]
    stages.append(('subtract_same_disease_CTD',x))
    x=x[x.disease_id.isin(pos)&x.gene_id.astype(str).isin(gix)].drop_duplicates(['disease_id','gene_id'])
    stages.append(('background_and_labelled_then_unique',x))
    x=x[[(d,str(g)) not in pos_pairs for d,g in zip(x.disease_id,x.gene_id)]]
    stages.append(('final_explicit_truth_subtraction',x))
    for stage,df in stages: funnel.append({'source':source,'stage':stage,**cnt(df)})
pd.DataFrame(funnel).to_csv(OUT/'external_truth_funnel.csv',index=False,encoding='utf-8-sig')
truth_overlaps=[(r.disease_id,str(r.gene_id)) for r in conf.itertuples() if str(r.gene_id) in pos.get(r.disease_id,set())]

# Reconstruct family assignment using the exact authoritative functions via AST.
mapping_source=ROOT/'src/gpu8/mapping.py'
tree=ast.parse(mapping_source.read_text(encoding='utf-8'))
nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in ['primary_tree_prefix','build_family_groups']]
scope={'pd':pd,'re':re,'defaultdict':defaultdict,'deque':deque}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(mapping_source),'exec'),scope)
reconstructed=scope['build_family_groups'](ds,hi)
family_equal=bool((reconstructed.family_group==ds.family_group).all())
family=dict(zip(ds.disease_id,ds.family_group));equiv=dict(zip(ds.disease_id,ds.equivalence_group))
mapping=ds[['disease_id','equivalence_group','tree_numbers','family_group','family_source','mondo_ids','mapping_type','parent_ids']].copy()
mapping=mapping.rename(columns={'equivalence_group':'equivalence_id','tree_numbers':'all_tree_ids','family_group':'assigned_family'})
mapping['fold_or_role']=mapping.disease_id.map(role).fillna('unlabelled_background')
mapping['canonical_own_branch']=mapping.all_tree_ids.map(scope['primary_tree_prefix'])
mapping.to_csv(OUT/'disease_family_fold_mapping.csv',index=False,encoding='utf-8-sig')
fam_cross=mapping.groupby('assigned_family').fold_or_role.nunique()
labelled=mapping[mapping.fold_or_role!='unlabelled_background']
fam_cross_labelled=labelled.groupby('assigned_family').fold_or_role.nunique()
eq_cross=labelled.groupby('equivalence_id').fold_or_role.nunique()

# Split the prior combined neighbour/family relation field.
par=hi.groupby('child_disease').parent_disease.apply(set).to_dict()
chi=hi.groupby('parent_disease').child_disease.apply(set).to_dict()
refs={}
for j in range(5):
    cc=next(c for c in ctxs if c['context_id']==f'f{j}_outer')
    held=set(spec['folds'][f'F{j}'])|set(spec['folds'][f'F{(j+1)%5}'])|set(spec['dev_diseases'])|set(spec['external_diseases'])
    forbidden={equiv.get(d,d) for d in held}
    refs[j]={d for d in cc['reference_diseases'] if equiv.get(d,d) not in forbidden}
rows=[]
for r in pairs.itertuples(index=False):
    d,g,j=r.disease_id,str(r.gene_id),int(r.fold)
    hn=(par.get(d,set())|chi.get(d,set())) & refs[j]
    fn={x for x in refs[j] if family.get(x)==family.get(d)}
    hmatch=sorted(x for x in hn if g in pos.get(x,set()))
    fmatch=sorted(x for x in fn if g in pos.get(x,set()))
    rows.append({**r._asdict(),'hierarchy_relation_seen':bool(hmatch),'same_assigned_family_relation_seen':bool(fmatch),'hierarchy_matching_diseases':'|'.join(hmatch),'family_matching_diseases':'|'.join(fmatch),'visible_same_family_diseases':len(fn)})
split=pd.DataFrame(rows)
split.to_csv(OUT/'external_relation_seen_split.csv',index=False,encoding='utf-8-sig')
split_summary=[]
for source,x in [('union_source_records',split),('union_unique_pairs',split.drop_duplicates(['disease_id','gene_id']))]+[(s,split[split.source==s]) for s in ['GenCC','ClinVar']]:
    split_summary.append({'source':source,'n':len(x),'hierarchy_only':int((x.hierarchy_relation_seen&~x.same_assigned_family_relation_seen).sum()),'family_only':int((~x.hierarchy_relation_seen&x.same_assigned_family_relation_seen).sum()),'both':int((x.hierarchy_relation_seen&x.same_assigned_family_relation_seen).sum()),'neither':int((~x.hierarchy_relation_seen&~x.same_assigned_family_relation_seen).sum())})

# Inspect every saved list actually used in external8.score; no model fit rerun.
candidate_rows=[]
for method in ['deployed','B','channel_lgb_v6','v71_direct_strict','F7_v6']:
  for seed in [42,52,62]:
    for j in range(5):
      p=ROOT/f'results_v8/formal/main/s{seed}/fold{j}/pred_{method}__test.npz'
      if not p.exists():
          candidate_rows.append({'method':method,'seed':seed,'fold':j,'missing_path':str(p)})
          continue
      with np.load(p) as z:
        ix={str(d):i for i,d in enumerate(z['ids'])}
        gg,ss=z['genes'],z['score']
        vv=z['valid'] if 'valid' in z.files else np.isfinite(ss)
        for d in pairs.loc[pairs.fold==j,'disease_id'].unique():
          if d not in ix: continue
          i=ix[d]; gs=gg[i][vv[i]].astype(int);scores=ss[i][vv[i]].astype(float)
          order=gs[np.lexsort((gs,-scores))]
          filtered=order[~np.isin(order,list(pos_rows.get(d,set())))]
          truth=set(conf.loc[conf.disease_id==d,'gene_row'].astype(int))
          candidate_rows.append({'method':method,'seed':seed,'fold':j,'disease_id':d,'before':len(order),'removed_ctd':len(order)-len(filtered),'after':len(filtered),'external_unique_positives':len(truth),'external_positive_ctd_overlap':len(truth&pos_rows.get(d,set())),'external_hits10':len(truth&set(filtered[:10]))})
candidates=pd.DataFrame(candidate_rows)
candidates.to_csv(OUT/'external_filtered_candidate_counts_per_seed_disease.csv',index=False,encoding='utf-8-sig')
stats=candidates.groupby('method').agg(n_runs=('after','count'),n_diseases=('disease_id','nunique'),before_min=('before','min'),before_median=('before','median'),before_mean=('before','mean'),before_max=('before','max'),after_min=('after','min'),after_p25=('after',lambda s:s.quantile(.25)),after_median=('after','median'),after_mean=('after','mean'),after_p75=('after',lambda s:s.quantile(.75)),after_max=('after','max'),removed_ctd_mean=('removed_ctd','mean'),runs_under10=('after',lambda s:int((s<10).sum())),runs_empty=('after',lambda s:int((s==0).sum())))
stats.reset_index().to_csv(OUT/'external_filtered_candidate_counts_summary.csv',index=False,encoding='utf-8-sig')

cv=pd.read_csv(ROOT/'results/external/clinvar_variant_level.csv',dtype=str).fillna('')
cv2=cv[pd.to_numeric(cv.review_stars)>=2]
result={
 'inputs': {str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [mapping_source,ROOT/'src/gpu8/external_labels.py',ROOT/'src/gpu8/protocol.py',ROOT/'src/gpu8v71/register71.py',ROOT/'src/gpu8v8final/external8.py',ROOT/'tables_v71/confirmation_set_v71.parquet',ROOT/'results_v8/external/external618/external_pairs_fields.parquet']},
 'upstream_clinvar_saved_audit':J('results/external/external_labels_audit.json')['clinvar_variant'],
 'clinvar_retained_dedup_rows':len(cv),'clinvar_retained_stars':cv.review_stars.value_counts().to_dict(),'clinvar_retained_types':cv.type.value_counts().to_dict(),'clinvar_retained_ge2_types':cv2.type.value_counts().to_dict(),
 'clinvar_filter': {'source':'variant_summary.txt.gz','assembly':'GRCh38','origin':'OriginSimple.lower().startswith(germline)','clinical':'contains pathogenic or likely pathogenic; exclude conflict,uncertain,benign','review':'initial >=1; high tier >=2 on retained gene-condition representative','dedup_key':['GeneID','normalised condition CURIE'],'dedup_order':'first qualifying source row retained, not maximum-star aggregation','multi_gene':'missing GeneID or -1 rejected; other IDs outside project gene background rejected; no Type-specific CNV exclusion'},
 'truth_funnel':funnel,'confirmation':cnt(conf),'scoring_source_records':cnt(pairs),'confirmation_CTD_overlap':len(truth_overlaps),'unique_pair_difference_source_dedup':len(pairs)-len(conf),
 'family_assignment':{'reconstructed_all_current_matches':family_equal,'n_diseases':len(ds),'sources':ds.family_source.value_counts().to_dict(),'all_have_own_tree':bool((ds.family_source=='own_tree').all()),'family_labels_cross_roles':int((fam_cross_labelled>1).sum()),'strict_equivalence_groups_cross_roles':int((eq_cross>1).sum()),'family_source':'CTD vocabulary tree_numbers; parents from CTD ParentIDs, not DO or Mondo is_a','equivalence_before_family':True,'equivalence_members_collapsed_before_branch_choice':False,'frozen_split_seed':spec['seed'],'fold_counts':{k:len(v) for k,v in spec['folds'].items()}},
 'relation_seen_split':split_summary,'same_relation_reproduces_original':bool((split.same_relation_seen.astype(bool)==(split.hierarchy_relation_seen|split.same_assigned_family_relation_seen)).all()),
 'candidate_count_summary':stats.reset_index().to_dict('records'),'diseases_without_external_new_truth':0 if not truth_overlaps else None,
}
(OUT/'external_family_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=lambda o:int(o) if isinstance(o,np.integer) else float(o) if isinstance(o,np.floating) else str(o)),encoding='utf-8')
print(json.dumps({k:result[k] for k in ['confirmation','scoring_source_records','confirmation_CTD_overlap','family_assignment','relation_seen_split','same_relation_reproduces_original','candidate_count_summary']},ensure_ascii=False,indent=2))
