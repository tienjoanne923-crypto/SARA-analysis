from pathlib import Path
import sys,json,re,gzip,ast,time
from collections import defaultdict,Counter
sys.path.insert(0,r'D:\信\sara_revision_20261014\lib')
import pandas as pd
ROOT=Path(r'D:\信\gpu8');OUT=Path(__file__).parent
gene_ids=set(pd.read_parquet(ROOT/'tables/genes.parquet').gene_id.astype(str))
ds=pd.read_parquet(ROOT/'tables/diseases.parquet').fillna('')
scope={'re':re,'pd':pd,'defaultdict':defaultdict}
for fn,names in [('mapping.py',['normalize_curie']),('external_labels.py',['parse_phenotype_ids','build_disease_lookup'])]:
    tree=ast.parse((ROOT/f'src/gpu8/{fn}').read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),fn,'exec'),scope)
norm=scope['normalize_curie'];parse=scope['parse_phenotype_ids'];lookup=scope['build_disease_lookup'](ds)
stars_map={'practice guideline':4,'reviewed by expert panel':3,'criteria provided, multiple submitters, no conflicts':2,'criteria provided, conflicting classifications':1,'criteria provided, conflicting interpretations':1,'criteria provided, single submitter':1}
counts=Counter();first={};maxstar={};first_type={};max_type={};classes=Counter();origins=Counter();types2=Counter();gene_rejected=Counter();high_unique_variations=set();record2_mapped_pairs=set();record2_mapped_diseases=set();conditions2=set();firstmeta={};origins2=Counter();classes2=Counter()
t0=time.time()
with gzip.open(ROOT/'external/clinvar/variant_summary.txt.gz','rt',encoding='utf-8',errors='replace') as fh:
    header=[c.strip().lstrip('#') for c in next(fh).rstrip('\n').split('\t')]
    col={c:i for i,c in enumerate(header)}
    for lineno,line in enumerate(fh,1):
        counts['rows_seen']+=1
        if lineno%2000000==0: print(f'rows={lineno} seconds={time.time()-t0:.1f}',flush=True)
        fs=line.rstrip('\n').split('\t')
        if len(fs)<len(header): counts['short_rows']+=1;continue
        if fs[col['Assembly']]!='GRCh38':continue
        counts['kept_assembly']+=1
        origin=fs[col['OriginSimple']].lower()
        if not origin.startswith('germline'):continue
        counts['kept_germline']+=1
        sig=fs[col['ClinicalSignificance']].lower()
        if not any(p in sig for p in {'pathogenic','likely pathogenic','pathogenic/likely pathogenic'}):continue
        if any(p in sig for p in ['conflict','uncertain','benign']):continue
        counts['kept_pathogenic']+=1;classes[fs[col['ClinicalSignificance']]]+=1;origins[fs[col['OriginSimple']]]+=1
        stars=stars_map.get(fs[col['ReviewStatus']].strip().lower(),0)
        if stars<1:continue
        counts['kept_review_ge1']+=1
        if stars>=2:counts['kept_review_ge2']+=1
        gid=fs[col['GeneID']].strip()
        if not gid or gid=='-1':gene_rejected[gid or 'empty']+=1;continue
        counts['kept_single_gene_ge1']+=1
        if stars>=2:counts['kept_single_gene_ge2']+=1
        if gid not in gene_ids:gene_rejected['not_project_geneID']+=1;continue
        counts['kept_mapped_gene_ge1']+=1
        typ=fs[col['Type']]
        if stars>=2:
            counts['kept_mapped_gene_ge2']+=1;types2[typ]+=1;high_unique_variations.add(fs[col['VariationID']]);origins2[fs[col['OriginSimple']]]+=1;classes2[fs[col['ClinicalSignificance']]]+=1
        for curie in parse(fs[col['PhenotypeIDS']]):
            if curie.startswith('MedGen:CN') or curie=='MedGen:C3661900':continue
            key=(gid,curie)
            if key not in first:
                first[key]=stars;first_type[key]=typ
                firstmeta[key]={'gene_id':gid,'condition_curie':curie,'review_stars':stars,'origin':fs[col['OriginSimple']],'clinical_significance':fs[col['ClinicalSignificance']],'type':typ,'variation_id':fs[col['VariationID']],'rcv_accessions':fs[col['RCVaccession']]}
            if stars>maxstar.get(key,-1):maxstar[key]=stars;max_type[key]=typ
            if stars>=2:
                conditions2.add(key)
                for did in lookup.get(curie,[]):
                    record2_mapped_pairs.add((did,gid));record2_mapped_diseases.add(did)

first_high={k for k,v in first.items() if v>=2};max_high={k for k,v in maxstar.items() if v>=2}
missed=max_high-first_high
missed_mapped={(d,g) for g,c in missed for d in lookup.get(c,[])}
first_mapped={(d,g) for g,c in first_high for d in lookup.get(c,[])}
mapped=[]
for (g,c),meta in firstmeta.items():
    for did in lookup.get(c,[]):
        mapped.append({**meta,'disease_id':did,'tier':'high' if meta['review_stars']>=2 else 'sensitivity'})
mapped=pd.DataFrame(mapped).drop_duplicates(['disease_id','gene_id','tier'])
conf=pd.read_parquet(ROOT/'tables_v71/confirmation_set_v71.parquet')
confkeys=set(zip(conf.disease_id,conf.gene_id.astype(str)))
frozen=mapped[(mapped.tier=='high') & pd.Series([(d,g) in confkeys for d,g in zip(mapped.disease_id,mapped.gene_id)],index=mapped.index)]
frozen.to_csv(OUT/'clinvar_frozen_confirmation_record_provenance.csv',index=False,encoding='utf-8-sig')
strictclasses={'Pathogenic','Likely pathogenic','Pathogenic/Likely pathogenic'}
result={'counts':dict(counts),'pathogenic_exact_classes':dict(classes),'pathogenic_origins':dict(origins),'ge2_mapped_gene_origins':dict(origins2),'ge2_mapped_gene_classes':dict(classes2),'single_gene_ge2_record_types':dict(types2),'gene_rejects':dict(gene_rejected),'ge2_unique_variation_ids':len(high_unique_variations),'gene_condition_unique_ge1':len(first),'gene_condition_unique_ge2_first':len(first_high),'gene_condition_unique_ge2_max':len(max_high),'eligible_ge2_gene_conditions_lost_by_first_row':len(missed),'mapped_ge2_pairs_if_max_star':len(record2_mapped_pairs),'mapped_ge2_diseases_if_max_star':len(record2_mapped_diseases),'mapped_ge2_pairs_from_current_first':len(first_mapped),'mapped_ge2_pairs_lost_to_first_row':len(record2_mapped_pairs-first_mapped),'current_first_high_types':dict(Counter(first_type[k] for k in first_high)),'source_header':header,'seconds':time.time()-t0,'frozen_clinvar_provenance':{'records':len(frozen),'diseases':int(frozen.disease_id.nunique()),'origins':frozen.origin.value_counts().to_dict(),'clinical_significance':frozen.clinical_significance.value_counts().to_dict(),'types':frozen.type.value_counts().to_dict(),'exact_PLP_only_records':int(frozen.clinical_significance.isin(strictclasses).sum()),'nonexact_PLP_records':int((~frozen.clinical_significance.isin(strictclasses)).sum())}}
(OUT/'clinvar_raw_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
pd.DataFrame([{'gene_id':g,'condition_curie':c,'first_stars':first[(g,c)],'max_stars':maxstar[(g,c)],'mapped_diseases':'|'.join(lookup.get(c,[]))} for g,c in sorted(missed)]).to_csv(OUT/'clinvar_first_row_star_attrition.csv',index=False,encoding='utf-8-sig')
print(json.dumps(result,ensure_ascii=False,indent=2))
