from pathlib import Path
import json,hashlib,zipfile
ROOT=Path(r'D:\信\gpu8');OUT=Path(__file__).parent
names=['external_family_audit.py','clinvar_raw_audit.py','equivalence_training_audit.py','external_strict_sensitivity.py','equivalence_main_sensitivity.py','verify_teacher_equivalence_fit_hashes.py','external_family_ready_text.md','external_family_audit.json','clinvar_raw_audit.json','external_truth_funnel.csv','disease_family_fold_mapping.csv','external_relation_seen_split.csv','external_filtered_candidate_counts_per_seed_disease.csv','external_filtered_candidate_counts_summary.csv','clinvar_first_row_star_attrition.csv','clinvar_frozen_confirmation_record_provenance.csv','strict_equivalence_cross_roles.csv','strict_equivalence_reranker_training_overlap.csv','strict_equivalence_training_audit.json','external_clinvar_definition_sensitivity_per_disease.csv','external_clinvar_definition_sensitivity_table.csv','external_clinvar_definition_sensitivity_contrasts.csv','external_clinvar_definition_sensitivity.json','cross_role_equivalence_excluded_main_diseases.csv','main_equivalence_exclusion_sensitivity_table.csv','main_equivalence_exclusion_sensitivity_contrasts.csv','main_equivalence_exclusion_sensitivity.json','teacher_equivalence_fit_hash_verification.csv','teacher_equivalence_fit_hash_verification.json']
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
sources=[ROOT/p for p in ['tables/diseases.parquet','tables/genes.parquet','tables/relations.parquet','tables/hierarchy.parquet','tables/external_labels.parquet','tables_v71/confirmation_set_v71.parquet','results_v8/external/external618/external_pairs_fields.parquet','results_v8/external/external618/external_per_disease.parquet','results_v8/pub/query_ranks.parquet','results_v8/eval/eval_test_main.parquet','manifests/splits_main_5fold.json','manifests/contexts.json','configs/frozen_config.json','results/external/clinvar_variant_level.csv','results/external/external_labels_audit.json','external/clinvar/variant_summary.txt.gz','src/gpu8/external_labels.py','src/gpu8/mapping.py','src/gpu8/protocol.py','src/gpu8v71/register71.py','src/gpu8v71/evaluate71.py','src/gpu8v8final/external8.py','src/gpu8v8final/roles8.py','src/gpu8v8final/teacher8.py','src/gpu8v8final/chain8.py','src/gpu8v8final/feats8.py','src/gpu8v8final/models8.py']]
for method in ['deployed','B','channel_lgb_v6','v71_direct_strict','F7_v6']:
 for seed in [42,52,62]:
  for j in range(5):sources.append(ROOT/f'results_v8/formal/main/s{seed}/fold{j}/pred_{method}__test.npz')
for seed in [42,52,62]:
 for j in range(5):
  for b in range(3):sources.append(ROOT/f'results_v8/formal/main/s{seed}/fold{j}/pools_f{j}_inner{b}_oof.npz')
sources.extend(sorted((ROOT/'runs_v8').glob('V8T__main__*/result.json')))
sources.append(ROOT/'caches_v71/featurepack_v71.npz')
manifest={'scope':'2026-10-18 external-label and disease-family implementation audit; saved prediction re-scoring only; no model retraining','source_root':str(ROOT),'sources':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sources],'outputs':[{'path':n,'bytes':(OUT/n).stat().st_size,'sha256':sha(OUT/n)} for n in names]}
(OUT/'external_family_bundle_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
readme='''# SARA 外部标签与疾病家族可复核数据，2026-10-18

本包包含本轮代码/保存结果审计与两项事后敏感性重评分。不包含原始数据库或重新训练的模型，也不改变冻结预测。所有疾病、基因、变异标识均来自论文既有公开资源。

先读 external_family_ready_text.md；实现与计数汇总见各 JSON，逐病/逐名单/来源代表记录见 CSV。

主要文件：
- disease_family_fold_mapping.csv：13,321 个疾病的等价ID、所有树号、唯一家族、来源、开发/外部/折角色。
- strict_equivalence_cross_roles.csv：4 跨角色等价组、8 标识。
- strict_equivalence_reranker_training_overlap.csv：跨角色等价病进入排序器折外候选的上下文与种子核对。
- teacher_equivalence_fit_hash_verification.csv：65 条已执行语义拟合日志的训练ID哈希重建，20上下文，全部相符。
- external_truth_funnel.csv：映射、高置信、五折范围、同病CTD剔除的来源行/唯一关系/疾病漏斗。
- external_relation_seen_split.csv：2,441来源记录的直接父子/同已分配家族两字段拆分。
- external_filtered_candidate_counts*.csv：不补位过滤后每方法的实际名单数，含全618病×3种子。
- clinvar_frozen_confirmation_record_provenance.csv：冻结1,980条ClinVar代表记录的来源/分类/变异类型/VariationID/RCVaccession。
- clinvar_first_row_star_attrition.csv：全项目首条代表规则与最大星级规则的差异，仅供审计，未扩充冻结真值。
- external_clinvar_definition_sensitivity*.csv/json：精确P/LP及纯胚系精确P/LP的保存预测敏感性。
- main_equivalence_exclusion_sensitivity*.csv/json：排除跨角色等价组5测试病的保存名次敏感性。
- external_family_bundle_manifest.json：原始输入和本包输出的字节数与SHA-256。

可复现脚本仅读取 GPU8 既有源数据和保存预测；需要 Python、numpy、pandas、pyarrow。当前目录为 D:\\信\\sara_revision_20261018\\audit，源根为 D:\\信\\gpu8。迁移到其他机器时请调整各脚本 ROOT、依赖路径和读取本目录脚本的路径。

依次运行 external_family_audit.py、clinvar_raw_audit.py、equivalence_training_audit.py、verify_teacher_equivalence_fit_hashes.py、external_strict_sensitivity.py、equivalence_main_sensitivity.py。后两项只重算评价；原定义逐病结果与既有评价最大绝对差均为0。家族聚类自助法为10,000次、随机种子20260923。

解释边界：家族跨角色为0，参照/语义训练实施等价排除，排序器全链等价闭合隔离未实施。删病敏感性不等同于重训。ClinVar当前规则含混合胚系/体细胞与少量复合分类；更严格子集为事后分析，不能称新独立确认。
'''
(OUT/'README_external_family_reproducibility.md').write_text(readme,encoding='utf-8')
archive=OUT/'external_family_reproducibility_20261018.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for n in names+['external_family_bundle_manifest.json','README_external_family_reproducibility.md']:
  z.write(OUT/n,arcname=n)
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 assert len(z.namelist())==len(names)+2
print(json.dumps({'zip':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'files':len(names)+2,'source_files_hashed':len(sources)},ensure_ascii=False,indent=2))
