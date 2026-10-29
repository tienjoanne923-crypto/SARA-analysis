SARA 分析包与 GitHub 候选版本说明

本包对应10月18日科学稿及10月28日BioData Mining英文修订稿。原实验数值未修改，未合并10月21/22日扩展与P1分析。

已包含原始代码及依赖模块、冻结配置与开发选择、预处理脚本、可移植运行入口、环境版本说明和输入SHA-256。原始数据库、处理后数据、模型权重及保存预测不在此代码包内；按来源条款取得输入后再校验。

先运行 python scripts/verify_package.py 和 python scripts/check_python_sources.py。完整步骤见 README.md、docs/REPRODUCING.md、environment/README.md。核验包内文件与输入哈希不等于完整重训通过。

当前公开候选版本为 v1.0.0-rc2，仓库为 https://github.com/tienjoanne923-crypto/SARA-analysis。存档作者为徐嘉璐（Jialu Xu），原创代码采用MIT许可证。DOI尚未取得。Declarations分成“GitHub当前状态文本”和“取得持久DOI后使用的替换模板”；不得把待填模板当成已经取得DOI的声明。

已有缺口保留：历史CTD完整原始转储缺失；V8分段SapBERT原始缓存缺失，可按固定模型重建并记录新哈希；seed42依赖开发阶段预测。没有用最新下载代替冻结数据，也没有把重新开发选参称为独立复现。
