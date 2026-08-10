# MedCPT + BM25 全文索引

该程序流式读取 `chunks/part-*.jsonl`，使用 MedCPT Article Encoder 生成768维向量并写入Qdrant，同时把同一批子块写入OpenSearch BM25。两个索引都以稳定的 `chunk_id` 为共同主键，并完整保留父块、章节及图表元数据。

## 固定方案

- Qdrant集合：`fulltext_medcpt_v1`，768维Cosine。
- OpenSearch索引：`fulltext_bm25_v1`。
- Embedding输入：`Title + Section/Subsection + Text`，运行时临时拼接。
- PMID必须来自官方映射SQLite中的 `paper_id_mapping` 表；程序同时校验 `index_metadata`、NCBI来源和快照SHA，缺失或冲突立即失败。
- 仅索引 `chunks/`；`parents/` 用于召回后的上下文扩展，`figures_tables/` 用于图片详情查询，不重复向量化。
- Figure/Table caption已经是chunk，`image_paths`、`figure_ids` 和 `table_ids` 会进入两个索引。

## 186服务器准备

安装GPU版PyTorch、Transformers、Qdrant客户端和OpenSearch客户端，并确保Qdrant与OpenSearch已启动：

```bash
python -m pip install torch transformers qdrant-client opensearch-py
```

186服务器必须能够读取117服务器生成的 `PDF/chunks/` 和官方PMID映射SQLite；可使用NFS挂载或只复制这两项。Embedding阶段不需要复制 `parents/` 和图片。

## 输入验证

验证不加载模型、不连接Qdrant/OpenSearch：

```bash
python scripts/index_medcpt_fulltext.py \
  --chunks-dir /path/to/PDF/chunks \
  --mapping-db /home/dell/synbiogpt3/data/id_mapping/pmid_pmcid_full.sqlite3 \
  --validate-only
```

## 首分片试运行

```bash
python scripts/index_medcpt_fulltext.py \
  --chunks-dir /path/to/PDF/chunks \
  --mapping-db /home/dell/synbiogpt3/data/id_mapping/pmid_pmcid_full.sqlite3 \
  --model /path/to/MedCPT/Article-Encoder \
  --local-files-only \
  --device cuda \
  --dtype float16 \
  --encode-batch-size 128 \
  --upload-batch-size 1024 \
  --qdrant-url http://localhost:6333 \
  --opensearch-url http://localhost:9200 \
  --limit-shards 1
```

显存不足时只降低 `--encode-batch-size`；`--upload-batch-size` 主要影响网络吞吐。首分片完成后检查Qdrant和OpenSearch计数、768维向量及图表payload。

## 继续全量

使用相同参数删除 `--limit-shards 1` 即可。`medcpt_index_manifest.json` 只在Qdrant和OpenSearch均写入并校验成功后标记分片完成；中断后重跑会跳过已完成分片，未完成分片通过稳定ID安全覆盖。

```bash
python scripts/index_medcpt_fulltext.py \
  --chunks-dir /path/to/PDF/chunks \
  --mapping-db /home/dell/synbiogpt3/data/id_mapping/pmid_pmcid_full.sqlite3 \
  --model /path/to/MedCPT/Article-Encoder \
  --local-files-only \
  --device cuda \
  --dtype float16 \
  --encode-batch-size 128 \
  --upload-batch-size 1024
```

不要复用manifest去构建不同模型或不同集合；新版本应使用新的Qdrant集合名、OpenSearch索引名和state文件。

## 117 GPU节点

仓库提供 `run_117.sh`，默认直接读取117的生产分片、模型和官方映射，并写入 `58.19.38.186` 上的Qdrant/OpenSearch。首分片试运行：

```bash
LIMIT_SHARDS=1 bash scripts/medcpt_indexing/run_117.sh
```

确认首分片后，使用同一state文件继续全量：

```bash
bash scripts/medcpt_indexing/run_117.sh
```

可通过环境变量覆盖路径、服务地址与batch大小；默认 `ENCODE_BATCH_SIZE=256`、`UPLOAD_BATCH_SIZE=2048`，适用于当前A800 80GB节点。
