# MinerU Markdown 全文切块（500篇试运行）

该工具只生成全文子块、父块和图表记录，不生成 SPECTER2 论文级索引，也不会修改原始 Markdown 或图片。

## 输入与选择规则

- 递归扫描 `--input-dir` 下文件名为 `PMC*.md` 的文件，因此兼容真实目录结构：
  `output/{任务UUID}/{PMCID}/auto/{PMCID}.md`。
- 默认只选择同级 `images/` 中至少有一个文件的论文；使用 `--include-without-images` 可关闭该限制。
- 按 PMCID 文件名排序并取前 `--limit` 个唯一 PMCID，默认 500。
- 同一 PMCID 有多次 MinerU 输出时，依次比较有效图片引用数、图片引用数、标题数和正文长度，确定性选择质量最高的一份。
- `--metadata-jsonl` 可指向单个 JSONL 或包含多个 JSONL 的目录。程序先确定500个PMCID，再流式查找这些PMCID，避免将全文元数据全部载入内存。
- 首次运行会建立可复用的 `article_inventory.sqlite3`。后续试跑、断点续跑和2万篇生产运行直接查询清单，不再递归扫描全部NFS目录；只有输入目录新增或删除论文后才使用 `--refresh-inventory`。

## MedCPT 长度保护

业务规则以 180–260 words 为目标、320 words 为硬上限。程序额外应用 448 MedCPT tokens 硬上限，为后续标题、章节前缀和 BERT 特殊token保留空间。若单个句子本身超限，先尝试在分号/冒号处拆分；仍超限时才在单词边界强制拆分并记录警告。

默认加载 `ncbi/MedCPT-Article-Encoder` tokenizer。生产试运行建议明确要求该 tokenizer，避免静默使用回退计数器：

```bash
python -m pip install -r scripts/medcpt_fulltext/requirements.txt
```

如果117服务器不能访问 Hugging Face，可先在可联网环境下载模型缓存，再复制缓存到117；或者首次测试允许回退 tokenizer。实际使用的名称会写入每个chunk、documents和statistics。

## 117服务器500篇命令

```bash
cd /path/to/synbiogpt3

python scripts/chunk_mineru_markdown.py \
  --input-dir /qiannanhu01_nfs/pdf_parse/jsonl_backup/output \
  --output-dir /qiannanhu01_nfs/pdf_parse/jsonl_backup/medcpt_visual_chunks_v1/pilot_500 \
  --inventory-db /qiannanhu01_nfs/pdf_parse/jsonl_backup/medcpt_visual_chunks_v1/article_inventory.sqlite3 \
  --refresh-inventory \
  --limit 500 \
  --workers 8 \
  --require-medcpt-tokenizer
```

可靠Markdown H1会直接作为论文标题。若确实需要补充解析耗时和 `source_file`，再增加：

```bash
--metadata-jsonl /qiannanhu01_nfs/pdf_parse/jsonl_backup
```

读取该目录可能需要流式扫描多个大型JSONL，因此建议先在不带该参数的500篇试运行中检查正文切块。

`--refresh-inventory` 只在首次构建或源目录内容变化后使用，第二次运行应删除该参数。未显式提供 `--inventory-db` 时，默认使用输出目录上一级的 `article_inventory.sqlite3`。

首次建议从 `--workers 4` 或 `8` 开始，根据CPU和NFS读取负载调整。切块只使用CPU。在117的Linux环境中，worker会继承父进程已经加载的tokenizer，避免每个进程重复加载；worker数量也不会超过实际待处理论文数。

## 断点续跑与输出

每篇论文先写入：

```text
OUTPUT/.spool/{PMCID末三位}/{PMCID}.json.gz
OUTPUT/.spool/{PMCID末三位}/{PMCID}.done.json
```

压缩结果完整落盘后才原子写入完成标记。再次运行相同输入、参数和tokenizer时默认跳过成功论文；使用 `--force` 可重新处理。全部论文处理结束后，程序按PMCID逐篇读取spool并流式写出，不会把2万篇的全部chunk同时载入内存；输出先写临时文件，再原子替换：

```text
chunks.jsonl
parents.jsonl
figures_tables.jsonl
documents.jsonl
errors.jsonl
statistics.json
inspection_samples.jsonl
duplicate_resolution.jsonl
```

`figures_tables.jsonl` 只记录 Markdown 中能够识别的图表、相对图片路径、caption和相邻引用上下文。表格只有JPG时始终设置 `table_text_missing=true`，程序不调用OCR，也不会虚构表格单元格。

## 检查顺序

1. 查看 `errors.jsonl`，确认失败原因。
2. 查看 `statistics.json` 的超长块、短块、未知标题和标题异常数量。
   同时比较 `discovery_seconds`、`tokenizer_load_seconds`、`document_processing_wall_seconds` 和 `merge_seconds`，判断耗时发生在哪个阶段。
3. 人工检查 `inspection_samples.jsonl` 中固定随机种子抽取的20篇。
4. 检查 `figures_tables.jsonl` 的图片路径、图号/表号和caption绑定。
5. 500篇通过后再使用新的输出目录启动约2万篇，继续复用同一个inventory数据库，并保留pilot结果用于规则对比。
