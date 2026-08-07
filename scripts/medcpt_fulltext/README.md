# MinerU Markdown 全文切块

该工具为 MedCPT/BM25 生成全文子块、父块和图表记录，不生成 SPECTER2 论文级索引，也不会修改原始 Markdown 或图片。

## 输入与处理规则

- 递归扫描 `--input-dir` 下文件名为 `PMC*.md` 的论文。
- 默认只选择同级 `images/` 非空的论文；`--include-without-images` 可关闭限制。
- 按 PMCID 排序并处理前 `--limit` 篇；`--limit 0` 表示全部。
- 同一 PMCID 有多个 MinerU 结果时，确定性选择有效图片引用和正文结构更完整的一份。
- `--metadata-jsonl` 可指向单个 JSONL 或目录，用于补充标题和来源元数据。
- `article_inventory.sqlite3` 保存 Markdown 清单，避免每次递归扫描大型 NFS 目录。输入目录变化后必须使用 `--refresh-inventory`。

无标准章节的旧论文仅在常规解析没有产生任何正文时，才保守恢复标题之后、References 之前的叙述文本，并标记为 `Unassigned`。仍然无法产生正文的论文写入 `errors.jsonl`，不会被静默计为成功。文档内完全相同且不关联图表的长文本块只保留第一次出现。

## MedCPT 长度

正文目标为 180–260 words，320 words 为硬上限。448 tokens 的硬上限应用于完整的 `Title + Section + Text`。`token_count` 记录完整输入长度，`text_token_count` 仅记录正文长度。

生产运行必须使用 MedCPT Article Encoder 的真实 tokenizer：

```bash
python -m pip install -r scripts/medcpt_fulltext/requirements.txt
```

## 117服务器生产命令

首次使用正确的 `_nfs` 输入路径时刷新清单：

```bash
cd /path/to/synbiogpt3

python scripts/chunk_mineru_markdown.py \
  --input-dir /qiannanhu01_nfs/pdf_parse/jsonl_backup/output \
  --output-dir /qiannanhu01_nfs/synbiogpt/backend/data/20000PDF_v1 \
  --inventory-db /qiannanhu01_nfs/synbiogpt/backend/data/article_inventory.sqlite3 \
  --refresh-inventory \
  --limit 20000 \
  --documents-per-shard 500 \
  --workers 8 \
  --tokenizer /qiannanhu01_nfs/models/MedCPT/Article-Encoder \
  --require-medcpt-tokenizer \
  --local-files-only
```

确认清单路径正确后，断点续跑时删除 `--refresh-inventory`。相同输入、参数和 tokenizer 会跳过已经成功提交到 `.spool/` 的论文；`--force` 才会强制重算。

## 生产输出

每500个已选择PMCID对应一个固定分片边界。失败论文不会改变后续论文的分片编号：

```text
OUTPUT/
  chunks/part-00000.jsonl
  parents/part-00000.jsonl
  figures_tables/part-00000.jsonl
  documents.jsonl
  errors.jsonl
  statistics.json
  manifest.json
  inspection_samples.jsonl
  duplicate_resolution.jsonl
  .spool/
```

程序流式归并，不会把2万篇的全部chunk载入内存。每个文件先写 `.partial`，完成后原子替换；相同输入重复运行产生稳定内容。

`manifest.json` 是下游唯一权威清单，记录每个分片的PMCID范围、成功/失败论文数、行数、逻辑内容SHA-256和文件大小。Embedding程序应按 `manifest.shards` 顺序读取 `files.chunks.path`，不要自行扫描目录。

`figures_tables` 保存已有caption、相对图片路径和邻近正文；只有JPG的表格保持 `table_text_missing=true`，不调用OCR。Embedding阶段按batch读取chunk的 `paper_title`、`section`、`subsection` 和 `text`，调用 `build_embedding_text()` 临时拼接，不在JSONL中重复保存 `embedding_text`。

## 验收顺序

1. 检查 `errors.jsonl`；命令退出码为2表示至少一篇失败，但成功分片仍会完整生成。
2. 检查 `statistics.json` 中的超长块、短块、未知标题和零失败要求。
3. 检查 `manifest.json` 的分片数、行数和PMCID边界。
4. 人工查看 `inspection_samples.jsonl` 中固定种子抽取的论文。
