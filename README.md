# SynBioGPT

SynBioGPT 是合成生物学问答平台，核心增强点是知识库检索链路（向量检索 + 可选混合检索）。

## 功能概览
- 聊天式问答与知识库管理
- 文档入库、切块、向量化
- 可选混合检索（Hybrid）：向量召回 + 词法一级检索（BM25 全文/TOC）+ 重排
- 多模型后端兼容（OpenAI / Ollama 等）

## 目录结构
- `src/`：前端（SvelteKit）
- `backend/open_webui/`：后端（FastAPI）
- `backend/data/`：本地数据目录（数据库、索引、上传内容等）
- `backend/open_webui/apps/retrieval/synbio/`：SynBioGPT 正式在线检索编排与 WebUI 薄适配
- `scripts/medcpt_fulltext/`：MinerU Markdown 全文处理与稳定切块
- `scripts/medcpt_indexing/`：MedCPT/Qdrant/OpenSearch 生产索引管线
- `scripts/medcpt_images/`：现有图表恢复、访问清单和索引补充流程
- `scripts/query_medcpt_fulltext.py`：调用正式 RetrievalPipeline 的离线验证入口

## 环境要求
- Node.js `>=18.13 <=22`
- Python `3.11`

## 快速启动（开发）

### 1) 启动前端
在仓库根目录执行：

```bash
npm install
npm run dev
```

### 2) 启动后端
在 `backend/` 目录执行：

```bash
uv run uvicorn open_webui.main:app --host 0.0.0.0 --port 8080 --reload
```

如果你使用现有脚本：

```bash
cd backend
bash dev.sh
```

## 检索模式说明

### 关闭 Hybrid（默认原始链路）
- 走纯向量检索（原 RAG 检索路径）

### 开启 Hybrid
- `Query Processor → MedCPT Dense + BM25 → RRF → Cross Encoder`
- 命中后恢复 Parent/Previous/Next，并扩展 Figure/Table 证据
- WebUI 与离线验证 CLI 共用同一 `RetrievalPipeline`

后台可通过 `/retrieval/api/v1/query/settings` 与 `/retrieval/api/v1/query/settings/update` 查看和更新 `hybrid` 开关。

## 测试

```bash
python -m pytest backend/open_webui/test/apps/retrieval/test_lexical_index.py backend/open_webui/test/apps/retrieval/test_utils_hybrid.py -q
```

## 说明
- `backend/data/` 是业务数据目录，删除前请先备份。
- `node_modules/`、`.svelte-kit/`、`build/`、`__pycache__/` 都可按需清理并可再生成。
