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
- Docker Engine、Docker Compose v2（Docker部署）
- NVIDIA Container Toolkit（后端使用CUDA版MedCPT时）

## Docker Compose

Docker只负责启动SynBioGPT前端和后端。Qwen、Qdrant、OpenSearch及已有索引仍由宿主机或外部服务器提供，不会重新建库或重新向量化。

首次使用先创建配置：

```bash
cp .env.example .env
```

至少确认以下配置：

```env
OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1
OPENAI_API_KEY=
OPENAI_MODEL=Qwen3.5-4B
QDRANT_URI=http://host.docker.internal:6333
OPENSEARCH_URI=http://host.docker.internal:9200
MEDCPT_MODEL_DIR=/qiannanhu01/models/MedCPT
SPECTER2_MODEL_DIR=/qiannanhu01_nfs/models/SPECTER2
SPECTER2_PMID_MAPPING_DB=/path/to/pmid_pmcid_full.sqlite3
SYNBIO_DATA_DIR=./backend/data
```

`host.docker.internal`表示Docker宿主机；外部服务应填写容器可访问的真实IP或DNS。`PAPER_ASSET_BASE_URL`必须填写浏览器也能访问的图片服务地址。

### 开发模式

```bash
docker compose -f compose.yaml -f compose.dev.yaml up -d
```

首次运行若本地尚无镜像，Compose会自动构建；也可以显式追加`--build`。

- 前端地址：`http://localhost:5173`，`src/`和`static/`已挂载并支持Vite HMR。
- 后端地址：`http://localhost:8080`，`backend/`已挂载但不启用`uvicorn --reload`。
- 修改Python后执行：

```bash
docker compose restart backend
```

普通源码修改无需重新构建。仅在`backend/requirements.txt`、`pyproject.toml`、`uv.lock`、`package.json`、`package-lock.json`或Dockerfile变化后重建对应镜像：

```bash
docker compose -f compose.yaml -f compose.dev.yaml build backend
docker compose -f compose.yaml -f compose.dev.yaml build frontend
```

### 生产模式

```bash
docker compose up -d --build
```

默认访问地址为`http://localhost:8080`。生产前端由Nginx提供静态文件，并将API、SSE和WebSocket请求转发给后端。

```bash
docker compose logs -f backend
docker compose down
```

`docker compose down`不会删除`${SYNBIO_DATA_DIR}`中的`webui.db`、上传文件和缓存；MedCPT目录以只读方式挂载到后端容器。

如果Qwen运行在Docker宿主机，服务必须监听容器可达的地址（例如`0.0.0.0:8000`），而不是只监听宿主机回环地址；SynBioGPT容器仍使用`http://host.docker.internal:8000/v1`访问它。

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

SPECTER2 是独立的论文级通道，不参与全文 chunk 融合。Query Processor 生成
英文 `semantic_query` 后，可调用 `/retrieval/api/v1/papers/search` 搜索论文；
`/retrieval/api/v1/papers/related` 根据 PMID 推荐相关论文。两者固定读取 Qdrant
别名 `synbiogpt_papers_specter2`，不会重建论文向量。

后台可通过 `/retrieval/api/v1/query/settings` 与 `/retrieval/api/v1/query/settings/update` 查看和更新 `hybrid` 开关。

## 测试

日常后端测试直接使用已经创建好的项目虚拟环境，避免触发 Hatchling 的前端构建钩子。

Windows PowerShell：

```powershell
& .\.venv\Scripts\python.exe -m pytest test/ -q
```

Linux/macOS：

```bash
./.venv/bin/python -m pytest test/ -q
```

运行单个模块时，例如：

```bash
./.venv/bin/python -m pytest test/test_dialogue_routing.py -q
```

也可以跨平台执行 `uv run --no-sync python -m pytest test/ -q`。不要将裸
`uv run python -m pytest ...` 用作日常测试命令；项目元数据变化后它可能执行
`npm install` 和完整前端构建，首次运行会明显更慢。

## 说明
- `backend/data/` 是业务数据目录，删除前请先备份。
- `node_modules/`、`.svelte-kit/`、`build/`、`__pycache__/` 都可按需清理并可再生成。
