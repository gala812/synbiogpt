- `CHANGELOG.md`：后端启动时会读取，并通过 changelog API / 前端弹窗使用。
- `README.md`：被 `pyproject.toml` 作为项目说明文件引用。


- 前端开发：`npm run dev`
- 前端生产：nginx 访问静态文件
- 后端开发：`uvicorn`
- 后端生产：`dev.sh`

## 说明

原 `TROUBLESHOOTING.md` 主要解释 Open WebUI 与 Ollama 的连接方式。

关键模型：

```text
浏览器前端
  -> 后端 /ollama 路由
  -> OLLAMA_BASE_URL
  -> Ollama 服务
```

也就是说，前端不是直接访问 Ollama，而是通过后端代理。这么做可以避免 CORS 问题，也避免把 Ollama API 直接暴露给前端。

文档还提到：

- Docker 容器里访问宿主机 `127.0.0.1:11434` 经常失败。
- Docker 场景下可能需要 `--network=host` 或 `host.docker.internal`。
- Ollama 慢响应可能触发默认超时，可通过 `AIOHTTP_CLIENT_TIMEOUT` 调整。
- 需要确认 Ollama URL 配置和 Ollama 版本。


## 知识库功能结构

原 `知识库.md` 记录了知识库功能的前后端结构、数据流，以及标签检索扩展思路。

### 前端主要文件

- `src/lib/components/workspace/Knowledge.svelte`
  - 知识库列表页。
  - 展示知识库。
  - 支持搜索、进入详情、创建和删除。

- `src/lib/components/workspace/Knowledge/CreateKnowledgeBase.svelte`
  - 创建知识库页面。
  - 输入名称、描述、访问控制等信息。
  - 可扩展为创建时选择标签。

- `src/lib/components/workspace/Knowledge/KnowledgeBase.svelte`
  - 知识库详情和编辑页。
  - 支持编辑名称、描述。
  - 管理上传文件、目录同步、拖拽上传、文件编辑、文件删除和访问控制。

- `src/lib/components/workspace/Knowledge/KnowledgeBase/Files.svelte`
  - 知识库内文件列表。

- `src/lib/components/workspace/Knowledge/KnowledgeBase/AddContentMenu.svelte`
  - 添加内容菜单。

- `src/lib/components/workspace/Knowledge/KnowledgeBase/AddTextContentModal.svelte`
  - 添加文本内容弹窗。

- `src/lib/apis/knowledge/index.ts`
  - 知识库 API 调用层。
  - 包括创建、查询、更新、删除知识库，以及添加、更新、移除文件。

### 后端主要文件

- `backend/open_webui/apps/webui/routers/knowledge.py`
  - 知识库 HTTP 路由。
  - 处理知识库 CRUD、文件添加、文件更新、文件移除、重置、删除等接口。

- `backend/open_webui/apps/webui/models/knowledge.py`
  - 知识库数据库模型和数据库操作。
  - 主要字段包括 `id`、`user_id`、`name`、`description`、`data`、`meta`、`access_control`、`created_at`、`updated_at`。

- `backend/open_webui/apps/webui/main.py`
  - 注册知识库路由到 FastAPI 应用。

- `backend/open_webui/apps/retrieval/main.py`
  - 文件处理、文本切分、embedding 生成、向量库写入。

- `backend/open_webui/apps/retrieval/vector/connector.py`
  - 根据 `VECTOR_DB` 选择具体向量数据库客户端。
  - 上层检索代码通过统一的 `VECTOR_DB_CLIENT` 调用向量库。

- `backend/open_webui/utils/access_control.py`
  - 访问控制工具。
  - 判断用户或用户组是否有访问权限。

### 知识库数据流

```text
前端知识库页面
  -> src/lib/apis/knowledge/index.ts
  -> 后端 knowledge router
  -> knowledge 数据库模型
  -> SQLite / PostgreSQL 元数据
  -> retrieval 文件处理
  -> vector connector
  -> 向量数据库
```

当前项目的实际演进方向是：

- 向量检索运行时从 Chroma 切换到 Qdrant。
- OpenSearch BM25 作为 hybrid search 的 sidecar 保留。
- RAG / reranker / sources / prompt 主流程保持不变。

### 标签检索扩展思路

原文档中还记录了一个标签检索设计：

- 创建知识库时选择标签。
- 将标签保存到 `meta.tags`。
- 前端 API 创建知识库时携带标签。
- 后端创建 / 更新知识库时接收并返回标签。
- RAG 检索时可以根据标签筛选候选内容，再与当前知识库内容合并。
- 向量库客户端可以扩展 `list_metadata_values` 或 metadata query 能力。

这部分是架构设计记录，不是当前运行必需逻辑。后续如果要重新做标签驱动检索，可以参考这部分。

## Pyodide 静态包说明

原 `static/pyodide/README.md` 是 Pyodide 上游包说明。

主要内容：

- Pyodide 是运行在 WebAssembly 中的 CPython。
- JavaScript 可以通过 `loadPyodide` 加载 Python 运行时。
- 可以在 Node.js 或浏览器中执行 Python 代码。
- 支持通过 Pyodide 包加载机制和 `micropip` 加载 Python 包。
- Pyodide artifact 版本需要和 npm 包版本一致。

当前项目中，前端脚本包含：

```bash
npm run pyodide:fetch
```

因此 Pyodide 资产本身可能仍然和前端构建有关。但这个 README 只是上游说明文档，不被代码读取，也不影响运行。

- `CHANGELOG.md` 被后端启动代码读取。
- `README.md` 被项目元数据引用。
