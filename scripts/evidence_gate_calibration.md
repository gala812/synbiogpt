# Evidence Gate 标注与校准

该流程只校准 MedCPT Cross Encoder 的 raw logit，不修改 MedCPT、BM25、RRF
或 Cross Encoder 算法，也不会自动修改生产阈值。

## 1. 开启受控采集

采集默认关闭。仅在已确认数据存储位置和访问权限后设置：

```env
MEDCPT_EVIDENCE_CALIBRATION_LOG_PATH=C:/secure/synbiogpt/evidence_samples.jsonl
MEDCPT_EVIDENCE_CALIBRATION_SAMPLE_RATE=0.10
MEDCPT_EVIDENCE_CALIBRATION_MAX_TEXT_CHARS=4000
```

重启后端后，系统会在 Cross Encoder 完成、Evidence Gate 执行之前，对抽中的
query-document 候选写入 JSONL。记录包含用户原始问题、重写查询、文献片段、
raw logit 和文献标识。该文件可能含敏感内容，应使用受限目录，不应提交 Git。

采样 Query 应同时覆盖：

- 能在知识库中找到直接证据的科研问题；
- 主题相关但没有直接答案的问题；
- 知识库范围外的科研问题；
- “为什么”“有什么限制”等连续追问；
- 中英文、长短问题及不同生物学实体。

普通寒暄不会进入 Cross Encoder，因此不属于 Evidence Gate 校准集。

## 2. 导出标注表

```powershell
.\.venv\Scripts\python.exe scripts\calibrate_evidence_gate.py export `
  --input C:\secure\synbiogpt\evidence_samples.jsonl `
  --output C:\secure\synbiogpt\evidence_labels.csv
```

在 CSV 中填写：

- `relevance_label=1`：文献片段能直接支持回答中的至少一项实质性结论；
- `relevance_label=0`：仅关键词相似、只有宽泛背景、无法回答问题或完全无关；
- 无法判断时保持空白，不要强行标注；
- `labeler_id` 填标注者代号，分歧和判断依据写入 `label_notes`。

为降低分数锚定偏差，标注时应隐藏最后一列 `raw_logit`。建议抽取一部分样本由
两名标注者独立标注，再对分歧进行复核。

## 3. 离线校准

`target-precision` 是显式业务目标，不是预设的 Cross Encoder 分数阈值。下面的
0.90 仅为命令示例，应由项目负责人根据“误注入无关证据”和“漏掉可用证据”的
成本决定。

```powershell
.\.venv\Scripts\python.exe scripts\calibrate_evidence_gate.py calibrate `
  --input C:\secure\synbiogpt\evidence_labels.csv `
  --target-precision 0.90 `
  --collection fulltext_medcpt_ip_v1 `
  --cross-encoder-model C:\models\MedCPT\Cross-Encoder `
  --validation-fraction 0.20 `
  --output C:\secure\synbiogpt\evidence_calibration_report.json
```

脚本按规范化后的 `semantic_query` 做查询级拆分，确保同一问题的候选不会同时
出现在校准集和验证集。报告包括：

- 相关/无关 raw-logit 分布；
- 每个候选阈值的 precision、recall、F1 和接受率；
- 有答案 Query 的证据召回率；
- 无答案 Query 的错误证据注入率；
- 校准集候选阈值及独立验证指标；
- 样本不足、验证未达标等警告。

只有 `recommendation_ready=true` 时报告才会给出 `suggested_environment`。即便如此，
仍应人工审核错误案例并在灰度环境验证，然后才把建议值配置为：

```env
MEDCPT_EVIDENCE_GATE_MIN_SCORE=<审核通过的 raw-logit 阈值>
```

阈值必须绑定 Cross Encoder 模型版本和知识库分布。模型、分词参数或文献库发生
显著变化后，应重新采集并校准。
