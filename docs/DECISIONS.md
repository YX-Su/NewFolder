# 技术决策文档

> 这份文档是面试时讲项目的核心物料。每个决策都包含：选了什么、为什么、放弃了什么、风险点。

## 0. 总体原则

1. **数据质量 > 框架花活**：PDF 解析是项目上限，先死磕这一段。
2. **评测先行**：第一周就有可跑的 eval pipeline，否则后续改进无据可依。
3. **手搓 vs 框架**：能用 100 行代码说清楚的，不引入 LangChain。
   - 求职场景下，手搓 ReAct loop 比"调 LangChain Agent"更能体现深度。
   - 代价是没法白嫖框架升级，但 MVP 周期内不是问题。
4. **API-first**：所有重模型（embedding、rerank、LLM）走 API，本地零 GPU 依赖。

---

## 1. LLM 选型

**选用**：DeepSeek-V3 作为主求解模型；DeepSeek-Reasoner 作为难题求解 / LLM-judge；
GPT-4o-mini 或 Qwen-Max 作为对照基线。

**理由**：
- DeepSeek-V3 在 MATH / CMATH / Gaokao-Math 上接近 GPT-4，**单价是 GPT-4o 的 1/30**。
- DeepSeek-Reasoner（R1 类）在大学数学题上链式推理质量显著好于 V3，单价仍可控。
- 200 元预算下，按 V3 约 ¥1/百万输入 token 算，足够跑 ~500 次完整评测 + 上千次开发联调。

**放弃**：
- ❌ Kimi：上下文超长，但数学推理略弱，定价没有 DeepSeek 划算。
- ❌ 本地 Qwen2.5-Math-7B：无 GPU 跑起来 token/s 太低，且我们要的是"应用层"信号。

**风险**：
- DeepSeek API 稳定性，需要在 `llm.py` 里加 `retry + 多 provider fallback`。

---

## 2. PDF 解析

**选用**：**MinerU (`magic-pdf`)** 作为主解析器；Mathpix 作为关键页 fallback。

**理由**：
- MinerU 由上海人工智能实验室开源，**专门针对中文学术内容 + 公式**，
  输出结构化 Markdown，公式自动转 LaTeX。
- 完全开源、无 API 费用，CPU 也能跑（慢但可用）。
- 对比：
  - **Mathpix**：精度最高但 $0.004/页起，一本 600 页教材 = ¥17，3 本就吃掉预算 25%。仅作 fallback。
  - **Marker**：英文学术好，中文教材公式抽取明显逊于 MinerU。
  - **Nougat (Meta)**：仅英文学术论文优化，中文教材不可用。
  - **`pdfplumber` / `PyMuPDF`**：基础文本可以，但**公式会变成乱码**，直接 PASS。

**关键实现细节**：
- MinerU 输出的 Markdown 里，公式被包成 `$...$` 或 `$$...$$`，**这就是我们后续检索的关键载体**。
- 跨页公式 / 表格在 MinerU v1.0+ 已较好处理，但仍需后处理：
  - 合并被切断的 `证明` 块
  - 修正 `定理 1.2.3` 这种编号的 OCR 错误（数字 1/l/I 混淆）

**风险**：
- 教材扫描质量不一，老版教材的 PDF 可能是扫描件而非数字 PDF —— 这种情况 MinerU 也吃力。
- **若 PDF 是扫描件**，需要先用 OCR 走一遍（MinerU 内置了 PaddleOCR，但中文公式 OCR 精度还是有限）。

---

## 3. 分块（Chunking）

**选用**：**语义结构分块**——按"定义 / 定理 / 命题 / 推论 / 例 / 证明 / 解"切。

**理由**：
- 数学教材的语义单元天然就是这些块，**按字数切会把"定理陈述"和"证明"切散**，严重伤检索。
- MinerU 输出的 Markdown 里这些标记都很规整（"定义 1.2.3"、"例 4.5"、"证明"），用正则就能命中。

**Chunk schema**：
```python
{
    "id": "数学分析_ch3_thm_2_1",
    "text": "...",  # 包含 LaTeX
    "natural_language": "...",  # 公式转中文口语化（提升 embedding 召回）
    "metadata": {
        "type": "theorem|definition|proposition|corollary|example|proof|solution",
        "book": "数学分析（华师大第五版）",
        "chapter": "第三章 函数极限",
        "section": "§3.2 函数极限的性质",
        "number": "定理 3.2.1",
        "page": 87,
        "linked_id": "数学分析_ch3_thm_2_1",  # 证明指向定理
    },
}
```

**关键细节**：
- 长证明（>800 字）做**滑窗切割但保留 metadata**，子块共享同一个 `linked_id`，
  检索到子块时整段回填。
- "natural_language" 字段：把 `$\lim_{x\to 0} \frac{\sin x}{x} = 1$` 转成
  "x 趋于 0 时 sin x 比 x 的极限等于 1" —— 这一步显著提升对**自然语言提问**的召回。
  实现上先用正则覆盖 80% 常见模式，剩下兜底用 LLM batch 转一次。

**放弃**：
- ❌ 固定字数切：常见 RAG 默认做法，但数学场景完全不适用。
- ❌ Semantic chunking by embedding similarity：开销大、且对结构化文本反而不如规则。

---

## 4. 检索

**选用**：**Hybrid (BM25 + Dense) + Rerank + Query Rewrite**。

### 4.1 Embedding
- **模型**：`BAAI/bge-m3` —— 中文 SOTA、支持长文本、且同模型可同时给 dense / sparse / multi-vec。
- **服务**：SiliconFlow API（有免费额度，超出后 ¥0.5/百万 token）。
- 备选：`bge-large-zh-v1.5`（更小更快，精度略低）。

### 4.2 BM25
- **分词**：`jieba` + 自定义数学词典（"极限"、"连续"、"特征值"、`\int`、`\sum` 等作为词条）。
- **关键决策**：LaTeX 命令 `\lim`、`\int` **作为独立 token** 保留——这让用户用 LaTeX 提问时也能命中。

### 4.3 Rerank
- `BAAI/bge-reranker-v2-m3`，对 BM25+dense 召回的 top-30 重排到 top-5。

### 4.4 Query Rewrite
- 用 LLM 把用户题目重写成 2~3 个**检索友好的查询**，例如：
  - 原题：`求 $\lim_{x\to 0} \frac{\sin 3x}{x}$`
  - 重写：["sin x 比 x 极限定理"、"等价无穷小替换"、"重要极限 sin x / x"]
- 也会让 LLM 输出"这道题可能用到的定理名"，作为高权重检索锚点。

### 4.5 为什么是这个组合
- 数学题里**专有名词**（定理名、概念名）权重极高，纯 dense 容易被语义"漂移"误导，BM25 兜底必要。
- 用户提问常是**符号题面**而非概念名，dense 又比 BM25 强 —— 所以 hybrid。
- Rerank 解决 hybrid 召回噪声大的问题。
- Query Rewrite 解决"题目"和"教材"之间的**领域 gap**（题面是符号，教材是论述）。

---

## 5. Agent 设计

**选用**：**单 Agent + 工具调用**（手搓 ReAct loop，不上 LangChain）。

**工具集**：
1. `retrieve_textbook(query: str, k: int = 5)` —— 走第 4 节那套检索
2. `verify_with_sympy(expression: str, claim: str)` —— 把 LLM 给的结论用 SymPy 跑一遍
3. `search_related_examples(theorem_name: str)` —— 按定理名拉所有例题（命中 metadata.linked_id）

**Loop**：
```
while step < MAX_STEPS:
    决策 = LLM.plan(problem, history)
    if 决策.action == "answer":
        return 决策.final_answer
    result = 执行工具(决策.tool, 决策.args)
    history.append((决策, result))
```

**为什么不上多 Agent（检索/解题/验证三 Agent）**：
- MVP 阶段加 Agent 数 = 加调试成本 + 加 token 成本，**收益未验证**。
- 评测体系建好后，做一个"single vs multi-agent" 的对比实验，**有数据再决定** —— 这本身就是一个简历加分点。
- 当前阶段保留多 Agent 接口（`agent.py` 里 `SolverAgent` 是 base class），但默认走 single。

---

## 6. 评测

### 6.1 评测集
- **种子集**：`data/eval/seed.jsonl`，10 道手工标注题
- **目标规模**：4 周内扩到 50~100 题，覆盖：
  - 数学分析：极限 / 连续 / 微分 / 积分 / 级数 / 多元微积分
  - 高等代数：矩阵 / 行列式 / 线性方程组 / 线性空间 / 特征值 / 二次型
- 每题标注：
  ```json
  {
    "id": "...",
    "problem": "...",          // LaTeX
    "topic": "数学分析/极限",
    "ground_truth_answer": "...",
    "ground_truth_chunks": ["chunk_id_1", "chunk_id_2"],  // 该题应该命中的教材片段
    "rubric": "应使用夹逼定理；最终答案 = 1"
  }
  ```

### 6.2 指标
| 指标 | 含义 | 实现 |
| --- | --- | --- |
| `retrieval_recall@k` | top-k 是否包含标注 chunk | 集合交集 |
| `final_answer_match` | 答案数值/符号是否匹配 | `sympy.simplify(ans - gt) == 0` |
| `faithfulness@judge` | 解题过程是否正确、引用是否合理 | DeepSeek-Reasoner 按 rubric 打分 0-5 |
| `tokens_per_problem` | 每题平均消耗 token | 直接统计 |
| `latency_p50/p95` | 延迟 | 直接统计 |

### 6.3 对照
| 变体 | 检索 | Agent | SymPy |
| --- | --- | --- | --- |
| `baseline` | ❌ | ❌（直接调 LLM） | ❌ |
| `rag` | ✅ | ❌（context 拼进 prompt） | ❌ |
| `agent` | ✅ | ✅ | ✅ |

**消融实验**：
- `rag` 内部再切：`bm25_only` / `dense_only` / `hybrid` / `hybrid+rerank` / `hybrid+rerank+rewrite`
- Chunking：`fixed_500` vs `semantic`
- 这些数字就是最后写在 README / 博客里的**核心结果表**。

---

## 7. 公式检索的专门处理（已知最大坑）

**问题**：
1. 用户问 `求 $\int_0^1 x^2 dx$`，教材里写 `定积分的牛顿-莱布尼茨公式`，
   纯文本 embedding 几乎对不上。
2. LaTeX 在 embedding 模型里通常作为"噪声 token"被忽略。

**对策（按效果递增）**：
1. **公式→自然语言 dual indexing**（见 §3）：每个 chunk 存两份文本，embedding 用拼接版。
2. **题面 → 概念词**：query rewrite 阶段强制 LLM 输出"涉及的数学概念名"，这一支走 BM25。
3. **Symbol-aware tokenizer**：jieba 词典里塞入 `\int`、`\sum`、`\lim`、`\partial`、`\nabla` 等。
4. （后续）**公式归一化**：`\int_0^1 x^2 \,dx` → `INTEGRAL_DEF(LOWER=0, UPPER=1, INTEGRAND=x^2, VAR=x)`
   作为一个独立的检索字段。MVP 不做，留作 v2。

---

## 8. 取舍清单（写简历用）

| 做了什么 | 没做什么 | 理由 |
| --- | --- | --- |
| 手搓 ReAct Agent | 没用 LangChain | 200 行代码 vs 整套框架，求职场景下手搓更值钱 |
| Hybrid + Rerank + Rewrite | 没做 GraphRAG | 教材结构清晰，GraphRAG 复杂度收益不明显 |
| Eval 先行 | 没急着做 UI | 没数据的 demo 是空中楼阁 |
| 单 Agent | 没上多 Agent | 留作消融实验的对照 |
| MinerU | 没用 Mathpix | 预算 |
| SymPy 验证 | 没接 Wolfram Alpha | 离线、免费、对大学数学够用 |
| Chroma | 没上 Milvus / Qdrant | 数据量 < 10 万 chunks，Chroma 足够 |

---

## 9. 后续计划

- **v0.1（本周）**：骨架 + 评测框架 + 种子评测集（10 题）
- **v0.2**：接入 1 本真实教材 PDF，跑通端到端，出第一份评测数字
- **v0.3**：完成消融实验，写技术博客
- **v0.4**：扩到 3 本教材 + 50 题评测集，录演示视频
- **v1.0**：（如有时间）公式归一化、多 Agent 对比、PR 给 MinerU 修 bug
