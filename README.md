# MathRAG — 中国大学数学题辅助求解工具

面向中国大学数学（数学分析、高等代数）的 **RAG + Agent** 系统。输入一道题，
检索教材中相关的定义、定理、例题，调用 LLM 给出解题思路，并可选用 SymPy 做符号验证。

> 定位：录题工作流加速 + 大模型应用层岗位求职作品集。

## 项目状态

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| 决策文档 | ✅ | `docs/DECISIONS.md` — 所有选型理由 |
| LLM 客户端 | ✅ | `src/mathrag/llm.py` — OpenAI 兼容协议，可切 DeepSeek/Qwen/Kimi |
| Prompt 模板 | ✅ | `src/mathrag/prompts.py` |
| 语义分块 | ✅ | `src/mathrag/chunking.py` — 按"定义/定理/例题/证明"切 |
| Embedding | ✅ | `src/mathrag/embeddings.py` — SiliconFlow `bge-m3` |
| 索引 | ✅ | `src/mathrag/indexing.py` — Chroma + BM25 (jieba) |
| 检索 | ✅ | `src/mathrag/retrieval.py` — Hybrid + Rerank + Query Rewrite |
| Agent | ✅ | `src/mathrag/agent.py` — 手搓 ReAct，工具：检索 + SymPy |
| 评测框架 | ✅ | `src/mathrag/eval.py` + `data/eval/seed.jsonl` |
| Streamlit | ✅ | `app.py` |
| PDF 解析 | 🟡 | `src/mathrag/parsing.py` — MinerU 集成（需用户单独安装 magic-pdf） |
| 真实教材数据 | ⏳ | 待用户提供 PDF |
| Benchmark 跑通 | ⏳ | 拿到数据后第一件事 |

## 快速开始

```bash
# 1. 装依赖
pip install -e .

# 2. 配 .env（拷 .env.example，填 DeepSeek + SiliconFlow key）
cp .env.example .env

# 3. 解析 PDF -> Markdown（需先 pip install magic-pdf[full]）
python scripts/parse_pdf.py --pdf data/raw/数学分析.pdf --out data/parsed/

# 4. 构建索引
python scripts/build_index.py --src data/parsed/ --out data/index/

# 5. 跑评测
python scripts/run_eval.py --eval data/eval/seed.jsonl --variants baseline,rag,agent

# 6. 起 Demo
streamlit run app.py
```

## 架构

```
PDF (教材)
  ↓ MinerU (magic-pdf)
Markdown + LaTeX 公式
  ↓ 语义切块 (按 定义/定理/例/证明)
Chunks (带 metadata: type, chapter, section)
  ↓ Embedding (bge-m3) + BM25 (jieba)
Vector Store (Chroma) + BM25 Index
  ↓
[用户问题]
  ↓ Query Rewrite → 多路检索 → Rerank (bge-reranker-v2-m3) → Top-K
  ↓
Solver Agent (手搓 ReAct loop)
  ├── tool: retrieve_textbook(query) — 检索教材
  ├── tool: verify_with_sympy(expr) — 符号验证（可选）
  └── final answer (带引用)
```

## 评测

- **种子集**：`data/eval/seed.jsonl`，10 道手工标注题（数学分析 + 高等代数）。
- **指标**：
  - 检索：`recall@k` —— 标注的 ground-truth 章节是否在 top-k 中
  - 答案：`final_answer_match` —— 数值/符号答案匹配（用 SymPy）
  - 解答忠实度：`faithfulness@LLM-judge` —— 用强模型按 rubric 打分
- **对照**：
  - `baseline`：裸调用 LLM
  - `rag`：检索 + LLM（无 Agent）
  - `agent`：检索 + Agent + SymPy

详见 `docs/DECISIONS.md` §6。
