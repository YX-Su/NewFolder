"""Streamlit demo. Run: `streamlit run app.py`."""
from __future__ import annotations

import streamlit as st

from mathrag.agent import SolverAgent, solve_baseline, solve_rag
from mathrag.config import paths
from mathrag.indexing import load_index
from mathrag.retrieval import Retriever


st.set_page_config(page_title="MathRAG · 大学数学解题助手", layout="wide")
st.title("MathRAG · 大学数学解题助手")
st.caption("RAG + Agent · DeepSeek + bge-m3 · 数学分析 / 高等代数")


@st.cache_resource
def get_retriever() -> Retriever | None:
    p = paths()
    if not (p.index / "chunks.jsonl").exists():
        return None
    idx = load_index(p.index)
    return Retriever(index=idx)


retriever = get_retriever()
if retriever is None:
    st.warning("索引未构建。先跑 `python scripts/build_index.py`。当前只能用 baseline 模式。")

with st.sidebar:
    variant = st.radio("Variant", ["agent", "rag", "baseline"], index=0)
    st.markdown("---")
    st.markdown("**Variant 说明**")
    st.markdown("- `baseline` — 裸调 LLM\n- `rag` — 检索 + LLM\n- `agent` — 检索 + SymPy 验证 + Agent loop")

problem = st.text_area(
    "题目（支持 LaTeX）",
    height=160,
    placeholder=r"例：求 $\lim_{x\to 0} \frac{\sin 3x}{x}$",
)

if st.button("求解", type="primary", disabled=not problem):
    with st.spinner("求解中..."):
        if variant == "baseline":
            result = solve_baseline(problem)
        elif variant == "rag" and retriever is not None:
            result = solve_rag(problem, retriever)
        elif variant == "agent" and retriever is not None:
            result = SolverAgent(retriever=retriever).solve(problem)
        else:
            st.error("当前 variant 需要索引，请先构建。")
            st.stop()

    st.markdown("### 解答")
    st.markdown(result.answer)

    cols = st.columns(3)
    cols[0].metric("prompt tokens", result.prompt_tokens)
    cols[1].metric("completion tokens", result.completion_tokens)
    cols[2].metric("variant", result.variant)

    if result.retrieved_chunk_ids:
        with st.expander(f"检索到的教材片段 ({len(result.retrieved_chunk_ids)})"):
            for cid in result.retrieved_chunk_ids:
                st.code(cid)

    if result.trace:
        with st.expander(f"Agent trace ({len(result.trace)} steps)"):
            for i, step in enumerate(result.trace, 1):
                st.markdown(f"**Step {i} · {step.action}**")
                st.markdown(f"> {step.thought}")
                st.json({"args": step.args, "observation": step.observation})
