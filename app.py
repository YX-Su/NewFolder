"""Streamlit demo. Run: `streamlit run app.py`.

Architecture: users bring their own API keys (DeepSeek + SiliconFlow). The index
is prebuilt by the project author and loaded read-only. No backend secrets.
"""
from __future__ import annotations

import os

import streamlit as st

from mathrag.agent import SolverAgent, solve_baseline, solve_rag
from mathrag.config import EmbeddingConfig, LLMConfig, paths
from mathrag.embeddings import EmbeddingClient, RerankClient
from mathrag.indexing import load_index
from mathrag.llm import LLMClient
from mathrag.retrieval import Retriever


st.set_page_config(page_title="MathRAG · 大学数学解题助手", layout="wide")
st.title("MathRAG · 大学数学解题助手")
st.caption("RAG + Agent · DeepSeek + bge-m3 · 数学分析 / 高等代数")


# ---------- Key management ----------

with st.sidebar:
    st.header("🔑 API Keys")
    st.markdown(
        "**自带 key 使用**：所有调用走你的账号，你付费。"
        "key 只在浏览器会话内有效，不会上传到服务端。"
    )

    deepseek_key = st.text_input(
        "DeepSeek API Key",
        value=st.session_state.get("deepseek_key", os.getenv("DEEPSEEK_API_KEY", "")),
        type="password",
        help="https://platform.deepseek.com",
    )
    siliconflow_key = st.text_input(
        "SiliconFlow API Key",
        value=st.session_state.get("siliconflow_key", os.getenv("SILICONFLOW_API_KEY", "")),
        type="password",
        help="https://siliconflow.cn （embedding + rerank）",
    )
    st.session_state["deepseek_key"] = deepseek_key
    st.session_state["siliconflow_key"] = siliconflow_key

    st.markdown("---")
    variant = st.radio("Variant", ["agent", "rag", "baseline"], index=0)
    st.markdown(
        "- `baseline` — 裸调 LLM（只需 DeepSeek key）\n"
        "- `rag` — 检索 + LLM（需要两个 key）\n"
        "- `agent` — 检索 + SymPy + Agent loop（需要两个 key）"
    )


# ---------- Build clients from user-provided keys ----------

def make_llm(key: str) -> LLMClient | None:
    if not key:
        return None
    return LLMClient(
        cfg=LLMConfig(
            api_key=key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            reasoner_model=os.getenv("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner"),
        )
    )


def make_embed_clients(key: str) -> tuple[EmbeddingClient, RerankClient] | None:
    if not key:
        return None
    cfg = EmbeddingConfig(
        api_key=key,
        base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        rerank_model=os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
    )
    return EmbeddingClient(cfg=cfg), RerankClient(cfg=cfg)


@st.cache_resource
def load_prebuilt_index():
    p = paths()
    if not (p.index / "chunks.jsonl").exists():
        return None
    return load_index(p.index)


index = load_prebuilt_index()
if index is None:
    st.info(
        "ℹ️ 索引未构建。`baseline` 模式不受影响；`rag` / `agent` 需要项目作者先跑 "
        "`python scripts/build_index.py` 构建索引。"
    )


# ---------- Gate ----------

def required_keys_ok() -> tuple[bool, str]:
    if not deepseek_key:
        return False, "缺 DeepSeek key（左侧填一下）"
    if variant != "baseline":
        if not siliconflow_key:
            return False, "缺 SiliconFlow key（rag / agent 需要做 embedding + rerank）"
        if index is None:
            return False, "索引未构建，当前部署只能跑 baseline"
    return True, ""


ok, gate_msg = required_keys_ok()


# ---------- Main UI ----------

problem = st.text_area(
    "题目（支持 LaTeX）",
    height=160,
    placeholder=r"例：求 $\lim_{x\to 0} \frac{\sin 3x}{x}$",
)

if not ok and problem:
    st.warning(gate_msg)

if st.button("求解", type="primary", disabled=not (problem and ok)):
    with st.spinner("求解中..."):
        llm = make_llm(deepseek_key)
        assert llm is not None  # gate guarantees

        if variant == "baseline":
            result = solve_baseline(problem, llm=llm)
        else:
            embed_clients = make_embed_clients(siliconflow_key)
            assert embed_clients is not None and index is not None
            embedder, reranker = embed_clients
            retriever = Retriever(
                index=index,
                embedder=embedder,
                reranker=reranker,
                llm=llm,
            )
            if variant == "rag":
                result = solve_rag(problem, retriever=retriever, llm=llm)
            else:
                result = SolverAgent(retriever=retriever, llm=llm).solve(problem)

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
