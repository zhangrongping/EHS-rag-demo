"""
EHS 规范智能问答 - Streamlit 网页版
====================================
运行：
  pip install streamlit requests chromadb fastembed
  $env:DEEPSEEK_API_KEY="sk-你的key"
  streamlit run streamlit_app.py
浏览器会自动打开 http://localhost:8501
"""

import os
import streamlit as st
import requests
import chromadb
from fastembed import TextEmbedding

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
DOC_PATH = "ehs_docs.txt"
COLLECTION_NAME = "ehs_kb"


# ---- 切块（和 rag_demo.py 一致）----
def chunk_text(text, max_len=400, overlap=50):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) + 1 <= max_len:
            current = (current + "\n" + p) if current else p
        else:
            if current:
                chunks.append(current)
            if len(p) > max_len:
                start = 0
                while start < len(p):
                    chunks.append(p[start:start + max_len])
                    start += max_len - overlap
                current = ""
            else:
                current = p
    if current:
        chunks.append(current)
    return chunks


# ---- 初始化向量库和模型（Streamlit 缓存，避免每次交互重载）----
@st.cache_resource
def init_store():
    client = chromadb.PersistentClient(path="./chroma_db")
    emb_model = TextEmbedding(model_name=EMBED_MODEL)
    return client, emb_model


def ensure_index(client, emb_model):
    try:
        col = client.get_collection(COLLECTION_NAME)
        if col.count() > 0:
            return col
    except Exception:
        pass
    if not os.path.exists(DOC_PATH):
        return None
    with open(DOC_PATH, encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_text(text)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    col = client.create_collection(COLLECTION_NAME)
    embs = [e.tolist() for e in emb_model.embed(chunks)]
    col.add(ids=[f"chunk_{i}" for i in range(len(chunks))],
            documents=chunks, embeddings=embs)
    return col


def retrieve(query, emb_model, col, top_k=3):
    q_emb = list(emb_model.embed([query]))[0].tolist()
    res = col.query(query_embeddings=[q_emb], n_results=top_k)
    return res["documents"][0]


def ask_llm(context, question, api_key):
    prompt = f"""请严格根据下面参考资料回答。资料中没有的内容，直接说"资料中未提及"，不要编造。

参考资料：
{context}

用户问题：{question}"""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False}
    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    except requests.exceptions.ConnectionError:
        return "【网络错误】连不上 API"
    if r.status_code != 200:
        return f"【HTTP {r.status_code}】{r.text[:300]}"
    return r.json()["choices"][0]["message"]["content"]


# ---- 界面 ----
st.set_page_config(page_title="EHS 规范智能问答", page_icon="📘")
st.title("📘 EHS 规范智能问答")
st.caption("基于 RAG：先检索 EHS 文档相关条款，再由大模型生成回答，避免凭空编造")

api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    st.error("未检测到 DEEPSEEK_API_KEY。请在启动本应用前设置环境变量：`$env:DEEPSEEK_API_KEY=\"sk-...\"`")
    st.stop()

client, emb_model = init_store()
col = ensure_index(client, emb_model)
if col is None:
    st.warning(f"未找到 {DOC_PATH}，请把 EHS 规范文本放到本脚本同目录。")
    st.stop()

st.sidebar.write(f"知识库已加载：{col.count()} 个文本块")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

q = st.chat_input("输入关于 EHS 规范的问题，例如：高处作业审批要求是什么？")
if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    docs = retrieve(q, emb_model, col, top_k=3)
    with st.expander("查看本次检索到的参考条款"):
        for i, d in enumerate(docs):
            st.write(f"**[{i+1}]** {d}")

    context = "\n---\n".join(docs)
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            ans = ask_llm(context, q, api_key)
        st.markdown(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})
