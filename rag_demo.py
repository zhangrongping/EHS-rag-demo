"""
第 2 周：一个最朴素的 RAG —— EHS 规范问答
=============================================
它做的事：
  你的 EHS 文档 → 切成小块 → 每块转成向量存起来
  你提问      → 问题也转成向量 → 找出最相关的 3 段 → 拼进 Prompt → 问大模型

用法：
1. pip install requests chromadb fastembed
2. 把一份 EHS/安全生产规范文本保存为 ehs_docs.txt，放在本脚本同目录
   （PDF 的话用 Word 打开另存为 txt；先用网上找的公开规范也行，别用客户涉密原文）
3. $env:DEEPSEEK_API_KEY="sk-你的key"
4. python rag_demo.py
"""

import os
import sys
import requests
import chromadb
from fastembed import TextEmbedding

# ===== 配置 =====
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"   # 中文 embedding 模型，第一次运行自动下载
DOC_PATH = "ehs_docs.txt"
COLLECTION_NAME = "ehs_kb"


def get_api_key():
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("错误：先在当前 PowerShell 窗口执行：")
        print('  $env:DEEPSEEK_API_KEY="sk-你的key"')
        sys.exit(1)
    return key


# 1) 读文档
def load_doc(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# 2) 切块：短段落先攒起来，攒到接近 max_len 再成一块；
#    单段超过 max_len 的才按字符硬切，带重叠避免切断语义。
def chunk_text(text, max_len=400, overlap=50):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""
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


# 3) 初始化向量库（持久化到 ./chroma_db 文件夹）和 embedding 模型
def init_store():
    client = chromadb.PersistentClient(path="./chroma_db")
    emb_model = TextEmbedding(model_name=EMBED_MODEL)
    return client, emb_model


def build_index(chunks, emb_model, client):
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    col = client.create_collection(COLLECTION_NAME)

    embeddings = list(emb_model.embed(chunks))
    embeddings_list = [e.tolist() for e in embeddings]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    col.add(ids=ids, documents=chunks, embeddings=embeddings_list)
    print(f"已索引 {len(chunks)} 个文本块")


# 4) 检索：把问题转成向量，找最相近的 top_k 段
def retrieve(query, emb_model, col, top_k=3):
    q_emb = list(emb_model.embed([query]))[0].tolist()
    res = col.query(query_embeddings=[q_emb], n_results=top_k)
    return res["documents"][0]


# 5) 把检索到的片段塞进 Prompt，问大模型
def ask_llm(context, question, api_key):
    prompt = f"""请严格根据下面提供的参考资料回答用户问题。
如果参考资料里没有相关内容，就直接回答"资料中未提及"，不要自己编造。

参考资料：
{context}

用户问题：{question}
"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    except requests.exceptions.ConnectionError:
        return "【网络错误】连不上 API"
    if resp.status_code != 200:
        return f"【HTTP {resp.status_code}】{resp.text[:300]}"
    return resp.json()["choices"][0]["message"]["content"]


def main():
    api_key = get_api_key()

    if not os.path.exists(DOC_PATH):
        print(f"没找到 {DOC_PATH}。请把 EHS 规范文本存成 txt 放到脚本同目录。")
        sys.exit(1)

    text = load_doc(DOC_PATH)
    chunks = chunk_text(text)
    print(f"文档共 {len(text)} 字，切成 {len(chunks)} 块")

    client, emb_model = init_store()
    build_index(chunks, emb_model, client)
    col = client.get_collection(COLLECTION_NAME)

    print("\nEHS 知识库问答已就绪（输入 exit 退出）")
    print("试试问：高处作业审批要求是什么？/ 安全帽佩戴有什么规定？\n")

    while True:
        q = input("你的问题：").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit", "退出"):
            break
        docs = retrieve(q, emb_model, col, top_k=3)
        context = "\n---\n".join(docs)
        print("（检索到的参考片段：）")
        for i, d in enumerate(docs):
            print(f"  [{i+1}] {d[:60]}...")
        ans = ask_llm(context, q, api_key)
        print(f"\n回答：{ans}\n")


if __name__ == "__main__":
    main()
