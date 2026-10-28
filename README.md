# EHS 规范智能问答（RAG 知识库 Demo）

## 项目背景

在 EHS（环境健康安全）管理平台交付过程中，客户现场人员经常反复查询各类安全规范条款，既依赖人工翻阅文档,以及责任无法落实到业务线、管理退化成“留痕”、专业能力与资源不足，大模型直接回答又容易编造条款号和内容。本项目是一个私有化知识库问答应用：上传企业自己的规范文档后，用户用自然语言提问，系统先检索文档中相关条款，再由大模型据此生成回答，并标注参考来源，避免幻觉。

## 功能

- 加载本地 EHS 规范文档（txt），自动切块并向量化入库
- 自然语言提问，返回带参考条款的回答
- 网页界面（Streamlit），可展开查看本次命中的原始条款
- 容器化打包（Docker），可部署到内网环境

## 技术栈

| 环节 | 选型 | 说明 |
|---|---|---|
| LLM | DeepSeek Chat API | 生成最终回答 |
| Embedding | bge-small-zh-v1.5（fastembed，本地） | 中文文本向量化 |
| 向量库 | ChromaDB（本地持久化） | 存储与相似度检索 |
| 切块 | 自研：短段落聚合 + 长段重叠硬切 | |
| 界面 | Streamlit | 快速构建对话式网页 |
| 部署 | Docker | 容器化交付 |

## 本地运行

```powershell
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置大模型 API Key
$env:DEEPSEEK_API_KEY="sk-你的key"

# 3. 把规范文档保存为 ehs_docs.txt 放到项目目录

# 4. 启动
streamlit run streamlit_app.py
# 浏览器打开 http://localhost:8501
```

## Docker 运行

```powershell
docker build -t ehs-rag .
docker run -p 8501:8501 -e DEEPSEEK_API_KEY=sk-你的key ehs-rag
```

## 项目结构

```
├── streamlit_app.py      # 网页应用主程序
├── rag_demo.py           # 命令行版（理解 RAG 流程用）
├── ehs_docs.txt          # 知识库文档（需自备，勿提交涉密内容）
├── requirements.txt
├── Dockerfile
└── README.md
```

## 实现要点与踩过的坑

1. **切块策略**：规范文档每行很短，最初按行切导致平均块仅 50 字、语义不完整；改为短段落聚合到约 400 字再切块后，检索相关性明显提升。
2. **检索防幻觉**：Prompt 中明确约束"资料中没有就回答未提及"，并在界面展示命中的原始条款，便于人工核对。
3. **网络问题**：fastembed 默认从 HuggingFace 下载模型，国内需设置 `HF_ENDPOINT=https://hf-mirror.com` 走镜像。
