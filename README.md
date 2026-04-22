---

# ⛏️ Minecraft Wiki 知识库智能助手 (RAG-MCwiki)

本项目是一个专为 **Minecraft** 玩家和研究者设计的本地化知识检索增强生成（RAG）系统。通过采集并结构化 Minecraft Wiki 数据（2026/4/22），结合本地大模型（Local LLM）与云端 API，实现精准、低延迟、无幻觉的游戏机制问答。

## 🌟 核心特性

   **全本地化隐私安全**：支持通过 LM Studio 驱动本地模型（如 Qwen3.5），数据无需上传云端。
   **混合检索优化**：结合了 `Jieba` 分词的关键词提取与 `ChromaDB` 的语义向量检索，大幅提升专有名词召回率。
   **结构化知识感知**：针对 Wiki 的 JSON 数据进行分层解析，保留 Markdown 标题结构，增强模型对上下文的理解。
   **实时流式响应**：基于 Streamlit 的流式输出技术，实现毫秒级的首字响应（TTFT）。
   **透明化溯源**：每条回答均附带参考来源折叠面板，支持查看原始 Wiki 链接。
   **开发者调试模式**：内置 Debug 面板，实时展示发送给 AI 的实际 Prompt 组装内容。

## 🛠️ 技术栈

   **前端交互**：Streamlit
   **大模型编排**：LangChain (Core / Community / Classic)
   **向量数据库**：ChromaDB (Persistent Storage)
   **词向量模型**：`shibing624/text2vec-base-chinese` (HuggingFace)
   **分词引擎**：Jieba
   **模型支持**：LM Studio (OpenAI 兼容接口) / DeepSeek API

## 📂 项目结构

```text
.
├── rag_app.py              # Streamlit 主程序
├── structured_output/      # 存放清洗后的结构化 Wiki JSON 文件
├── chroma_db/              # 向量数据库持久化目录（自动生成）
├── requirements.txt        # 依赖列表
└── README.md               # 项目说明文档
```

## 🚀 快速开始

### 1. 环境准备
确保你的环境中已安装 Python 3.10+。建议使用虚拟环境：

```bash
python -m venv venv
source venv/bin/activate  # Windows 使用 venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 数据准备
将结构化的 Minecraft Wiki 数据（JSON 格式）放入 `./structured_output` 文件夹中。数据应包含 `title`, `source_url`, `structured_content` 或 `text` 字段。

### 3. 配置大模型
   **本地模式**：启动 **LM Studio**，加载模型，并开启 Local Server（默认端口 1234）。
   **云端模式**：获取 **DeepSeek API Key**。

### 4. 运行应用
```bash
streamlit run rag_app.py
```

## 📝 技术要点说明

### 语义检索逻辑
系统采用 `RecursiveCharacterTextSplitter` 将文档切分为 500 字的片段，并保留 50 字的重叠（Overlap）。检索时，系统会先对用户提问进行 `Jieba` 分词，将“原句 + 核心关键词”共同投入向量空间进行多路召回，通过内容去重后选取最相关的 Top-3 片段。

### 双后端切换
在 `get_qa_chain` 函数中，系统通过修改 `base_url` 实现了对标准 OpenAI 协议的“接口劫持”。无论使用本地算力还是云端算力，业务逻辑均保持一致。

### 缓存优化
利用 `@st.cache_resource` 装饰器，确保高权重的 Embedding 模型和向量数据库仅在启动时加载一次，避免 Streamlit 响应式机制导致的重复加载开销。

---

**提示**：在使用本地模型时，请确保显存充足。若遇到网络连接问题（如无法下载 Embedding 模型），代码已内置镜像站 `hf-mirror.com` 的重定向逻辑。
