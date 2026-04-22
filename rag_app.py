import os
import json
import streamlit as st
import jieba
from langchain_core.documents import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Minecraft Wiki 知识库助手", page_icon="⛏️", layout="wide")

# ==========================================
# 1. 向量库与检索器初始化
# ==========================================
@st.cache_resource(show_spinner=False)
def init_retriever():
    def load_data(folder_path):
        documents = []
        if not os.path.exists(folder_path):
            return documents
            
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                filepath = os.path.join(folder_path, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    content_lines = []
                    
                    if 'structured_content' in data:
                        for section_title, text_list in data['structured_content'].items():
                            content_lines.append(f"### {section_title} ###")
                            for text_item in text_list:
                                if text_item.strip():  
                                    content_lines.append(text_item)
                            content_lines.append("") 
                    else:
                        content_lines.append(data.get('text', ''))
                        
                    full_text = "\n".join(content_lines)
                    
                    doc = Document(
                        page_content=full_text,
                        metadata={
                            'title': data.get('title', '未知标题'),
                            'source_url': data.get('source_url', '')
                        }
                    )
                    documents.append(doc)
        return documents

    embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    persist_dir = "./chroma_db"

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        docs = load_data('./structured_output') 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_dir)

    return vectorstore.as_retriever(search_kwargs={"k": 3})

# ==========================================
# 2. 动态创建 QA 链
# ==========================================
def get_qa_chain(model_type, ds_api_key=""):
    if model_type == "本地模型 (LM Studio)":
        llm = ChatOpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="Qwen/Qwen3.5-9B",
            temperature=0.7,
            top_p=0.8,
            presence_penalty=1.5,
            max_tokens=32768,
            streaming=True,
            model_kwargs={
                "extra_body": {
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            }
        )
    else:
        llm = ChatOpenAI(
            base_url="https://api.deepseek.com",
            api_key=ds_api_key,
            model="deepseek-chat",
            temperature=0.7,
            streaming=True
        )

    system_prompt = (
        "你是一个Minecraft知识库智能助手。请使用以下检索到的背景信息来回答问题。"
        "如果你不知道答案，就说你不知道，不要编造。"
        "\n\n"
        "背景信息："
        "\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    return create_stuff_documents_chain(llm, prompt)

# ==========================================
# 3. 自定义分词与检索策略
# ==========================================
@st.cache_data
def get_stopwords():
    return {"的", "是", "了", "在", "怎么", "什么", "和", "与", "如何", "帮我", "查一下", "告诉我", "有哪些"}

def retrieve_by_keywords(retriever, user_input):
    stopwords = get_stopwords()
    words = jieba.lcut(user_input)
    # 过滤停用词和空字符，得到纯净的关键词列表
    keywords = [w for w in words if w not in stopwords and len(w.strip()) > 0]
    
    # 【核心改动】将用户原始的完整输入加入到检索列表的首位
    original_input = user_input.strip()
    if original_input and original_input not in keywords:
        keywords.insert(0, original_input)
    
    # 兜底逻辑：如果输入全是停用词或为空
    if not keywords:
        keywords = [user_input]
        
    st.toast(f"🔍 触发扩展检索，检索项: {keywords}", icon="🧩")
    
    all_docs = []
    # 针对每个检索项分别进行检索（原句会优先检索）
    for kw in keywords:
        docs = retriever.invoke(kw)
        all_docs.extend(docs)
        
    # 文档去重
    unique_docs = []
    seen_content = set()
    for doc in all_docs:
        if doc.page_content not in seen_content:
            unique_docs.append(doc)
            seen_content.add(doc.page_content)
            
    return unique_docs

# ==========================================
# 4. 页面 UI & 侧边栏
# ==========================================
with st.sidebar:
    st.image("https://zh.minecraft.wiki/images/Wiki.png", width=150)
    st.title("系统设置")
    
    model_choice = st.radio("选择大模型后端:", ["本地模型 (LM Studio)", "云端模型 (DeepSeek API)"])
    deepseek_api_key = ""
    
    if model_choice == "云端模型 (DeepSeek API)":
        deepseek_api_key = st.text_input("请输入 DeepSeek API Key", type="password", placeholder="sk-...")
        if not deepseek_api_key:
            st.warning("使用云端模型需输入 API Key")
            
    st.markdown("---")
    st.title("系统状态")
    with st.spinner("正在连接大脑与记忆库..."):
        retriever = init_retriever()
    st.success("向量库加载完毕")
    st.success(f"当前选中: {model_choice.split(' ')[0]}")
    
    st.markdown("---")
    st.markdown("### 使用说明")
    st.markdown("基于本地 ChromaDB 构建的 RAG 问答系统。可在侧边栏无缝切换本地计算与云端 API。")

# ==========================================
# 5. 主界面与聊天逻辑
# ==========================================
st.title("⛏️ Minecraft 知识库智能助手")
st.caption("我是专属 RAG 助手，有什么关于游戏机制的问题都可以问我！")

if model_choice == "云端模型 (DeepSeek API)" and not deepseek_api_key:
    st.info("👈 请在左侧边栏输入 DeepSeek API Key 以开始对话。")
    st.stop()

qa_chain = get_qa_chain(model_choice, deepseek_api_key)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "你好，今天想查询点什么？"}]

# 【核心改动1】渲染历史消息时，增加对 debug 信息的展示
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # 渲染发送给 AI 的 Prompt (如果有)
        if "debug_prompt" in msg and msg["debug_prompt"]:
            with st.expander("🛠️ 查看发送给 AI 的实际内容 (Debug)"):
                st.code(msg["debug_prompt"], language="markdown")
                
        # 渲染参考来源
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 查看参考来源"):
                for source in msg["sources"]:
                    st.markdown(f"- [{source['title']}]({source['source_url']})")

if user_input := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        source_list = []
        debug_prompt_text = "" # 用于保存当前的 prompt 文本
        
        with st.spinner("正在拆解关键词并检索记忆库..."):
            try:
                retrieved_docs = retrieve_by_keywords(retriever, user_input)
                
                for doc in retrieved_docs:
                    source_list.append({
                        "title": doc.metadata.get('title', '未知标题'),
                        "source_url": doc.metadata.get('source_url', '#')
                    })
                
                # 【核心改动2】将我们要喂给 AI 的内容格式化成字符串，并在 UI 上展示
                context_str = "\n\n".join([f"【片段 {i+1}】\n{doc.page_content}" for i, doc in enumerate(retrieved_docs)])
                debug_prompt_text = (
                    "**[System]**\n"
                    "你是一个Minecraft知识库智能助手。请使用以下检索到的背景信息来回答问题。"
                    "如果你不知道答案，就说你不知道，不要编造。\n\n"
                    "**[Context]**\n"
                    f"{context_str}\n\n"
                    "**[User]**\n"
                    f"{user_input}"
                )
                
                with st.expander("查看发送给 AI 的实际合成检索内容"):
                    st.code(debug_prompt_text, language="markdown")
                
                for chunk in qa_chain.stream({
                    "context": retrieved_docs, 
                    "input": user_input
                }):
                    if isinstance(chunk, str):
                        full_response += chunk
                    else:
                        full_response += chunk.get("answer", "")
                        
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                unique_sources = [dict(t) for t in {tuple(d.items()) for d in source_list}]

                if unique_sources:
                    with st.expander(f"查看参考来源 (共召回 {len(retrieved_docs)} 个片段)"):
                        for source in unique_sources:
                            st.markdown(f"- [{source['title']}]({source['source_url']})")
                
                # 【核心改动3】将 debug_prompt_text 存入历史记录，防止页面刷新后丢失
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "sources": unique_sources,
                    "debug_prompt": debug_prompt_text
                })
                
            except Exception as e:
                st.error(f"生成回复时出错！详细报错: {str(e)}")