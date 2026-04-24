import os
import re
import json
import shutil
import sqlite3
import time
import hashlib
from datetime import datetime
import requests
import streamlit as st
import jieba
from langchain_core.documents import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 静默 transformers 路径访问日志
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

st.set_page_config(page_title="Minecraft Wiki 知识库助手", page_icon="⛏️", layout="wide")

# ==========================================
# 辅助函数与常量定义
# ==========================================
THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
REASONING_BLOCK_TYPES = {"reasoning", "reasoning_content", "thinking", "thought"}

DETAIL_REQUIREMENTS = {
    "简洁": "回答控制在3到5句，聚焦关键结论。",
    "标准": "先给结论，再给步骤，步骤不少于5条，并补充1到2条注意事项。",
    "详细": "按“结论 -> 具体步骤 -> 材料清单 -> 常见错误 -> 进阶优化”输出，步骤不少于8条。"
}

def strip_reasoning_content(text):
    if not text: return ""
    return THINK_TAG_PATTERN.sub("", text)

def extract_context_docs(context_docs):
    raw_context_docs = []
    for doc in context_docs or []:
        metadata = getattr(doc, "metadata", {}) or {}
        content = getattr(doc, "page_content", "") or ""
        raw_context_docs.append({
            "title": metadata.get('title', '未知标题'),
            "source_url": metadata.get('source_url', '#'),
            "content": content,
            "content_length": len(content)
        })
    return raw_context_docs

def render_professional_trace(trace, key_prefix, show_full_context):
    with st.expander("🧪 专业模式：处理流程", expanded=True):
        st.markdown("#### 1) 输入")
        st.code(trace["user_input"], language="text")
        st.json({
            "请求时间": trace["request_time"],
            "模型后端": trace["model_choice"],
            "检索项": trace["keywords"],
            "回答详细度": trace["answer_detail"],
            "输入字符数": trace["input_chars"]
        })

        st.markdown("#### 2) 检索")
        st.json({
            "检索项数量": len(trace["keywords"]),
            "去重后片段数": trace["retrieved_chunk_count"]
        })

        if trace["retrieved_chunks"]:
            rows = []
            for idx, chunk in enumerate(trace["retrieved_chunks"], 1):
                rows.append({
                    "序号": idx,
                    "标题": chunk["title"],
                    "长度": chunk["content_length"],
                    "来源": chunk["source_url"]
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

            for idx, chunk in enumerate(trace["retrieved_chunks"], 1):
                with st.expander(f"片段 {idx}: {chunk['title']} ({chunk['content_length']} 字符)"):
                    context_text = chunk["content"]
                    if not show_full_context and len(context_text) > 400:
                        context_text = context_text[:400] + "..."
                    st.text_area("内容预览", context_text, height=150, key=f"{key_prefix}_chunk_{idx}")
        
        st.markdown("#### 3) 系统提示词 (System Prompt)")
        st.code(trace["system_prompt"], language="markdown")

        st.markdown("#### 4) 性能指标")
        if trace.get("step_durations_ms"):
            timing_rows = []
            for step_name, step_ms in trace["step_durations_ms"].items():
                timing_rows.append({"阶段": step_name, "耗时(ms)": step_ms})
            st.dataframe(timing_rows, use_container_width=True, hide_index=True)

        st.json({
            "首字耗时(ms)": trace["first_token_ms"],
            "总耗时(ms)": trace["total_time_ms"],
            "输出字符数": trace["output_chars"]
        })

def _log_backend_progress(message: str):
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] [RAG INIT] {message}", flush=True)


def _vector_state_path(persist_dir: str) -> str:
    return os.path.join(persist_dir, "build_state.json")


def _compute_data_signature(folder_path: str) -> str:
    if not os.path.isdir(folder_path):
        return ""

    hasher = hashlib.sha256()
    json_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".json"))
    for filename in json_files:
        file_path = os.path.join(folder_path, filename)
        try:
            stat = os.stat(file_path)
        except OSError:
            continue
        hasher.update(filename.encode("utf-8"))
        hasher.update(str(stat.st_size).encode("ascii"))
        hasher.update(str(stat.st_mtime_ns).encode("ascii"))

    return hasher.hexdigest()


def _load_vector_state(persist_dir: str) -> dict:
    state_path = _vector_state_path(persist_dir)
    if not os.path.isfile(state_path):
        return {}

    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_vector_state(persist_dir: str, state: dict):
    os.makedirs(persist_dir, exist_ok=True)
    state_path = _vector_state_path(persist_dir)
    tmp_path = f"{state_path}.tmp"
    payload = dict(state)
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, state_path)


def _as_nonnegative_int(value, default=0) -> int:
    try:
        number = int(value)
        return number if number >= 0 else default
    except (TypeError, ValueError):
        return default


def _is_vectorstore_ready(persist_dir: str, data_signature: str) -> bool:
    if not os.path.isdir(persist_dir):
        return False

    state = _load_vector_state(persist_dir)
    if not state:
        return False
    if state.get("status") != "complete":
        return False
    if state.get("data_signature") != data_signature:
        return False

    sqlite_path = os.path.join(persist_dir, "chroma.sqlite3")
    if not os.path.isfile(sqlite_path):
        return False
    try:
        with sqlite3.connect(sqlite_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return bool(row and row[0] > 0)
    except Exception:
        return False


# 向量库与检索器初始化（含前后端进度提示）
@st.cache_resource(show_spinner="正在加载 Embedding 模型与向量库，请稍候...")
def init_retriever(retrieval_k=3):
    # 在当前容器（通常是 sidebar）中创建进度占位
    progress_placeholder = st.empty()

    def report(progress: float, text: str):
        _log_backend_progress(text)
        # 更新进度条
        progress_placeholder.progress(progress, text=f"⏳ {text}")

    def load_data(folder_path):
        documents = []
        if not os.path.exists(folder_path):
            return documents

        json_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".json"))
        total_files = len(json_files)
        for idx, filename in enumerate(json_files, start=1):
            if idx == 1 or idx % 200 == 0 or idx == total_files:
                report(0.05 + 0.35 * (idx / max(total_files, 1)), f"加载知识文件: {idx}/{total_files}")

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

    report(0.02, "初始化 Embedding 模型 (首次运行需下载模型数据，请务必查看终端获取实时下载进度条)...")
    embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B", encode_kwargs={"normalize_embeddings": True})
    persist_dir = "./chroma_db"
    data_dir = "./structured_output"
    data_signature = _compute_data_signature(data_dir)

    if _is_vectorstore_ready(persist_dir, data_signature):
        report(0.30, "检测到已存在向量库，正在连接...")
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        report(0.35, "开始加载结构化知识文件...")
        docs = load_data(data_dir)
        if not docs:
            raise RuntimeError("structured_output 目录为空或不存在，无法构建向量库。")

        report(0.45, f"知识文件加载完成，共 {len(docs)} 篇文档，开始切分...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        total_splits = len(splits)
        report(0.55, f"文档切分完成，共 {total_splits} 个片段，开始写入向量库...")

        existing_state = _load_vector_state(persist_dir)
        has_sqlite = os.path.isfile(os.path.join(persist_dir, "chroma.sqlite3"))
        can_resume = (
            bool(existing_state)
            and has_sqlite
            and existing_state.get("status") == "building"
            and existing_state.get("data_signature") == data_signature
            and _as_nonnegative_int(existing_state.get("total_splits"), -1) == total_splits
        )

        if can_resume:
            completed_splits = _as_nonnegative_int(existing_state.get("completed_splits"), 0)
            completed_splits = min(completed_splits, total_splits)
            report(
                0.55 + 0.40 * (completed_splits / max(total_splits, 1)),
                f"检测到中断记录，正在断点续建: {completed_splits}/{total_splits}",
            )
            build_state = {
                "status": "building",
                "data_signature": data_signature,
                "total_splits": total_splits,
                "completed_splits": completed_splits,
            }
        else:
            if os.path.exists(persist_dir):
                report(0.30, "检测到未完成/过期向量库，正在清理后重建...")
                shutil.rmtree(persist_dir, ignore_errors=True)
            completed_splits = 0
            build_state = {
                "status": "building",
                "data_signature": data_signature,
                "total_splits": total_splits,
                "completed_splits": 0,
            }
            _save_vector_state(persist_dir, build_state)

        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        batch_size = 128
        for start in range(completed_splits, total_splits, batch_size):
            batch = splits[start:start + batch_size]
            vectorstore.add_documents(batch)
            done = min(start + batch_size, total_splits)
            build_state["completed_splits"] = done
            _save_vector_state(persist_dir, build_state)
            report(0.55 + 0.40 * (done / max(total_splits, 1)), f"向量入库进度: {done}/{total_splits}")

        build_state["status"] = "complete"
        build_state["completed_splits"] = total_splits
        _save_vector_state(persist_dir, build_state)

    report(1.0, "向量库初始化完成。")
    progress_placeholder.empty()
    return vectorstore.as_retriever(search_kwargs={"k": retrieval_k})

# 动态创建 QA 链
def get_qa_chain(model_type, ds_api_key="", answer_detail="标准"):
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
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False}
            }
        )
    else:
        llm = ChatOpenAI(
            base_url="https://api.deepseek.com",
            api_key=ds_api_key,
            model="deepseek-v4-flash",
            temperature=0.1,
            streaming=True,
            extra_body={
                "thinking": {"type": "disabled"}
            }
        )

    system_prompt = (
        "你是一个Minecraft知识库智能助手。请使用以下检索到的背景信息来回答问题。"
        "如果你不知道答案，就说你不知道，不要编造。"
        f"\n回答详细度要求：{DETAIL_REQUIREMENTS.get(answer_detail, DETAIL_REQUIREMENTS['标准'])}"
        "\n\n"
        "背景信息："
        "\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    return create_stuff_documents_chain(llm, prompt), system_prompt

# 分词与检索策略
@st.cache_data
def get_stopwords():
    return {"的", "是", "了", "在", "怎么", "什么", "和", "与", "如何", "帮我", "查一下", "告诉我", "有哪些"}

def retrieve_by_keywords(retriever, user_input):
    stopwords = get_stopwords()
    words = jieba.lcut(user_input)
    # 过滤停用词和空字符，得到纯净的关键词列表
    keywords = [w for w in words if w not in stopwords and len(w.strip()) > 0]
    
    # 将原始的完整输入加入到检索列表的首位
    original_input = user_input.strip()
    if original_input and original_input not in keywords:
        keywords.insert(0, original_input)
    
    # 输入全是停用词或为空
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

# 页面 UI & 侧边栏
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
    st.title("检索与回答设置")
    retrieval_k = st.slider("检索片段数量 (k)", min_value=1, max_value=8, value=3, help="k越大信息越全，但处理更慢")
    answer_detail = st.select_slider("回答详细度", options=["简洁", "标准", "详细"], value="标准")
    professional_mode = st.toggle("专业模式（显示完整流程）", value=False)
    show_full_context = False
    if professional_mode:
        show_full_context = st.checkbox("显示检索片段全文", value=False)

    st.markdown("---")
    st.title("系统状态")
    
    with st.spinner("正在唤醒系统..."):
        retriever = init_retriever(retrieval_k=retrieval_k)
        
    st.success("向量库加载完毕")
    st.success(f"当前选中: {model_choice.split(' ')[0]}")
    
    st.markdown("---")
    st.markdown("### 使用说明")
    st.markdown("基于本地 ChromaDB 构建的 RAG 问答系统。可在侧边栏无缝切换本地计算与云端 API。")

# 主界面与聊天逻辑
st.title("⛏️ Minecraft 知识库智能助手")
st.caption("我是专属 RAG 助手，有什么关于游戏机制的问题都可以问我！")

if model_choice == "云端模型 (DeepSeek API)" and not deepseek_api_key:
    st.info("👈 请在左侧边栏输入 DeepSeek API Key 以开始对话。")
    st.stop()

qa_chain, system_prompt = get_qa_chain(model_choice, deepseek_api_key, answer_detail=answer_detail)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "你好，今天想查询点什么？"}]

# debug 信息展示
for msg_index, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # 渲染发送给 AI 的 Prompt (旧逻辑，由于兼容性暂时保留预览)
        if "debug_prompt" in msg and msg["debug_prompt"] and not professional_mode:
            with st.expander("🛠️ 查看发送给 AI 的内容 (简易版)"):
                st.code(msg["debug_prompt"], language="markdown")
                
        # 渲染参考来源
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 查看参考来源"):
                for source in msg["sources"]:
                    st.markdown(f"- [{source['title']}]({source['source_url']})")
        
        # 渲染专业 Trace
        if professional_mode and "trace" in msg:
            render_professional_trace(msg["trace"], f"hist_{msg_index}", show_full_context)

if user_input := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        source_list = []
        
        # 性能追踪变量
        request_start = time.perf_counter()
        step_durations_ms = {}
        first_token_ms = None
        
        with st.spinner("正在拆解关键词并检索记忆库..."):
            try:
                # 步骤 1: 关键词提取与检索
                step_start = time.perf_counter()
                stopwords = get_stopwords()
                words = jieba.lcut(user_input)
                keywords = [w for w in words if w not in stopwords and len(w.strip()) > 0]
                if user_input.strip() not in keywords:
                    keywords.insert(0, user_input.strip())
                
                retrieved_docs = []
                seen_content = set()
                for kw in keywords:
                    docs = retriever.invoke(kw)
                    for d in docs:
                        if d.page_content not in seen_content:
                            retrieved_docs.append(d)
                            seen_content.add(d.page_content)
                step_durations_ms["关键词提取与检索"] = round((time.perf_counter() - step_start) * 1000, 2)
                
                # 整理来源
                for doc in retrieved_docs:
                    source_list.append({
                        "title": doc.metadata.get('title', '未知标题'),
                        "source_url": doc.metadata.get('source_url', '#'),
                        "content": doc.page_content,
                        "content_length": len(doc.page_content)
                    })
                
                # 步骤 2: 构造内容预览 (Debug 文本)
                context_str = "\n\n".join([f"【片段 {i+1}】\n{doc.page_content}" for i, doc in enumerate(retrieved_docs)])
                debug_prompt_text = (
                    "**[System]**\n"
                    f"{system_prompt}\n\n"
                    "**[Context]**\n"
                    f"{context_str}\n\n"
                    "**[User]**\n"
                    f"{user_input}"
                )
                
                # 步骤 3: 模型流式输出
                step_start = time.perf_counter()
                for chunk in qa_chain.stream({
                    "context": retrieved_docs, 
                    "input": user_input
                }):
                    content = ""
                    if isinstance(chunk, str):
                        content = chunk
                    else:
                        # 兼容不同 chain 输出格式
                        content = chunk.get("answer", "") if isinstance(chunk, dict) else str(chunk)
                    
                    if content:
                        if first_token_ms is None:
                            first_token_ms = round((time.perf_counter() - request_start) * 1000, 2)
                        full_response += content
                        message_placeholder.markdown(full_response + "▌")
                
                step_durations_ms["模型推理生成"] = round((time.perf_counter() - step_start) * 1000, 2)
                message_placeholder.markdown(full_response)
                
                # 结果后处理
                unique_sources = []
                src_seen = set()
                for s in source_list:
                    if s["title"] not in src_seen:
                        unique_sources.append({"title": s["title"], "source_url": s["source_url"]})
                        src_seen.add(s["title"])

                total_time_ms = round((time.perf_counter() - request_start) * 1000, 2)
                
                # 构造 Trace 对象
                trace = {
                    "request_time": datetime.now().strftime("%H:%M:%S"),
                    "user_input": user_input,
                    "input_chars": len(user_input),
                    "model_choice": model_choice,
                    "keywords": keywords,
                    "answer_detail": answer_detail,
                    "system_prompt": system_prompt,
                    "retrieved_chunk_count": len(retrieved_docs),
                    "retrieved_chunks": source_list,
                    "step_durations_ms": step_durations_ms,
                    "first_token_ms": first_token_ms,
                    "total_time_ms": total_time_ms,
                    "output_chars": len(full_response),
                    "final_answer": full_response
                }

                if professional_mode:
                    render_professional_trace(trace, f"curr_{len(st.session_state.messages)}", show_full_context)

                if unique_sources:
                    with st.expander(f"查看参考来源 (共召回 {len(retrieved_docs)} 个片段)"):
                        for source in unique_sources:
                            st.markdown(f"- [{source['title']}]({source['source_url']})")
                
                # 存入历史
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "sources": unique_sources,
                    "debug_prompt": debug_prompt_text,
                    "trace": trace
                })
                
            except Exception as e:
                st.error(f"生成回复时出错！详细报错: {str(e)}")
