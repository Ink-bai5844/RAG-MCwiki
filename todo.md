# Todo List

- [ ] 前后端分离
- [ ] 优化log消息
- [ ] 开发调用方式
- [ ] 无头模式
- [ ] 增加文本向量化状态，初始时要自己选择执行分词操作
- [ ] 增加可选文本向量化 Qwen3-Embedding-0.6B
- [ ] 模型增加删除选项
- [ ] 增加本项目独立数据源，并且用户可以直接下载处理好的数据库或者元数据，在前端显示当前数据时间和是否有更新，防止用户大批量对mcwiki进行爬取

  🌟 阶段一：核心架构解耦（前后端分离与无头模式）
  这是解决你 todo.md 中“前后端分离”、“无头模式”和“开发调用方式”的关键步骤。目前 rag_app.py 将 UI 和 LangChain 逻辑耦合在一起，这限制了系统的扩展性。

   * 重构方案：引入 FastAPI 作为核心后端框架。
       * 后端 (FastAPI)：封装所有 RAG 逻辑。提供如 /api/v1/chat/completions（兼容 OpenAI
         格式的对话接口）、/api/v1/knowledge_base/build（触发构建向量库）、/api/v1/models 等 RESTful API。这即是你的无头模式 (Headless)。
       * 前端 (Streamlit 保持或重构)：精简后的 rag_app.py 仅作为前端 UI，通过 requests 或 aiohttp 调用 FastAPI 后端。未来甚至可以轻松替换为 Vue/React
         客户端。
   * 收益：系统可以作为独立 API 服务运行（如接入 QQ 机器人、Discord Bot、或者游戏内 Mod），彻底实现解耦。

  🌟 阶段二：知识库与向量化管控面板
  针对你提到的“增加文本向量化状态，初始时自己选择执行”、“可选 Qwen3-Embedding”以及“模型删除选项”。

   * 分离初始化逻辑：将当前启动时自动执行的 init_retriever 逻辑抽离。系统启动时不自动建库，而是检查状态。
   * 开发“知识库管理”面板 (Data Pipeline UI)：
       * 手动触发器：在 UI 侧边栏增加【构建向量库】的按钮及进度条（可通过后端 WebSocket 或轮询获取进度）。
       * Embedding 模型热切换：提供下拉菜单，让用户自由选择 shibing624/text2vec-base-chinese 或 Qwen/Qwen3-Embedding-0.6B（并允许填写自定义 HuggingFace
         模型路径）。
       * 向量库与模型清理功能：提供一键清理 chroma_db/ 缓存目录和对应向量数据的按钮（“模型增加删除选项”）。

  🌟 阶段三：云端数据源与防滥用机制
  针对“增加本项目独立数据源，防止大批量爬取 mcwiki”。你的 crawler.py
  写得非常专业（支持并发、限流、断点续传），但让每个用户自己去爬取几千个页面不仅费时，也会给 Minecraft Wiki 服务器造成负担。

   * 中心化数据分发：将你爬取并用 clean_data.py 清洗好的 structured_output 压缩包，甚至打包好的 chroma_db 数据库，上传到 GitHub Releases、HuggingFace
     Datasets 或阿里云 OSS 等免费/廉价存储中。
   * 前端状态显示：
       * 在应用主页展示“当前本地知识库版本：2026-04-22”。
       * 系统启动时请求远端 JSON 接口（例如 version.json），比对版本号。如果有更新，提示“发现新版本知识库 (2026-05-01)”。
   * 一键下载同步：提供一个【一键同步最新知识库】的按钮，后台使用流式下载解压覆盖本地库，直接跳过繁琐的爬虫和耗时的向量化步骤。

  🌟 阶段四：工程化体验与日志优化
  针对“优化 log 消息”和代码清理。

   * 引入标准 logging 模块：淘汰 crawler.py 和 rag_app.py 中的 print()。配置统一的 logger，输出格式化的日志（如 [2026-04-24 10:00:00] [INFO] [RAG]
     正在召回片段...），并支持写入文件（使用 RotatingFileHandler）。
   * 废弃文件清理：整合 rag_app.py 与 rag_app1.py。目前看起来 rag_app.py 更加先进（实现了 Jieba 多路召回和良好的打点 Trace），将不再需要的旧逻辑剔除，保持
     codebase 整洁。

# 修改意见
## RAG
  1. 废除当前的“碎裂化”关键词检索（首要任务）
  现状问题： 现在的代码把 java版26.1 拆成了 java、版、26.1。搜索 版 会召回成百上千条包含“基岩版”、“教育版”的文档，这是噪声的最大来源。

  修改意见：使用 LLM 辅助的 Query 改写（Query Transformation）
   * 做法： 在进入 retriever 之前，先让 LLM 把用户的口语改写为 3 个精准的“搜索短语”。
   * 示例： 输入“java版26.1更新了什么”，让 LLM 输出："Java版 26.1 更新日志", "Minecraft 26.1 Java Edition changes"。
   * 效果： 彻底杜绝搜索“版”、“的”这种废词，召回数量会从 26 条骤降至 5-8 条，且全是干货。

  2. 引入重排序（Reranker）机制
  现状问题： 向量搜索（ChromaDB）只计算“字面向量”的相似度，它无法理解“基岩版26.1”和“Java版26.1”语义上的对立，导致它把两个版本的东西混在一起。

  修改意见：添加 Cross-Encoder（交叉编码器）进行二次打分
   * 做法： 先从数据库拿 20 个片段，然后使用一个专门的小模型（如 bge-reranker-base）对这 20 个片段与用户问题的“相关性”进行精准打分。
   * 效果： 它可以识别出即使包含了关键词但逻辑无关的噪声，并将它们过滤掉。只给 AI 提供评分最高的 Top-5。

  3. 从“按字符切分”改为“按语义结构切分”
  现状问题： 您现在的切分是 RecursiveCharacterTextSplitter(chunk_size=500)。这会把 Wiki 里的一个完整表格或一个更新条目从中截断，导致 AI
  拿到的信息是不完整的。

  修改意见：利用 JSON 的 structured_content 进行结构化分段
   * 做法： 既然您的 output 文件夹里有结构化的 JSON，应该直接以 JSON 中的每个 section（章节）为一个 Chunk。
   * 效果： 保证了每个知识片段的完整性。AI 读到的是“整个 26.1 的漏洞修复列表”，而不是“修复列表的前 3 行”。

  4. 增加元数据过滤（Metadata Filtering）
  现状问题： 系统无法区分“Java版”和“基岩版”这两个硬属性，全靠向量去“猜”。

  修改意见：在检索时添加硬性过滤条件
   * 做法： 利用 jieba 或简单的正则识别出提问中的“Java”或“基岩”关键词，然后在 retriever.invoke 时传入 filter 参数：

   1     retriever.invoke(query, filter={"platform": "Java"})
   * 效果： 如果用户问 Java 版，数据库连一条基岩版的文档都不会吐出来。