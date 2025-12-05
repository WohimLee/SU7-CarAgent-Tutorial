
```
ai-trade-system/
├── apps/
│   ├── gateway/                  # API Gateway / BFF 层
│   │   ├── main.py               # FastAPI 入口（对外HTTP接口）
│   │   ├── routes/
│   │   │   ├── chat.py           # /api/chat 对话接口
│   │   │   ├── strategy.py       # /api/strategy 策略配置/调用
│   │   │   ├── trade.py          # /api/trade 下单/撤单
│   │   │   └── alert.py          # /api/alert 订阅&提醒
│   │   └── deps.py               # 依赖注入（auth, user, etc）
│   │
│   ├── langgraph_app/            # LangGraph 主应用
│   │   ├── __init__.py
│   │   ├── config.py             # LangGraph 配置（store, checkpointer等）
│   │   ├── graph_builder.py      # 总图构建（把各子图组合起来）
│   │   ├── graph_runner.py       # 封装 run_graph / streaming
│   │   │
│   │   ├── agents/               # 所有 Agent (角色/智能体)
│   │   │   ├── coordinator_agent.py   # 协调/路由/对话中枢
│   │   │   ├── intent_agent.py        # 意图识别（问答 / 下单 / 选股 / 资讯）
│   │   │   ├── advisor_agent.py       # 投顾 Agent（给出建议/解释）
│   │   │   ├── stock_picker_agent.py  # 智能选股 Agent
│   │   │   ├── monitor_agent.py       # 盯盘&预警 Agent
│   │   │   └── trade_agent.py         # 交易执行 Agent
│   │   │
│   │   ├── subgraphs/            # 子图 (复用流程)
│   │   │   ├── market_data_subgraph.py     # 行情数据获取+聚合
│   │   │   ├── fundamental_data_subgraph.py# 财报&指标处理
│   │   │   ├── factor_analysis_subgraph.py # 因子/回测流程
│   │   │   ├── rag_retrieval_subgraph.py   # 研报/公告/政策 RAG
│   │   │   ├── risk_check_subgraph.py      # 合规&风控流程
│   │   │   ├── order_execution_subgraph.py # 下单/撤单/查状态
│   │   │   ├── report_generate_subgraph.py # 组合/个股报告生成
│   │   │   └── notification_subgraph.py    # 触发提醒/消息推送
│   │   │
│   │   ├── tools/                # LangGraph Tool 定义
│   │   │   ├── market_data_tools.py        # 调行情/深度数据
│   │   │   ├── account_tools.py            # 查资金/持仓/订单
│   │   │   ├── trade_tools.py              # 下单/撤单/改单
│   │   │   ├── risk_tools.py               # 单笔风控/额度检查
│   │   │   ├── rag_tools.py                # 文本检索/RAG工具
│   │   │   ├── analysis_tools.py           # 估值/技术指标计算
│   │   │   └── notify_tools.py             # 调用短信/邮件/Push
│   │   │
│   │   ├── prompts/              # Prompt 模板
│   │   │   ├── system/
│   │   │   │   ├── base_system_prompt.txt   # 全局系统提示词
│   │   │   │   ├── advisor_system_prompt.txt
│   │   │   │   └── trade_system_prompt.txt
│   │   │   ├── templates/
│   │   │   │   ├── report_template.md       # 个股/组合报告模板
│   │   │   │   └── alert_message_template.md
│   │   │   └── few_shot_examples/          # 示例对话/格式例子
│   │   │
│   │   └── state/                 # LangGraph 状态&类型定义
│   │       ├── base_state.py      # 全局 GraphState 定义
│   │       ├── user_session_state.py
│   │       └── trade_context_state.py
│   │
│   ├── services/                 # 业务服务封装（对外服务调用）
│   │   ├── market_data_service.py    # 调行情源
│   │   ├── account_service.py        # 账户/持仓/资金
│   │   ├── trade_service.py          # 券商交易接口封装
│   │   ├── risk_service.py           # 风控&合规检查
│   │   ├── rag_service.py            # 向量库/文档检索
│   │   ├── user_profile_service.py   # 用户画像/偏好
│   │   └── notification_service.py   # 推送服务（短信/邮件/App push）
│   │
│   └── workers/                  # 后台任务/定时任务（盯盘等）
│       ├── monitor_worker.py         # 行情监控+触发预警
│       ├── sync_data_worker.py       # 夜间批处理/数据同步
│       └── backtest_worker.py        # 策略回测任务
│
├── infra/
│   ├── db/
│   │   ├── models.py             # ORM 实体（用户/持仓快照/日志等）
│   │   ├── schema/               # Pydantic schemas
│   │   └── migrations/           # 数据库迁移
│   ├── config/                   # 配置管理（环境变量/多环境）
│   │   ├── settings.py
│   │   └── logging_conf.yaml
│   ├── llm/
│   │   ├── llm_client.py         # 大模型客户端包装（OpenAI/自建等）
│   │   └── embeddings_client.py  # 向量化
│   ├── vectorstore/
│   │   ├── client.py             # 向量库客户端（Milvus, pgvector 等）
│   │   └── index_manager.py
│   └── utils/
│       ├── time_utils.py
│       ├── id_utils.py
│       └── monitoring.py         # Prometheus/日志埋点
│
├── tests/
│   ├── test_agents/
│   ├── test_subgraphs/
│   ├── test_tools/
│   └── test_api/
│
├── scripts/
│   ├── load_demo_data.py
│   ├── build_indexes.py          # 构建RAG向量索引
│   └── start_dev.sh
│
├── pyproject.toml / requirements.txt
└── README.md
```