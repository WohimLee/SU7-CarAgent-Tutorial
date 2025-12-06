"""手册 RAG 子图定义模块。"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from li_assistant.core.state import CarAssistantState


def _manual_rag_node(state: CarAssistantState) -> CarAssistantState:
    """占位节点：后续在此实现手册 RAG 流程。"""

    return state


def build_manual_rag_subgraph() -> Any:
    """构建并返回“手册问答”子图。"""

    graph = StateGraph(CarAssistantState)

    graph.add_node("manual_rag", _manual_rag_node)
    graph.add_edge(START, "manual_rag")
    graph.add_edge("manual_rag", END)

    return graph.compile()


graph = build_manual_rag_subgraph()

