"""导航 / 路线规划子图定义模块。"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from li_assistant.core.state import CarAssistantState
from li_assistant.langgraph_app.graphs.subgraphs import manual_rag_graph



def _navigation_node(state: CarAssistantState) -> CarAssistantState:
    """占位节点：后续在此实现路线解析与导航规划流程。"""

    return state


def build_navigation_subgraph() -> Any:
    """构建并返回“导航 / 路线规划”子图。"""

    graph = StateGraph(CarAssistantState)

    graph.add_node("navigation", _navigation_node)
    graph.add_edge(START, "navigation")
    graph.add_edge("navigation", END)

    return graph.compile()


graph = build_navigation_subgraph()

