"""安全保护 / 风险控制子图定义模块。"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from li_assistant.core.state import CarAssistantState


def _safety_guard_node(state: CarAssistantState) -> CarAssistantState:
    """占位节点：后续在此实现安全检查与规则拦截逻辑。"""

    return state


def build_safety_guard_subgraph() -> Any:
    """构建并返回“安全保护 / 风险控制”子图。"""

    graph = StateGraph(CarAssistantState)

    graph.add_node("safety_guard", _safety_guard_node)
    graph.add_edge(START, "safety_guard")
    graph.add_edge("safety_guard", END)

    return graph.compile()


graph = build_safety_guard_subgraph()

