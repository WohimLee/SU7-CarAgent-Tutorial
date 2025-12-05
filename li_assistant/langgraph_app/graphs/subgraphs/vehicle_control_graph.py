"""车辆状态与控制子图定义模块。"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from li_assistant.core.state import CarAssistantState


def _vehicle_control_node(state: CarAssistantState) -> CarAssistantState:
    """占位节点：后续在此实现车辆状态查询与控制流程。"""

    return state


def build_vehicle_control_subgraph() -> Any:
    """构建并返回“车辆状态与控制”子图。"""

    graph = StateGraph(CarAssistantState)

    graph.add_node("vehicle_control", _vehicle_control_node)
    graph.add_edge(START, "vehicle_control")
    graph.add_edge("vehicle_control", END)

    return graph.compile()


graph = build_vehicle_control_subgraph()

