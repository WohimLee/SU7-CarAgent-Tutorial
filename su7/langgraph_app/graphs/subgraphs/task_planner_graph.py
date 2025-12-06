"""多步任务规划子图定义模块。"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from li_assistant.core.state import CarAssistantState


def _task_planner_node(state: CarAssistantState) -> CarAssistantState:
    """占位节点：后续在此实现多步任务拆解与编排流程。"""

    return state


def build_task_planner_subgraph() -> Any:
    """构建并返回“多步任务规划”子图。"""

    graph = StateGraph(CarAssistantState)

    graph.add_node("task_planner", _task_planner_node)
    graph.add_edge(START, "task_planner")
    graph.add_edge("task_planner", END)

    return graph.compile()


graph = build_task_planner_subgraph()

