"""主 LangGraph 图定义模块。"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from li_assistant.core.state import CarAssistantState

from li_assistant.langgraph_app.graphs.subgraphs.manual_rag_graph import graph as manual_rag_graph
from li_assistant.langgraph_app.graphs.subgraphs.navigation_graph import graph as navigation_graph
from li_assistant.langgraph_app.graphs.subgraphs.safety_guard_graph import graph as safety_guard_graph
from li_assistant.langgraph_app.graphs.subgraphs.task_planner_graph import graph as task_planner_graph
from li_assistant.langgraph_app.graphs.subgraphs.vehicle_control_graph import graph as vehicle_control_graph



def _passthrough_node(state: CarAssistantState) -> CarAssistantState:
    """占位节点：目前只原样返回 state，方便后续替换为真正的 Orchestrator。"""

    return state


def _route_by_intent(state: CarAssistantState) -> str:
    """根据 intent 字段决定下一步走向哪个子图或直接结束。"""

    intent = state.get("intent")  # type: ignore[call-arg]

    if intent == "manual_faq":
        return "manual_rag"
    if intent == "vehicle_control":
        return "vehicle_control"
    if intent == "navigation":
        return "navigation"
    if intent == "task_planning":
        return "task_planner"
    if intent is None or intent == "unknown" or intent == "chitchat":
        return "end"

    return "end"


def build_main_graph() -> Any:
    """构建并返回主图（用于 langgraph.json 中引用）。"""

    graph = StateGraph(CarAssistantState)  # 使用我们在 core.state 中定义的状态类型

    # 顶层 Orchestrator 节点（后续可替换为真正的 LLM 路由逻辑）
    graph.add_node("orchestrator", _passthrough_node)

    # 各功能子图作为节点挂到主图上
    graph.add_node("manual_rag", manual_rag_graph)
    graph.add_node("vehicle_control", vehicle_control_graph)
    graph.add_node("navigation", navigation_graph)
    graph.add_node("task_planner", task_planner_graph)
    graph.add_node("safety_guard", safety_guard_graph)

    # 从 START 进入 orchestrator
    graph.add_edge(START, "orchestrator")

    # orchestrator 根据 intent 决定下一步走哪个子图或直接结束
    graph.add_conditional_edges(
        "orchestrator",
        _route_by_intent,
        {
            "manual_rag": "manual_rag",
            "vehicle_control": "vehicle_control",
            "navigation": "navigation",
            "task_planner": "task_planner",
            # 安全子图可以在以后从其他节点跳转进入，这里先保留映射
            "safety_guard": "safety_guard",
            "end": END,
        },
    )

    # 各子图执行完后直接结束本轮
    graph.add_edge("manual_rag", END)
    graph.add_edge("vehicle_control", END)
    graph.add_edge("navigation", END)
    graph.add_edge("task_planner", END)
    graph.add_edge("safety_guard", END)

    compiled = graph.compile()  # 编译 StateGraph，得到可执行的图对象
    return compiled


# 约定导出名称，便于被 CLI 或其他模块直接引用
main_graph = build_main_graph()
