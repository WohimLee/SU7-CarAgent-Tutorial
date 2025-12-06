from __future__ import annotations  # 启用未来注解特性，避免前置类型声明顺序限制

from pydantic import BaseModel
from langgraph.graph import MessagesState  # 从 LangGraph 导入带有 messages 聚合逻辑的基础状态类型
from typing import Any, Dict, Literal, Optional, TypedDict, List

class ConversationTurn(BaseModel):
    role: str
    content: str

class TaskStep(BaseModel):
    name: str
    status: str  # pending / running / done / failed
    result: Optional[Dict[str, Any]] = None

class GraphState(BaseModel):
    history: List[ConversationTurn] = []
    intent: Optional[str] = None
    domain: Optional[str] = None
    user_profile: Dict[str, Any] = {}
    task_plan: List[TaskStep] = []
    last_tool_result: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None


Intent = Literal[  # 定义用户意图的枚举类型（使用 Literal 做静态约束）
    "manual_faq",       # 代表“车辆手册 / 使用说明”相关的问答
    "vehicle_control",  # 代表“车辆控制 / 车辆状态查询”相关的意图
    "navigation",       # 代表“导航 / 路线规划”相关的意图
    "task_planning",    # 代表“复杂多步任务规划”相关的意图
    "chitchat",         # 代表“闲聊 / 非任务型对话”相关的意图
    "unknown",          # 代表“暂时无法归类 / 未识别”的兜底意图
]  # Intent 类型用于 Orchestrator 节点在路由不同子图时做分支判断

# 定义用户偏好的路线类型枚举
RoutePreference = Literal[  
    "fastest",      # 表示偏好“最快到达”的路线策略
    "shortest",     # 表示偏好“里程最短”的路线策略
    "scenic",       # 表示偏好“更美景观”的路线策略
]  # RoutePreference 主要被导航子图和用户画像一起使用

# 定义用户画像信息的结构（total=False 表示字段都是可选）
class UserProfile(TypedDict, total=False):  
    preferred_temperature_celsius: float    # 用户偏好的车内温度（摄氏度），用于空调控制子图
    prefers_quiet_mode: bool                # 用户是否偏好“简洁 / 少说话”的回复风格
    preferred_route: RoutePreference        # 用户在导航时默认偏好的路线策略
    language: str                           # 用户偏好的语言，例如 'zh'、'en' 等


# 定义一次“车辆状态快照”的结构
class VehicleStatusSnapshot(TypedDict, total=False):  
    fuel_level_percent: float           # 剩余油量百分比（燃油车或增程器油箱）
    battery_level_percent: float        # 剩余电量百分比（纯电或混动车型）
    inside_temperature_celsius: float   # 车内当前温度（摄氏度）
    outside_temperature_celsius: float  # 车外当前温度（摄氏度）
    doors_locked: bool                  # 车门是否全部上锁
    is_moving: bool                     # 车辆当前是否处于行驶状态


class NavigationPlan(TypedDict, total=False):  # 定义导航 / 路线规划结果的结构
    origin: str  # 导航起点描述（可以是地名、POI 名称或经纬度字符串）
    destination: str  # 导航终点描述（与 origin 同样的表示方式）
    distance_km: float  # 规划路线的总距离（单位：公里）
    estimated_time_min: float  # 预计行驶时间（单位：分钟）
    route_strategy: RoutePreference  # 实际采用的路线策略（可能与用户默认偏好不同）
    traffic_summary: str  # 路况概况说明（例如“整体通畅，部分路段轻微拥堵”）


class CarAssistantState(MessagesState):  # 继承 LangGraph 自带的 MessagesState，增加车载助手特有字段
    intent: Optional[Intent]  # 当前轮对话解析出的用户意图，用于 Orchestrator 路由子图
    domain: Optional[str]  # 更细粒度的领域标签，例如 'air_conditioning'、'charging' 等
    need_clarification: bool  # 当前是否需要向用户追问 / 澄清需求
    user_profile: Optional[UserProfile]  # 当前会话绑定的用户画像信息（可从外部持久化加载）
    vehicle_status: Optional[VehicleStatusSnapshot]  # 最近一次从车辆 API 获取到的状态快照
    navigation_plan: Optional[NavigationPlan]  # 最近一次导航规划的结果，便于后续确认或修改
    last_tool_name: Optional[str]  # 最近一次调用的工具 / 子图名称，用于调试和决策
    debug_info: Dict[str, Any]  # 额外的调试信息或临时数据，不建议持久化，仅在开发阶段使用


# 使用 __all__ 有助于在 from li_assistant.core.state import * 时控制导出内容
__all__ = [  # 显式声明本模块向外暴露的符号列表
    "Intent",           # 暴露 Intent 类型枚举
    "RoutePreference",  # 暴露 RoutePreference 路线偏好类型枚举
    "UserProfile",      # 暴露 UserProfile 用户画像结构
    "VehicleStatusSnapshot",    # 暴露 VehicleStatusSnapshot 车辆状态快照结构
    "NavigationPlan",   # 暴露 NavigationPlan 导航规划结构
    "CarAssistantState",        # 暴露 CarAssistantState 主状态 TypedDict
]  
