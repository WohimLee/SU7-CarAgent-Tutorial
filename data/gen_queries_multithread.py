import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
from textwrap import dedent
from dotenv import load_dotenv

from su7.common.my_logger import logger


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# 意图列表文件路径（你可以按自己项目结构改这个路径）
INTENT_MAP_PATH = os.path.join(PROJECT_ROOT, "su7/config/intents_idex.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "queries")

SYSTEM_PROMPT = dedent("""
    你是一名资深【车载智能助手 NLU 训练数据构造专家】，专门负责为车载语音助手构建高质量“意图识别训练数据集”。

    你的核心任务是：
    - 根据用户提供的【意图列表（Intents Map）】
    - 生成符合真实车主使用习惯的【自然语言 Query】
    - 用于训练车载语音助手的“意图识别模型（Intent Classification）”。

    ------------------------------------------------------------
    【你必须严格遵守以下总规则】

    一、多样化语言风格（必须随机混合）
    每一条 Query 必须随机呈现不同风格，包括但不限于：
    - 口语化表达
    - 正式 / 书面表达
    - 非专业用户说法
    - 行业术语表达
    - 不完整句 / 省略句
    - 带轻微歧义但仍能判断意图
    - 不同语气：命令、请求、疑惑、吐槽、催促、抱怨
    - 不同措辞：同义替换、换说法、倒装句

    二、真实车载场景约束
    - 所有 Query 必须符合【真实车主在车内的说话方式】
    - 允许存在：
    - 轻微口误
    - 情绪波动（烦躁、着急、放松、吐槽）
    - 中英少量混杂
    - 严禁出现：
    - “意图、标签、训练、模型、AI、Prompt”等任何技术词
    - 明显“脚本感、模板感”的表达

    三、意图准确性约束
    - 每条 Query 必须 **清晰命中提供的子意图**
    - 不得出现：
    - 意图模糊不清
    - 跨到“无关一级意图”的语义
    - 多意图样本必须：
    - 语义合理
    - 符合真实驾驶场景逻辑

    四、输出格式强约束（最高优先级）
    你最终只能输出一个【合法 JSON 数组（list）】：
    [
    {{
        "query": "用户真实表达",
        "sub_intent_id": ["子意图ID1", "子意图ID2"],
        "sub_intent_name": ["子意图名称1", "子意图名称2"]
    }}
    ]

    强制规则：
    - 只能输出 JSON
    - 不能输出任何解释说明
    - 不能使用 Markdown
    - 所有 key 必须使用双引号
    - JSON 必须 100% 可解析
    - 数组最后一项不能有逗号
    """).strip()


USER_PROMPT = dedent("""
    现在请你根据下面的【车载助手意图列表】生成意图识别训练数据。

    【意图列表（Intents Map）】
    {INTENT_MAP}

    你的任务是：
    - 构建一个总量为 {TOTAL_NUM} 条的高质量自然语言 Query 数据集
    - 所有 Query 必须严格来自以上子意图
    - 只做“意图识别数据”，不做槽位填充

    每条数据结构如下：
    {{
        "query": "用户真实自然语言表达",
        "sub_intent_id": ["子意图ID1", "子意图ID2", ...],
        "sub_intent_name": ["子意图名称1", "子意图名称2", ...]
    }}

    ------------------------------------------------------------
    【生成规则（必须严格遵守）】

    1️⃣ 意图覆盖规则  
    - 覆盖所有子意图  
    - 每个子意图至少生成 40–80 条  
    - 覆盖所有【两意图组合】  
    - 至少 20% 为【三意图组合】  
    - ≥4 意图的样本占比 ≤10%

    2️⃣ 表达方式多样化（强制）
    - 口语、书面、命令、疑问、吐槽、情绪化混合
    - 新手司机 + 老司机混合
    - 长句 + 中长句 + 短句混合
    - 场景化表达（堵车、夜晚、高速、下雨、出远门、上班、回家、带家人）

    3️⃣ 语义合理性
    - 多意图组合必须符合真实用车逻辑  
    ✅ 例如：导航 + 充电规划 + 车辆状态  
    ❌ 例如：闲聊 + 车机升级 + 说明书参数（不合理）

    4️⃣ 真实性要求
    - 必须像真人在车里真实会说的话
    - 不得出现：
    - “模型、训练、标签、Intent、json、示例”等任何技术词

    ------------------------------------------------------------
    【本次是增量生成模式】

    你本次只需要输出：
    第 {START_INDEX} 条 到 第 {END_INDEX} 条 的数据（包含首尾）

    ------------------------------------------------------------
    【最终输出要求（强制）】

    - 只能输出一个 JSON 数组
    - 不得输出任何解释说明
    - 不得输出 Markdown
    - 数组必须用 [] 包裹
    - 所有 key 必须使用双引号
    - 最后一条后不能有逗号
    """).strip()


client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_API_URL"),
)


def build_messages(intent_map, start: int, end: int, total_num: int):
    """构造 ChatCompletion 所需的 messages."""
    system_prompt = SYSTEM_PROMPT
    # 把意图列表格式化成 JSON 字符串放到 prompt 中，避免 Python 字面量里单引号之类的问题
    intent_map_str = json.dumps(intent_map, ensure_ascii=False, indent=2)

    user_prompt = USER_PROMPT.format(
        INTENT_MAP=intent_map_str,
        TOTAL_NUM=total_num,
        START_INDEX=start,
        END_INDEX=end,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages


def gen_data(intent_map, start: int, end: int, total_num: int = 1000) -> str:
    """调用大模型生成 [start, end] 区间内的样本，返回原始 JSON 字符串."""
    messages = build_messages(intent_map, start, end, total_num)

    completion = client.chat.completions.create(
        model="qwen3-max",
        messages=messages,
        stream=False
    )
    return completion.choices[0].message.content


def process_batch(start_idx: int, end_idx: int, intent_map, total_num: int) -> str:
    """
    在线程中处理一个 batch：
    - 调大模型
    - 解析 JSON
    - 写出到对应文件
    返回写出的文件路径，用于日志/调试。
    """
    logger.info(f"[线程] 生成第 {start_idx} 条到第 {end_idx} 条的数据")

    samples_str = gen_data(intent_map=intent_map, start=start_idx, end=end_idx, total_num=total_num)

    try:
        data = json.loads(samples_str)
    except json.JSONDecodeError as e:
        logger.error(f"[线程] JSON 解析失败: {e}")
        logger.error(f"[线程] 原始输出: {samples_str}")
        # 这里直接抛出，让主线程感知到失败
        raise

    out_path = os.path.join(OUTPUT_DIR, f"{start_idx}_{end_idx}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"[线程] 已写入文件: {out_path}")
    return out_path


if __name__ == "__main__":
    # 读取意图列表
    if not os.path.exists(INTENT_MAP_PATH):
        raise FileNotFoundError(f"意图文件不存在，请检查路径: {INTENT_MAP_PATH}")

    with open(INTENT_MAP_PATH, "r", encoding="utf-8") as f:
        intent_map = json.load(f)

    # 生成总数 & 每批数量
    N = 5000  # 总样本数
    batch = 5  # 每批生成条数

    # 线程数，可通过环境变量控制
    max_workers = 10

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info(f"开始生成意图识别数据，总数={N}，步长={batch}，线程数={max_workers}")

    # 预先计算好所有 batch 的区间
    batches = []
    for i in range(0, N, batch):
        start_idx = i
        end_idx = min(i + batch, N)
        batches.append((start_idx, end_idx))

    # 使用线程池并行生成
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(
                process_batch,
                start_idx,
                end_idx,
                intent_map,
                N,
            ): (start_idx, end_idx)
            for (start_idx, end_idx) in batches
        }

        # 用 tqdm 包一层进度条
        for future in tqdm(as_completed(future_to_batch), total=len(future_to_batch), desc="Generating data (multi-thread)"):
            start_idx, end_idx = future_to_batch[future]
            try:
                out_path = future.result()
                logger.info(f"batch [{start_idx}, {end_idx}] 完成，输出文件: {out_path}")
            except Exception as e:
                logger.error(f"batch [{start_idx}, {end_idx}] 失败: {repr(e)}")
