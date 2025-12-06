import os
import re
import base64
import json

from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
from textwrap import dedent
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from su7.common.my_logger import logger


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
load_dotenv(os.path.join(f"{PROJECT_ROOT}", ".env"))

SYSTEM_PROMPT = dedent("""
你是一名专业的汽车技术文档专家和多模态助手，负责根据《小米 SU7 用户手册》的各个小节生成用于检索问答系统的问答对（Q&A）。

总体要求：
1. 你会同时接收到一段 Markdown 文本和若干与该小节相关的图片。
2. 所有问题和答案必须严格依据当前小节的文字和图片，不要引入外部知识，不要猜测文中或图中没有给出的信息。
3. 既要有面向普通用户的基础操作类问题（例如“如何…？”、“什么是…？”），也要有涉及注意事项、安全警示或多步操作流程的综合性问题。
4. 如果图片中出现按键编号、部件名称、图示步骤、表格信息等，可以据此设计问题；文本中的 Markdown 表格或 HTML <table> 视为正文的一部分，也可以用来出题。
5. 答案需使用自然、口语化但专业准确的中文完整句子，如有安全相关提示，需要在答案中强调风险和注意事项。
6. 如果某条信息只出现在图片中而未在文字中描述，也可以根据图片内容出题；但不要在无法从图中看出的地方进行想象或发挥。
7. 输出格式必须为 JSON 数组，每个元素包含字段：
   - "question"：问题
   - "answer"：答案
   - "source_title"：本小节标题
   - "source_type"：固定写 "manual_chunk"
   - "language"：固定写 "zh-CN"
8. 只输出 JSON 数组，不要包含额外说明文本。
""").strip()


USER_PROMPT = dedent("""
下面是本次需要处理的手册小节：

【小节标题】
{chunk_title}

【小节正文】（Markdown）
```markdown
{chunk_text}
请结合系统说明和（若有）随消息附带的图片，为本小节生成尽量全面且实用的问答对，并按要求输出 JSON 数组。
""").strip()


client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_API_URL"),
)

def image_file_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def build_messages(chunk_title, chunk_text):
    # 1. 先把所有图片转成 image_url

    images_root = os.path.join(PROJECT_ROOT, "output")

    res = re.findall(r"(images/[^\s)]+)", chunk_text)

    image_paths = [os.path.join(images_root, path) for path in res]
    image_contents = []
    for p in image_paths:
        data_url = image_file_to_data_url(Path(p))
        image_contents.append({
            "type": "image_url",
            "image_url": {"url": data_url},
        })

    # 2. 再拼 user 的文本 prompt
    user_text = USER_PROMPT.format(
        chunk_title=chunk_title,
        chunk_text=chunk_text,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                *image_contents,                # 所有图片
                {"type": "text", "text": user_text},  # 本次 chunk 的文字说明
            ],
        },
    ]
    return messages


def gen_data(chunk_title, chunk_text):

    messages = build_messages(chunk_title, chunk_text)

    completion = client.chat.completions.create(
        model="qwen3-vl-plus",
        messages=messages,
        stream=False,
        extra_body={
            "enable_thinking": False,
            "thinking_budget": 81920,
        },
    )
    return completion.choices[0].message.content




if __name__ == "__main__":
    all_chunks = os.listdir(os.path.join(PROJECT_ROOT, "output/mineru_parse/chunks"))
    for chunk in all_chunks:
        with open(os.path.join(PROJECT_ROOT, "output/mineru_parse/chunks", chunk)) as f:
            
            chunk_title = re.split(r"[_.]", chunk)[1]
            chunk_text = f.read()
        build_messages(chunk_title, chunk_text)

