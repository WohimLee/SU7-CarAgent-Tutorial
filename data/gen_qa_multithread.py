import os
import re
import base64
import json

from pathlib import Path
from textwrap import dedent
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

from su7.common.my_logger import logger


# ---------------- 基础配置 ----------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

CHUNKS_DIR = os.path.join(PROJECT_ROOT, "output/mineru_parse/chunks")
QA_DIR = os.path.join(PROJECT_ROOT, "output/mineru_parse/qa")
IMAGES_ROOT = os.path.join(PROJECT_ROOT, "output/mineru_parse")

os.makedirs(QA_DIR, exist_ok=True)

SYSTEM_PROMPT = dedent("""
你是一名专业的汽车技术文档专家和多模态助手，负责根据《小米 SU7 用户手册》的各个小节生成用于检索问答系统的问答对（Q&A）。

【核心输出要求（必须严格遵守）】
1. 最终输出必须是 **纯粹的 JSON 数组**，不能包含任何额外字符。
2. **禁止输出 Markdown 代码块**，如 ```json、```、`json`、``` 等，一律不允许出现。
3. **禁止输出空行、注释、说明文字、前后缀、标签、提示语**。只能输出 JSON 内容本身。
4. JSON 数组格式示例（注意：你不能输出示例，只能输出正式内容）：
   [
        {
        "question": "...",
        "answer": "...",
        "source_title": "...",
        "source_type": "manual_chunk",
        "language": "zh-CN"
        }
   ]
5. 所有 JSON 字段必须符合以下结构：
   - "question"：问题
   - "answer"：答案
   - "source_title"：本小节标题
   - "source_type"：固定写 "manual_chunk"
   - "language"：固定写 "zh-CN"
6. JSON 中不得出现其他字段，也不得缺少字段。
7. 数组中的条目数量通常为 **5–15 条**（视章节信息量可多可少）。
8. 如果本章内容几乎没有信息，可输出空数组：[]

【内容要求】
1. 所有问答必须完全基于传入的小节 Markdown 文本和图片，不允许使用外部知识。
2. 既要包含基础操作类问题（如「如何…？」、「什么是…？」），也要包含注意事项、安全提示、流程步骤类问题。
3. 当问题或答案中引用了来自图片的信息时，必须在文本中明确写出对应图片的文件名或相对路径，例如：
   - “…如图片 images/xxx.png 所示…”
   - “…详见 images/xxx/yyy.jpg 中标注的按键…”
   不要只说“如图所示”、“见上图”等泛化表述，而是要把图片路径一并写出来。
4. 答案必须自然、口语化但专业准确；涉及操作时优先使用步骤描述（例如 1、2、3…）。
5. 问题之间不能重复或高度相似，应覆盖不同的知识点或使用场景。

【绝对禁止】
- 输出 markdown 代码块（例如 ```json）
- 输出除 JSON 外的任何文字
- 输出多余的包装结构，如 {"result": [...]}
- 输出解释说明
- 在 JSON 前后添加反引号、缩进提示或前缀后缀

你必须只返回**合法 JSON 数组**。
""").strip()


USER_PROMPT = dedent("""
下面是本次需要处理的手册小节：

【小节标题】
{chunk_title}

【小节正文】（Markdown）
```markdown
{chunk_text}
```

请结合系统说明和（若有）随消息附带的图片，为本小节生成尽量全面且实用的问答对，并按要求输出 JSON 数组。
""").strip()


client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_API_URL"),
)


# ---------------- 工具函数 ----------------

def guess_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    return "image/jpeg"


def image_file_to_data_url(path: Path) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    mime = guess_mime_type(path)
    return f"data:{mime};base64,{b64}"


def extract_image_paths_from_markdown(chunk_text: str) -> list[Path]:
    matches = re.findall(r"\((images/[^)\s]+)\)", chunk_text)
    image_paths: list[Path] = []
    for rel_path in matches:
        abs_path = Path(IMAGES_ROOT) / rel_path
        image_paths.append(abs_path)
    return image_paths


def build_messages(chunk_title: str, chunk_text: str):
    image_paths = extract_image_paths_from_markdown(chunk_text)
    image_contents = []

    for p in image_paths:
        if not p.exists():
            logger.warning(f"Image not found, skip: {p}")
            continue
        try:
            data_url = image_file_to_data_url(p)
        except Exception as e:
            logger.exception(f"Failed to encode image {p}: {e}")
            continue

        image_contents.append({
            "type": "image_url",
            "image_url": {"url": data_url},
        })

    user_text = USER_PROMPT.format(
        chunk_title=chunk_title,
        chunk_text=chunk_text,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                *image_contents,
                {"type": "text", "text": user_text},
            ],
        },
    ]
    return messages


def gen_data(chunk_title: str, chunk_text: str) -> str:
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

    content = completion.choices[0].message.content
    if isinstance(content, str):
        return content.strip()

    try:
        return "".join(part.get("text", "") for part in content).strip()
    except Exception:
        return str(content).strip()


def process_single_chunk(chunk: str):
    chunk_path = os.path.join(CHUNKS_DIR, chunk)

    try:
        with open(chunk_path, "r", encoding="utf-8") as f:
            chunk_text = f.read()
    except Exception as e:
        logger.exception(f"读取 chunk 失败: {chunk_path}, error: {e}")
        return

    if len(chunk_text) < 10:
        return

    try:
        chunk_title = re.split(r"[_.]", chunk)[1]
    except Exception:
        chunk_title = chunk

    try:
        logger.info(f"正在生成: 【{chunk}】 QA 对...")
        qa_json_str = gen_data(chunk_title, chunk_text)
    except Exception as e:
        logger.exception(f"生成 QA 失败: {chunk_path}, error: {e}")
        return

    try:
        data = json.loads(qa_json_str)
        if not isinstance(data, list):
            logger.warning(f"返回的 JSON 不是数组，文件: {chunk}")
    except json.JSONDecodeError:
        logger.warning(f"模型返回的内容不是合法 JSON，原样写入: {chunk}")

    out_path = os.path.join(QA_DIR, f"{chunk}.qa.json")
    try:
        with open(out_path, "w", encoding="utf-8") as fw:
            fw.write(qa_json_str)
        logger.info(f"已生成 QA: {out_path}")
    except Exception as e:
        logger.exception(f"写入 QA 文件失败: {out_path}, error: {e}")


def process_all_chunks(max_workers: int | None = None):
    if not os.path.isdir(CHUNKS_DIR):
        raise FileNotFoundError(f"chunks 目录不存在: {CHUNKS_DIR}")

    all_chunks = sorted(
        f for f in os.listdir(CHUNKS_DIR)
        if os.path.isfile(os.path.join(CHUNKS_DIR, f))
    )

    if not all_chunks:
        logger.warning("未找到任何 chunk 文件")
        return

    logger.info(f"共发现 {len(all_chunks)} 个 chunk 文件，开始并发生成 QA …")

    if max_workers is None:
        cpu_cnt = os.cpu_count() or 4
        max_workers = min(8, cpu_cnt)

    logger.info(f"使用线程数: {max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_chunk, chunk): chunk
            for chunk in all_chunks
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating QA for chunks"):
            try:
                future.result()
            except Exception as e:
                chunk = futures[future]
                logger.exception(f"处理 chunk 过程中未捕获的异常: {chunk}, error: {e}")


if __name__ == "__main__":
    worker = 10
    process_all_chunks(worker)
