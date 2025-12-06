# data/chunk_manual.py

from pathlib import Path
import json
import re


OUTLINE_PATH = Path("data/outline.md")
MANUAL_PATH = Path("output/mineru_parse/2024 小米SU7 Pro Max 用户手册.md")
OUT_DIR = Path("output/mineru_parse/chunks")


def parse_outline_md(path: Path):
    """解析 data/outline.md，得到按顺序排列的 (level, title) 列表。"""
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("#"):
                continue
            level = len(line) - len(line.lstrip("#"))
            title = line[level:].strip()
            if title:
                items.append({"level": level, "title": title})
    return items


def load_manual_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return f.readlines()


def find_section_positions(outline_items, manual_lines):
    """
    在手册中按顺序查找每个大纲标题出现的位置（行号）。
    匹配规则：行以 # 开头，且包含标题字符串。
    """
    positions = []
    last_idx = 0
    for item in outline_items:
        if item["level"] == 1:
            continue
        title = item["title"]
        found_idx = None
        for idx in range(last_idx, len(manual_lines)):
            line = manual_lines[idx].strip()
            if line.startswith("#") and title in line:
                found_idx = idx
                last_idx = idx + 1
                break
        if found_idx is not None:
            positions.append({**item, "start_line": found_idx})
        else:
            # 如果没找到，可以根据需要打印一下提示
            print(f"[WARN] 没找到标题：{title}")
    return positions


def build_chunks(positions, manual_lines):
    """根据每个标题的 start_line，把手册切成一个个 chunk。"""
    chunks = []
    for i, pos in enumerate(positions):
        start = pos["start_line"]
        end = positions[i + 1]["start_line"] if i + 1 < len(positions) else len(manual_lines)
        content = "".join(manual_lines[start:end]).strip()
        chunks.append(
            {
                "title": pos["title"],
                "level": pos["level"],
                "start_line": start + 1,  # 转成 1-based，便于调试
                "end_line": end,
                "content": content,
            }
        )
    return chunks


def normalize_filename(title: str) -> str:
    """
    把中文标题转成适合作为文件名的一部分：
    - 保留中文、数字、字母，下划线
    - 其它字符变成下划线
    """
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "_", title)
    return s.strip("_") or "section"


def save_chunks_as_files(chunks, out_dir: Path):
    """每个 chunk 单独保存为一个 markdown 文件。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, ch in enumerate(chunks, start=1):
        fname = f"{i:03d}_{normalize_filename(ch['title'])}.md"
        path = out_dir / fname
        with path.open("w", encoding="utf-8") as f:
            f.write(ch["content"])
        print(f"saved: {path}")


def save_chunks_jsonl(chunks, path: Path):
    """可选：同时保存一份 JSONL，方便做 RAG。"""
    with path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    print(f"JSONL saved to: {path}")


def main():
    outline_items = parse_outline_md(OUTLINE_PATH)
    manual_lines = load_manual_lines(MANUAL_PATH)

    positions = find_section_positions(outline_items, manual_lines)
    print(f"找到 {len(positions)} 个标题匹配。")

    chunks = build_chunks(positions, manual_lines)

    # 1) 每段保存成单独的 markdown 文件
    save_chunks_as_files(chunks, OUT_DIR)

    # 2) 可选：保存一份 JSONL 汇总
    save_chunks_jsonl(chunks, Path("output/mineru_parse/manual_chunks.jsonl"))


if __name__ == "__main__":
    main()
