

# pip install pymupdf


import fitz  # PyMuPDF
import json

def get_outline(pdf_path):
    doc = fitz.open(pdf_path)
    # 获取目录（TOC，Table of Contents）
    # 返回格式：[[level, title, page], ...]
    toc = doc.get_toc()  # 默认 simple=True

    for level, title, page in toc:
        indent = "  " * (level - 1)  # 用缩进表示层级
        print(f"{indent}{title} ...... 第 {page} 页")

    # 保存为 markdown
    out_path = "data/outline.md"
    with open(out_path, "w", encoding="utf-8") as f:
        for level, title, page in toc:
            hashes = "#" * level  # 生成 #、##、### ...
            # f.write(f"{hashes} {title}（p.{page}）\n")
            f.write(f"{hashes} {title}\n")


    # 保存为 json
    data = [
        {"level": level, "title": title, "page": page}
        for level, title, page in toc
    ]

    out_path = "data/outline.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("JSON 目录已保存到:", out_path)

    # 保存为 txt
    out_path = "data/outline.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for level, title, page in toc:
            indent = "  " * (level - 1)
            f.write(f"{indent}- {title}（p.{page}）\n")

    print("Markdown 目录已保存到:", out_path)

if __name__ == "__main__":

    file = "data/raw/2024 小米SU7 Pro Max 用户手册.pdf"

    res = get_outline(file)

    pass