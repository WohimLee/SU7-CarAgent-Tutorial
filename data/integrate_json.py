import json
import glob
import os

# 要合并的目录
INPUT_DIR = "output/queries"
# 合并结果输出文件
OUTPUT_FILE = "output/merged_queries.json"

def main():
    # 找到所有 .json 文件并排序（按文件名）
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))

    merged = []

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 如果每个文件本身就是一个列表，就把它展开
        if isinstance(data, list):
            merged.extend(data)
        else:
            # 否则直接把对象 append 进去
            merged.append(data)

    # 保存合并后的结果
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"合并完成，共 {len(merged)} 条，写入 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
