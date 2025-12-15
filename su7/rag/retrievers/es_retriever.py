import os
from pathlib import Path
from typing import Any, Dict, List

from elasticsearch import Elasticsearch, helpers

from su7.common.my_logger import logger


# ====== 路径 & 常量 ======
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

DEFAULT_CHUNKS_DIR = os.path.join(PROJECT_ROOT, "output", "mineru_parse", "chunks")

# ES 相关配置，可通过环境变量覆盖
DEFAULT_ES_URL = "http://localhost:9200"
BASIC_AUTH = ("elastic", "QPe9yYcr")
DEFAULT_ES_INDEX = "su7_manual_es"


# ====== 工具函数：读取 chunks ======
def clean_markdown(text: str) -> str:
    """简单清洗 Markdown 文本。"""
    lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        # 丢掉图片行等噪声
        if stripped.startswith("![]("):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def load_chunks(chunks_dir: str) -> List[Dict[str, Any]]:
    """从 output/mineru_parse/chunks 读取所有 .md chunk。

    同时从文件名中解析一个“标题”，用于领域专业词增强（例如 095_挂钩.md → 标题为“挂钩”）。
    """
    path = Path(chunks_dir)
    if not path.exists():
        raise FileNotFoundError(f"Chunks dir not found: {chunks_dir}")

    items: List[Dict[str, Any]] = []
    for file_path in sorted(path.glob("*.md")):
        raw = file_path.read_text(encoding="utf-8")
        content = clean_markdown(raw)
        if not content:
            continue

        stem = file_path.stem
        # 形如 "095_挂钩" → 取下划线后的部分作为标题
        if "_" in stem:
            _, title = stem.split("_", 1)
        else:
            title = stem

        items.append(
            {
                "chunk_id": stem,
                "title": title,
                "content": content,
                "source": str(file_path),
            }
        )

    logger.info("Loaded %d chunks from %s", len(items), chunks_dir)
    return items


# ====== 检索器：基于 Elasticsearch 的关键词 / 领域词召回 ======
class ESManualKeywordRetriever:
    """使用 Elasticsearch + IK 分词对手册 chunks 做关键词 & 领域专业词召回。"""

    def __init__(
        self,
        index_name: str = DEFAULT_ES_INDEX,
        es_url: str = DEFAULT_ES_URL,
        basic_auth: tuple = BASIC_AUTH
    ) -> None:
        self.index_name = index_name
        self.client = Elasticsearch(es_url, basic_auth=basic_auth)

    # ---- 索引构建 ----
    def create_index(self, recreate: bool = False) -> None:
        """创建（或重建）ES 索引，使用 IK 分词器。"""
        if recreate and self.client.indices.exists(index=self.index_name):
            logger.info("Deleting existing ES index %s", self.index_name)
            self.client.indices.delete(index=self.index_name)

        if self.client.indices.exists(index=self.index_name):
            return

        body = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "source": {"type": "keyword"},
                     # 从文件名解析出的标题，如“挂钩”“远程控制”，偏领域级专业词
                    "title": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "search_analyzer": "ik_max_word",
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "search_analyzer": "ik_max_word",
                    },
                }
            },
        }

        self.client.indices.create(index=self.index_name, body=body)
        logger.info("Created ES index %s", self.index_name)

    def build_index_from_chunks(
        self,
        chunks_dir: str = DEFAULT_CHUNKS_DIR,
        recreate: bool = False,
        batch_size: int = 500,
    ) -> None:
        """遍历 chunks，写入 ES（包含 title + content）。"""
        self.create_index(recreate=recreate)

        chunks = load_chunks(chunks_dir)
        logger.info(
            "Start indexing %d chunks into ES index '%s'",
            len(chunks),
            self.index_name,
        )

        actions = []
        for item in chunks:
            doc = {
                "chunk_id": item["chunk_id"],
                "source": item["source"],
                "title": item["title"],
                "content": item["content"],
            }

            actions.append(
                {
                    "_index": self.index_name,
                    "_id": item["chunk_id"],
                    "_source": doc,
                }
            )

            if len(actions) >= batch_size:
                helpers.bulk(self.client, actions)
                actions = []

        if actions:
            helpers.bulk(self.client, actions)

        self.client.indices.refresh(index=self.index_name)
        logger.info("Finished indexing chunks into ES index '%s'", self.index_name)

    # ---- 检索 ----
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """在 ES 中按关键词 / 领域专业词检索最相关的 chunk。"""
        if not query:
            return []

        body = {
            "query": {
                # 把分词、倒排和相关性交给 ES + IK 分词器处理，
                # 对标题字段（领域词）赋予更高权重。
                "multi_match": {
                    "query": query,
                    "fields": [
                        "title^3.0",   # 领域专业词（来自文件名）权重更高
                        "content^1.0",  # 正文内容
                    ],
                    "type": "best_fields",
                }
            },
            "_source": ["chunk_id", "title", "content", "source"],
            "size": top_k,
        }

        resp = self.client.search(index=self.index_name, body=body)
        hits_raw = resp.get("hits", {}).get("hits", [])

        results: List[Dict[str, Any]] = []
        for hit in hits_raw:
            source = hit.get("_source", {})
            score = float(hit.get("_score", 0.0))
            results.append(
                {
                    "score": score,
                    "chunk_id": source.get("chunk_id"),
                    "title": source.get("title"),
                    "content": source.get("content"),
                    "source": source.get("source"),
                }
            )

        return results


if __name__ == "__main__":
    # 示例：先（可选）构建索引，再做一次检索
    retriever = ESManualKeywordRetriever()

    # 首次使用时可以取消下一行注释，完成索引构建
    retriever.build_index_from_chunks(recreate=True)

    query = "挂钩有什么注意事项"
    hits = retriever.retrieve(query=query, top_k=10)
    for i, hit in enumerate(hits, start=1):
        logger.info(
            f"[{i}] score={hit['score']:.4f}, "
            f"chunk_id={hit['chunk_id']}, source={hit['source']}"
        )
