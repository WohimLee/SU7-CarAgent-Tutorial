import os
import re
from pathlib import Path
from typing import Any, Dict, List

from elasticsearch import Elasticsearch, helpers

from su7.common.my_logger import logger


# ====== 路径 & 常量 ======
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

DEFAULT_CHUNKS_DIR = os.path.join(PROJECT_ROOT, "output", "mineru_parse", "chunks")

# ES 相关配置，可通过环境变量覆盖
DEFAULT_ES_URL = "http://localhost:9200"
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
    """从 output/mineru_parse/chunks 读取所有 .md chunk。"""
    path = Path(chunks_dir)
    if not path.exists():
        raise FileNotFoundError(f"Chunks dir not found: {chunks_dir}")

    items: List[Dict[str, Any]] = []
    for file_path in sorted(path.glob("*.md")):
        raw = file_path.read_text(encoding="utf-8")
        content = clean_markdown(raw)
        if not content:
            continue
        items.append(
            {
                "chunk_id": file_path.stem,
                "content": content,
                "source": str(file_path),
            }
        )

    logger.info("Loaded %d chunks from %s", len(items), chunks_dir)
    return items


# ====== 简单分词 & 领域词抽取（用于构造 ES 查询 / 字段） ======
_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+")
_CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> List[str]:
    """把文本切成粗粒度 token（中文连串 / 英文数字串）。"""
    tokens: List[str] = []
    for match in _TOKEN_PATTERN.finditer(text):
        token = match.group(0).strip()
        if not token:
            continue
        tokens.append(token.lower())
    return tokens


def extract_domain_terms(tokens: List[str]) -> List[str]:
    """从 token 列表中抽取“领域专业词”。

    简单规则：
    - 中文：连续汉字长度 >= 2
    - 英文/数字：长度 >= 3
    """
    terms: List[str] = []
    for token in tokens:
        if _CHINESE_PATTERN.fullmatch(token):
            if len(token) >= 2:
                terms.append(token)
        else:
            if len(token) >= 3:
                terms.append(token)
    return terms


# ====== 检索器：基于 Elasticsearch 的关键词 / 领域词召回 ======
class ESManualKeywordRetriever:
    """使用 Elasticsearch 对手册 chunks 做关键词 & 领域词召回。"""

    def __init__(
        self,
        index_name: str = DEFAULT_ES_INDEX,
        es_url: str = DEFAULT_ES_URL,
    ) -> None:
        self.index_name = index_name
        self.client = Elasticsearch(es_url)

    # ---- 索引构建 ----
    def create_index(self, recreate: bool = False) -> None:
        """创建（或重建）ES 索引，包含 content / domain_terms 等字段。"""
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
                    "content": {
                        "type": "text",
                        "analyzer": "standard",
                    },
                    # 将抽取出的领域词列表写入这里，使用 text 提升召回权重
                    "domain_terms": {
                        "type": "text",
                        "analyzer": "standard",
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
        """遍历 chunks，写入 ES（包含 content + domain_terms）。"""
        self.create_index(recreate=recreate)

        chunks = load_chunks(chunks_dir)
        logger.info(
            "Start indexing %d chunks into ES index '%s'",
            len(chunks),
            self.index_name,
        )

        actions = []
        for item in chunks:
            tokens = tokenize(item["content"])
            domain_terms = extract_domain_terms(tokens)

            doc = {
                "chunk_id": item["chunk_id"],
                "source": item["source"],
                "content": item["content"],
                # 作为空格分隔的字符串写入，便于用 match 查询
                "domain_terms": " ".join(domain_terms),
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
        """在 ES 中按关键词 / 领域词检索最相关的 chunk。"""
        if not query:
            return []

        query_tokens = tokenize(query)
        query_domain_terms = extract_domain_terms(query_tokens)

        should_clauses: List[Dict[str, Any]] = []

        # 原始 query 在 content 上做全文检索
        should_clauses.append(
            {
                "match": {
                    "content": {
                        "query": query,
                        "boost": 1.0,
                    }
                }
            }
        )

        # token 合成的查询
        if query_tokens:
            should_clauses.append(
                {
                    "match": {
                        "content": {
                            "query": " ".join(query_tokens),
                            "boost": 1.2,
                        }
                    }
                }
            )

        # 领域词单独一个字段，给更高权重
        if query_domain_terms:
            should_clauses.append(
                {
                    "match": {
                        "domain_terms": {
                            "query": " ".join(query_domain_terms),
                            "boost": 2.0,
                        }
                    }
                }
            )

        body = {
            "query": {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1,
                }
            },
            "_source": ["chunk_id", "content", "source"],
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
                    "content": source.get("content"),
                    "source": source.get("source"),
                }
            )

        return results


if __name__ == "__main__":
    # 示例：先（可选）构建索引，再做一次检索
    retriever = ESManualKeywordRetriever()

    # 首次使用时可以取消下一行注释，完成索引构建
    # retriever.build_index_from_chunks(recreate=True)

    query = "挂钩有什么注意事项"
    hits = retriever.retrieve(query=query, top_k=10)
    for i, hit in enumerate(hits, start=1):
        logger.info(
            f"[{i}] score={hit['score']:.4f}, "
            f"chunk_id={hit['chunk_id']}, source={hit['source']}"
        )
