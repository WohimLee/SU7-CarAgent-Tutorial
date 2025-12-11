import os
from pathlib import Path
from typing import List, Dict, Any

from pymilvus import (
    MilvusClient,
    FieldSchema,
    CollectionSchema,
    DataType,
)
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from su7.common.my_logger import logger


# ====== 路径 & 常量 ======
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

DEFAULT_CHUNKS_DIR = os.path.join(PROJECT_ROOT, "output", "mineru_parse", "chunks")
DEFAULT_COLLECTION_NAME = "su7_manual"

# MilvusClient（HTTP 客户端）
client = MilvusClient(uri="http://localhost:19530")

# 嵌入模型名称和维度需要与你实际使用的服务保持一致
EMBEDDING_MODEL = "/Users/azen/Desktop/llm/models/bge-m3"
EMBEDDING_DIM = 1024


# ====== 工具函数 ======
def clean_markdown(text: str) -> str:
    """
    简单清洗 Markdown：
    - 去掉图片行
    - 保留标题和正文
    """
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("![]("):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def load_chunks(chunks_dir: str) -> List[Dict[str, Any]]:
    """
    从 output/mineru_parse/chunks 读取所有 .md 文件。

    返回：
    [
        {
            "chunk_id": "095_挂钩",
            "content": "……",
            "source": "output/mineru_parse/chunks/095_挂钩.md",
        },
        ...
    ]
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
        items.append(
            {
                "chunk_id": file_path.stem,
                "content": content,
                "source": str(file_path),
            }
        )

    logger.info(f"Loaded {len(items)} chunks from {chunks_dir}")
    return items


# ====== 检索器 ======
class MilvusManualRetriever:
    """
    Milvus 检索器：
    - build_index_from_chunks: 把 output/mineru_parse/chunks 写入 Milvus
    - retrieve: 根据 query 做相似度检索
    """

    def __init__(
        self,
        collection_name: str,
        client: MilvusClient,
        emb_model: SentenceTransformer,
        recreate: bool = False,
    ):
        self.collection_name = collection_name
        self.client = client
        self.emb_model = emb_model

        # 如需保留历史数据，把 recreate 改成 False
        self.create_collection(collection_name, recreate=recreate)

    def create_collection(self, collection_name: str, recreate: bool = True):
        """
        创建（或重建）集合，并建立索引。
        """
        if recreate and self.client.has_collection(collection_name=collection_name):
            self.client.drop_collection(collection_name=collection_name)

        if not self.client.has_collection(collection_name=collection_name):
            fields = [
                FieldSchema(
                    name="primary_key",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=True,
                ),
                FieldSchema(
                    name="chunk_id",
                    dtype=DataType.VARCHAR,
                    max_length=256,
                ),
                FieldSchema(
                    name="content",
                    dtype=DataType.VARCHAR,
                    max_length=1024 * 16,
                ),
                FieldSchema(
                    name="source",
                    dtype=DataType.VARCHAR,
                    max_length=1024,
                ),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=EMBEDDING_DIM,
                ),
            ]

            schema = CollectionSchema(
                fields=fields,
                description="SU7 manual",
                enable_dynamic_field=False,
            )

            # 创建集合
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
            )

            # 建索引
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type="HNSW",
                metric_type="IP",   # 模型一般是余弦/内积
                params={"M": 16, "efConstruction": 200},
            )

            self.client.create_index(
                collection_name=collection_name,
                index_params=index_params,
            )

        # load 到内存
        self.client.load_collection(collection_name=collection_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        用 SentenceTransformer 计算向量。
        返回：List[List[float]]
        """
        # 默认 convert_to_numpy=True，返回 np.ndarray
        vecs = self.emb_model.encode(texts)
        try:
            return vecs.tolist()
        except AttributeError:
            # 已经是 list 了
            return vecs

    def build_index_from_chunks(self, chunks_dir: str, batch_size: int = 8):
        """
        遍历 chunks 目录，计算 embedding 并批量写入 Milvus。
        """
        chunks = load_chunks(chunks_dir)

        texts = [item["content"] for item in chunks]
        chunk_ids = [item["chunk_id"] for item in chunks]
        sources = [item["source"] for item in chunks]

        logger.info(
            f"Start inserting {len(chunks)} chunks into Milvus collection '{self.collection_name}'"
        )

        for start in tqdm(range(0, len(chunks), batch_size)):
            end = min(start + batch_size, len(chunks))
            batch_texts = texts[start:end]
            batch_chunk_ids = chunk_ids[start:end]
            batch_sources = sources[start:end]

            embeddings = self.embed_texts(batch_texts)

            if not embeddings:
                continue

            if len(embeddings[0]) != EMBEDDING_DIM:
                raise ValueError(
                    f"Embedding dim mismatch: got {len(embeddings[0])}, expected {EMBEDDING_DIM}"
                )

            entities = [
                {
                    "chunk_id": batch_chunk_ids[i],
                    "content": batch_texts[i],
                    "source": batch_sources[i],
                    "embedding": embeddings[i],
                }
                for i in range(len(batch_texts))
            ]

            self.client.insert(self.collection_name, entities)

            logger.info(
                f"Inserted batch {start}-{end} into collection '{self.collection_name}'"
            )

        # 一定要带 collection_name
        self.client.flush(collection_name=self.collection_name)
        logger.info(f"Finished building index for collection '{self.collection_name}'")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        在 Milvus 中按向量相似度检索最相关的 chunk。
        返回结构：
        [
            {
                "score": 0.9,
                "chunk_id": "...",
                "content": "...",
                "source": "...",
            },
            ...
        ]
        """
        if not query:
            return []

        query_vec = self.embed_texts([query])[0]
        if len(query_vec) != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dim mismatch: got {len(query_vec)}, expected {EMBEDDING_DIM}"
            )

        self.client.load_collection(collection_name=self.collection_name)

        search_params = {
            "metric_type": "IP",
            "params": {
                "ef": 64,  # HNSW 搜索参数
            },
        }

        # MilvusClient API：返回 list[list[dict]]，每个 query 对应一个 list
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vec],
            limit=top_k,
            output_fields=["chunk_id", "content", "source"],
            search_params=search_params,
            # filter='company_code == "600000.SH" && year >= 2023 && year <= 2024'
        )

        hits: List[Dict[str, Any]] = []

        # 只搜索了一个 query，所以用 results[0]
        for hit in results[0]:
            # 常见结构：{"id": ..., "distance": ..., "entity": {...}}
            entity = hit.get("entity", {})
            score = float(hit.get("distance", 0.0))
            hits.append(
                {
                    "score": score,
                    "chunk_id": entity.get("chunk_id"),
                    "content": entity.get("content"),
                    "source": entity.get("source"),
                }
            )

        return hits


# ====== 脚本入口 ======
if __name__ == "__main__":
    # 如需清空所有 collection，可取消下面注释：
    # collections = client.list_collections()
    # print(collections)
    # for c in collections:
    #     client.drop_collection(c)

    chunks_dir = DEFAULT_CHUNKS_DIR
    emb_model = SentenceTransformer(EMBEDDING_MODEL)

    retriever = MilvusManualRetriever(
        collection_name=DEFAULT_COLLECTION_NAME,
        client=client,
        emb_model=emb_model,
        recreate=False,  # 不想每次重建可以改成 False
    )

    # retriever.build_index_from_chunks(chunks_dir=chunks_dir)

    query = "挂钩有什么注意事项"

    hits = retriever.retrieve(query=query, top_k=10)
    for i, hit in enumerate(hits, start=1):
        logger.info(
            f"[{i}] score={hit['score']:.4f}, chunk_id={hit['chunk_id']}, source={hit['source']}"
        )
