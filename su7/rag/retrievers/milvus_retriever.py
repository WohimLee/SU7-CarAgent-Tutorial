import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from openai import OpenAI
from pymilvus import (
    MilvusClient,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)
from sent

from su7.common.my_logger import logger



PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

DEFAULT_CHUNKS_DIR = os.path.join(PROJECT_ROOT, "output", "mineru_parse", "chunks")

DEFAULT_COLLECTION_NAME = "su7_manual"


client = MilvusClient(uri="http://localhost:19530")

# 嵌入模型名称和维度需要与你实际使用的服务保持一致
EMBEDDING_MODEL = "/Users/azen/Desktop/llm/models/bge-m3"
EMBEDDING_DIM = 1024




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


def load_chunks(chunks_dir: str):
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


class MilvusManualRetriever:
    """
    Milvus 检索器：
    - build_index_from_chunks: 把 output/mineru_parse/chunks 写入 Milvus
    - retrieve: 根据 query 做相似度检索
    """

    def __init__(self, collection_name, client: MilvusClient):
        self.collection_name = collection_name
        self.client = client

        self.create_collection(collection_name)
        
    def create_collection(self, collection_name):
        if self.client.has_collection(collection_name=collection_name):
            self.client.drop_collection(collection_name=collection_name)

        fields = [
            FieldSchema(
                name="primary_key",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
            ),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
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

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="IP",   # 或 "L2" 看你向量
            params={"M": 16, "efConstruction": 200},
        )

        self.client.create_index(
            collection_name=collection_name,
            index_params=index_params,
        )

        # load
        self.client.load_collection(collection_name=collection_name)





    def build_index_from_chunks(self, chunks_dir, batch_size: int = 16):
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

        for start in range(0, len(chunks), batch_size):
            end = min(start + batch_size, len(chunks))
            batch_texts = texts[start:end]
            batch_chunk_ids = chunk_ids[start:end]
            batch_sources = sources[start:end]

            embeddings = self.embed_texts(batch_texts)
            if not embeddings:
                logger.warning(f"Empty embeddings for batch {start}-{end}, skip.")
                continue

            if len(embeddings[0]) != EMBEDDING_DIM:
                raise ValueError(
                    f"Embedding dim mismatch: got {len(embeddings[0])}, expected {EMBEDDING_DIM}"
                )

            self.collection.insert(
                [
                    batch_chunk_ids,
                    batch_texts,
                    batch_sources,
                    embeddings,
                ],
                field_names=["chunk_id", "content", "source", "embedding"],
            )

            logger.info(
                f"Inserted batch {start}-{end} into collection '{self.collection_name}'"
            )
        
        self.collection.flush()
        self.collection.load()
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

        self.collection.load()

        search_params = {
            "metric_type": "IP",
            "params": {"ef": 64},
        }

        results = self.collection.search(
            data=[query_vec],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["chunk_id", "content", "source"],
        )

        hits = []
        for hit in results[0]:
            hits.append(
                {
                    "score": float(hit.score),
                    "chunk_id": hit.entity.get("chunk_id"),
                    "content": hit.entity.get("content"),
                    "source": hit.entity.get("source"),
                }
            )

        return hits
    
    def embed_texts(self):

        return


if __name__ == "__main__":

    # collections = client.list_collections()
    # print(collections)
    # for collection in collections:
    #     client.drop_collection(collection)


    chunks_dir = DEFAULT_CHUNKS_DIR
    
    retriever = MilvusManualRetriever(collection_name=DEFAULT_COLLECTION_NAME, client=client)

    emb_model = 
    retriever.build_index_from_chunks(chunks_dir=chunks_dir)

    query = "挂钩有什么注意事项"

    hits = retriever.retrieve(query=query, top_k=10)
    for i, hit in enumerate(hits, start=1):
        logger.info(
            f"[{i}] score={hit['score']:.4f}, chunk_id={hit['chunk_id']}, source={hit['source']}"
        )
