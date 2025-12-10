import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    get_linear_schedule_with_warmup,
)


class IntentDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[List[int]],
        tokenizer: BertTokenizerFast,
        max_len: int = 64,
    ) -> None:
        self.texts = texts
        self.labels = labels  # 每条样本是一个多热向量
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label_vec = self.labels[idx]  # List[int]，长度 = num_labels

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        # multi-label：float 类型，BCEWithLogitsLoss
        item["labels"] = torch.tensor(label_vec, dtype=torch.float)
        return item


def read_data(
    file: str,
) -> Tuple[List[str], List[List[str]], Dict[str, int]]:
    """
    从 merged_queries.json 读取数据（multi-label）：
    [
      {
        "query": "...",
        "sub_intent_id": ["manual_feature_usage", "xxx", ...],
        ...
      },
      ...
    ]
    """
    queries: List[str] = []
    intents: List[List[str]] = []
    all_intents: set[str] = set()

    with open(file, encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        query = item["query"]
        sub_intents = item["sub_intent_id"]

        queries.append(query)
        intents.append(sub_intents)
        all_intents.update(sub_intents)

    # 固定排序，保证 label_id 稳定
    label_map = {intent: idx for idx, intent in enumerate(sorted(all_intents))}
    return queries, intents, label_map


def build_label_vectors(
    intents: List[List[str]],
    label_map: Dict[str, int],
) -> List[List[int]]:
    """
    把 sub_intent_id 列表转成多热向量：
    例如 num_labels=5, 标签 ["a", "c"] -> [1,0,1,0,0]
    """
    num_labels = len(label_map)
    label_vectors: List[List[int]] = []

    for sub_intents in intents:
        vec = [0] * num_labels
        for intent in sub_intents:
            if intent not in label_map:
                continue
            vec[label_map[intent]] = 1
        if sum(vec) == 0:
            # 没有任何标签的样本可以选择丢弃或保留全 0，这里保留
            pass
        label_vectors.append(vec)

    return label_vectors


def train_val_split(
    texts: List[str],
    labels: List[List[int]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[List[int]], List[List[int]]]:
    indices = list(range(len(texts)))
    random.Random(seed).shuffle(indices)

    split_idx = int(len(indices) * (1 - val_ratio))
    # 样本太少就不划分验证集
    if split_idx <= 0 or split_idx >= len(indices):
        return texts, [], labels, []

    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    return train_texts, val_texts, train_labels, val_labels


def evaluate_f1(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> float:
    """
    简单的 micro-F1 评价，用于多标签。
    """
    if len(dataloader) == 0:
        return 0.0

    model.eval()
    tp = fp = fn = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]  # (B, C)，float
            outputs = model(**{k: v for k, v in batch.items() if k != "labels"})
            logits = outputs.logits  # (B, C)

            probs = logits.sigmoid()
            preds = (probs >= threshold).float()

            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(
    data_path: str = "output/merged_queries.json",
    model_name: str = "bert-large-uncased",
    output_dir: str = "output/intent_model",
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 2e-5,
    max_len: int = 64,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 读数据，构建 label_map
    queries, intents, label_map = read_data(data_path)
    label_vectors = build_label_vectors(intents, label_map)
    num_labels = len(label_map)
    print(f"样本数: {len(queries)}，类别数: {num_labels}")

    # 2. 划分训练 / 验证
    train_texts, val_texts, train_labels, val_labels = train_val_split(
        queries,
        label_vectors,
        val_ratio=val_ratio,
        seed=seed,
    )

    # 3. 加载 tokenizer 和模型（多标签）
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",  # 关键：启用多标签
    )
    model.to(device)

    # 4. 构建 DataLoader
    train_dataset = IntentDataset(train_texts, train_labels, tokenizer, max_len=max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if val_texts:
        val_dataset = IntentDataset(val_texts, val_labels, tokenizer, max_len=max_len)
        val_loader: DataLoader | None = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
    else:
        val_loader = None

    # 5. 优化器 + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # 6. 训练循环（loss 由 HF 内部用 BCEWithLogitsLoss 计算）
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            outputs = model(**batch)  # labels 是 (B, C) float
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))

        if val_loader is not None:
            f1 = evaluate_f1(model, val_loader, device)
            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"- train_loss: {avg_loss:.4f}, val_micro_f1: {f1:.4f}"
            )
        else:
            print(f"Epoch {epoch + 1}/{epochs} - train_loss: {avg_loss:.4f}")

    # 7. 保存模型和 label_map
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    label_map_path = os.path.join(output_dir, "label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"模型已保存到: {output_dir}")
    print(f"标签映射已保存到: {label_map_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于 bert-large-uncased 的多标签意图识别训练脚本",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="output/merged_queries.json",
        help="训练数据路径（merged_queries.json）。",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-large-uncased",
        help="预训练模型名，例如 bert-large-uncased。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/intent_model",
        help="训练结束后模型保存目录。",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="训练轮数。",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size。",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="学习率。",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=64,
        help="最大序列长度。",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="验证集占比 (0-1)。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        data_path=args.data_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_len=args.max_len,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
