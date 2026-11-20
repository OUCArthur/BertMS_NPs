#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于BERT向量的质谱数据集

使用预训练的BERT模型提取质谱特征，然后缓存特征向量
这样可以大大加快训练速度（因为BERT是冻结的）

Created by: BertMS Project
Date: 2025-01-24
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import pickle


class BertSpectrumDataset(Dataset):
    """
    基于BERT向量的质谱数据集

    策略：
    1. 加载GNPS数据
    2. 使用预训练的BERT模型提取每个质谱的特征向量
    3. 缓存特征向量到磁盘，避免重复计算
    4. 训练时直接使用缓存的特征向量
    """

    def __init__(
        self,
        data_path: str,
        bert_vectors_cache_path: Optional[str] = None,
        bert_model=None,
        tokenizer=None,
        device: str = 'cuda',
        force_recompute: bool = False
    ):
        """
        初始化数据集

        Args:
            data_path: GNPS数据JSON文件路径
            bert_vectors_cache_path: BERT向量缓存路径（.pkl文件）
            bert_model: 预训练的BERT模型（如果需要重新计算向量）
            tokenizer: Tokenizer（如果需要重新计算向量）
            device: 计算设备
            force_recompute: 强制重新计算BERT向量
        """
        self.data_path = data_path
        self.device = device

        # 加载数据
        print(f"加载数据: {data_path}")
        self.data = self._load_data(data_path)
        print(f"✓ 加载了 {len(self.data)} 个样本")

        # 构建标签映射
        self.label_to_idx, self.idx_to_label = self._build_label_mapping()
        print(f"✓ 类别数: {len(self.label_to_idx)}")
        print(f"✓ 类别分布: {self._get_class_distribution()}")

        # 设置缓存路径
        if bert_vectors_cache_path is None:
            # 自动生成缓存文件名
            data_name = Path(data_path).stem
            bert_vectors_cache_path = f"data/ms/{data_name}_bert_vectors.pkl"

        self.bert_vectors_cache_path = bert_vectors_cache_path

        # 加载或计算BERT向量
        self.bert_vectors = self._load_or_compute_bert_vectors(
            bert_model=bert_model,
            tokenizer=tokenizer,
            force_recompute=force_recompute
        )

        print(f"✓ BERT向量形状: {self.bert_vectors.shape}")

    def _load_data(self, data_path: str) -> List[Dict]:
        """加载GNPS数据"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 过滤有效数据
        valid_data = []
        for item in data:
            if 'peaks_json' in item and item['peaks_json']:
                if 'detailed_class' in item or 'library_class' in item:
                    valid_data.append(item)

        return valid_data

    def _build_label_mapping(self) -> Tuple[Dict, Dict]:
        """构建标签映射"""
        unique_labels = sorted(set(
            item.get('detailed_class', item.get('library_class')) for item in self.data
        ))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        return label_to_idx, idx_to_label

    def _get_class_distribution(self) -> Dict:
        """获取类别分布"""
        from collections import Counter
        labels = [item.get('detailed_class', item.get('library_class')) for item in self.data]
        return dict(Counter(labels))

    def _load_or_compute_bert_vectors(
        self,
        bert_model,
        tokenizer,
        force_recompute: bool = False
    ) -> torch.Tensor:
        """
        加载或计算BERT向量

        Args:
            bert_model: BERT模型
            tokenizer: Tokenizer
            force_recompute: 强制重新计算

        Returns:
            bert_vectors: [num_samples, hidden_size]
        """
        cache_path = Path(self.bert_vectors_cache_path)

        # 尝试从缓存加载
        if not force_recompute and cache_path.exists():
            print(f"从缓存加载BERT向量: {cache_path}")
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                bert_vectors = cached_data['bert_vectors']
                print(f"✓ 加载了 {bert_vectors.shape[0]} 个BERT向量")
                return bert_vectors

        # 需要计算BERT向量
        if bert_model is None or tokenizer is None:
            raise ValueError(
                f"缓存文件不存在: {cache_path}\n"
                "请提供 bert_model 和 tokenizer 来计算BERT向量"
            )

        print(f"计算BERT向量...")
        bert_vectors = self._compute_bert_vectors(bert_model, tokenizer)

        # 保存缓存
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"保存BERT向量缓存: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'bert_vectors': bert_vectors,
                'data_path': self.data_path,
                'num_samples': len(self.data)
            }, f)

        return bert_vectors

    def _compute_bert_vectors(self, bert_model, tokenizer) -> torch.Tensor:
        """
        使用BERT计算质谱特征向量

        Args:
            bert_model: BERT模型
            tokenizer: Tokenizer

        Returns:
            bert_vectors: [num_samples, hidden_size]
        """
        bert_model = bert_model.to(self.device)
        bert_model.eval()

        bert_vectors_list = []

        print("正在提取BERT特征向量...")
        with torch.no_grad():
            for item in tqdm(self.data, desc="BERT编码"):
                # 从peaks_json获取质谱数据
                peaks = item['peaks_json']

                # 这里需要将质谱转换为BERT输入
                # 暂时返回一个占位符，实际需要根据您的tokenizer实现
                # TODO: 实现质谱到BERT token的转换
                # input_ids, attention_mask = tokenizer.encode_spectrum(peaks)

                # 临时方案：创建一个全零向量作为占位符
                # 实际使用时需要替换为真实的BERT编码
                hidden_size = bert_model.config.hidden_size if hasattr(bert_model, 'config') else 768
                vector = torch.zeros(hidden_size, dtype=torch.float32)

                bert_vectors_list.append(vector)

        bert_vectors = torch.stack(bert_vectors_list, dim=0)
        return bert_vectors

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本

        Returns:
            {
                'bert_vector': [hidden_size],
                'label': scalar,
                'compound_name': str (可选)
            }
        """
        item = self.data[idx]

        # BERT特征向量
        bert_vector = self.bert_vectors[idx]

        # 标签
        label_name = item.get('detailed_class', item.get('library_class'))
        label = self.label_to_idx[label_name]

        return {
            'bert_vector': bert_vector,
            'label': torch.tensor(label, dtype=torch.long),
            'compound_name': item.get('compound_name', 'Unknown')
        }


def create_bert_dataloaders(
    data_path: str,
    bert_vectors_cache_path: Optional[str] = None,
    bert_model=None,
    tokenizer=None,
    batch_size: int = 64,
    val_split: float = 0.2,
    random_seed: int = 42,
    num_workers: int = 0,
    device: str = 'cuda',
    force_recompute: bool = False
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    创建训练和验证数据加载器

    Args:
        data_path: 数据文件路径
        bert_vectors_cache_path: BERT向量缓存路径
        bert_model: BERT模型
        tokenizer: Tokenizer
        batch_size: 批大小
        val_split: 验证集比例
        random_seed: 随机种子
        num_workers: DataLoader workers数量
        device: 计算设备
        force_recompute: 强制重新计算BERT向量

    Returns:
        (train_loader, val_loader, info_dict)
    """
    # 创建完整数据集
    full_dataset = BertSpectrumDataset(
        data_path=data_path,
        bert_vectors_cache_path=bert_vectors_cache_path,
        bert_model=bert_model,
        tokenizer=tokenizer,
        device=device,
        force_recompute=force_recompute
    )

    # 划分训练集和验证集
    from torch.utils.data import random_split

    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size

    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False
    )

    # 信息字典
    info = {
        'num_classes': len(full_dataset.label_to_idx),
        'label_to_idx': full_dataset.label_to_idx,
        'idx_to_label': full_dataset.idx_to_label,
        'class_distribution': full_dataset._get_class_distribution(),
        'train_size': train_size,
        'val_size': val_size,
        'bert_vector_dim': full_dataset.bert_vectors.shape[1]
    }

    print(f"\n数据集信息:")
    print(f"  训练集: {train_size} 样本")
    print(f"  验证集: {val_size} 样本")
    print(f"  类别数: {info['num_classes']}")
    print(f"  BERT向量维度: {info['bert_vector_dim']}")

    return train_loader, val_loader, info
